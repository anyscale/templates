import dsp
import dspy
import os
import dotenv
import traceback
import ray
import vllm
from vllm import LLM
from vllm.logger import init_logger
import dspy.dsp.deploy_dspy
from dspy.dsp.deploy_dspy.async_llm import AsyncLLMWrapper
from dspy.dsp.deploy_dspy.download import download_model
from transformers import AutoTokenizer
from dspy.evaluate import Evaluate
from concurrent.futures import ThreadPoolExecutor, Future
import time
import math
import tqdm
import pandas as pd
import types
import json
from dspy.dsp.utils.utils import deduplicate
from dspy.datasets import HotPotQA
from dotenv import load_dotenv
from IPython.display import HTML, display as ipython_display
from dspy.utils.display import configure_dataframe_display
from dspy.utils.utils import merge_dicts, truncate_cell
logger = init_logger(__name__)
 #, "env_vars": {"CUDA_VISIBLE_DEVICES": "0"}

class BasicMH(dspy.Module):
    def __init__(self, passages_per_hop=3, num_hops=2):
        super().__init__()
        self.num_hops = num_hops
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = [dspy.ChainOfThought("context, question -> search_query") for _ in range(self.num_hops)]
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = []
        
        for hop in range(self.num_hops):
            search_query = self.generate_query[hop](context=context, question=question).search_query
            passages = self.retrieve(search_query).passages
            context = deduplicate(context + passages)

        answer = self.generate_answer(context=context, question=question).copy(context=context)
        return answer

class DSPyActor:
    def __init__(self, batch_size=5, program=None, num_threads=1):
        dotenv.load_dotenv()
        model = "meta-llama/Meta-Llama-3-8B-Instruct"
        engine_args = {
            "max_pending_requests": 512,
            "enforce_eager": True,
            "engine_use_ray": False,
            "worker_use_ray": False,
            "enable_prefix_caching": True,
            "tensor_parallel_size": 1
        }
        self.lm = dspy.VLLMOfflineEngine.instantiate_with_llm(model=model, engine_args=engine_args)
        COLBERT_V2_ENDPOINT = "http://20.102.90.50:2017/wiki17_abstracts"
        self.rm = dspy.ColBERTv2(url=COLBERT_V2_ENDPOINT)
        self.num_threads = num_threads

        dspy.settings.configure(lm=self.lm, rm=self.rm, experimental=True)

        self.basic_pred = program
        self.thread_pool = ThreadPoolExecutor(max_workers=num_threads)
        self.batch_size = batch_size

    def __call__(self, inputs):
        # inputs is of form {"item": [example1, example2, ...]}
        results = {"results": []}
        futures: list[Future] = []

        for idx, question in enumerate(inputs["item"]):
            future = self.thread_pool.submit(self.process_batch, question, idx)
            futures.append(future)
        
        for future in futures:
            result = future.result()
            results["results"].append(result)

        self.lm.shutdown()
        return results

    def process_example(self, example, idx):
        # time.sleep(10 * idx)
        with dspy.context(lm=self.lm):
            try:
                pred = self.basic_pred(**example.inputs())
                return {"pred": pred}
            except Exception as e:
                print("error", traceback.print_exception(e))
                return {"question": example.inputs(), "error": str(e)}


class EvaluateRay(Evaluate):
    def __init__(self,
        *,
        devset,
        metric=None,
        concurrency=1,
        num_threads_per_worker=1,
        display_progress=False,
        display_table=False,
        max_errors=5,
        return_all_scores=False,
        return_outputs=False,
        **_kwargs,
        ):
        super().__init__(devset=devset, metric=metric, num_threads=num_threads_per_worker*concurrency, display_progress=display_progress, display_table=display_table, max_errors=max_errors, return_all_scores=return_all_scores, return_outputs=return_outputs, **_kwargs)

    def __call__(
        self,
        wrapped_program,
        metric=None,
        devset=None,
        concurrency=None,
        num_threads_per_worker=None,
        display_progress=None,
        display_table=None,
        return_all_scores=None,
        return_outputs=None,
    ):
        metric = metric if metric is not None else self.metric
        devset = devset if devset is not None else self.devset
        concurrency = concurrency if concurrency is not None else self.concurrency
        num_threads_per_worker = num_threads_per_worker if num_threads_per_worker is not None else self.num_threads_per_worker
        display_progress = display_progress if display_progress is not None else self.display_progress
        display_table = display_table if display_table is not None else self.display_table
        return_all_scores = return_all_scores if return_all_scores is not None else self.return_all_scores
        return_outputs = return_outputs if return_outputs is not None else self.return_outputs
        results = []

        

        devset = list(enumerate(devset))
        tqdm.tqdm._instances.clear()

        if num_threads_per_worker == 1:
            reordered_devset, ncorrect, ntotal = self._execute_single_thread(wrapped_program, devset, display_progress)
        else:
            # reordered_devset, ncorrect, ntotal = self._execute_multi_thread(
            #     wrapped_program,
            #     devset,
            #     num_threads,
            #     display_progress,
            # )
            ds = ray.data.from_items([x.inputs() for x in devset])
            batch_size = math.ceil(ds.count() / concurrency)
            results = ds.map_batches(DSPyActor,
                    batch_size=batch_size,
                    num_gpus=1,
                    concurrency=concurrency,
                    fn_constructor_kwargs={"batch_size": batch_size, "program": wrapped_program, "num_threads": num_threads_per_worker}
                ).take_all()

        dspy.logger.info(f"Average Metric: {ncorrect} / {ntotal} ({round(100 * ncorrect / ntotal, 1)}%)")

        predicted_devset = sorted(reordered_devset)

        if return_outputs:  # Handle the return_outputs logic
            results = [(example, prediction, score) for _, example, prediction, score in predicted_devset]

        data = [
            merge_dicts(example, prediction) | {"correct": score} for _, example, prediction, score in predicted_devset
        ]

        result_df = pd.DataFrame(data)

        # Truncate every cell in the DataFrame (DataFrame.applymap was renamed to DataFrame.map in Pandas 2.1.0)
        result_df = result_df.map(truncate_cell) if hasattr(result_df, "map") else result_df.applymap(truncate_cell)

        # Rename the 'correct' column to the name of the metric object
        metric_name = metric.__name__ if isinstance(metric, types.FunctionType) else metric.__class__.__name__
        result_df = result_df.rename(columns={"correct": metric_name})

        if display_table:
            if isinstance(display_table, bool):
                df_to_display = result_df.copy()
                truncated_rows = 0
            else:
                df_to_display = result_df.head(display_table).copy()
                truncated_rows = len(result_df) - display_table

            styled_df = configure_dataframe_display(df_to_display, metric_name)

            ipython_display(styled_df)

            if truncated_rows > 0:
                # Simplified message about the truncated rows
                message = f"""
                <div style='
                    text-align: center;
                    font-size: 16px;
                    font-weight: bold;
                    color: #555;
                    margin: 10px 0;'>
                    ... {truncated_rows} more rows not displayed ...
                </div>
                """
                ipython_display(HTML(message))

        if return_all_scores and return_outputs:
            return round(100 * ncorrect / ntotal, 2), results, [score for *_, score in predicted_devset]
        if return_all_scores:
            return round(100 * ncorrect / ntotal, 2), [score for *_, score in predicted_devset]
        if return_outputs:
            return round(100 * ncorrect / ntotal, 2), results

        return round(100 * ncorrect / ntotal, 2)
    
def main(dataset):
    # ds = ray.data.from_items([f"What is 1 + {num}" for num in range(10000)])
    start_idx = 0
    end_idx = 20
    
    # ds = ray.data.from_items([x.question for x in dataset[start_idx:end_idx]])
    LOAD_RESULTS = False

    if not LOAD_RESULTS:
        start = time.time()
        evaluate = EvaluateRay(devset=dataset, metric=dspy.evaluate.answer_exact_match)
        results = evaluate(program=BasicMH())

        end = time.time()
        print("results: ", results)
        print(f"Time taken: {end - start}")
        print(f"Total items: {len(results)}")
        print(f"Time taken per item: {(end - start) / len(results)}")
    #     with open(f"results_8b_{start_idx}_{end_idx}.json", "w") as f:
    #         # results is of form [{"results": {"question": ..., "answer": ...}}]
    #         # we need to flatten it to [{"question": ..., "answer": ...}]
    #         results = [result["results"] for result in results]
    #         json.dump(results, f)

    # if LOAD_RESULTS:
    #     results = json.load(open(f"results_8b_{start_idx}_{end_idx}.json", "r"))
    
    
    # QA_map = {x.question: x for x in dataset[start_idx:end_idx]}
    # predictions = [dspy.Prediction(answer=x["answer"], question=x["question"]) for x in results]
    # metric = dspy.evaluate.answer_exact_match
    # results = [metric(x, QA_map[x.question]) for x in predictions]
    # print("results: ", sum(results), "/", len(results), f"= {sum(results) / len(results)}")

if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env={"py_modules": [dspy, dsp], "env_vars": {"HF_HOME": "/mnt/local_storage/.cache/huggingface/", "HF_TOKEN": os.environ["HF_TOKEN"]}})

    dataset = HotPotQA(train_seed=1, eval_seed=2023, test_size=0, only_hard_examples=True)
    trainset = [x.with_inputs('question') for x in dataset.train]
    devset = [x.with_inputs('question') for x in dataset.dev]
    main(devset)