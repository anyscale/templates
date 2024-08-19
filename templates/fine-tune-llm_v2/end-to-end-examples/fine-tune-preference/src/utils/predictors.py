"""
Ray Data Actors for online and offline inference
"""

import os
from typing import TYPE_CHECKING, Union

from openai import OpenAI
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.utils.common import get_completion, init_logger
from src.utils.download import download_model
from src.utils.models import OfflineInferenceConfig, OnlineInferenceConfig

if TYPE_CHECKING:
    from ray.data import Dataset

logger = init_logger()

# TODO: See if this parameter can be removed entirely
VLLM_MAX_MODEL_LEN = 8192


class OfflinePredictor:
    def __init__(
        self,
        model_config: OfflineInferenceConfig,
        col_in: str,
        col_out: str,
    ):
        logger = init_logger()

        model_id_or_path = download_model(model_config.model_id_or_path)

        adapter_path = None
        if model_config.adapter_id_or_path:
            logger.info("Downloading LoRA:", model_config.adapter_id_or_path)
            adapter_path = download_model(model_config.adapter_id_or_path)

        self.col_in = col_in
        self.col_out = col_out
        self.lora_location = adapter_path
        self.vllm_settings = dict(
            tensor_parallel_size=model_config.scaling_config.num_gpus_per_instance,
            max_model_len=VLLM_MAX_MODEL_LEN,
            dtype="bfloat16",
        )

        # Create a sampling params object.
        self.sampling_params = SamplingParams(
            n=1,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            top_p=model_config.top_p,
            stop=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"],
        )

        llm_args = dict(model=model_id_or_path)

        if self.lora_location is not None:
            llm_args.update(
                dict(
                    enable_lora=True,
                    max_loras=1,
                    max_lora_rank=64,
                )
            )

        if self.vllm_settings is not None:
            llm_args.update(self.vllm_settings)

        # Create an LLM.
        self.llm = LLM(**llm_args)

    def __call__(self, batch):
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        if self.lora_location is not None:
            outputs = self.llm.generate(
                list(batch[self.col_in]),
                self.sampling_params,
                lora_request=LoRARequest("lora", 1, self.lora_location),
            )
        else:
            outputs = self.llm.generate(list(batch[self.col_in]), self.sampling_params)

        generated_text = []
        for i, output in enumerate(outputs):
            generated_text.append(output.outputs[0].text.strip())

        return {
            **batch,
            self.col_out: generated_text,
        }


class OnlinePredictor:
    def __init__(
        self,
        model_config: OnlineInferenceConfig,
        col_in: str,
        col_out: str,
    ):
        if not os.environ.get(model_config.api_key_env_var):
            raise ValueError(
                f"API Key must be set through {model_config.api_key_env_var}"
            )
        self.client = OpenAI(
            base_url=model_config.base_url,
            api_key=os.environ[model_config.api_key_env_var],
        )
        self.model = model_config.model_id
        self.col_in = col_in
        self.col_out = col_out
        self.temperature = model_config.temperature
        self.max_tokens = model_config.max_tokens

    def __call__(self, example):
        try:
            resp = get_completion(
                client=self.client,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=list(example[self.col_in]),
            )
            example[self.col_out] = resp.choices[0].message.content
        except Exception as e:
            logger.error(
                f"Error generating response:  {e} for input {example[self.col_in]}"
            )
            example[self.col_out] = None
        return example


def get_predictions_on_dataset(
    ds: "Dataset",
    model_config: Union[OnlineInferenceConfig, OfflineInferenceConfig],
    col_in: str,
    col_out: str,
):
    """Get predictions for a model on the given dataset using Ray data

    Supports online/offline inference given the model config.
    Args:
        ds: The input dataset
        model_config: Model inference config. Can be online/ offline.
        col_in: Input column in the dataset.
        col_out: Output column to write the results to.
    """
    if isinstance(model_config, OfflineInferenceConfig):
        ds = ds.map_batches(
            OfflinePredictor,
            fn_constructor_kwargs=dict(
                col_in=col_in,
                col_out=col_out,
                model_config=model_config,
            ),
            num_gpus=model_config.scaling_config.num_gpus_per_instance,
            concurrency=model_config.scaling_config.concurrency,
            batch_size=model_config.scaling_config.batch_size,
            accelerator_type=model_config.scaling_config.accelerator_type,
            zero_copy_batch=True,
            batch_format="numpy",
        )
    else:
        ds = ds.map(
            OnlinePredictor,
            fn_constructor_kwargs=dict(
                col_in=col_in,
                col_out=col_out,
                model_config=model_config,
            ),
            concurrency=model_config.concurrency,
        )
    return ds
