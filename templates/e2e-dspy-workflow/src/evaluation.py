import dspy
from src.constants import MAX_BOOTSTRAPPED_DEMOS, MAX_LABELED_DEMOS, NUM_CANDIDATE_PROGRAMS, NUM_THREADS, MAX_ERRORS
import yaml
import os

def split_into_devset_and_optimizer_sets(collected_data_examples, dev_size, optimizer_num_val):
    devset_synthetic = collected_data_examples[:dev_size]
    ft_optimizer_devset = collected_data_examples[dev_size:dev_size+optimizer_num_val]
    ft_optimizer_trainset = collected_data_examples[dev_size+optimizer_num_val:]
    return devset_synthetic, ft_optimizer_trainset, ft_optimizer_devset

def evaluate_and_prompt_optimize(devset, optimizer_trainset, optimizer_valset, module_class, models, metric, labels_in_use):
    ft_results = {}
    for folder, llama in models.items():
        optimizer = dspy.BootstrapFewShotWithRandomSearch(metric=metric, max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS, max_labeled_demos=MAX_LABELED_DEMOS, num_candidate_programs=NUM_CANDIDATE_PROGRAMS, num_threads=NUM_THREADS, max_errors=MAX_ERRORS)
        print("Evaluating", llama.model)
        ft_results[folder] = {}
        with dspy.context(lm=llama):
            evaluate_devset = dspy.Evaluate(devset=devset, metric=metric, num_threads=NUM_THREADS, display_progress=False, max_errors=MAX_ERRORS)

            vanilla_program = module_class(labels_in_use)
            devset_result = evaluate_devset(vanilla_program)
            ft_results[folder]["vanilla"] = {"devset": devset_result}

            bfrs_finetuned_program = optimizer.compile(vanilla_program, trainset=optimizer_trainset, valset=optimizer_valset)
            bfrs_finetuned_program.save(f"simpleintent_1b_32_ft_bfrs_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}_{folder.split('/')[-1]}.json")

            llama_8b_bfrs_finetuned_eval = evaluate_devset(bfrs_finetuned_program)
            ft_results[folder]["bfrs"] = {"devset": llama_8b_bfrs_finetuned_eval}
            print(f"Evaluation result for {folder} on devset: {llama_8b_bfrs_finetuned_eval}")

    return ft_results

def run_testset_evaluation(ft_results, all_llamas, labels_in_use, testset, metric, module_class):
    best_non_base_model = max([x for x in ft_results.keys() if x != "base"], key=lambda x: ft_results[x]["bfrs"]["devset"])
    print("Best non-base model:", best_non_base_model)
    base_and_best = {"base": all_llamas["base"], best_non_base_model: all_llamas[best_non_base_model]}
    best_program_path, best_model, best_score = None, None, 0
    evaluate_testset = dspy.Evaluate(devset=testset, metric=metric, num_threads=NUM_THREADS, display_progress=False, max_errors=MAX_ERRORS)
    for folder, llama in base_and_best.items():
        print("Evaluating", folder)
        vanilla_program = module_class(labels_in_use)

        with dspy.context(lm=llama):
            testset_result_vanilla = evaluate_testset(vanilla_program)
            if testset_result_vanilla > best_score:
                best_score = testset_result_vanilla
                best_program_path = program_path
                best_model = folder

            ft_results[folder]["vanilla"]["testset"] = testset_result_vanilla
            program_path = f"simpleintent_1b_32_ft_bfrs_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}_{folder.split('/')[-1]}.json"
            vanilla_program.load(program_path)
            testset_result = evaluate_testset(vanilla_program)
            if testset_result > best_score:
                best_score = testset_result
                best_program_path = program_path
                best_model = folder
            ft_results[folder]["bfrs"]["testset"] = testset_result

    return ft_results, (best_program_path, best_model, best_score)

def update_serve_config_hf_token(serve_config_path: str):
    """Helper function to update the provided serve config with the current HF_TOKEN env variable"""
    with open(serve_config_path, "r") as f:
        serve_config = yaml.safe_load(f)

    model_config_location = serve_config["applications"][0]["args"]["llm_configs"][0]

    with open(model_config_location, "r") as f:
        model_config = yaml.safe_load(f)

    if not os.environ.get("HUGGING_FACE_HUB_TOKEN") and not os.environ.get("HF_TOKEN"):
        raise ValueError("HUGGING_FACE_HUB_TOKEN or HF_TOKEN must be set")
    model_config["runtime_env"]["env_vars"]["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    with open(model_config_location, "w") as f:
        yaml.safe_dump(model_config, f)
