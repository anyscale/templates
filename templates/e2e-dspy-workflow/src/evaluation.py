import dspy
from src import IntentClassificationModule

bootstrap_fewshot_random_search_parameters = {
    "max_bootstrapped_demos": 3,
    "max_labeled_demos": 3,
    "num_candidate_programs": 6,
}
MAX_BOOTSTRAPPED_DEMOS = bootstrap_fewshot_random_search_parameters["max_bootstrapped_demos"]
MAX_LABELED_DEMOS = bootstrap_fewshot_random_search_parameters["max_labeled_demos"]
NUM_CANDIDATE_PROGRAMS = bootstrap_fewshot_random_search_parameters["num_candidate_programs"]

def split_into_devset_and_optimizer_sets(collected_data_examples, dev_size, optimizer_num_val):

    devset_synthetic = collected_data_examples[:dev_size]
    ft_optimizer_devset = collected_data_examples[dev_size:dev_size+optimizer_num_val]
    ft_optimizer_trainset = collected_data_examples[dev_size+optimizer_num_val:]
    return devset_synthetic, ft_optimizer_trainset, ft_optimizer_devset

def evaluate_and_prompt_optimize(devset, optimizer_trainset, optimizer_valset, program, models, metric, labels_in_use):
    MAX_ERRORS = 1000

    optimizer = dspy.BootstrapFewShotWithRandomSearch(metric=metric, max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS, max_labeled_demos=MAX_LABELED_DEMOS, num_candidate_programs=NUM_CANDIDATE_PROGRAMS, num_threads=NUM_THREADS, max_errors=MAX_ERRORS)
    ft_results = {}
    for folder, llama in models.items():
        print("Evaluating", llama.model)
        ft_results[folder] = {}
        with dspy.context(lm=llama):
            evaluate_devset = dspy.Evaluate(devset=devset, metric=metric, num_threads=NUM_THREADS, display_progress=True, max_errors=MAX_ERRORS)
            
            vanilla_program = IntentClassificationModule(labels_in_use)
            devset_result = evaluate_devset(vanilla_program)
            ft_results[folder]["vanilla"] = {"devset": devset_result}

            bfrs_finetuned_program = optimizer.compile(vanilla_program, trainset=optimizer_trainset, valset=optimizer_valset)
            bfrs_finetuned_program.save(f"simpleintent_1b_32_ft_bfrs_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}_{folder.split('/')[-1]}.json")
            
            llama_8b_bfrs_finetuned_eval = evaluate_devset(bfrs_finetuned_program)
            ft_results[folder]["bfrs"] = {"devset": llama_8b_bfrs_finetuned_eval}
            print(f"Evaluation result for {folder} on devset: {llama_8b_bfrs_finetuned_eval}")

    return ft_results

def run_testset_evaluation(ft_results, all_llamas, labels_in_use, testset):
    MAX_ERRORS = 10000

    best_non_base_model = max([x for x in ft_results.keys() if x != "base"], key=lambda x: ft_results[x]["bfrs"]["devset"])
    print("Best non-base model:", best_non_base_model)
    base_and_best = {"base": all_llamas["base"], best_non_base_model: all_llamas[best_non_base_model]}

    evaluate_testset = dspy.Evaluate(devset=testset, metric=metric, num_threads=NUM_THREADS, display_progress=True, max_errors=MAX_ERRORS)
    for folder, llama in base_and_best.items():
        print("Evaluating", folder)
        vanilla_program = IntentClassificationModule(labels_in_use)
        
        with dspy.context(lm=llama):
            testset_result_vanilla = evaluate_testset(vanilla_program)
            ft_results[folder]["vanilla"]["testset"] = testset_result_vanilla
            vanilla_program.load(f"simpleintent_1b_32_ft_bfrs_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}_{folder.split('/')[-1]}.json")
            testset_result = evaluate_testset(vanilla_program)
            ft_results[folder]["bfrs"]["testset"] = testset_result

    return ft_results