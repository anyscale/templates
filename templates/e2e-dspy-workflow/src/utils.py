import ujson
import dspy
import dsp
from dspy.evaluate import answer_exact_match
import os
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
import litellm

litellm.set_verbose=False
litellm.suppress_debug_info=True

# -------------------------------
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------
def delete_labels(dataset):
    for example in dataset:
        if "label" in example:
            del example["label"]
    return dataset

def read_jsonl(filename):
    with open(filename, "r") as f:
        return [ujson.loads(line) for line in f]

def write_jsonl(filename, data):
    with open(filename, "w") as f:
        for item in data:
            f.write(ujson.dumps(item) + "\n")

# -------------------------------
def adjusted_exact_match(example, pred, trace=None, frac=1.0):
    example.answer = example.label
    pred.answer = pred.label
    return answer_exact_match(example, pred, trace, frac)

metric = adjusted_exact_match
# -------------------------------
def set_random_seed(seed = 0):
    import random

    rng = random.Random(seed)
    return rng

# -------------------------------
NUM_THREADS = 300
common_kwargs = dict(metric=adjusted_exact_match, num_threads=NUM_THREADS, display_progress=True, max_errors=10000)

# -------------------------------
int_to_label_dict = {
    0: "activate_my_card",
    1: "age_limit",
    2: "apple_pay_or_google_pay",
    3: "atm_support",
    4: "automatic_top_up",
    5: "balance_not_updated_after_bank_transfer",
    6: "balance_not_updated_after_cheque_or_cash_deposit",
    7: "beneficiary_not_allowed",
    8: "cancel_transfer",
    9: "card_about_to_expire",
    10: "card_acceptance",
    11: "card_arrival",
    12: "card_delivery_estimate",
    13: "card_linking",
    14: "card_not_working",
    15: "card_payment_fee_charged",
    16: "card_payment_not_recognised",
    17: "card_payment_wrong_exchange_rate",
    18: "card_swallowed",
    19: "cash_withdrawal_charge",
    20: "cash_withdrawal_not_recognised",
    21: "change_pin",
    22: "compromised_card",
    23: "contactless_not_working",
    24: "country_support",
    25: "declined_card_payment",
    26: "declined_cash_withdrawal",
    27: "declined_transfer",
    28: "direct_debit_payment_not_recognised",
    29: "disposable_card_limits",
    30: "edit_personal_details",
    31: "exchange_charge",
    32: "exchange_rate",
    33: "exchange_via_app",
    34: "extra_charge_on_statement",
    35: "failed_transfer",
    36: "fiat_currency_support",
    37: "get_disposable_virtual_card",
    38: "get_physical_card",
    39: "getting_spare_card",
    40: "getting_virtual_card",
    41: "lost_or_stolen_card",
    42: "lost_or_stolen_phone",
    43: "order_physical_card",
    44: "passcode_forgotten",
    45: "pending_card_payment",
    46: "pending_cash_withdrawal",
    47: "pending_top_up",
    48: "pending_transfer",
    49: "pin_blocked",
    50: "receiving_money",
    51: "refund_not_showing_up",
    52: "request_refund",
    53: "reverted_card_payment",
    54: "supported_cards_and_currencies",
    55: "terminate_account",
    56: "top_up_by_bank_transfer_charge",
    57: "top_up_by_card_charge",
    58: "top_up_by_cash_or_cheque",
    59: "top_up_failed",
    60: "top_up_limits",
    61: "top_up_reverted",
    62: "topping_up_by_card",
    63: "transaction_charged_twice",
    64: "transfer_fee_charged",
    65: "transfer_into_account",
    66: "transfer_not_received_by_recipient",
    67: "transfer_timing",
    68: "unable_to_verify_identity",
    69: "verify_my_identity",
    70: "verify_source_of_funds",
    71: "verify_top_up",
    72: "virtual_card_not_working",
    73: "visa_or_mastercard",
    74: "why_verify_identity",
    75: "wrong_amount_of_cash_received",
    76: "wrong_exchange_rate_for_cash_withdrawal"
}

label_to_int_dict = {v: k for k, v in int_to_label_dict.items()}
def preprocess_data(trainset, testset):
    def convert_int_to_label(example):
        example["label"] = int_to_label_dict[example["label"]]
        return example

    trainset_processed = [convert_int_to_label(example) for example in trainset]
    testset_processed = [convert_int_to_label(example) for example in testset]
    return trainset_processed, testset_processed

def load_data_from_huggingface():
    from dspy.datasets import DataLoader

    dl = DataLoader()
    full_trainset = dl.from_huggingface(
        dataset_name="PolyAI/banking77", # Dataset name from Huggingface
        fields=("label", "text"), # Fields needed
        input_keys=("text",), # What our model expects to recieve to generate an output
        split="train"
    )

    full_testset = dl.from_huggingface(
        dataset_name="PolyAI/banking77", # Dataset name from Huggingface
        fields=("label", "text"), # Fields needed
        input_keys=("text",), # What our model expects to recieve to generate an output
        split="test"
    )
    return full_trainset, full_testset

# -------------------------------
def init_ray():
    import ray

    ray.init(runtime_env={"env_vars": os.environ, "py_modules": [dspy, dsp]})

# -------------------------------
def set_dspy_cache():
    cache_dir = "/home/ray/default/dspy/cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # I have included a .env.example with the necessary environment variables to be set
    # You can also set them manually if you prefer

    os.environ["DSP_CACHEDIR"] = cache_dir

# -------------------------------
def check_env_vars():
    necessary_env_vars = [
        "DSP_CACHEDIR",
        "HF_TOKEN",
        "HF_HOME"
    ]

    for var in necessary_env_vars:
        assert os.environ[var], f"{var} is not set"

# -------------------------------
def filter_to_top_n_labels(trainset, testset, n=25):
    from collections import Counter

    label_counts = Counter(example['label'] for example in trainset)

    # get the top 25 labels sorted by name for any ties
    top_25_labels = sorted(label_counts.keys(), key=lambda x: (-label_counts[x], x))[:25]

    # Filter the datasets to only include examples with the top 25 labels
    trainset_filtered = [example for example in trainset if example['label'] in top_25_labels]
    testset_filtered = [example for example in testset if example['label'] in top_25_labels]
    return trainset_filtered, testset_filtered, top_25_labels

# -------------------------------
MAX_TOKENS = 1000
MODEL_PARAMETERS = {
  "max_tokens": MAX_TOKENS,
  "temperature": 0,
}

LOCAL_API_PARAMETERS = {
  "api_base": "http://localhost:8000/v1",
  "api_key": "fake-key-doesnt-matter"
}

# -------------------------------
def sanity_check_program(model, program, item):
    with dspy.context(lm=model):
        sample_input = item
        print(f"Program input: {sample_input}")
        print(f"Program output label: {program(**sample_input.inputs()).label}")

# -------------------------------

# Optimization hyperparameters

# Define the hyperparameters for prompt optimization
bootstrap_fewshot_random_search_parameters = {
    "max_bootstrapped_demos": 3,
    "max_labeled_demos": 3,
    "num_candidate_programs": 6,
}

# -------------------------------
def get_llama_lms_from_model_names(model_names):
    llama_1b = dspy.LM(model="openai/meta-llama/Llama-3.2-1B-Instruct", **LOCAL_API_PARAMETERS, **MODEL_PARAMETERS)
    finetuned_llamas_1b = {f: dspy.LM(model="openai/" + f, **LOCAL_API_PARAMETERS, **MODEL_PARAMETERS) for f in model_names}
    all_llamas = {**finetuned_llamas_1b, "base": llama_1b}
    return all_llamas

finetuning_kwargs = {
    "hyperparameters": {
        "num_devices": 4,
        "trainer_resources": None,
        "worker_resources": None,
        "generation_config": {
            "prompt_format": {
                "system": "<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>",
                "user": "<|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|>",
                "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{instruction}<|eot_id|>",
                "trailing_assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "bos": "<|begin_of_text|>",
                "system_in_user": False,
                "default_system_message": ""
            },
        },
        "learning_rate": 3e-5,
        "num_epochs": 6,
        "train_batch_size_per_device": 32
    },
    "use_lora": True,
}

def split_into_devset_and_optimizer_sets(collected_data_examples, dev_size, optimizer_num_val):

    devset_synthetic = collected_data_examples[:dev_size]
    ft_optimizer_devset = collected_data_examples[dev_size:dev_size+optimizer_num_val]
    ft_optimizer_trainset = collected_data_examples[dev_size+optimizer_num_val:]
    return devset_synthetic, ft_optimizer_trainset, ft_optimizer_devset

def evaluate_and_prompt_optimize(devset, optimizer_trainset, optimizer_valset, program, models, metric, labels_in_use):
    COMPILE_PROGRAM = True
    MAX_BOOTSTRAPPED_DEMOS = bootstrap_fewshot_random_search_parameters["max_bootstrapped_demos"]
    MAX_LABELED_DEMOS = bootstrap_fewshot_random_search_parameters["max_labeled_demos"]
    NUM_CANDIDATE_PROGRAMS = bootstrap_fewshot_random_search_parameters["num_candidate_programs"]
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
            ft_results[folder]["bfrs"] = {"devset": llama_8b_bfrs_finetuned_eval, "true_labels": None, "testset": None}
            print(f"result for {folder}: {llama_8b_bfrs_finetuned_eval}, None, None")

    return ft_results

def run_testset_evaluation(ft_results, all_llamas, labels_in_use, testset):
    MAX_BOOTSTRAPPED_DEMOS = bootstrap_fewshot_random_search_parameters["max_bootstrapped_demos"]
    MAX_LABELED_DEMOS = bootstrap_fewshot_random_search_parameters["max_labeled_demos"]
    NUM_CANDIDATE_PROGRAMS = bootstrap_fewshot_random_search_parameters["num_candidate_programs"]
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

def graph_testset_results(ft_results):
    # Prepare data for the graph
    models = []
    vanilla_testset = []
    bfrs_testset = []

    for model, results in ft_results.items():
        if model == "base":
            models.append("Base Model")
        else:
            models.append(f"Epoch {model.split(':')[1].split('-')[1]}")
        vanilla_testset.append(results['vanilla'].get('testset', None))
        bfrs_testset.append(results['bfrs'].get('testset', None))
    
    vanilla_testset = [x if x is not None else 0 for x in vanilla_testset]
    bfrs_testset = [x if x is not None else 0 for x in bfrs_testset]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35

    # Plot bars
    vanilla_bars = ax.bar(x - width/2, vanilla_testset, width, label='No Prompt Optimization', color='skyblue')
    bfrs_bars = ax.bar(x + width/2, bfrs_testset, width, label='BootstrapFewShotRandomSearch', color='lightgreen')

    # Add value labels on top of each bar
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

    add_labels(vanilla_bars)
    add_labels(bfrs_bars)

    # Customize the plot
    ax.set_ylabel('Test Set Scores')
    ax.set_title('Model Performance Comparison (Real Test Set; N=1000)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def graph_devset_results(ft_results):
    models = []
    vanilla_devset = []
    bfrs_devset = []

    for model, results in ft_results.items():
        if model == "base":
            models.append("base")
        else:
            models.append("Epoch " + model.split(':')[1].split('-')[1])  # Extract epoch information
        vanilla_devset.append(results['vanilla']['devset'])
        bfrs_devset.append(results['bfrs']['devset'])

    # Sort the data by epoch, keeping "base" at the beginning
    sorted_data = sorted(zip(models, vanilla_devset, bfrs_devset),
                         key=lambda x: (x[0] != "base", x[0]))
    models, vanilla_devset, bfrs_devset = zip(*sorted_data)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Adjust bar positions and width
    x = np.arange(len(models))
    width = 0.35

    # Plot bars for Dev Set
    vanilla_bars = ax.bar(x - width/2, vanilla_devset, width, label='No Prompt Optimization', color='skyblue')
    bfrs_bars = ax.bar(x + width/2, bfrs_devset, width, label='BootstrapFewShotRandomSearch', color='lightgreen')

    # Add value labels on top of each bar
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

    add_labels(vanilla_bars)
    add_labels(bfrs_bars)

    # Customize the plot
    ax.set_ylabel('Dev Set Scores')
    ax.set_title('Model Performance Comparison Across Epochs (Synthetic Dev Set; N=1000)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Find the highest devset score and its corresponding model
    highest_devset_score = max(bfrs_devset)
    highest_score_model = models[bfrs_devset.index(highest_devset_score)]

    print(f"Highest Dev Set Score: {highest_devset_score:.1f}, Model: {highest_score_model}")
# -------------------------------

class IntentClassification(dspy.Signature):
    """As a part of a banking issue traiging system, classify the intent of a natural language query into one of the 25 labels.
    The intent should exactly match one of the following:
    ['card_payment_fee_charged', 'direct_debit_payment_not_recognised', 'balance_not_updated_after_cheque_or_cash_deposit', 'wrong_amount_of_cash_received', 'cash_withdrawal_charge', 'transaction_charged_twice', 'declined_cash_withdrawal', 'transfer_fee_charged', 'balance_not_updated_after_bank_transfer', 'transfer_not_received_by_recipient', 'request_refund', 'card_payment_not_recognised', 'card_payment_wrong_exchange_rate', 'extra_charge_on_statement', 'wrong_exchange_rate_for_cash_withdrawal', 'refund_not_showing_up', 'reverted_card_payment', 'cash_withdrawal_not_recognised', 'activate_my_card', 'pending_card_payment', 'cancel_transfer', 'beneficiary_not_allowed', 'card_arrival', 'declined_card_payment', 'pending_top_up']
    """

    intent = dspy.InputField(desc="Intent of the query")
    label = dspy.OutputField(desc="Type of the intent; Should just be one of the 25 labels with no other text")

class IntentClassificationModule(dspy.Module):
    def __init__(self, labels_in_use):
        self.intent_classifier = dspy.ChainOfThought(IntentClassification)
        self.valid_labels = set(labels_in_use)

    def forward(self, text):
        prediction = self.intent_classifier(intent=text)
        sanitized_prediction = dspy.Prediction(label=prediction.label.lower().strip().replace(" ", "_"), reasoning=prediction.reasoning)
        return sanitized_prediction