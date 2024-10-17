import dspy
import dsp
import os

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

# -------------------------------
import litellm

litellm.set_verbose=False
litellm.suppress_debug_info=True

# -------------------------------
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------
def init_ray():
    import ray

    ray.init(runtime_env={"env_vars": os.environ, "py_modules": [dspy, dsp]})

# -------------------------------
def set_dspy_cache_location(local_cache_dir=None):
    cache_dir = local_cache_dir if local_cache_dir is not None else "/home/ray/default/dspy/cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

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
