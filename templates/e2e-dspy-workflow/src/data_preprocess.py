import ujson
from dspy.evaluate import answer_exact_match
import dspy

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

def valid_label_metric(example, prediction, trace=None, frac=1.0):
    if prediction.label not in label_to_int_dict:
        return False
    return True

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
def filter_to_top_n_labels(trainset, testset, n=25):
    from collections import Counter

    label_counts = Counter(example['label'] for example in trainset)

    # get the top 25 labels sorted by name for any ties
    top_25_labels = sorted(label_counts.keys(), key=lambda x: (-label_counts[x], x))[:n]

    # Filter the datasets to only include examples with the top 25 labels
    trainset_filtered = [example for example in trainset if example['label'] in top_25_labels]
    testset_filtered = [example for example in testset if example['label'] in top_25_labels]
    return trainset_filtered, testset_filtered, top_25_labels
