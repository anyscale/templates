import ujson
from dspy.evaluate import answer_exact_match
from src.constants import INT_TO_LABEL_DICT, LABEL_TO_INT_DICT, NUM_THREADS
from collections import Counter
import random


def get_valid_label_metric_fn(labels_in_use):
    def valid_label_metric(example, prediction, trace=None, frac=1.0):
        if prediction.label not in labels_in_use:
            return False
        return True
    return valid_label_metric

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

def adjusted_exact_match(example, pred, trace=None, frac=1.0):
    example.answer = example.label
    pred.answer = pred.label
    return answer_exact_match(example, pred, trace, frac)

def set_random_seed(seed = 0):

    rng = random.Random(seed)
    return rng

def convert_int_label_to_string(trainset, testset):
    def convert_int_to_label(example):
        example["label"] = INT_TO_LABEL_DICT[example["label"]]
        return example

    trainset_processed = [convert_int_to_label(example) for example in trainset]
    testset_processed = [convert_int_to_label(example) for example in testset]
    return trainset_processed, testset_processed

def load_data_from_huggingface():
    from dspy.datasets import DataLoader

    dl = DataLoader()
    full_trainset = dl.from_huggingface(
        dataset_name="PolyAI/banking77",
        fields=("label", "text"),
        input_keys=("text",),
        split="train"
    )

    full_testset = dl.from_huggingface(
        dataset_name="PolyAI/banking77",
        fields=("label", "text"),
        input_keys=("text",),
        split="test"
    )
    return full_trainset, full_testset

def filter_to_top_n_labels(trainset, testset, n=25):

    label_counts = Counter(example['label'] for example in trainset)

    # get the top 25 labels sorted by name for any ties
    top_25_labels = sorted(label_counts.keys(), key=lambda x: (-label_counts[x], x))[:n]

    # Filter the datasets to only include examples with the top 25 labels
    trainset_filtered = [example for example in trainset if example['label'] in top_25_labels]
    testset_filtered = [example for example in testset if example['label'] in top_25_labels]
    return trainset_filtered, testset_filtered, top_25_labels
