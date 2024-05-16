"""
Test a fine-tuned model against GSM8k test set.

This script assumes a local endpoint with a model `ADAPTER_NAME` fine-tuned with GSM8k.
You can set up a local endpoint with the serving template and code to start from in the LoRA example section of it.
You need to provide your own ADAPTER_NAME that you used to set up the local endpoint with the fine-tuned model.

You can also use this script to query Anyscale's Endpoints API instead if you fine-tuned there, in which case you are required to use your own key.
"""

from openai import OpenAI
import ray
import numpy as np

ADAPTER_NAME = None # Put the name of the adapter you want to test here
DATASET = "s3://air-example-data/gsm8k/test.jsonl"
ENDPOINTS_URL = "http://127.0.0.1:8000/m/v1"
ENDPOINTS_KEY = "put key here if needed" # We don't need a key here since we query localhost


def batched_process_fn(batch):
    client = OpenAI(
        base_url=ENDPOINTS_URL,
        api_key=ENDPOINTS_KEY
    )

    # Sanity check: This function only looks at the first set of messages per batch
    assert len(batch) == 1
    messages = batch["messages"][0]
    prompt = [messages[0]]
    label = messages[1]["content"]
    model_output = client.chat.completions.create(
        model=ADAPTER_NAME,
        messages=prompt
    )
    model_output = model_output.choices[0].message.content

    batch["outputs"] = np.array([model_output])

    # Go through the model outputs, looking for the `####` that signifies the answer in the GSM8k dataset
    success = False
    idx = model_output.find("####")
    if idx != -1:
        # 5 is the length of the "####" token.
        actualoutput = model_output[idx+5:]
        actualanswer = label[label.find("####")+5:]
        if actualanswer == actualoutput:
            success = True

    batch["successes"] = np.array([success])
    return batch

ds = ray.data.read_json(DATASET)
ds = ds.repartition(ds.count())

ds = ds.map_batches(
    batched_process_fn,
    concurrency=20,
    # Use batch size of 1 so that each batch is a single sample
    batch_size=1,
    num_cpus=0.5,
)

successes_dict = ds.take_all()

successes = failures = 0
for elem in successes_dict:
    if elem["successes"] == True:
        successes += 1
    else:
        failures += 1

print("Results:")
print("Total samples: ", successes + failures)
print("Successes: ", successes)
print("Failures: ", failures)
print("Success rate: ", successes / (successes + failures))
