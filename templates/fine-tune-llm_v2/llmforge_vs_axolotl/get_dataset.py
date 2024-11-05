import datasets
import json
def messages_to_jsonl(messages, output_file):
    """
    Convert a list of message pairs into JSONL format and write to a file.
    Each line will be wrapped in a "messages" field while preserving the original format.

    Args:
        messages (list): List of message pairs (user and assistant messages)
        output_file (str): Path to the output JSONL file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for message_pair in messages:
            # Create a wrapper with the "messages" field containing the original format
            conversation = {
                "messages": message_pair
            }
            # Write the JSON object as a line in the file
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

data = datasets.load_dataset("eatang/alpaca_long_subset")["train"]
messages_to_jsonl(data["messages"], "alpaca_long_subset.jsonl")
