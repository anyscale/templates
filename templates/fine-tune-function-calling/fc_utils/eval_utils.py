from typing import Union

from fc_utils.response_parsers import (
    ERROR_OUTPUT,
    AnyscaleResponseParser,
    OpenAIResponseParser,
)


def is_match(response: dict, ground_truth: dict):
    """
    Checks if the response matches the ground truth.
    """
    if ground_truth["tool_calls"] is None:
        if response["tool_calls"] is None:
            if len(ground_truth["content"]):
                return len(response["content"]) > 0  # non zero content
            return True
        else:  # explicit else for clarity
            return False

    if response["tool_calls"] is None:
        return False
    if len(response["tool_calls"]) != len(ground_truth["tool_calls"]):
        return False
    for expected_tool_call, actual_tool_call in zip(ground_truth["tool_calls"], response["tool_calls"]):
        if expected_tool_call != actual_tool_call:
            return False


def parse_and_eval(
    parser: Union[OpenAIResponseParser, AnyscaleResponseParser],
    user_messages,
    expected_responses,
    tools=None,
):
    """
    Parse and eval loop to parse the assistant responses and evaluate them against the ground truth in the conversation.
    This assumes that an assistant response is expected after every user/tool message.
    Args:
    parser: OpenAIResponseParser or AnyscaleResponseParser object
    user_messages: list of user messages to send to the chat completion model (contains system message and tool responses if any)
    expected_responses: list of ground truth responses
    tools: list of tools to available, used by OpenAIResponseParser
    """
    assert user_messages[0]["role"] == "system", "First message must be from system"
    match = True
    current_conv = user_messages[:1]
    for i in range(1, len(user_messages)):
        current_conv.append(user_messages[i])
        if current_conv[-1]["role"] == "tool":
            # For a tool, we need to have a valid tool call id. This can only be retrieved from the previous assistant response
            # because our dataset is synthetically constructed
            if (
                current_conv[-2]["role"] == "assistant"
                and current_conv[-2]["tool_calls"]
            ):
                current_conv[-1]["tool_call_id"] = current_conv[-2]["tool_calls"][0].id
            else:
                return current_conv, False
        if isinstance(parser, OpenAIResponseParser):
            parsed_response = parser.get_parsed_response(current_conv, tools)
        else:
            parsed_response = parser.get_parsed_response(current_conv)
        if parsed_response["content"] == ERROR_OUTPUT:
            return None, None  # return None if there's an error
        match = match and (is_match(parsed_response, expected_responses[i - 1]))
        original_assistant_response = parsed_response["original_response"]
        current_conv.append(dict(original_assistant_response))
        if not match:  # return right away if model output is incorrect
            return current_conv, match
    return current_conv, match


def evaluate_gpt4(ds):
    openai_accuracy = 0
    pbar = tqdm(total=len(modified_ds), desc="Evaluating GPT4..")
    for example in ds:
        openai_messages = example["openai_messages"]
        openai_tools = [
            {"type": "function", "function": fn} for fn in example["openai_functions"]
        ]
        openai_conv, openai_is_match = parse_and_eval(
            openai_parser, openai_messages, example["expected_responses"], tools=openai_tools
        )
        if openai_conv is None:  # skip if api errors out
            continue
        if openai_is_match:
            openai_accuracy += 1
    return openai_accuracy

if __name__ == "__main__":
    # Evaluation code in full
    import pickle

    import datasets
    from tqdm import tqdm

    OPENAI_API_KEY = "yourKey"
    ANYSCALE_API_KEY = "yourKey"
    # evaluate gpt-4
    openai_parser = OpenAIResponseParser(
        api_key=OPENAI_API_KEY, api_base="https://api.openai.com/v1", model="gpt-4"
    )
    anyscale_parser = AnyscaleResponseParser(
        api_key=ANYSCALE_API_KEY,
        api_base="https://serve-session-ljyv94qdghjth7ldigbzfhhjxx.i.anyscaleuserdata.com/v1",
        model="meta-llama/Meta-Llama-3-8B-Instruct:glaiveai_v1:1234",
    )  # make sure to not add a stray / at the end!

    from fc_utils.response_parsers import AnyscaleResponseParser, OpenAIResponseParser

    modified_ds = datasets.load_dataset(
        "SumanthRH/glaiveai-function-calling-v2-test", split="test"
    )  # load preprocessed test dataset
    anyscale_accuracy = 0
    openai_accuracy = 0
    total_count = 0  # count can exclude api error outputs
    save_dicts = []
    pbar = tqdm(total=len(modified_ds), desc="Evaluating models..")
    save_interval = 50
    i = 0
    while i < len(modified_ds):
        example = modified_ds[i]
        i += 1
        openai_messages = example["openai_user_messages"]
        anyscale_messages = example["anyscale_user_messages"]
        openai_tools = [
            {"type": "function", "function": fn} for fn in example["openai_functions"]
        ]
        openai_conv, openai_is_match = parse_and_eval(
            openai_parser, openai_messages, example["ground_truths"], tools=openai_tools
        )
        anyscale_conv, anyscale_is_match = parse_and_eval(
            anyscale_parser, anyscale_messages, example["ground_truths"]
        )
        save_dicts.append(
            {
                "openai": openai_conv,
                "anyscale": anyscale_conv,
                "ground_truths": example["ground_truths"],
                "example": example,
                "anyscale_is_match": anyscale_is_match,
                "openai_is_match": openai_is_match,
            }
        )
        pbar.set_description(
            "Iter: %d, OpenAI: %d, Anyscale: %d"
            % (i, openai_accuracy, anyscale_accuracy)
        )
        pbar.update(1)

        if openai_conv is None or anyscale_conv is None:  # skip if api errors out
            continue

        if anyscale_is_match:
            anyscale_accuracy += 1

        if openai_is_match:
            openai_accuracy += 1

        if (i + 1) % save_interval == 0:
            with open("evaluation_results.pkl", "wb") as f:
                pickle.dump(save_dicts, f)

        total_count += 1
