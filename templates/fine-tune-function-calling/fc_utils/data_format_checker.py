"""
Data format validation function from https://docs.endpoints.anyscale.com/endpoints/fine-tuning/dataset-prep/#context-length
"""


class FormattingError(Exception):
    pass


def convert_message_list_to_text(messages: list) -> str:
    # special tokens are for the Llama-2 model
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    text = ""

    if messages[0]["role"] == "system":
        messages = [
            {
                "role": messages[1]["role"],
                "content": B_SYS
                + messages[0]["content"]
                + E_SYS
                + messages[1]["content"],
            }
        ] + messages[2:]

    assert all([msg["role"] == "user" for msg in messages[::2]]) and all(
        [msg["role"] == "assistant" for msg in messages[1::2]]
    ), (
        "model only supports 'system','user' and 'assistant' roles, "
        "starting with user and alternating (u/a/u/a/u...)"
    )

    texts = []
    for prompt, answer in zip(messages[::2], messages[1::2]):
        texts.append(
            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
        )

    text = "</s><s>".join(texts)
    # add the bos and eos token at the beginning of the first turn and the end of the last turn
    text = "<s>" + text + " </s>"
    # During training last message should be from assistant (not from a user)
    assert (
        messages[-1]["role"] == "assistant"
    ), f"Last message must be from assistant, got {messages[-1]['role']}"

    return text
