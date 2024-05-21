"""
Printing utilities
"""

import json
from typing import Dict, Any, Optional
from IPython.display import display, HTML, DisplayHandle
from colorama import Fore, Style

from fc_utils.data_format import DatasetFormat

# Define a dictionary to map Colorama ANSI codes to CSS styles
COLORAMA_TO_CSS_DICT = {
    Fore.RED: '<span style="color: red;">',  # Red
    Fore.GREEN: '<span style="color: green;">',  # Green
    Fore.YELLOW: '<span style="color: yellow;">',  # Yellow
    Fore.BLUE: '<span style="color: blue;">',  # Blue
    Fore.MAGENTA: '<span style="color: magenta;">',  # Magenta
    Fore.CYAN: '<span style="color: cyan;">',  # Cyan
    Style.RESET_ALL: "</span>",  # Reset
}

# Maps keys to colorama colors
KEYS_TO_COLORS = {
    "system": Fore.RED,
    "user": Fore.GREEN,
    "assistant": Fore.BLUE,
    "tool": Fore.YELLOW,
    "tools": Fore.MAGENTA,
    "chat": Fore.CYAN,
}


def colorama_to_css(ansi_text):
    """Converts Colorama ANSI codes to CSS styles."""
    # Replace ANSI codes with corresponding HTML tags
    for code, css_tag in COLORAMA_TO_CSS_DICT.items():
        ansi_text = ansi_text.replace(code, css_tag)

    return ansi_text


def _pprint_as_str(example: Dict[str, Any], dataset_format: DatasetFormat) -> str:
    """Formats an example to pretty print with colors for different roles."""
    pprint_str = ""
    # Define colors used for different keys/roles
    reset = Style.RESET_ALL
    for key in example.keys():
        if key == "messages":
            pprint_str += f"{KEYS_TO_COLORS['chat']}Messages: {reset}\n"
            for msg in example["messages"]:
                role = msg["role"]
                content = msg["content"]
                color = KEYS_TO_COLORS.get(role, reset)
                string = f"\t{color}{role}: {reset}{content}\n"
                if dataset_format == DatasetFormat.OPENAI:
                    if role == "assistant":
                        # If the format is OpenAI, include the tool_calls field
                        tool_calls = msg.get("tool_calls", "")
                        string = f"\t{color}{role}: \n\t\tcontent: {reset}{content}\n"
                        string += f"\t\t{color}tool_calls: {reset}{tool_calls}\n"
                    elif role == "tool":
                        # If the format is OpenAI, include the name and tool_call_id fields
                        response_str = json.dumps(
                            {
                                "name": msg["name"],
                                "content": content,
                                "tool_call_id": msg["tool_call_id"],
                            }
                        )
                        string = f"\t{color}{role}: {reset}{response_str}\n"
                pprint_str += string
        else:
            color = KEYS_TO_COLORS.get(key, reset)
            string = f"{color}{key.capitalize()}: {reset}{example[key]}\n"
            pprint_str += string
    return pprint_str


def pprint_example(
    example: Dict[str, Any], dataset_format: DatasetFormat
) -> Optional[DisplayHandle]:
    """Formats an example to pretty print in html."""
    pprint_str = _pprint_as_str(example, dataset_format)
    html_str = colorama_to_css(pprint_str)
    html_str = f"<pre>{html_str}</pre>"
    return display(HTML(html_str))
