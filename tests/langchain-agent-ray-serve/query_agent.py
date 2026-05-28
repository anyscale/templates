"""Query the locally-served LangGraph agent and assert a real LLM->MCP->agent answer."""
import sys

import httpx

URL = "http://127.0.0.1:8000/chat"
PROMPT = "What's the weather in Palo Alto?"
# Evidence that the weather MCP tool was actually exercised (not an LLM-only reply).
TOOL_MARKERS = ("get_forecast", "get_alerts", "tool_calls", '"tools"', "ToolMessage", '"type": "tool"')


def main() -> int:
    lines: list[str] = []
    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        with client.stream("POST", URL, json={"user_request": PROMPT}) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    lines.append(line)

    blob = "\n".join(lines)
    print(f"[query_agent] {len(lines)} SSE line(s), {len(blob)} chars")
    print(blob[:3000])

    if "event: error" in blob:
        print("[query_agent] FAIL: agent streamed an error frame", file=sys.stderr)
        return 1
    if not lines:
        print("[query_agent] FAIL: no SSE events from /chat", file=sys.stderr)
        return 1
    if not any(m in blob for m in TOOL_MARKERS):
        print("[query_agent] FAIL: no evidence the weather MCP tool was invoked", file=sys.stderr)
        return 1

    print("[query_agent] PASS: agent answered via LLM -> weather MCP tool -> response")
    return 0


if __name__ == "__main__":
    sys.exit(main())
