import json

import httpx

from serve_llm.smoke import chat


def test_chat_sends_openai_request_and_returns_content() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hi there"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reply = chat("Hello", base_url="http://test/v1/", model="m", max_tokens=8, client=client)

    assert reply == "Hi there"
    assert seen["url"] == "http://test/v1/chat/completions"
    assert seen["body"]["model"] == "m"
    assert seen["body"]["messages"] == [{"role": "user", "content": "Hello"}]
    assert seen["body"]["max_tokens"] == 8
