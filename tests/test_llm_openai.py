import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_openai_completion_uses_openai_sdk(monkeypatch):
    from delta_context2.infomation import llm

    captured = {}

    class FakeChatCompletions:
        @staticmethod
        def create(**kwargs):
            captured["completion_args"] = kwargs
            message = SimpleNamespace(content="done")
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_args"] = kwargs
            self.chat = SimpleNamespace(completions=FakeChatCompletions())

    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm, "OPENAI_KEY", "test-key")
    monkeypatch.setattr(llm, "OPENAI_URL", "https://aiberm.com/v1/chat/completions")
    monkeypatch.setattr(llm, "OPENAI_TIMEOUT", 123)

    result = llm.openai_completion(
        "user prompt",
        system_message="system prompt",
        temperature=0.2,
        model="test-model",
        json_output=True,
    )

    assert result == "done"
    assert captured["client_args"] == {
        "api_key": "test-key",
        "base_url": "https://aiberm.com/v1",
        "timeout": 123,
    }
    assert captured["completion_args"] == {
        "model": "test-model",
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ],
        "stream": False,
        "response_format": {"type": "json_object"},
    }
