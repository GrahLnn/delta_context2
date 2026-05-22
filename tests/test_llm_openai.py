import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_project_dotenv_overrides_existing_process_env(monkeypatch, tmp_path):
    from delta_context2.infomation import llm

    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_URL=https://project.example/v1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_URL", "https://system.example/v1")

    llm._load_project_dotenv()

    assert llm.os.getenv("OPENAI_API_URL") == "https://project.example/v1"


def test_openai_base_url_accepts_base_or_chat_completion_endpoint(monkeypatch):
    from delta_context2.infomation import llm

    for raw_url in (
        "https://aiberm.com/v1",
        "https://aiberm.com/v1/",
        "https://aiberm.com/v1/chat/completions",
        "https://aiberm.com/v1/chat/completions/",
    ):
        monkeypatch.setattr(llm, "OPENAI_URL", raw_url)
        assert llm._openai_base_url() == "https://aiberm.com/v1"


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
