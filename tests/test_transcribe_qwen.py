import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class FakeResult:
    def __init__(self):
        self.text = "Hello world"
        self.language = "English"
        self.time_stamps = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5},
            {"text": "world", "start_time": 0.6, "end_time": 1.0},
        ]


class FakeModel:
    def transcribe(self, audio, language=None, return_time_stamps=False):
        return [FakeResult()]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def test_transcribe_audio_returns_expected_shape(monkeypatch, tmp_path):
    fake_qwen_module = types.SimpleNamespace(Qwen3ASRModel=FakeModel)
    fake_torch_module = types.SimpleNamespace(bfloat16="bfloat16")

    monkeypatch.setenv("GEMINI_API_KEY", "")

    from delta_context2.audio import transcribe as transcribe_mod

    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_module)
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module)
    monkeypatch.setattr(transcribe_mod, "read_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(transcribe_mod.whisper, "load_audio", lambda *args, **kwargs: "audio-array")

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    result = transcribe_mod.transcribe_audio(
        str(tmp_path),
        "fake.wav",
    )

    assert result["ord_text"] == "Hello world"
    assert result["language"] == "English"
    assert result["ord_words"] == [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.0},
    ]
    assert result["audio"] == "audio-array"


def test_format_words_handles_unspaced_tokens(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")

    from delta_context2.audio.transcribe import format_words

    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.0},
    ]

    result = format_words(words)

    assert [w["word"] for w in result] == ["Hello", "world"]


def test_words_to_text_inserts_spaces_for_stripped_tokens(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")

    from delta_context2.audio import transcribe as transcribe_mod

    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.0},
    ]

    assert transcribe_mod._words_to_text(words) == "Hello world"


@pytest.mark.integration
def test_transcribe_audio_on_fixture(tmp_path, monkeypatch):
    pytest.importorskip("qwen_asr")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    monkeypatch.setenv("GEMINI_API_KEY", "")

    from delta_context2.audio import transcribe as transcribe_mod

    audio_path = ROOT / "tests" / "fixtures" / "vocal.wav"
    if not audio_path.exists():
        pytest.skip("Fixture audio not found")

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    result = transcribe_mod.transcribe_audio(
        str(tmp_path),
        str(audio_path),
    )

    assert isinstance(result.get("ord_text"), str)
    assert isinstance(result.get("ord_words"), list)
