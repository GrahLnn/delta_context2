import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_transcribe_audio_returns_expected_shape(monkeypatch, tmp_path):
    from delta_context2.audio import transcribe as transcribe_mod

    monkeypatch.setattr(transcribe_mod, "read_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(transcribe_mod, "load_mono_audio", lambda *args, **kwargs: "audio-array")
    monkeypatch.setattr(
        transcribe_mod,
        "transcribe_with_mega_asr",
        lambda *args, **kwargs: {
            "text": "Hello world",
            "language": "English",
            "time_stamps": [
                {"text": "Hello", "start_time": 0.0, "end_time": 0.5},
                {"text": "world", "start_time": 0.6, "end_time": 1.0},
            ],
        },
    )

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


def test_transcribe_audio_rejects_text_without_timestamps(monkeypatch, tmp_path):
    from delta_context2.audio import transcribe as transcribe_mod

    monkeypatch.setattr(transcribe_mod, "read_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(transcribe_mod, "load_mono_audio", lambda *args, **kwargs: "audio-array")
    monkeypatch.setattr(
        transcribe_mod,
        "transcribe_with_mega_asr",
        lambda *args, **kwargs: {
            "text": "Hello world",
            "language": "English",
            "time_stamps": None,
        },
    )

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="without word timestamps"):
        transcribe_mod.transcribe_audio(str(tmp_path), "fake.wav")


def test_router_windows_keep_long_audio_below_position_limit():
    from delta_context2.audio.mega_asr import AudioQualityRouter

    offsets = AudioQualityRouter._window_offsets(
        total_samples=16000 * 120,
        window_samples=16000 * 30,
        max_windows=4,
    )

    assert offsets == [0, 480000, 960000, 1440000]


def test_normalize_time_stamps_accepts_forced_aligner_items():
    from delta_context2.audio.mega_asr import normalize_time_stamps

    align_result = SimpleNamespace(
        items=[
            SimpleNamespace(text="Hello", start_time=0.0, end_time=0.5),
            SimpleNamespace(text="world", start_time=0.6, end_time=1.0),
        ]
    )

    assert normalize_time_stamps(align_result) == [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.0},
    ]


def test_mega_asr_transcriber_mounts_lora_on_qwen_model(monkeypatch, tmp_path):
    from delta_context2.audio import mega_asr

    captured = {}
    qwen_inner_model = object()

    class FakeSwitch:
        def __init__(self, keep_delta_on_gpu=True):
            pass

        def add_adapter(self, **kwargs):
            captured["parent_module"] = kwargs["parent_module"]

        def set_active(self, active):
            captured["active"] = active

    class FakeTranscriber(mega_asr.MegaASRTranscriber):
        def _load_qwen_model(self):
            return SimpleNamespace(model=qwen_inner_model)

    monkeypatch.setattr(mega_asr, "ensure_mega_asr_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(mega_asr, "LoRADeltaSwitch", FakeSwitch)

    settings = mega_asr.MegaASRSettings(
        ckpt_dir=tmp_path,
        routing_enabled=False,
    )
    FakeTranscriber(settings)

    assert captured["parent_module"] is qwen_inner_model
    assert captured["active"] is True


def test_format_words_handles_unspaced_tokens():
    from delta_context2.audio.transcribe import format_words

    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.0},
    ]

    result = format_words(words)

    assert [w["word"] for w in result] == ["Hello", "world"]


def test_words_to_text_inserts_spaces_for_stripped_tokens():
    from delta_context2.audio import transcribe as transcribe_mod

    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.0},
    ]

    assert transcribe_mod._words_to_text(words) == "Hello world"


@pytest.mark.integration
def test_transcribe_audio_on_fixture(tmp_path, monkeypatch):
    pytest.skip("Mega-ASR integration downloads large model weights.")
    pytest.importorskip("qwen_asr")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

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
