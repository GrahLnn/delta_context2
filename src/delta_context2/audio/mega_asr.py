import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import safe_open


MEGA_ASR_REPO_ID = "zhifeixie/Mega-ASR"
QWEN_FORCED_ALIGNER_REPO_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_CKPT_DIR = Path("ckpt") / "Mega-ASR"
DEFAULT_ROUTER_WINDOW_SECONDS = 30.0
DEFAULT_ROUTER_MAX_WINDOWS = 12
DEFAULT_ROUTER_DEGRADED_RATIO = 0.5
DEFAULT_MAX_NEW_TOKENS = 2048
SAMPLE_RATE = 16000


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


@dataclass
class MegaASRSettings:
    ckpt_dir: Path = DEFAULT_CKPT_DIR
    routing_enabled: bool = True
    quality_threshold: float = 0.5
    router_window_seconds: float = DEFAULT_ROUTER_WINDOW_SECONDS
    router_max_windows: int = DEFAULT_ROUTER_MAX_WINDOWS
    router_degraded_ratio: float = DEFAULT_ROUTER_DEGRADED_RATIO
    device_map: str | None = None
    quality_device: str | None = None
    max_inference_batch_size: int = 16
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    keep_delta_on_gpu: bool = True

    @classmethod
    def from_env(cls) -> "MegaASRSettings":
        return cls(
            ckpt_dir=Path(os.getenv("MEGA_ASR_CKPT_DIR", str(DEFAULT_CKPT_DIR))),
            routing_enabled=_env_bool("MEGA_ASR_ROUTING_ENABLED", True),
            quality_threshold=_env_float("MEGA_ASR_ROUTER_THRESHOLD", 0.5),
            router_window_seconds=_env_float(
                "MEGA_ASR_ROUTER_WINDOW_SECONDS",
                DEFAULT_ROUTER_WINDOW_SECONDS,
            ),
            router_max_windows=_env_int(
                "MEGA_ASR_ROUTER_MAX_WINDOWS",
                DEFAULT_ROUTER_MAX_WINDOWS,
            ),
            router_degraded_ratio=_env_float(
                "MEGA_ASR_ROUTER_DEGRADED_RATIO",
                DEFAULT_ROUTER_DEGRADED_RATIO,
            ),
            device_map=os.getenv("MEGA_ASR_DEVICE_MAP") or None,
            quality_device=os.getenv("MEGA_ASR_QUALITY_DEVICE") or None,
            max_inference_batch_size=_env_int("MEGA_ASR_BATCH_SIZE", 16),
            max_new_tokens=_env_int("MEGA_ASR_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS),
            keep_delta_on_gpu=_env_bool("MEGA_ASR_KEEP_DELTA_ON_GPU", True),
        )


class LoRADeltaSwitch:
    def __init__(self, keep_delta_on_gpu: bool = True) -> None:
        self.keep_delta_on_gpu = keep_delta_on_gpu
        self.items: list[dict[str, Any]] = []
        self.active = False

    @staticmethod
    def _load_adapter_state(adapter_dir: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
        adapter_dir = str(adapter_dir)
        safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        bin_path = os.path.join(adapter_dir, "adapter_model.bin")

        if os.path.exists(safetensors_path):
            return safe_load_file(safetensors_path)
        return torch.load(bin_path, map_location="cpu")

    @staticmethod
    def _load_adapter_config(adapter_dir: str | os.PathLike[str]) -> dict[str, Any]:
        config_path = os.path.join(str(adapter_dir), "adapter_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_adapter_blocks(adapter_dir: str | os.PathLike[str]) -> dict[str, Any]:
        blocks_path = os.path.join(str(adapter_dir), "mega_lora_blocks.json")
        if not os.path.exists(blocks_path):
            return {}

        with open(blocks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _normalize_module_name(name: str) -> str:
        for prefix in ("base_model.model.",):
            if name.startswith(prefix):
                name = name[len(prefix) :]

        if name.startswith("thinker.layers."):
            name = name.replace("thinker.layers.", "thinker.model.layers.", 1)

        return name

    @staticmethod
    def _module_name_candidates(name: str) -> list[str]:
        candidates = [name]

        if name.startswith("model."):
            candidates.append(name[len("model.") :])

        if name.startswith("thinker.layers."):
            candidates.append(name.replace("thinker.layers.", "thinker.model.layers.", 1))

        if name.startswith("thinker.model."):
            candidates.append(name.replace("thinker.model.", "thinker.", 1))

        return list(dict.fromkeys(candidates))

    @staticmethod
    def _raw_module_name(key: str, marker: str) -> str:
        name = key.split(marker)[0]
        for prefix in ("base_model.model.", "model."):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    def _split_lora_key(self, key: str) -> tuple[str | None, str | None, str | None]:
        raw_key = key
        key = self._normalize_module_name(key)

        for marker in (".lora_A.", ".lora_B."):
            if marker in key:
                module_name = key.split(marker)[0]
                raw_module_name = self._raw_module_name(raw_key, marker)
                kind = "A" if marker == ".lora_A." else "B"
                return module_name, raw_module_name, kind

        return None, None, None

    def add_adapter(
        self,
        parent_module: torch.nn.Module,
        adapter_dir: str | os.PathLike[str],
        name: str,
        strip_prefixes: list[str] | None = None,
    ) -> None:
        config = self._load_adapter_config(adapter_dir)
        state = self._load_adapter_state(adapter_dir)
        blocks = self._load_adapter_blocks(adapter_dir)

        lora_alpha = config.get("lora_alpha", 1)
        rank = config.get("r")
        alpha_pattern = config.get("alpha_pattern") or {}
        rank_pattern = config.get("rank_pattern") or {}
        fan_in_fan_out = bool(config.get("fan_in_fan_out", False))

        module_dict = dict(parent_module.named_modules())
        grouped: dict[str, dict[str, Any]] = {}

        for key, tensor in state.items():
            module_name, raw_module_name, kind = self._split_lora_key(key)
            if module_name is None or raw_module_name is None or kind is None:
                continue

            if strip_prefixes:
                for prefix in strip_prefixes:
                    if module_name.startswith(prefix):
                        module_name = module_name[len(prefix) :]
                    if raw_module_name.startswith(prefix):
                        raw_module_name = raw_module_name[len(prefix) :]

            matched_name = None
            for candidate in self._module_name_candidates(module_name):
                if candidate in module_dict:
                    matched_name = candidate
                    break

            target_name = matched_name or module_name
            group_key = f"{target_name}\0{raw_module_name}"
            item = grouped.setdefault(
                group_key,
                {
                    "target_module_name": target_name,
                    "raw_module_name": raw_module_name,
                },
            )
            item[kind] = tensor.cpu()

        loaded = 0
        missing = []

        for pair in grouped.values():
            if "A" not in pair or "B" not in pair:
                continue
            module_name = pair["target_module_name"]
            raw_module_name = pair["raw_module_name"]
            if module_name not in module_dict:
                missing.append(module_name)
                continue

            module = module_dict[module_name]
            if not hasattr(module, "weight"):
                missing.append(module_name)
                continue

            weight = module.weight
            a_matrix = pair["A"].to(device=weight.device, dtype=torch.float32)
            b_matrix = pair["B"].to(device=weight.device, dtype=torch.float32)
            module_blocks = blocks.get(raw_module_name) or blocks.get(module_name)

            if module_blocks:
                deltas = []
                for block in module_blocks:
                    start = int(block["start"])
                    end = int(block["end"])
                    block_rank = int(block.get("rank", end - start))
                    block_alpha = int(block.get("alpha", block_rank))
                    delta = torch.matmul(b_matrix[:, start:end], a_matrix[start:end])
                    delta = delta * (float(block_alpha) / float(block_rank))
                    if fan_in_fan_out:
                        delta = delta.T
                    deltas.append(delta)
            else:
                adapter_rank = rank_pattern.get(raw_module_name, rank_pattern.get(module_name, rank))
                if adapter_rank is None:
                    adapter_rank = a_matrix.shape[0]
                adapter_alpha = alpha_pattern.get(
                    raw_module_name,
                    alpha_pattern.get(module_name, lora_alpha),
                )
                scaling = float(adapter_alpha) / float(adapter_rank)
                delta = torch.matmul(b_matrix, a_matrix) * scaling
                if fan_in_fan_out:
                    delta = delta.T
                deltas = [delta]

            for delta in deltas:
                if delta.shape != weight.shape:
                    try:
                        delta = delta.reshape(weight.shape)
                    except Exception:
                        missing.append(
                            f"{module_name}: delta shape {tuple(delta.shape)} != "
                            f"weight shape {tuple(weight.shape)}"
                        )
                        continue

                delta = delta.to(dtype=weight.dtype)
                if self.keep_delta_on_gpu:
                    delta = delta.to(device=weight.device)
                else:
                    delta = delta.cpu()

                self.items.append(
                    {
                        "name": name,
                        "module_name": module_name,
                        "weight": weight,
                        "delta": delta,
                    }
                )
                loaded += 1

        if missing:
            warnings.warn(
                f"LoRA adapter {name} loaded {loaded} deltas, "
                f"missing {len(missing)} modules. Examples: {missing[:5]}",
                stacklevel=2,
            )

    @torch.no_grad()
    def set_active(self, active: bool) -> float:
        if self.active == active:
            return 0.0

        start = time.perf_counter()
        sign = 1.0 if active else -1.0

        for item in self.items:
            weight = item["weight"]
            delta = item["delta"]
            if delta.device != weight.device:
                delta = delta.to(device=weight.device)
            weight.data.add_(delta, alpha=sign)

        self.active = active
        return time.perf_counter() - start


class AudioQualityRouter:
    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        device: str | None = None,
        threshold: float = 0.5,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model, self.mel_extractor = self._load_model()

    def _load_model(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        from .mega_asr_audio_quality import LogMelSpectrogram, create_audio_quality_model

        checkpoint_path = Path(self.checkpoint_path)
        if checkpoint_path.suffix == ".safetensors":
            with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
                metadata = f.metadata()
            checkpoint_config = json.loads(metadata.get("config", "{}"))
            config = checkpoint_config.get("model", {})
            state_dict = safe_load_file(str(checkpoint_path), device=self.device)
        else:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            config = checkpoint.get("config", {}).get("model", {})
            state_dict = checkpoint["model_state_dict"]

        model = create_audio_quality_model(config)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        mel_extractor = LogMelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=config.get("n_mels", 80),
        ).to(self.device)
        mel_extractor.eval()

        return model, mel_extractor

    @staticmethod
    def _window_offsets(total_samples: int, window_samples: int, max_windows: int) -> list[int]:
        if total_samples <= window_samples:
            return [0]
        if max_windows <= 1:
            return [0]

        last_start = total_samples - window_samples
        offsets = np.linspace(0, last_start, num=max_windows)
        return [int(offset) for offset in offsets]

    def _predict_waveform(self, waveform: torch.Tensor) -> tuple[bool, float]:
        mel = self.mel_extractor(waveform)
        mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0)

        logits = self.model(mel, mask=None)
        probs = torch.softmax(logits, dim=-1)
        degraded_prob = float(probs[0, 1].item())
        return degraded_prob >= self.threshold, degraded_prob

    @torch.no_grad()
    def predict(
        self,
        audio: str | os.PathLike[str],
        *,
        window_seconds: float = DEFAULT_ROUTER_WINDOW_SECONDS,
        max_windows: int = DEFAULT_ROUTER_MAX_WINDOWS,
        degraded_ratio: float = DEFAULT_ROUTER_DEGRADED_RATIO,
    ) -> tuple[bool, float]:
        waveform_np = load_mono_audio(audio, sample_rate=self.sample_rate)
        window_samples = max(1, int(window_seconds * self.sample_rate))
        offsets = self._window_offsets(len(waveform_np), window_samples, max_windows)

        degraded_count = 0
        probabilities = []
        for offset in offsets:
            window = waveform_np[offset : offset + window_samples]
            if len(window) < window_samples:
                window = np.pad(window, (0, window_samples - len(window)))
            waveform = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
            is_degraded, degraded_prob = self._predict_waveform(waveform)
            degraded_count += int(is_degraded)
            probabilities.append(degraded_prob)

        if not probabilities:
            return False, 0.0

        ratio = degraded_count / len(probabilities)
        return ratio >= degraded_ratio, max(probabilities)


def load_mono_audio(audio: str | os.PathLike[str], *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    import librosa

    audio_np, _ = librosa.load(str(audio), sr=sample_rate, mono=True)
    return np.asarray(audio_np, dtype=np.float32)


def _checkpoint_has_model(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def ensure_mega_asr_weights(ckpt_dir: Path) -> None:
    model_dir = ckpt_dir / "Qwen3-ASR-1.7B"
    lora_dir = ckpt_dir / "mega-asr-merged"
    router_path = ckpt_dir / "audio_quality_router" / "best_acc_model.safetensors"

    if _checkpoint_has_model(model_dir) and lora_dir.is_dir() and router_path.is_file():
        return

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MEGA_ASR_REPO_ID,
        repo_type="model",
        local_dir=str(ckpt_dir),
        local_dir_use_symlinks=False,
    )


def _result_items(result: Any) -> list[Any]:
    if result is None:
        return []
    items = getattr(result, "items", None)
    if items is not None:
        return list(items)
    if isinstance(result, list):
        return result
    return []


def normalize_time_stamps(time_stamps: Any) -> list[dict[str, Any]]:
    words = []
    for stamp in _result_items(time_stamps):
        if isinstance(stamp, dict):
            text = stamp.get("text") or stamp.get("word") or stamp.get("token")
            start = stamp.get("start") or stamp.get("start_time")
            end = stamp.get("end") or stamp.get("end_time")
        else:
            text = getattr(stamp, "text", None) or getattr(stamp, "word", None)
            start = getattr(stamp, "start", None) or getattr(stamp, "start_time", None)
            end = getattr(stamp, "end", None) or getattr(stamp, "end_time", None)
        if text is None:
            continue
        words.append({"word": text, "start": start, "end": end})
    return words


class MegaASRTranscriber:
    def __init__(self, settings: MegaASRSettings | None = None) -> None:
        self.settings = settings or MegaASRSettings.from_env()
        self.ckpt_dir = self.settings.ckpt_dir.expanduser()
        ensure_mega_asr_weights(self.ckpt_dir)

        self.model_path = self.ckpt_dir / "Qwen3-ASR-1.7B"
        self.lora_dir = self.ckpt_dir / "mega-asr-merged"
        self.router_checkpoint = (
            self.ckpt_dir / "audio_quality_router" / "best_acc_model.safetensors"
        )
        self.router = None
        if self.settings.routing_enabled:
            self.router = AudioQualityRouter(
                self.router_checkpoint,
                device=self.settings.quality_device,
                threshold=self.settings.quality_threshold,
            )
        self.model = self._load_qwen_model()
        self.lora_switch = LoRADeltaSwitch(
            keep_delta_on_gpu=self.settings.keep_delta_on_gpu
        )
        self.lora_switch.add_adapter(
            parent_module=self.model.model,
            adapter_dir=self.lora_dir,
            name="mega_asr_merged_adapter",
        )
        self._set_lora(True)

    def _load_qwen_model(self) -> Any:
        from qwen_asr import Qwen3ASRModel

        device_map = self.settings.device_map
        if device_map is None:
            device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device_map != "cpu" else torch.float32

        return Qwen3ASRModel.from_pretrained(
            str(self.model_path),
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=self.settings.max_inference_batch_size,
            max_new_tokens=self.settings.max_new_tokens,
            forced_aligner=QWEN_FORCED_ALIGNER_REPO_ID,
            forced_aligner_kwargs={
                "dtype": dtype,
                "device_map": device_map,
            },
        )

    def _set_lora(self, active: bool) -> None:
        self.lora_switch.set_active(active)

    def _route(self, audio_path: str | os.PathLike[str]) -> tuple[bool, float | None]:
        if self.router is None:
            return True, None
        return self.router.predict(
            audio_path,
            window_seconds=self.settings.router_window_seconds,
            max_windows=self.settings.router_max_windows,
            degraded_ratio=self.settings.router_degraded_ratio,
        )

    def transcribe(self, audio_path: str | os.PathLike[str]) -> dict[str, Any]:
        use_lora, degraded_prob = self._route(audio_path)
        self._set_lora(use_lora)

        results = self.model.transcribe(
            audio=str(audio_path),
            language=None,
            return_time_stamps=True,
        )
        if not results:
            raise ValueError("Mega-ASR returned no results")

        first = results[0]
        return {
            "text": getattr(first, "text", ""),
            "language": getattr(first, "language", None),
            "time_stamps": getattr(first, "time_stamps", None),
            "use_lora": use_lora,
            "degraded_prob": degraded_prob,
        }


_TRANSCRIBER: MegaASRTranscriber | None = None


def get_mega_asr_transcriber() -> MegaASRTranscriber:
    global _TRANSCRIBER
    if _TRANSCRIBER is None:
        _TRANSCRIBER = MegaASRTranscriber()
    return _TRANSCRIBER


def transcribe_with_mega_asr(audio_path: str | os.PathLike[str]) -> dict[str, Any]:
    return get_mega_asr_transcriber().transcribe(audio_path)
