from pathlib import Path
import hashlib

import httpx
from alive_progress import alive_bar


def download_file(url: str, save_path: Path):
    """
    从指定的 URL 下载文件并保存到指定路径，显示进度条。

    :param url: 要下载的文件的 URL
    :param save_path: 保存下载文件的路径
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        return str(save_path)

    with httpx.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(save_path, "wb") as file:
            with alive_bar(total_size, title=f"Downloading {save_path.name}") as bar:
                for chunk in response.iter_bytes(chunk_size=8192):
                    size = file.write(chunk)
                    bar(size)


def check_model_exist(model_type: str) -> tuple[Path, Path]:
    """
    检查模型文件是否存在，如果不存在则下载。验证文件完整性。

    :return: 本地模型权重文件路径和配置文件路径
    """
    match model_type:
        case "bs_roformer":
            weight_url = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
            weight_sha256 = (
                "5B84F37E8D444C8CB30C79D77F613A41C05868FF9C9AC6C7049C00AEFAE115AA"
            )
            config_url = "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml"
        case "mel_band_roformer":
            weight_url = "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt"
            weight_sha256 = (
                "87201F4D31AFB5BC79993230FC49446918425574DB48C01C405E44F365C7559E"
            )
            config_url = "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml"

        case _:
            raise ValueError(f"Invalid model type: {model_type}")

    cache_dir = Path.home() / ".cache" / "delta_context2" / model_type
    local_weight_path = cache_dir / weight_url.split("/")[-1]
    local_config_path = cache_dir / config_url.split("/")[-1]

    cache_dir.mkdir(parents=True, exist_ok=True)

    download_and_verify(weight_url, local_weight_path, weight_sha256)
    download_file(config_url, local_config_path)

    return local_weight_path, local_config_path


def download_and_verify(url: str, save_path: Path, expected_sha256: str):
    """
    下载文件并验证其SHA256哈希值。如果哈希值不匹配，尝试恢复下载。

    :param url: 要下载的文件的URL
    :param save_path: 保存下载文件的路径
    :param expected_sha256: 预期的SHA256哈希值
    """
    download_file(url, save_path)

    if not verify_file(save_path, expected_sha256):
        print(f"文件 {save_path} 哈希值不匹配，尝试恢复下载...")
        resume_download(url, save_path)

        if not verify_file(save_path, expected_sha256):
            raise ValueError(f"下载的文件 {save_path} 哈希值仍然不正确")


def verify_file(file_path: Path, expected_sha256: str) -> bool:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().lower() == expected_sha256.lower()


def resume_download(url: str, save_path: Path):
    file_size = save_path.stat().st_size
    headers = {"Range": f"bytes={file_size}-"}

    with httpx.stream("GET", url, headers=headers, follow_redirects=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0)) + file_size

        with open(save_path, "ab") as file:
            with alive_bar(
                total_size, title=f"恢复下载 {save_path.name}", start=file_size
            ) as bar:
                for chunk in response.iter_bytes(chunk_size=8192):
                    size = file.write(chunk)
                    bar(size)
