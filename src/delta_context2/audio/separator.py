import contextlib
import io
import os
import shutil
import subprocess
from pathlib import Path

from museper.inference import separate_audio
from ..utils.decorator import show_progress


def separate_audio_from_video(video_path: str, output_audio_path: str = None) -> str:
    """
    从视频文件中分离出音频。

    :param video_path: 输入视频文件的路径
    :param output_audio_path: 输出音频文件的路径（可选）
    :return: 输出音频文件的路径
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_audio_path is None:
        output_audio_path = video_path.with_suffix(".mp3")
    else:
        output_audio_path = Path(output_audio_path)

    if os.path.exists(output_audio_path):
        return str(output_audio_path)

    # 使用 FFmpeg 分离音频
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                str(output_audio_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while separating audio: {e.stderr.decode()}")
        raise

    return str(output_audio_path)

@show_progress("Extracting")
def extract_vocal(audio_path: str, config: str, weight: str) -> None:
    audio_path: Path = Path(audio_path)
    model_give_name = audio_path.with_name(f"{audio_path.stem}_vocals.wav")
    target_audio_path = audio_path.with_suffix(".wav")
    if os.path.exists(target_audio_path):
        return str(target_audio_path)

    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        separate_audio(
            config_path=config,
            check_point=weight,
            input_file=audio_path,
            store_dir=None,
            device_id=0,
            extract_instrumental=False,
        )

    shutil.move(model_give_name, target_audio_path)
    os.remove(audio_path)

    return str(target_audio_path)
