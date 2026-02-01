import contextlib
import io
import os
import shutil
import subprocess
import time
from pathlib import Path

import librosa

# from museper.inference import separate_audio
from ..utils.decorator import show_progress
from ..utils.network import check_model_exist


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


def remove_file_with_retry(file_path, max_retries=3, delay=1):
    for i in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except PermissionError:
            if i < max_retries - 1:
                time.sleep(delay)
            else:
                print(f"警告：无法删除文件 {file_path}，该文件可能正被其他程序使用")
                return False


@show_progress("Extracting")
def extract_vocal(audio_path: str) -> str:
    raise NotImplementedError("Function not implemented")
    # model_type = "mel_band_roformer"
    # audio_path: Path = Path(audio_path)
    # model_give_name = audio_path.with_name(f"{audio_path.stem}_vocals.wav")
    # target_audio_path = audio_path.with_name("vocal.wav")

    # if os.path.exists(target_audio_path):
    #     return str(target_audio_path)

    # f = io.StringIO()
    # with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
    #     separate_audio(
    #         input_file=audio_path,
    #         store_dir=None,
    #         device_id=0,
    #         extract_instrumental=False,
    #         model_type=model_type,
    #     )

    # # 获取原始音频长度
    # original_duration = get_audio_duration(audio_path)

    # # 使用ffmpeg只保留音频数据
    # temp_audio_path = audio_path.with_name(f"{audio_path.stem}_temp.wav")
    # subprocess.run(
    #     [
    #         "ffmpeg",
    #         "-i",
    #         str(model_give_name),
    #         "-map",
    #         "0:a",
    #         "-c",
    #         "copy",
    #         str(temp_audio_path),
    #     ],
    #     check=True,
    # )

    # # 获取提取后音频长度
    # extracted_duration = get_audio_duration(temp_audio_path)

    # if abs(original_duration - extracted_duration) > 0.1:  # 容许误差0.1秒
    #     shutil.move(audio_path, target_audio_path)
    #     remove_file_with_retry(temp_audio_path)
    # else:
    #     shutil.move(temp_audio_path, target_audio_path)

    # # 删除原始文件

    # remove_file_with_retry(model_give_name)
    # remove_file_with_retry(audio_path)

    # return str(target_audio_path)


def get_audio_duration(file_path: Path) -> float:
    """使用librosa获取音频长度"""

    return librosa.get_duration(path=str(file_path))
