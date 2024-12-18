import glob
import os
import shutil
import time
from pathlib import Path

import yt_dlp
from tenacity import retry, stop_after_attempt, wait_exponential

from ..text.utils import sanitize_filename


@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=4, max=15),
)
def download_ytb_mp4(
    video_url: str, item_dir: Path, file_name: str, ytb_cookies: Path = None
) -> str:
    """
    下载 YouTube 视频并转换为 MP4 格式，显示下载进度条。

    :param video_url: 要下载的 YouTube 视频的 URL
    :param item_dir: 视频存储的根目录
    :param file_name: 输出文件的名称（不带扩展名）
    :param ytb_cookies: 可选，YouTube 的 cookies 文件路径
    :return: 下载的 MP4 文件路径
    :raises: 在下载过程中发生错误时引发异常
    """
    source_dir = item_dir / "source"
    temp_dir = Path("temp")
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    sanitized_filename = sanitize_filename(file_name)
    out_name = str(source_dir / sanitized_filename)

    temp_path = temp_dir / sanitized_filename

    ydl_opts = {
        "format": "bestvideo+bestaudio",
        "outtmpl": f"{temp_path}.%(ext)s",
        "merge_output_format": "mp4",
        "quiet": True,
        # 如果需要显示进度条，可以移除 'quiet': True 并配置进度钩子
        # "progress_hooks": [my_hook],
    }
    if ytb_cookies:
        ydl_opts["cookiefile"] = str(ytb_cookies)

    # 检查目标文件是否已存在
    final_mp4 = out_name + ".mp4"
    if os.path.exists(final_mp4):
        shutil.rmtree(temp_dir)
        return final_mp4

    # 检查临时文件是否已存在
    temp_files = glob.glob(f"{temp_path}.temp.mp4")
    if temp_files:
        shutil.move(temp_files[0], final_mp4)
        shutil.rmtree(temp_dir)
    else:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

    # 等待一段时间，确保文件句柄已释放
    time.sleep(1)

    # 查找下载后的文件（可能有不同的扩展名）
    downloaded_files = list(temp_dir.glob(f"{sanitized_filename}.*"))
    if not downloaded_files:
        raise FileNotFoundError(
            f"No files found in temp directory for {sanitized_filename}"
        )

    # 假设下载后只有一个文件，获取其路径
    downloaded_file = downloaded_files[0]

    # 移动文件到目标位置
    shutil.move(str(downloaded_file), final_mp4)
    shutil.rmtree(temp_dir)

    return final_mp4
