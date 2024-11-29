import glob
import os
import time
from pathlib import Path

import yt_dlp
from tenacity import retry, stop_after_attempt, wait_exponential

from ..text.utils import sanitize_filename


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def download_ytb_mp4(
    video_url: str, item_dir: Path, file_name: str, ytb_cookies: Path = None
) -> str:
    """
    下载 YouTube 视频并转换为 MP4 格式，显示下载进度条。

    :param video_url: 要下载的 YouTube 视频的 URL
    :param out_name: 输出文件的名称或路径
    :return: 下载的 MP4 文件路径
    :raises: 在下载过程中发生错误时引发异常
    """
    source_dir = item_dir / "source"
    path = source_dir / file_name
    os.makedirs(source_dir, exist_ok=True)

    basename = path.name
    file_name = sanitize_filename(basename)
    out_name = str(path.parent / file_name)

    ydl_opts = {
        "format": "bestvideo+bestaudio",
        "outtmpl": out_name + ".%(ext)s",
        "restrictfilenames": True,
        "merge_output_format": "mp4",
        "quiet": True,
    }
    if ytb_cookies:
        ydl_opts["cookiefile"] = ytb_cookies

    if os.path.exists(out_name + ".mp4"):
        return out_name + ".mp4"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return out_name + ".mp4"
