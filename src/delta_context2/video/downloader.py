import glob
import os
import time
from pathlib import Path

import yt_dlp
from alive_progress import alive_bar

from ..text.utils import sanitize_filename


def download_ytb_mp4(video_url: str, out_name: str | Path) -> str:
    """
    下载 YouTube 视频并转换为 MP4 格式，显示下载进度条。

    :param video_url: 要下载的 YouTube 视频的 URL
    :param out_name: 输出文件的名称或路径
    :return: 下载的 MP4 文件路径
    :raises: 在下载过程中发生错误时引发异常
    """

    def progress_hook(d):
        if d["status"] == "downloading":
            total_bytes = d.get("total_bytes", 0)
            downloaded_bytes = d.get("downloaded_bytes", 0)
            progress = downloaded_bytes / total_bytes if total_bytes else 0
            bar(progress)

    class MyLogger:
        def debug(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            print(msg)

    cost = 0
    max_retries = 50
    retry_count = 0
    path = Path(out_name)
    basename = path.name
    file_name = sanitize_filename(basename)
    out_name = str(path.parent / file_name)

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": out_name + ".%(ext)s",
        "restrictfilenames": True,
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
        "progress_hooks": [progress_hook],
        "logger": MyLogger(),
        "quiet": True,
    }

    if os.path.exists(out_name + ".mp4"):
        return out_name + ".mp4"

    while True:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl, alive_bar(
                100, title="Downloading", manual=True
            ) as bar:
                ydl.download([video_url])
            return out_name + ".mp4"

        except Exception as e:
            print(f"Error during download: {e}")
            retry_count += 1
            if cost == 3:
                print("Failed to download after 3 times.")
                exit(1)
            if retry_count < max_retries:
                print(f"Retrying {video_url}... attempt {retry_count}")
                time.sleep(5)
            elif retry_count == max_retries:
                file_list = glob.glob(out_name + ".*")
                for file in file_list:
                    os.remove(file)
                time.sleep(10)
                cost += 1
