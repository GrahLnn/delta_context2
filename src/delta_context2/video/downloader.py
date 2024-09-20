import glob
import os
import time
from pathlib import Path

import yt_dlp
from alive_progress import alive_bar
from yt_dlp.utils import DownloadError

from ..text.utils import formal_file_name, sanitize_filename


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
    # parent_path = formal_file_name(str(path.parent))
    out_name = str(path.parent / file_name)
    out_name = formal_file_name(out_name)

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
        "nopart": True,
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

        except DownloadError as e:
            error_message = str(e)
            print(f"Download error: {error_message}")
            retry_count += 1

            # 如果错误消息包含特定的内容，表明需要删除文件重新下载
            if (
                "HTTP Error 416" in error_message
                or "unable to download video data" in error_message
            ):
                print("Critical error, removing partial files and retrying...")
                file_list = glob.glob(out_name + ".*")
                for file in file_list:
                    try:
                        os.remove(file)
                    except OSError as remove_error:
                        print(f"Error removing file {file}: {remove_error}")
            else:
                print(f"Non-critical error, retrying without removing files...")

            if retry_count < max_retries:
                print(f"Retrying {video_url}... attempt {retry_count}")
                time.sleep(5)
            elif retry_count == max_retries:
                print("Maximum retry attempts reached. Cleaning up.")
                file_list = glob.glob(out_name + ".*")
                for file in file_list:
                    try:
                        os.remove(file)
                    except OSError as remove_error:
                        print(f"Error removing file {file}: {remove_error}")
                time.sleep(10)
                cost += 1
                retry_count = 0  # 重置重试计数器，以便在清理后再次尝试

        except Exception as e:
            print(f"Unexpected error: {e}")
            exit(1)
