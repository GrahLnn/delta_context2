import os
import re
import subprocess

from alive_progress import alive_bar

from ..utils.subtitle import get_seconds


def compress_video(input_file):
    # 获取输入文件的目录和文件名
    dir_name, base_name = os.path.split(input_file)
    # 构建输出文件名
    name, ext = os.path.splitext(base_name)

    output_file = os.path.join(dir_name, f"translated_video{ext}")

    if os.path.exists(output_file):
        os.remove(output_file)

    while True:
        # 构建ffmpeg命令
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-c:v",
            "libx265",
            "-tag:v",
            "hvc1",
            "-movflags",
            "faststart",
            "-crf",
            "30",
            "-preset",
            "superfast",
            "-c:a",
            "copy",
            output_file,
        ]
        # subprocess.run(cmd, check=True)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            encoding="utf-8",
            text=True,
        )

        duration = None
        progress = 0
        # 使用 alive_progress 显示进度条
        with alive_bar(100, title="compressing", manual=True) as bar:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if duration is None:
                    match = re.search(r"Duration: (\d{2}:\d{2}:\d{2}\.\d{2}),", line)
                    if match:
                        duration = get_seconds(match.group(1))

                match = re.search(r"time=(\d{2}:\d{2}:\d{2}\.\d{2})", line)
                if match:
                    elapsed_time = get_seconds(match.group(1))
                    if duration:
                        progress = round(elapsed_time / duration, 2)
                        bar(progress)

        process.wait()

        if progress >= 1.0:
            os.remove(input_file)
            break
        else:
            if os.path.exists(output_file):
                os.remove(output_file)
