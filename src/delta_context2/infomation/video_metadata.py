import json
from io import BytesIO
from pathlib import Path

import httpx
import yt_dlp
from alive_progress import alive_bar
from PIL import Image

from ..text.utils import formal_file_name, formal_folder_name, remove_illegal_chars
from .llm import openai_completion
from .prompt import SINGLE_TRANSLATION_PROMPT


def get_ytb_video_info(url: str, data_dir: Path, max_retries=3) -> dict:
    """
    Fetches metadata for a YouTube video.

    Args:
        url (str): The URL of the YouTube video.
        max_retries (int, optional): The maximum number of retries in case of failure. Defaults to 3.

    Returns:
        dict: A dictionary containing the video metadata, including title, description, uploader, thumbnail, and video URL.

    Raises:
        Exception: If the video metadata cannot be fetched after the maximum number of retries.
    """
    options = {
        "quiet": True,
        "extract_flat": True,
        "force_generic_extractor": True,
    }
    attempts = 0
    with alive_bar(1, bar=None, title="fetch video info", monitor=False) as bar:
        while attempts < max_retries:
            try:
                with yt_dlp.YoutubeDL(options) as ydl:
                    info_dict = ydl.extract_info(url, download=False)

                    title = remove_illegal_chars(info_dict.get("title", None))
                    video_url = info_dict.get("webpage_url", None)
                    description = info_dict.get("description", None)
                    uploader = info_dict.get("uploader", None)
                    thumbnail = info_dict.get("thumbnail", None)
                    name_formal = formal_folder_name(title)
                    data_path = data_dir / "videos" / name_formal / "metadata.json"
                    if data_path.exists():
                        with open(data_path, "r", encoding="utf-8") as file:
                            return json.load(file)

                    prompt = SINGLE_TRANSLATION_PROMPT.format(ORIGINAL_TEXT=title)
                    res = openai_completion(prompt)

                    download_and_resize_thumbnail(
                        thumbnail, data_dir / "videos" / name_formal / "source"
                    )

                    metadata = {
                        "title": title,
                        "title_zh": res,
                        "description": description,
                        "uploader": uploader,
                        "thumbnail": thumbnail,
                        "video_url": video_url,
                    }
                    data_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(data_path, "w", encoding="utf-8") as file:
                        json.dump(metadata, file, ensure_ascii=False, indent=4)
                    bar()
                    return metadata
            except Exception as e:
                print(f"Attempt {attempts + 1} failed with error: {e}")
                attempts += 1

    raise Exception("Failed to fetch video info after maximum retries.")


def download_and_resize_thumbnail(url: str, save_dir: Path) -> None:
    response = httpx.get(url)
    if response.status_code == 200:
        image_data = response.content
        image = Image.open(BytesIO(image_data))

        # 将图像放大到1080p级别
        resized_image = image.resize((1920, 1080), Image.Resampling.LANCZOS)

        save_dir.mkdir(parents=True, exist_ok=True)
        resized_image.save(save_dir / "thumbnail.jpg", "JPEG")
    else:
        raise Exception("Failed to download thumbnail.")
