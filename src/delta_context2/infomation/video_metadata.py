import json
from pathlib import Path
import yt_dlp
from alive_progress import alive_bar

from .prompt import SINGLE_TRANSLATION_PROMPT

from .llm import openai_completion
from ..text.utils import remove_illegal_chars


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
                    data_path = (
                        data_dir / "videos" / title.replace(" ", "_").replace(",", "").replace("#", "") / "metadata.json"
                    )
                    if data_path.exists():
                        with open(data_path, "r", encoding="utf-8") as file:
                            return json.load(file)
                        
                    prompt = SINGLE_TRANSLATION_PROMPT.format(TEXT=title)
                    res = openai_completion(prompt)
                    res = res.split("\n")[0]

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
