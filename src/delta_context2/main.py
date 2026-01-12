import os
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

# from .audio.force_align import force_align
from .audio.separator import extract_vocal, separate_audio_from_video
from .audio.transcribe import get_transcribe
from .infomation.llm import get_summary, get_tags
from .infomation.translate_agent import translate
from .infomation.video_metadata import get_ytb_video_info
from .text.utils import (
    formal_file_name,
    formal_folder_name,
    sanitize_filename,
    split_sentences_into_chunks,
)
from .utils.align import get_sentence_timestamps, split_to_atomic_part
from .utils.subtitle import render_video_with_subtitles, save_to_ass
from .video.downloader import download_ytb_mp4
from .video.utils import compress_video


class VideoProcessor:
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        country: str,
        ytb_cookies: Path = None,
        save_temp_file: bool = False,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.country = country
        self.DATA_DIR = Path("data")
        self.ytb_cookies = ytb_cookies
        self.save_temp_file = save_temp_file

    def process(self, ytb_url: str, compress: bool = True) -> dict:
        """
        return: the video dir
        """
        parsed_url = urlparse(ytb_url)
        is_http_url = parsed_url.scheme in ("http", "https")
        local_path = None
        if is_http_url:
            video_info = get_ytb_video_info(
                ytb_url, self.DATA_DIR, self.ytb_cookies
            )
        else:
            if parsed_url.scheme == "file":
                local_path = Path(parsed_url.path.lstrip("/"))
            else:
                local_path = Path(ytb_url)
            local_path = local_path.expanduser()
            video_info = {
                "title": local_path.stem,
                "description": "",
                "uploader": None,
                "thumbnail": None,
                "video_url": None,
            }
            local_missing = not local_path.is_file()
        print("processing: ", video_info["title"])
        formal_name = formal_folder_name(video_info["title"])
        item_dir = self.DATA_DIR / "videos" / formal_name
        if not os.path.exists(item_dir / "translated_video.mp4"):
            if is_http_url:
                video_path = download_ytb_mp4(
                    ytb_url, item_dir, formal_name, self.ytb_cookies
                )
            else:
                source_dir = item_dir / "source"
                source_dir.mkdir(parents=True, exist_ok=True)
                sanitized_name = (
                    sanitize_filename(local_path.stem) + local_path.suffix      
                )
                video_path = source_dir / sanitized_name
                if local_missing:
                    if video_path.exists():
                        local_path = video_path
                    else:
                        raise FileNotFoundError(
                            f"Local video file not found: {local_path}"
                        )
                if not video_path.exists():
                    shutil.move(str(local_path), video_path)
                thumbnail_path = source_dir / "thumbnail.jpg"
                if not thumbnail_path.exists():
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(video_path),
                        "-vf",
                        "scale=1920:1080",
                        "-frames:v",
                        "1",
                        str(thumbnail_path),
                    ]
                    subprocess.run(cmd, check=True)
                video_info["thumbnail"] = str(thumbnail_path)
                data_path = item_dir / "metadata.json"
                if not data_path.exists():
                    data_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(data_path, "w", encoding="utf-8") as file:
                        json.dump(
                            video_info, file, ensure_ascii=False, indent=4
                        )
            if not os.path.exists(item_dir / "source" / "vocal.wav"):
                audio_path = separate_audio_from_video(video_path)
                audio_path = extract_vocal(audio_path)
            else:
                audio_path = item_dir / "source" / "vocal.wav"
        else:
            return item_dir
        transcribe = get_transcribe(item_dir, audio_path, video_info["description"])
        language, audio_waveform = transcribe["language"], transcribe["audio"]

        sentences = transcribe["sentences"]
        words = transcribe["words"]
        source_text_chunks = split_sentences_into_chunks(sentences)
        summary = get_summary(item_dir, sentences)
        get_tags(item_dir, summary["summary"])
        result = translate(
            item_dir,
            self.source_lang,
            self.target_lang,
            source_text_chunks,
            self.country,
        )
        translated_chunks = [chunk["translation"] for chunk in result]
        atomic_part = split_to_atomic_part(
            item_dir, source_text_chunks, translated_chunks
        )
        atomic_zhs = [part["zh"] for part in atomic_part]
        atomic_ens = [part["en"] for part in atomic_part]
        # words = force_align(audio_waveform, " ".join(atomic_ens), language)
        sentences_timestamps = get_sentence_timestamps(
            item_dir, atomic_ens, words, atomic_zhs
        )
        subtitle_path = save_to_ass(sentences_timestamps, "subtitle", item_dir)
        if os.path.exists(item_dir / "translated_video.mp4"):
            return item_dir
        translate_video = render_video_with_subtitles(
            video_path, subtitle_path, item_dir
        )
        if compress:
            compress_video(translate_video)
        else:
            if os.path.exists(item_dir / "translated_video.mp4"):
                os.remove(item_dir / "translated_video.mp4")
            os.rename(translate_video, item_dir / "translated_video.mp4")
        if not self.save_temp_file:
            os.remove(audio_path)
            os.remove(video_path)
        return item_dir
