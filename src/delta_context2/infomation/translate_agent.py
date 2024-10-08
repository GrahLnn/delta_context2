import json
import os
import re
import shutil
import time
from datetime import datetime, timedelta, timezone
from typing import List

import tiktoken
from alive_progress import alive_it

from ..infomation.prompt import (
    TA_IMPROVEMENT_PROMPT,
    TA_INIT_TRANSLATION_PROMPT,
    TA_REFLECTION_PROMPT,
)
from ..infomation.read_metadata import read_metadata
from ..utils.decorator import update_metadata
from .llm import get_completion


def load_credentials(filename):
    with open(filename) as file:
        credentials_info = json.load(file)
    return credentials_info


def replace_multiple_newlines(text):
    cleaned_text = re.sub(r"\n{3,}", "\n\n", text)
    return cleaned_text


def replace_spaces_in_links(text):
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def replacer(match):
        text_inside_brackets = match.group(1)
        link_inside_parentheses = match.group(2).replace(" ", "%20")
        return f"[{text_inside_brackets}]({link_inside_parentheses})"

    return pattern.sub(replacer, text)


def replace_chinese_parentheses(text):
    pattern = re.compile(r"\[([^\]]+)\]（([^）]+)）")

    def replacer(match):
        text_inside_brackets = match.group(1)
        link_inside_parentheses = match.group(2)
        return f"[{text_inside_brackets}]({link_inside_parentheses})"

    return pattern.sub(replacer, text)


def store_token(auth_token: str, auth_time: datetime):
    with open("asset/google_auth.json", "w") as file:
        data = {
            "auth_token": auth_token,
            "auth_time": auth_time.astimezone(timezone.utc).isoformat(),
        }
        json.dump(data, file)


def read_stored_token():
    try:
        with open("asset/google_auth.json") as file:
            data: dict = json.load(file)
            auth_token = data.get("auth_token")
            auth_time = data.get("auth_time")
            if auth_token and auth_time:
                auth_time = datetime.fromisoformat(auth_time).astimezone(timezone.utc)
                return auth_token, auth_time
    except FileNotFoundError:
        pass
    return None, None


def remove_hash_lines(text):
    # 匹配包含一个或多个#的行
    pattern = r"^\s*#+\s*$"
    # 使用re.sub来将匹配的行替换为空字符串
    cleaned_text = re.sub(pattern, "", text, flags=re.MULTILINE)
    return cleaned_text


def sleep_to_next_refresh():
    now = datetime.now(timezone.utc)
    tomorrow = now + timedelta(days=1)
    tomorrow = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    sleep_time = (tomorrow - now).total_seconds()
    for i in reversed(range(int(sleep_time))):
        hours, remainder = divmod(i, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        print(f"Sleep to {tomorrow} refresh at {time_str}.", flush=True, end="\r")
        time.sleep(1)
    print("Wake up! Keep working!                    ", flush=True)


def save_cache(dir, text, name):
    os.makedirs(dir, exist_ok=True)
    output = dir + f"{name}.md"

    with open(output, "w", encoding="utf-8") as output_file:
        output_file.write(text)


def num_tokens_in_string(input_str: str, encoding_name: str = "o200k_base") -> int:
    """
    Calculate the number of tokens in a given string using a specified encoding.

    Args:
        str (str): The input string to be tokenized.
        encoding_name (str, optional): The name of the encoding to use. Defaults to "o200k_base",
            which is the most commonly used encoder (used by GPT-4).

    Returns:
        int: The number of tokens in the input string.

    Example:
        >>> text = "Hello, how are you?"
        >>> num_tokens = num_tokens_in_string(text)
        >>> print(num_tokens)
        5
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens


@update_metadata(
    ("chunk_translation_init", lambda r: r),
)
def multichunk_initial_translation(
    dir, source_lang: str, target_lang: str, source_text_chunks: List[str]
) -> List[str]:
    """
    Translate a text in multiple chunks from the source language to the target language.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): A list of text chunks to be translated.

    Returns:
        List[str]: A list of translated text chunks.
    """

    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."

    done_idx = -1
    translation_chunks = []

    cache_file = "cache/init_translation.json"
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            cache_data: dict = json.load(f)
            done_idx = cache_data.get("done_idx")
            translation_chunks = cache_data.get("translation_chunks", [])

    if done_idx == len(source_text_chunks) - 1:
        return translation_chunks

    for i in alive_it(
        range(done_idx + 1, len(source_text_chunks)), title="init translate"
    ):
        prompt = TA_INIT_TRANSLATION_PROMPT.format(
            SOURCE_LANG=source_lang,
            TARGET_LANG=target_lang,
            CHUNK_TO_TRANSLATE=source_text_chunks[i],
        )
        max_retry = 3
        for attempt in range(max_retry):
            translation = get_completion(prompt)
            cleaned_translation = re.sub(r"<[^>]*>|\n", "", translation)
            if cleaned_translation:
                break
            if attempt == max_retry - 1:
                raise ValueError(f"Failed to get a valid translation.\n{translation}")

        translation_chunks.append(cleaned_translation)

        cache_data = {
            "done_idx": i,
            "translation_chunks": translation_chunks,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4)

    return translation_chunks


@update_metadata(
    ("chunk_translation_reflect", lambda r: r),
)
def multichunk_reflect_on_translation(
    dir,
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    country: str = "",
) -> List[str]:
    """
    Provides constructive criticism and suggestions for improving a partial translation.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text_chunks (List[str]): The source text divided into chunks.
        translation_1_chunks (List[str]): The translated chunks corresponding to the source text chunks.
        country (str): Country specified for target language.

    Returns:
        List[str]: A list of reflections containing suggestions for improving each translated chunk.
    """

    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    done_idx = -1
    reflection_chunks = []

    cache_file = "cache/reflection_chunks.json"
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            cache_data: dict = json.load(f)
            done_idx = cache_data.get("done_idx", 0)
            reflection_chunks = cache_data.get("reflection_chunks", [])

    if done_idx == len(source_text_chunks) - 1:
        return reflection_chunks

    for i in alive_it(
        range(done_idx + 1, len(source_text_chunks)),
        title="reflect translate",
    ):
        # Will translate chunk i
        tagged_text = (
            ("".join(source_text_chunks[max(i - 2, 0) : i]) if i > 0 else "")
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + (
                "".join(source_text_chunks[i + 1 : min(i + 2, len(source_text_chunks))])
                if i < len(source_text_chunks) - 1
                else ""
            )
        )
        prompt = TA_REFLECTION_PROMPT.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
            translation_1_chunk=translation_1_chunks[i],
            country=country,
        )

        max_attempts = 3
        for attempt in range(max_attempts):
            reflection = get_completion(prompt)
            cleaned_reflection = re.sub(r"<[^>]*>", "", reflection).strip()
            if cleaned_reflection:
                break
            if attempt == max_attempts - 1:
                raise ValueError(f"Failed to get a valid reflection.\n{reflection}")

        reflection_chunks.append(cleaned_reflection)

        cache_data = {
            "done_idx": i,
            "reflection_chunks": reflection_chunks,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4)

    return reflection_chunks


@update_metadata(
    ("chunk_translation_improve", lambda r: r),
)
def multichunk_improve_translation(
    dir,
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    reflection_chunks: List[str],
) -> List[str]:
    """
    Improves the translation of a text from source language to target language by considering expert suggestions.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): The source text divided into chunks.
        translation_1_chunks (List[str]): The initial translation of each chunk.
        reflection_chunks (List[str]): Expert suggestions for improving each translated chunk.

    Returns:
        List[str]: The improved translation of each chunk.
    """

    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."

    done_idx = -1
    translation_2_chunks = []

    cache_file = "cache/imporove_chunks.json"
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            cache_data: dict = json.load(f)
            done_idx = cache_data.get("done_idx", 0)
            translation_2_chunks = cache_data.get("translation_2_chunks", [])

    if done_idx == len(source_text_chunks) - 1:
        return translation_2_chunks

    for i in alive_it(
        range(done_idx + 1, len(source_text_chunks)),
        title="improve translate",
    ):
        # Will translate chunk i
        tagged_text = "<TRANSLATE_THIS>" + source_text_chunks[i] + "</TRANSLATE_THIS>"

        prompt = TA_IMPROVEMENT_PROMPT.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
            translation_1_chunk=translation_1_chunks[i],
            reflection_chunk=reflection_chunks[i],
        )

        max_attempts = 3
        for attempt in range(max_attempts):
            translation_2 = get_completion(prompt)
            cleaned_translation = re.sub(r"<[^>]*>|\n", "", translation_2).strip()
            if cleaned_translation:
                break
            if attempt == max_attempts - 1:
                raise ValueError(f"Failed to get a valid translation.\n{translation_2}")

        translation_2_chunks.append(cleaned_translation)

        cache_data = {
            "done_idx": i,
            "translation_2_chunks": translation_2_chunks,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4)

    return translation_2_chunks


def multichunk_translation(
    dir, source_lang, target_lang, source_text_chunks, country: str = ""
):
    """
    Improves the translation of multiple text chunks based on the initial translation and reflection.

    Args:
        source_lang (str): The source language of the text chunks.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): The list of source text chunks to be translated.
        translation_1_chunks (List[str]): The list of initial translations for each source text chunk.
        reflection_chunks (List[str]): The list of reflections on the initial translations.
        country (str): Country specified for target language
    Returns:
        List[str]: The list of improved translations for each source text chunk.
    """

    translation_1_chunks = multichunk_initial_translation(
        dir, source_lang, target_lang, source_text_chunks
    )

    reflection_chunks = multichunk_reflect_on_translation(
        dir,
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        country,
    )

    translation_2_chunks = multichunk_improve_translation(
        dir,
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        reflection_chunks,
    )

    return translation_2_chunks


def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    """
    Calculate the chunk size based on the token count and token limit.

    Args:
        token_count (int): The total number of tokens.
        token_limit (int): The maximum number of tokens allowed per chunk.

    Returns:
        int: The calculated chunk size.

    Description:
        This function calculates the chunk size based on the given token count and token limit.
        If the token count is less than or equal to the token limit, the function returns the token count as the chunk size.
        Otherwise, it calculates the number of chunks needed to accommodate all the tokens within the token limit.
        The chunk size is determined by dividing the token limit by the number of chunks.
        If there are remaining tokens after dividing the token count by the token limit,
        the chunk size is adjusted by adding the remaining tokens divided by the number of chunks.

    Example:
        >>> calculate_chunk_size(1000, 500)
        500
        >>> calculate_chunk_size(1530, 500)
        389
        >>> calculate_chunk_size(2242, 500)
        496
    """

    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size


def replace_markdown_links(text):
    # 正则表达式匹配 [![.*?](.*?)](.*?) 的模式
    pattern = re.compile(r"\[(!\[.*?\]\(.*?\))\]\(.*?\)")
    replaced_text = pattern.sub(r"\1", text)

    return replaced_text


@update_metadata(
    ("chunk_translation", lambda r: r),
)
def translate(
    dir,
    source_lang,
    target_lang,
    sentences,
    country,
):
    """Translate the source_text from source_lang to target_lang."""
    os.makedirs("cache", exist_ok=True)
    check = read_metadata(dir, "chunk_translation")
    if check:
        return check

    translation_2_chunks = multichunk_translation(
        dir, source_lang, target_lang, sentences, country
    )

    translation_2_chunks = [chunk.replace("\n", "") for chunk in translation_2_chunks]

    result = [
        {
            "source": sentences[i],
            "translation": translation_2_chunks[i],
        }
        for i in range(len(sentences))
    ]

    shutil.rmtree("cache")

    return result
