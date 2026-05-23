import os
import re
import time
import logging
from urllib.parse import urlsplit, urlunsplit

import demjson3
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
import tiktoken
from dotenv import find_dotenv, load_dotenv
from retry import retry

from ..infomation.read_metadata import read_metadata
from ..text.utils import split_sentences_into_chunks
from ..utils.decorator import show_progress, update_metadata
from .prompt import (
    SINGLE_TRANSLATION_PROMPT,
    SINGLE_TRANSLATION_PROMPT_WITH_CONTEXT,
    SUMMARY_SYS_MESSAGE,
)

def _load_project_dotenv() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path or None, override=True)


_load_project_dotenv()
OPENAI_URL = os.getenv("OPENAI_API_URL")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TASK_MODEL = os.getenv("TASK_MODEL")
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "300"))
OPENAI_MAX_ATTEMPTS = int(os.getenv("OPENAI_MAX_ATTEMPTS", "3"))
OPENAI_RETRY_INITIAL_DELAY = float(os.getenv("OPENAI_RETRY_INITIAL_DELAY", "1"))
OPENAI_RETRY_BACKOFF = float(os.getenv("OPENAI_RETRY_BACKOFF", "2"))
_CHAT_COMPLETIONS_PATH = "/chat/completions"
_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def _quiet_openai_http_logs() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.ERROR)


_quiet_openai_http_logs()


@show_progress("getting summary")
@update_metadata(
    ("summary", lambda r: r["summary"]),
    ("summary_zh", lambda r: r["summary_zh"]),
    ("title_zh", lambda r: r["title_zh"]),
)
def get_summary(idir, sentences: list[str]) -> dict:
    check = read_metadata(
        idir,
        ["summary", "summary_zh", "title_zh"],
    )
    title = read_metadata(idir, ["title"])
    if check:
        return check

    chunks = split_sentences_into_chunks(sentences, 8000)

    tldrs = []
    for chunk in chunks:
        tldr = get_completion(
            chunk
            + "\n\nAbstract this paragraph, and remove any non-content related information, and ONLY keep the main points."
        )
        tldrs.append(tldr)

    summary_zh = get_completion(
        " ".join(tldrs)
        + "\n\n直接给这个视频一个中文摘要，不要任何非摘要信息，只直接给摘要，并移除非内容相关的东西"
    )

    # prompt = SINGLE_TRANSLATION_PROMPT.format(ORIGINAL_TEXT=summary)
    # summary_zh = get_completion(prompt)
    while len(summary_zh) > 2000:
        summary_zh = get_completion(
            summary_zh,
            "Please condense this summary to be more concise, omitting any irrelevant parts with same language.",
        )
    prompt = SINGLE_TRANSLATION_PROMPT_WITH_CONTEXT.format(
        ORIGINAL_TEXT=title, CONTEXT=summary_zh
    )
    title_zh = get_completion(prompt)
    return {"summary": "not suport", "summary_zh": summary_zh, "title_zh": title_zh}


@update_metadata(("tags", lambda r: r))
def get_tags(idir, summary: str) -> dict:
    check = read_metadata(idir, ["tags"])
    if check:
        return check

    prompt = "Select three main keywords based on the summary, separated by commas. Only return tags and nothing else.\n\nHere is the summary text you will be working with:\n\n<summary>\n{summary}\n</summary>".format(
        summary=summary
    )

    res = get_completion(prompt)
    tags = res.split(",")
    tags = [tag.strip() for tag in tags]
    return tags


@retry(tries=3, delay=2)
def get_json_completion(prompt, model=TRANSLATION_MODEL):
    result = get_completion(prompt)
    pattern = re.compile(r"^json")
    json_str = pattern.sub("", result.strip().strip("```"))
    result = demjson3.decode(json_str)

    return result


def tokenize(text: str):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    token_integers = encoding.encode(text)
    return token_integers


def _normalize_openai_base_url(url: str | None) -> str:
    if not url:
        raise ValueError("OPENAI_API_URL is required.")

    parsed_url = urlsplit(url.strip())
    path = parsed_url.path.rstrip("/")
    if path.endswith(_CHAT_COMPLETIONS_PATH):
        path = path[: -len(_CHAT_COMPLETIONS_PATH)] or "/"

    return urlunsplit(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            path.rstrip("/"),
            "",
            "",
        )
    )


def _openai_base_url() -> str:
    return _normalize_openai_base_url(OPENAI_URL)


def _is_retryable_openai_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    return False


def _create_chat_completion(client: OpenAI, completion_args: dict) -> object:
    attempts = max(1, OPENAI_MAX_ATTEMPTS)
    delay = max(0.0, OPENAI_RETRY_INITIAL_DELAY)
    backoff = max(1.0, OPENAI_RETRY_BACKOFF)

    for attempt in range(attempts):
        try:
            return client.chat.completions.create(**completion_args)
        except Exception as exc:
            if not _is_retryable_openai_error(exc) or attempt == attempts - 1:
                raise
            if delay > 0:
                time.sleep(delay)
            delay *= backoff

    raise RuntimeError("OpenAI completion retry loop exited unexpectedly.")


def get_completion(
    prompt: str,
    system_message: str = "",
    model: str = TRANSLATION_MODEL,
    temperature: int = 1,
) -> str:
    return openai_completion(
        prompt=prompt,
        system_message=system_message,
        temperature=temperature,
        model=model,
    )


def openai_completion(
    prompt, system_message=None, temperature=0.3, model=TASK_MODEL, json_output=False
) -> str:
    if not OPENAI_KEY:
        raise ValueError("OPENAI_API_KEY is required.")
    if not model:
        raise ValueError("OpenAI model is required.")

    if system_message:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    client = OpenAI(
        api_key=OPENAI_KEY,
        base_url=_openai_base_url(),
        timeout=OPENAI_TIMEOUT,
        max_retries=0,
    )
    completion_args = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "stream": False,
    }
    if json_output:
        completion_args["response_format"] = {"type": "json_object"}

    res = _create_chat_completion(client, completion_args)
    return res.choices[0].message.content or ""
