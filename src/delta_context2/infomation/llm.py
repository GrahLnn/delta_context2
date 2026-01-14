import os
import random
import re
import time

import demjson3
import requests
import tiktoken
from dotenv import load_dotenv
from retry import retry

from ..infomation.read_metadata import read_metadata
from ..text.utils import split_sentences_into_chunks
from ..utils.decorator import show_progress, update_metadata
from .prompt import (
    SINGLE_TRANSLATION_PROMPT,
    SINGLE_TRANSLATION_PROMPT_WITH_CONTEXT,
    SUMMARY_SYS_MESSAGE,
)
from poolctrl import Pool, RateLimitRule

load_dotenv()
OPENAI_URL = os.getenv("OPENAI_API_URL")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TASK_MODEL = os.getenv("TASK_MODEL")
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL")
GEMINI_API = os.getenv("GEMINI_API")
GEMINI_KEYS = list(map(str.strip, os.getenv("GEMINI_API_KEY").split(",")))

pool = Pool(
    task_id="gemini",
    persist=True,
    limits=[
        RateLimitRule(max_requests=2, interval=1, time_unit="minute"),
        RateLimitRule(max_requests=1300, interval=1.5, time_unit="day"),
    ],
)


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
        tldr = get_completion(chunk + "\n\nAbstract this paragraph.")
        tldrs.append(tldr)

    summary_zh = get_completion(" ".join(tldrs) + "\n\n直接给这个视频一个中文摘要")

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


def choose_key():
    random.shuffle(GEMINI_KEYS)
    return GEMINI_KEYS[0]


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


@retry(tries=3, delay=2)
def call_api_without_authhead(url, data):
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.post(
        url, headers=headers, json=data, timeout=300
    )  # Added 5-minute timeout
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
    res = response.json()
    return res


def gemini_completion(prompt, system_message, temperature, model, key):
    os.makedirs("asset", exist_ok=True)

    payload = {
        "contents": {"parts": {"text": prompt}},
        "systemInstruction": {"parts": {"text": system_message}},
        "generationConfig": {
            "temperature": 0.7,
            "topK": 64,
            "topP": 0.95,
            "maxOutputTokens": 65536,
            "responseMimeType": "text/plain",
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ],
    }
    max_retries = 10
    for _ in range(max_retries):
        api_url = GEMINI_API + f"{model}:generateContent" + "?key=" + key
        res = call_api_without_authhead(api_url, payload)

        try:
            answer = res["candidates"][0]["content"]["parts"][0]["text"]
            return answer
        except Exception:
            print(
                "unexpect output: ",
                res,
            )
    print("API call failed. Related payload is save in prompt.txt.")
    with open("prompt.txt", "w") as f:
        f.write(prompt)

    raise Exception("Exceeded maximum retries.")


def get_completion(
    prompt: str,
    system_message: str = "",
    model: str = TRANSLATION_MODEL,
    temperature: int = 1,
) -> str:
    answer = ""
    failed_key = []
    for _ in range(len(GEMINI_KEYS) + 1):
        if "gemini" in model:
            try:
                with pool.context(GEMINI_KEYS) as key:
                    answer = gemini_completion(
                        prompt=prompt,
                        system_message=system_message,
                        temperature=temperature,
                        model=model,
                        key=key,
                    )
            except Exception as e:
                print(e)
                # failed_key.append(key)
                continue
        else:
            answer = openai_completion(
                prompt=prompt,
                system_message=system_message,
                temperature=temperature,
                model=model,
            )

        return answer
    raise Exception("Failed to get completion.")


def openai_completion(
    prompt, system_message=None, temperature=0.3, model=TASK_MODEL, json_output=False
) -> str:
    if system_message:
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    else:
        message = [{"role": "user", "content": prompt}]
    data = {
        "model": model,
        "temperature": temperature,
        "messages": message,
        "stream": False,
        "response_format": {"type": "json_object"} if json_output else None,
    }
    res = call_api(OPENAI_URL, OPENAI_KEY, data)
    answer = res["choices"][0]["message"]["content"]

    return answer


@retry(tries=5, delay=2)
def call_api(url, access_token, data):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        url, headers=headers, json=data, timeout=300
    )  # 添加5分钟超时
    response.raise_for_status()
    res = response.json()
    return res
