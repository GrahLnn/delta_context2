import re
import unicodedata

import tiktoken

from ..utils.decorator import update_metadata


def extract_zh_char(text):
    # 使用正则表达式匹配中文字符
    chinese_characters = re.findall(r"[\u4e00-\u9fff]+", text)
    # 将所有匹配到的中文字符拼接成一个字符串
    return "".join(chinese_characters)


def remove_illegal_chars(filename):
    # 定义非法字符的正则表达式模式
    illegal_chars = r'[<>:"/\\|?*\x00-\x1F]'
    # 使用正则表达式替换非法字符为空字符串
    sanitized_filename = re.sub(illegal_chars, "", filename)
    return sanitized_filename


def sanitize_filename(filename):
    # 定义非法字符的正则表达式
    illegal_characters = r'[<>:"\\|?*]'
    replacement_dict = {
        "/": "-",
        "\\": "-",
        "|": "-",
        "#": "-",
    }

    # 替换两个斜杠与竖线为 '-'
    for char, replacement in replacement_dict.items():
        filename = filename.replace(char, replacement)

    # 替换其余非法字符为空
    filename = re.sub(illegal_characters, "", filename)

    return filename


def abs_uni_len(s):
    # 过滤出中文字符
    chinese_chars = [c for c in s if unicodedata.category(c) == "Lo"]

    # 过滤出非中文字符，并排除空白字符、标点符号和其他符号
    non_chinese_chars = [
        c
        for c in s
        if unicodedata.category(c) != "Lo"
        and not unicodedata.category(c).startswith("Zs")
        and not unicodedata.category(c).startswith("P")
        and not unicodedata.category(c).startswith("S")
    ]

    # 计算中文字符的长度
    chinese_length = len(chinese_chars)

    # 计算非中文字符的长度，字母字符计为0.5，其他字符计为1
    non_chinese_length = sum(
        0.5 if unicodedata.category(c).startswith("L") else 1 for c in non_chinese_chars
    )

    # 总长度
    total_length = chinese_length + non_chinese_length

    return int(total_length)


def split_sentences_into_chunks(sentences, max_tokens=1000):
    chunks = []
    chunk = []
    current_token_count = 0

    encoding = tiktoken.encoding_for_model("gpt-4o")

    sentences = merge_sentences_with_commas(sentences)

    for sentence in sentences:
        token_integers = encoding.encode(sentence)
        token_count = len(token_integers)

        # Check if adding this sentence would exceed the limit
        if current_token_count + token_count <= max_tokens:
            chunk.append(sentence)
            current_token_count += token_count
        else:
            # Save the current chunk and start a new one
            chunk_text = " ".join(chunk)
            if not chunk_text.strip():
                raise ValueError("Chunk is empty")
            chunks.append(chunk_text)
            chunk = [sentence]
            current_token_count = token_count

    # Add the last chunk if it's not empty
    if chunk:
        chunks.append(" ".join(chunk))

    return chunks


def split_text_into_chunks(text, max_tokens=5000, delimiter="."):
    parts = re.split(f"({re.escape(delimiter)})", text)
    # 将分隔符和前面的字符串合并
    combined_parts = [parts[i] + parts[i + 1] for i in range(0, len(parts) - 1, 2)]
    # 如果原始字符串不以分隔符结尾，则最后一个部分需要添加
    if len(parts) % 2 != 0:
        combined_parts.append(parts[-1])
    result = []
    for part in combined_parts:
        if result and part == delimiter:
            result[-1] += part
        elif part:
            result.append(part)

    chunks = split_sentences_into_chunks(result, max_tokens)
    return chunks


def normalize_to_10(value, max_value):
    return int(value / max_value * 10)


def lowercase_first_letter(s):
    if not s:
        return s
    return s[:2].lower() + s[2:]


def merge_sentences_with_commas(sentences):
    merged_sentences = []
    temp_sentence = ""

    for sentence in sentences:
        if temp_sentence:
            temp_sentence += lowercase_first_letter(sentence)
            if not sentence.strip().endswith(","):
                merged_sentences.append(temp_sentence.replace("  ", " "))
                temp_sentence = ""
        else:
            if sentence.strip().endswith(","):
                temp_sentence = sentence
            else:
                merged_sentences.append(sentence.replace("  ", " "))

    if temp_sentence:
        merged_sentences.append(temp_sentence)

    return merged_sentences


def split_paragraph_regex(paragraph):
    # 使用正则表达式匹配句子，考虑常见的句子结束符号
    sentences = re.findall(r"[^.!?;]*[.!?;]", paragraph)

    # 去除首尾的空白字符
    sentences = [sentence.strip() for sentence in sentences]

    return sentences


def rm_repeated_sequences(text):
    words = text.split()
    new_words = words[:]

    for idx, word in enumerate(words):
        if not re.search("[a-zA-Z]", word):
            continue
        for iidx, ww in enumerate(words):
            if iidx <= idx:
                continue
            if iidx - idx > 100:
                break
            if ww == word:
                if words[idx:iidx] == words[iidx : iidx + (iidx - idx)]:
                    new_words[iidx : iidx + (iidx - idx)] = [None] * (iidx - idx)
    new_text = " ".join([w for w in new_words if w])
    return new_text


def formal_file_name(name):
    return (
        name.replace(" ", "_")
        .replace(",", "")
        .replace("#", "")
        .replace("'", "")
        .replace("!", "")
    )
