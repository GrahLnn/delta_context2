import json
import os
import re
import shutil
from pathlib import Path

import demjson3
from alive_progress import alive_bar, alive_it
from retry import retry

from ..audio.transcribe import align_diff_words
from ..infomation.llm import get_json_completion, openai_completion
from ..infomation.prompt import (
    PARAGRAPH_ALIGNMENT_TO_SENTENCE_PROMPT,
    SHORT_SEGMENT_TEXT_ALIGN_SENTENCE_ARRAY_PROMPT,
    SPLIT_SMALL_SENTENCE_PROMPT,
)
from ..infomation.read_metadata import read_metadata
from ..text.utils import abs_uni_len, normalize_to_10
from ..utils.decorator import update_metadata
from ..utils.list import flatten


def modify_zh_list(zh_list):
    processed_list = []

    i = 0
    while i < len(zh_list):
        if len(zh_list[i]) <= 2 and i + 1 < len(zh_list):
            # 如果当前项字符数小于等于2，并且不是最后一项，则将其添加到后一项
            zh_list[i + 1] = zh_list[i] + "，" + zh_list[i + 1]
        else:
            # 否则将当前项添加到结果列表中
            processed_list.append(zh_list[i])
        i += 1
    return processed_list


def modify_en_list(en_list):
    processed_list = []

    i = 0
    while i < len(en_list):
        if len(en_list[i].split()) == 1 and i + 1 < len(en_list):
            # 如果当前项单词数为1，并且不是最后一项，则将其添加到后一项
            en_list[i + 1] = en_list[i] + "," + en_list[i + 1]
        else:
            # 否则将当前项添加到结果列表中
            processed_list.append(en_list[i])
        i += 1
    return processed_list


def custom_mod(x, y):
    div = x // y + 1
    if x % y > y / 2:
        div += 1
    return div


def secend_split(zh_list, len_limit):
    new_list = []
    bar_len_check = [abs_uni_len(t) for t in zh_list]

    for idx, bar_len in enumerate(bar_len_check):
        if bar_len > len_limit:
            mod = custom_mod(bar_len, len_limit)
            prompt = SPLIT_SMALL_SENTENCE_PROMPT.format(
                PARTS_NUM=mod, TEXT=zh_list[idx]
            )
            result = openai_completion(prompt)

            result = result.split("\n")
            result = [r.strip() for r in result]
            new_list.extend(result)
        else:
            new_list.append(zh_list[idx])

    return [elem for elem in new_list if elem]


@retry(tries=3, delay=2)
def get_aligned_sentences(prompt):
    result = openai_completion(prompt)
    print(result)
    pattern = re.compile(r"^json")
    result = demjson3.decode(pattern.sub("", result.strip("```")))["pair"]
    return result


def radio_split(target_str, reference_list):
    len_ref = [abs_uni_len(n) for n in reference_list]
    words = target_str.split()

    # 计算 len_b 的总和
    total_len_b = sum(len_ref)

    # 计算每段应该包含的单词数
    proportions = [length / total_len_b for length in len_ref]
    total_words = len(target_str.split())
    words_per_segment = [int(proportion * total_words) for proportion in proportions]

    # 确保总单词数与原文本一致（由于整除可能有误差）
    adjustment = total_words - sum(words_per_segment)
    words_per_segment[-1] += adjustment

    # 将 words 分段
    words = target_str.split()
    segments = []
    start = 0

    for count in words_per_segment:
        end = start + count
        segments.append(" ".join(words[start:end]))
        start = end

    return segments


@retry(tries=3, delay=2, exceptions=ValueError)
def llm_align_sentences(source_text, translated_snetence_array):
    prompt = SHORT_SEGMENT_TEXT_ALIGN_SENTENCE_ARRAY_PROMPT.format(
        SEGMENTED_SENTENCES_A=str(translated_snetence_array),
        UNSEGMENTED_TEXT_B=source_text,
    )
    result = get_aligned_sentences(prompt)
    zh_list = translated_snetence_array
    en_list = [pair["sentence_b"] for pair in result]

    len_split = abs_uni_len("".join(en_list))
    len_source = abs_uni_len(source_text)
    if len_split != len_source:
        en_list = radio_split(source_text, zh_list)

    return zh_list, en_list


def hand_repair(zh_list, en_list):
    en_check = [abs_uni_len(en) for en in en_list]

    check_zh_list = []
    check_en_list = []
    for zh, en in zip(zh_list, en_list):
        if not zh.strip():
            check_en_list[-1] += (
                (" " + en) if check_en_list else check_en_list.append(en)
            )
        else:
            check_zh_list.append(zh)
            check_en_list.append(en)
    zh_list, en_list = check_zh_list, check_en_list
    new_zh_list = []
    new_en_list = []
    if 0 in en_check:
        for idx, _ in enumerate(en_check):
            if en_check[idx] == 0:
                if idx == 0:
                    continue
                if abs_uni_len(zh_list[idx] + zh_list[idx - 1]) <= 27:
                    new_zh_list[-1] = new_zh_list[-1] + "，" + zh_list[idx]
                    new_en_list[-1] = new_en_list[-1] + "" + en_list[idx]
                else:
                    radio = [
                        abs_uni_len(zh_list[idx - 1]),
                        abs_uni_len(zh_list[idx]),
                    ]
                    total_len = sum(radio)
                    split_ratio = radio[0] / total_len
                    words = (en_list[idx - 1] + en_list[idx]).split()
                    split_index = int(len(words) * split_ratio)
                    part1 = " ".join(words[:split_index])
                    part2 = " ".join(words[split_index:])

                    if new_en_list:
                        new_en_list[-1] = part1
                    else:
                        new_en_list.append(part1)
                    new_en_list.append(part2)
                    new_zh_list.append(zh_list[idx])

            else:
                new_zh_list.append(zh_list[idx])
                new_en_list.append(en_list[idx])
    else:
        new_zh_list = zh_list
        new_en_list = en_list

    return new_zh_list, new_en_list


def radio_check(lst):
    whole_length = abs_uni_len("".join(lst))
    radios = [normalize_to_10(abs_uni_len(s), whole_length) for s in lst]
    return radios


def en_large_diff_radio_repair(zh_list, en_list):
    zh_rid = radio_check(zh_list)
    en_rid = radio_check(en_list)
    new_en_list = en_list[:]

    def adjust_split(idx1, idx2):
        radio = [abs_uni_len(zh_list[idx1]), abs_uni_len(zh_list[idx2])]
        total_len = sum(radio)
        split_ratio = radio[0] / total_len
        words = (new_en_list[idx1] + " " + new_en_list[idx2]).split()
        split_index = int(len(words) * split_ratio)
        new_en_list[idx1] = " ".join(words[:split_index])
        new_en_list[idx2] = " ".join(words[split_index:])

    for idx, (zr, er) in enumerate(zip(zh_rid, en_rid)):
        if zr - er >= 3:
            if idx == 0:
                adjust_split(idx, idx + 1)
            else:
                adjust_split(idx - 1, idx)

    return new_en_list


def move_commas(en_list):
    result = []
    for i in range(len(en_list)):
        if en_list[i].startswith(","):
            # 如果当前句子以逗号开头，将逗号移到上一个句子末尾
            if result:
                result[-1] += ","
            # 移除当前句子的开头逗号
            result.append(en_list[i][1:].strip())
        else:
            result.append(en_list[i].strip())
    return result


@update_metadata(("atomic_part", lambda result: result))
def split_to_atomic_part(dir, source_text_chunks, translated_chunks, subtitle_len=27):
    os.makedirs("cache", exist_ok=True)
    check = read_metadata(dir, ["atomic_part"])
    if check:
        shutil.rmtree("cache")
        return check
    atomic_zhs = []
    atomic_ens = []
    done_idx = -1
    if os.path.exists(Path("cache") / "split_to_atomic_part.json"):
        with open(
            Path("cache") / "split_to_atomic_part.json", encoding="utf-8"
        ) as file:
            file = file.read()
            cache_data = demjson3.decode(file)
        atomic_zhs = cache_data["atomic_zhs"]
        atomic_ens = cache_data["atomic_ens"]
        done_idx = cache_data.get("done_idx")
    not_belong_this_chunk_zh = ""
    for i in range(done_idx + 1, len(source_text_chunks)):
        sentence = source_text_chunks[i]
        translation = not_belong_this_chunk_zh + translated_chunks[i]
        not_belong_this_chunk_zh = ""
        prompt = PARAGRAPH_ALIGNMENT_TO_SENTENCE_PROMPT.format(
            PARAGRAPH_A="".join(sentence),
            PARAGRAPH_B=translation.strip().replace("。", " ").replace("，", " "),
        )
        with alive_bar(
            1,
            title=f"align chunk {i + 1}/{len(source_text_chunks)}",
            bar=None,
            stats=False,
            monitor=False,
        ) as bar:
            try_count = 0
            while True:
                try:
                    result = get_json_completion(prompt)
                    a_sentences = [pair["sentence_a"] for pair in result["pair"]]
                    b_sentences = [pair["sentence_b"] for pair in result["pair"]]
                    break
                except Exception as e:
                    print(e)
                    print(json.dumps(result, indent=4, ensure_ascii=False))
                    try_count += 1
                    if try_count == 3:
                        raise ValueError("can not get alignment")

            en_texts = []
            zh_texts = []
            for idx, (source_text, translated_text) in enumerate(
                zip(a_sentences, b_sentences)
            ):
                if translated_text.strip() == "":
                    if en_texts:
                        en_texts[-1] += " " + source_text
                    else:
                        en_texts.append(source_text)

                    # if len(source_text.split()) == 1:
                    #     continue
                    # max_retry = 5
                    # for count in range(max_retry):
                    #     prompt = SINGLE_TRANSLATION_PROMPT.format(
                    #         ORIGINAL_TEXT=source_text
                    #     )
                    #     res = openai_completion(prompt)
                    #     print("补偿：", source_text, "->", res)
                    #     if len(extract_zh_char(res)) != 0:
                    #         b_sentences[idx] = res
                    #         break
                    #     if count + 1 == max_retry:
                    #         raise ValueError("sentence can not translate")
                else:
                    en_texts.append(source_text)
                    zh_texts.append(translated_text)
            bar()
        empty_indices = []
        for emptyi in range(len(en_texts) - 1, -1, -1):
            if en_texts[emptyi] == "":
                empty_indices.append(emptyi)
            else:
                break

        # 根据空项的索引从 z 中移除并保存到另一个列表
        removed_items = []
        for index in sorted(empty_indices, reverse=True):
            removed_items.append(en_texts.pop(index))
            # en_texts.pop(index)

        # 逆序保存的 removed_items 列表，因为我们是从末尾开始移除的
        removed_items.reverse()

        not_belong_this_chunk_zh = (
            " ".join(removed_items) + " " if removed_items else ""
        )

        for en_src, zh_tsl in alive_it(
            zip(en_texts, zh_texts),
            total=len(zh_texts),
            title=f"split chunk {i + 1}/{len(source_text_chunks)}",
        ):
            if abs_uni_len(zh_tsl) > subtitle_len:
                if "，" in zh_tsl:
                    split_text = re.split("，|；", zh_tsl)
                    split_text = [s for s in split_text if s]
                    new_t = modify_zh_list(split_text)
                else:
                    new_t = [zh_tsl]
                new_t = secend_split(new_t, subtitle_len)
                llm_align_zh_list, llm_align_en_list = llm_align_sentences(
                    en_src, new_t
                )

                zh_list, en_list = hand_repair(llm_align_zh_list, llm_align_en_list)
                if abs_uni_len("".join(en_list)) == 0:
                    raise ValueError(f"empty translation\n{llm_align_en_list}")
                en_list = en_large_diff_radio_repair(zh_list, en_list)
                en_list = move_commas(en_list)

                atomic_zhs.extend(zh_list)
                atomic_ens.extend(en_list)

            else:
                if en_src:
                    atomic_zhs.append(zh_tsl)
                    atomic_ens.append(en_src)
                else:
                    atomic_zhs[-1] += "，" + zh_tsl

        cache_data = {
            "done_idx": i,
            "atomic_zhs": atomic_zhs,
            "atomic_ens": atomic_ens,
        }
        with open(
            Path("cache") / "split_to_atomic_part.json", "w", encoding="utf-8"
        ) as file:
            json.dump(cache_data, file, ensure_ascii=False, indent=4)

    with open(Path(dir) / "metadata.json", encoding="utf-8") as file:
        data = json.load(file)
    with open(Path(dir) / "metadata.json", "w", encoding="utf-8") as file:
        final_transcribe = " ".join(atomic_ens)
        data["final_transcribe"] = final_transcribe
        json.dump(data, file, ensure_ascii=False, indent=4)

    atomic_part = []
    for zh, en in zip(atomic_zhs, atomic_ens):
        atomic_part.append({"zh": zh, "en": en})
    shutil.rmtree("cache")
    return atomic_part


@update_metadata(("sentence_timestamps", lambda result: result))
def get_sentence_timestamps(dir, atomic_ens, words, atomic_zhs):
    split_atomic_ens = [s.split() for s in atomic_ens]

    if len(flatten(split_atomic_ens)) != len(words):
        words = align_diff_words(
            words,
            "".join([word["word"] for word in words]).strip(),
            " ".join(atomic_ens).replace("  ", " ").strip(),
        )

    sentence_timestamps = []

    word_index = 0

    for sentence, zh_stc in zip(split_atomic_ens, atomic_zhs):
        sentence_start = words[word_index]["start"]

        sentence_end = words[word_index + len(sentence) - 1]["end"]
        zh_stc = zh_stc.replace("，", " ").replace("。", "").replace("；", "").strip()

        if zh_stc:
            sentence_timestamps.append(
                {
                    "text": zh_stc,
                    "start": sentence_start,
                    "end": sentence_end,
                }
            )

        word_index += len(sentence)

    return sentence_timestamps
