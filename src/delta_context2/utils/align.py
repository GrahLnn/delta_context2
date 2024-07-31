import json
import re
import sys

from alive_progress import alive_bar, alive_it
from retry import retry

from ..infomation.llm import get_json_completion, openai_completion
from ..infomation.prompt import (
    PARAGRAPH_ALIGNMENT_TO_SENTENCE_PROMPT,
    SHORT_SEGMENT_TEXT_ALIGN_SENTENCE_ARRAY_PROMPT,
    SINGLE_TRANSLATION_PROMPT,
    SPLIT_SMALL_SENTENCE_PROMPT,
)
from ..infomation.read_metadata import read_metadata
from ..text.utils import abs_uni_len, extract_zh_char, normalize_to_10
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
    pattern = re.compile(r"^json")
    result = json.loads(pattern.sub("", result.strip("```")))["pair"]
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

    new_zh_list = []
    new_en_list = []
    if 0 in en_check:
        for idx, _ in enumerate(en_check):
            if en_check[idx] == 0:
                if idx == 0:
                    print("none english in zero idx")
                    sys.exit(0)
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

                    new_en_list[-1] = part1
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
    check = read_metadata(dir, ["atomic_zhs", "atomic_ens"])
    if check:
        return check["atomic_zhs"], check["atomic_ens"]
    atomic_zhs = []
    atomic_ens = []
    for i, (sentence, translation) in enumerate(
        zip(source_text_chunks, translated_chunks)
    ):
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
            result = get_json_completion(prompt)
            print(json.dumps(result, indent=4, ensure_ascii=False))
            a_sentences = [pair["sentence_a"] for pair in result["pair"]]
            b_sentences = [pair["sentence_b"] for pair in result["pair"]]
            for idx, (source_text, translated_text) in enumerate(
                zip(a_sentences, b_sentences)
            ):
                if translated_text.strip() == "":
                    if len(source_text.split()) == 1:
                        continue
                    max_retry = 5
                    for count in range(max_retry):
                        prompt = SINGLE_TRANSLATION_PROMPT.format(TEXT=source_text)
                        res = openai_completion(prompt)
                        if len(extract_zh_char(res)) != 0:
                            b_sentences[idx] = res
                            break
                        if count + 1 == max_retry:
                            raise ValueError("sentence can not translate")
                bar()
        print(abs_uni_len("".join(a_sentences)), abs_uni_len("".join(sentence)))
        if abs(abs_uni_len("".join(a_sentences)) - abs_uni_len("".join(sentence))) > 10:
            [print(s, t) for s, t in zip(a_sentences, b_sentences)]
            print(abs_uni_len("".join(a_sentences)), abs_uni_len("".join(sentence)))
            raise ValueError("abs_uni_len not equal")

        # if "" in trans:
        #     raise ValueError("empty translation")

        for en_src, zh_tsl in alive_it(
            zip(a_sentences, b_sentences),
            total=len(a_sentences),
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
                zh_list, en_list = llm_align_sentences(en_src, new_t)
                zh_list, en_list = hand_repair(zh_list, en_list)
                if abs(abs_uni_len("".join(en_list)) - abs_uni_len(en_src)) > 10:
                    print(en_list)
                    aa = abs_uni_len("".join(en_list))
                    bb = abs_uni_len(en_src)

                    raise ValueError(
                        f"abs_uni_len not equal {aa}/{bb} {en_list} {en_src} {zh_list}"
                    )
                en_list = en_large_diff_radio_repair(zh_list, en_list)
                if abs(abs_uni_len("".join(en_list)) - abs_uni_len(en_src)) > 10:
                    print(en_list)
                    aa = abs_uni_len("".join(en_list))
                    bb = abs_uni_len(en_src)

                    raise ValueError(f"abs_uni_len not equal {aa}/{bb} {en_list}")
                en_list = move_commas(en_list)

                if abs(abs_uni_len("".join(en_list)) - abs_uni_len(en_src)) > 10:
                    print(en_list)
                    aa = abs_uni_len("".join(en_list))
                    bb = abs_uni_len(en_src)

                    raise ValueError(f"abs_uni_len not equal {aa}/{bb} {en_list}")

                atomic_zhs.extend(zh_list)
                atomic_ens.extend(en_list)

            else:
                if en_src:
                    atomic_zhs.append(zh_tsl)
                    atomic_ens.append(en_src)
                else:
                    atomic_zhs[-1] += "，" + zh_tsl

    atomic_part = []
    for zh, en in zip(atomic_zhs, atomic_ens):
        atomic_part.append({"zh": zh, "en": en})
    return atomic_part


@update_metadata(("sentence_timestamps", lambda result: result))
def get_sentence_timestamps(dir, atomic_ens, words, atomic_zhs):
    # 拆分 atomic_ens 中的每个句子为单词
    split_atomic_ens = [s.split() for s in atomic_ens]

    # 检查单词数量是否匹配
    if len(flatten(split_atomic_ens)) != len(words):
        [print(e, w["word"]) for e, w in zip(flatten(split_atomic_ens), words)]
        raise ValueError(
            f"The number of atomic_ens({len(flatten(split_atomic_ens))}) is not equal to the number of words({len(words)})."
        )

    # 用于存储句子的时间戳
    sentence_timestamps = []

    # 当前处理的单词索引
    word_index = 0

    # 遍历每个句子
    for sentence, zh_stc in zip(split_atomic_ens, atomic_zhs):
        # 获取句子的开始时间戳
        sentence_start = words[word_index]["start"]

        # 获取句子的结束时间戳
        sentence_end = words[word_index + len(sentence) - 1]["end"]
        zh_stc = zh_stc.replace("，", " ").replace("。", "").replace("；", "").strip()

        # 将句子的时间戳存储到结果列表中
        if zh_stc:
            sentence_timestamps.append(
                {
                    "text": zh_stc,
                    "start": sentence_start,
                    "end": sentence_end,
                }
            )

        # 更新单词索引
        word_index += len(sentence)

    return sentence_timestamps
