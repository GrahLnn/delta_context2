import difflib
import json
import toml
import os
import re
import shutil
from pathlib import Path

import demjson3
from tenacity import retry, stop_after_attempt

import tiktoken

from ..audio.transcribe import align_diff_words
from ..infomation.llm import (
    get_json_completion,
    openai_completion,
    tokenize,
)
from ..infomation.prompt import (
    PARAGRAPH_ALIGNMENT_TO_SENTENCE_PROMPT,
    SHORT_SEGMENT_TEXT_ALIGN_SENTENCE_ARRAY_PROMPT,
    SPLIT_SMALL_SENTENCE_PROMPT,
    SUBTITLE_SEGMENT_REPAIR_PROMPT,
)
from ..infomation.read_metadata import read_metadata
from ..text.utils import abs_uni_len, normalize_to_10
from ..utils.decorator import update_metadata
from ..utils.list import flatten
from ..utils.progress import step_progress, track_progress


_ZH_NUMBER_ENDINGS = (
    "零",
    "一",
    "二",
    "两",
    "三",
    "四",
    "五",
    "六",
    "七",
    "八",
    "九",
    "十",
    "百",
    "千",
    "万",
    "几",
    "多",
    "半",
)
_ZH_CLASSIFIER_PREFIXES = (
    "个",
    "种",
    "件",
    "条",
    "位",
    "次",
    "层",
    "类",
    "段",
    "份",
    "组",
    "批",
    "些",
)


def _should_join_zh_split_fragment(current, next_item):
    current = current.strip()
    next_item = next_item.strip()
    if len(current) <= 2:
        return True
    return current.endswith(_ZH_NUMBER_ENDINGS) and next_item.startswith(
        _ZH_CLASSIFIER_PREFIXES
    )


def modify_zh_list(zh_list):
    processed_list = []

    i = 0
    while i < len(zh_list):
        if i + 1 < len(zh_list) and _should_join_zh_split_fragment(
            zh_list[i], zh_list[i + 1]
        ):
            zh_list[i + 1] = zh_list[i] + zh_list[i + 1]
        else:
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


def second_split(zh_list, len_limit):
    new_list = []
    bar_len_check = [abs_uni_len(t) for t in zh_list]

    for idx, bar_len in enumerate(bar_len_check):
        if bar_len > len_limit:
            mod = custom_mod(bar_len, len_limit)
            prompt = SPLIT_SMALL_SENTENCE_PROMPT.format(
                PARTS_NUM=mod, TEXT=zh_list[idx]
            )
            result = openai_completion(prompt)
            result = re.sub(r"<[^>]*>", "", result).strip()

            result = result.split("\n")
            result = [r.strip() for r in result]
            new_list.extend(result)
        else:
            new_list.append(zh_list[idx])

    return [elem for elem in new_list if elem]


def retry_handler(retry_state):
    print(f"All retries failed. Last error: {retry_state.outcome.exception()}")
    print(f"Last attempt result: {retry_state.outcome.result()}")
    raise


@retry(stop=stop_after_attempt(3), retry_error_callback=retry_handler)
def get_aligned_sentences(prompt):
    sys_msg = "You have a special preference for JSON, and all your responses will be in the form of ```json{...}``` for users."
    result = openai_completion(prompt, sys_msg)
    pattern = re.compile(r"^json")
    json_str = pattern.sub("", result.strip().strip("```"))

    answer = demjson3.decode(json_str)["pair"]
    return answer


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


@retry(stop=stop_after_attempt(3))
def llm_align_sentences(source_text, translated_snetence_array):
    translated_snetence_array = [
        t.replace('"', "\\'") for t in translated_snetence_array
    ]
    prompt = SHORT_SEGMENT_TEXT_ALIGN_SENTENCE_ARRAY_PROMPT.format(
        SEGMENTED_SENTENCES_A=str(translated_snetence_array),
        UNSEGMENTED_TEXT_B=source_text,
    )
    result = get_aligned_sentences(prompt)
    zh_list = translated_snetence_array
    seen = set()
    en_list = [pair["sentence_b"] for pair in result]

    # 对列表进行处理，移除重复项，将重复的项替换为空字符串
    for i in range(len(en_list)):
        if en_list[i] in seen:
            en_list[i] = ""  # 重复的项赋值为空字符串
        else:
            seen.add(en_list[i])  # 将未见过的项添加到set中

    en_list = insert_deletions_into_sentences(en_list, source_text)

    # zh_len_r = abs_uni_len("".join(translated_snetence_array))
    # zh_len_s = abs_uni_len("".join())

    en_len_split = abs_uni_len("".join(en_list))
    en_len_source = abs_uni_len(source_text)
    if en_len_split != en_len_source:
        # print("radio split", en_len_split, en_len_source)
        en_list = radio_split(source_text, zh_list)

    return zh_list, en_list


def insert_deletions_into_sentences(sentences, original_text):
    # 将句子数组中的句子组合成一个字符串
    combined_text = " ".join(sentences)

    # 获取句子数组中每个句子的长度（以单词数计算）
    len_senb = [len(sentence.split()) for sentence in sentences]

    # 将原文字符串和组合后的字符串分割成单词列表
    words1 = original_text.split()
    words2 = combined_text.split()

    # 使用SequenceMatcher比较两个单词列表
    s = difflib.SequenceMatcher(None, words1, words2)
    diff = s.get_opcodes()

    # 计算句子数组中每个句子的起始和结束位置
    start_positions = []
    end_positions = []
    current_pos = 0
    for length in len_senb:
        start_positions.append(current_pos)
        current_pos += length
        end_positions.append(current_pos)

    # 用于收集更新后的句子
    updated_sentences = sentences[:]

    # 处理diff，将delete的部分插入到对应的句子中
    for tag, i1, i2, j1, j2 in diff:
        if tag == "delete":
            delete_words = words1[i1:i2]

            # 找到对应的句子
            for idx, (start, end) in enumerate(zip(start_positions, end_positions)):
                if start <= i1 < end:
                    # 计算插入位置在该句子中的具体索引
                    insert_position = i1 - start
                    senb_words = updated_sentences[idx].split()
                    updated_sentences[idx] = " ".join(
                        senb_words[:insert_position]
                        + delete_words
                        + senb_words[insert_position:]
                    )
                    break

    return updated_sentences


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


def en_large_diff_ratio_repair(zh_list, en_list):
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


def _join_zh_segments(left, right):
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    if right.startswith(("的", "地", "得")):
        return f"{left}{right}"
    return f"{left} {right}"


def _zh_len(text):
    return abs_uni_len(text.replace(" ", ""))


def _normalize_zh_segments_for_compare(segments):
    return re.sub(r"\s+", "", "".join(segments))


def _find_known_continuation_break(text):
    markers = (
        " 你",
        " 我",
        " 他",
        " 她",
        " 它",
        " 我们",
        " 你们",
        " 他们",
        " 她们",
        " 它们",
        " 这",
        " 那",
        " 而",
        " 并",
        " 但",
        " 所以",
        " 因为",
        " 如果",
        " 当",
        " 那么",
    )
    positions = [text.find(marker) for marker in markers if text.find(marker) > 0]
    return min(positions) if positions else -1


def _zh_ends_with_definite_open_relation(text):
    text = text.rstrip(" ，,;；。")
    return text.endswith(
        (
            "对",
            "向",
            "给",
            "把",
            "被",
            "将",
            "让",
            "使",
            "从",
            "和",
            "与",
            "比",
            "到",
            "在",
            "接近",
            "靠近",
            "分散到",
            "分布到",
            "需要",
            "需要去",
            "团队里",
            "组里",
            "其中",
        )
    )


def _zh_starts_with_definite_tail(text):
    text = text.strip()
    if text.startswith(("的", "地", "得")):
        return True
    return _zh_len(text) <= 14 and text.startswith(
        (
            "完全不是",
            "根本不是",
            "并不是",
            "也不是",
            "不是",
            "0 ",
            "1 ",
        )
    )


def _split_definite_zh_tail(previous, current, subtitle_len):
    current = current.strip()
    if not previous or not current:
        return "", current

    max_merged_len = int(subtitle_len * 1.65)
    current_break = _find_known_continuation_break(current)

    if current.startswith(("的", "地", "得")):
        if current_break > 0:
            return current[:current_break].strip(), current[current_break:].strip()
        if _zh_len(_join_zh_segments(previous, current)) <= max_merged_len:
            return current, ""

    if _zh_ends_with_definite_open_relation(previous):
        if current_break > 0:
            prefix = current[:current_break].strip()
            if _zh_len(_join_zh_segments(previous, prefix)) <= max_merged_len:
                return prefix, current[current_break:].strip()
        if _zh_len(_join_zh_segments(previous, current)) <= max_merged_len:
            return current, ""

    if _zh_starts_with_definite_tail(current):
        if _zh_len(_join_zh_segments(previous, current)) <= max_merged_len:
            return current, ""

    return "", current


def mechanically_repair_zh_subtitle_segments(zh_list, subtitle_len=27):
    result = []
    for zh in zh_list:
        current = zh.strip()
        if not current:
            continue
        if result:
            moved, current = _split_definite_zh_tail(
                result[-1], current, subtitle_len
            )
            if moved:
                result[-1] = _join_zh_segments(result[-1], moved)
            if not current:
                continue
        result.append(current)
    return result


def _zh_can_merge_for_readability(left, right, subtitle_len):
    max_merged_len = int(subtitle_len * 1.5)
    return _zh_len(_join_zh_segments(left, right)) <= max_merged_len


def _zh_starts_contrast_pair(text):
    return text.strip().startswith(
        (
            "不只是",
            "不只",
            "不仅",
            "不僅",
            "不光",
            "不单",
            "不單",
        )
    )


def _zh_starts_contrast_tail(text):
    return text.strip().startswith(("而是", "而不是", "也不是"))


def _zh_short_segment_prefers_previous(text):
    text = text.strip()
    return text.startswith(
        (
            "而是",
            "而不是",
            "也就是",
            "也就是说",
            "也就是說",
            "从而",
            "從而",
            "于是",
            "於是",
            "因此",
            "所以",
            "同时",
            "同時",
            "并且",
            "並且",
            "再顺着",
            "再順著",
            "也就",
        )
    )


def _zh_short_segment_prefers_next(text):
    text = text.strip()
    return text.startswith(
        (
            "不只是",
            "不只",
            "不仅",
            "不僅",
            "不光",
            "不单",
            "不單",
            "当",
            "當",
            "如果",
            "比如",
            "也许",
            "也許",
            "或者",
        )
    )


def _zh_is_short_for_readability(text, subtitle_len):
    return _zh_len(text) <= min(13, max(8, subtitle_len // 2))


def merge_short_zh_subtitle_segments_for_readability(zh_list, subtitle_len=27):
    result = []
    idx = 0
    while idx < len(zh_list):
        current = zh_list[idx].strip()
        if not current:
            idx += 1
            continue

        next_item = zh_list[idx + 1].strip() if idx + 1 < len(zh_list) else ""
        if (
            next_item
            and _zh_is_short_for_readability(current, subtitle_len)
            and _zh_starts_contrast_pair(current)
            and _zh_starts_contrast_tail(next_item)
            and _zh_can_merge_for_readability(current, next_item, subtitle_len)
        ):
            result.append(_join_zh_segments(current, next_item))
            idx += 2
            continue

        if (
            result
            and _zh_is_short_for_readability(current, subtitle_len)
            and (
                _zh_short_segment_prefers_previous(current)
                or (
                    _zh_starts_contrast_tail(current)
                    and _zh_starts_contrast_pair(result[-1])
                )
            )
            and _zh_can_merge_for_readability(result[-1], current, subtitle_len)
        ):
            result[-1] = _join_zh_segments(result[-1], current)
            idx += 1
            continue

        if (
            next_item
            and _zh_is_short_for_readability(current, subtitle_len)
            and _zh_short_segment_prefers_next(current)
            and _zh_can_merge_for_readability(current, next_item, subtitle_len)
        ):
            result.append(_join_zh_segments(current, next_item))
            idx += 2
            continue

        result.append(current)
        idx += 1

    return result


def _zh_segment_has_uncertain_readability_issue(prev_text, current_text):
    current = current_text.strip()
    previous = prev_text.strip()
    if not current:
        return False
    if current.startswith(
        ("而当", "而如果", "但如果", "并且当", "也就是", "并使用")
    ):
        return True
    if current.startswith(("密不可分的是", "热身例子", "更多可能的")):
        return True
    if current.endswith(("的", "了一个", "一个", "一种", "一些", "这件事")):
        return True
    if previous.endswith(("顿号", "逗号", "一个", "一种", "一些")):
        return True
    return False


def needs_llm_subtitle_segment_repair(zh_list):
    for idx, current in enumerate(zh_list):
        previous = zh_list[idx - 1] if idx > 0 else ""
        if _zh_segment_has_uncertain_readability_issue(previous, current):
            return True
    return False


def _repair_zh_segments_with_llm(zh_list, subtitle_len):
    prompt = SUBTITLE_SEGMENT_REPAIR_PROMPT.format(
        CHINESE_SEGMENTS=json.dumps(zh_list, ensure_ascii=False),
        SUBTITLE_LEN=subtitle_len,
    )
    result = get_json_completion(prompt)
    repaired = result.get("segments")
    if not isinstance(repaired, list) or not all(isinstance(s, str) for s in repaired):
        return zh_list

    repaired = [s.strip() for s in repaired if s.strip()]
    if _normalize_zh_segments_for_compare(repaired) != _normalize_zh_segments_for_compare(
        zh_list
    ):
        return zh_list
    return repaired


def repair_subtitle_segments_for_readability(zh_list, subtitle_len=27, use_llm=True):
    repaired = mechanically_repair_zh_subtitle_segments(zh_list, subtitle_len)
    if use_llm and needs_llm_subtitle_segment_repair(repaired):
        repaired = _repair_zh_segments_with_llm(repaired, subtitle_len)
    return merge_short_zh_subtitle_segments_for_readability(repaired, subtitle_len)



def _weighted_word_counts(total_words, weights):
    if not weights:
        return []
    if total_words < len(weights):
        return []

    total_weight = sum(weights)
    if total_weight <= 0:
        return []

    raw_counts = [weight / total_weight * total_words for weight in weights]
    counts = [max(1, int(count)) for count in raw_counts]

    while sum(counts) > total_words:
        candidates = [
            (raw_counts[idx] - counts[idx], idx)
            for idx, count in enumerate(counts)
            if count > 1
        ]
        if not candidates:
            return []
        _, idx = min(candidates)
        counts[idx] -= 1

    while sum(counts) < total_words:
        deficits = [(raw_counts[idx] - counts[idx], idx) for idx in range(len(counts))]
        _, idx = max(deficits)
        counts[idx] += 1

    return counts


def _has_subtitle_pacing_mismatch(zh_list, en_list):
    for zh, en in zip(zh_list, en_list):
        if _zh_len(zh) >= 10 and len(en.split()) <= 2:
            return True
    return False


def rebalance_en_segments_for_subtitle_pacing(zh_list, en_list):
    words = " ".join(en_list).split()
    if len(zh_list) != len(en_list) or _has_subtitle_pacing_mismatch(zh_list, en_list):
        counts = _weighted_word_counts(len(words), [_zh_len(zh) for zh in zh_list])
        if counts:
            result = []
            offset = 0
            for count in counts:
                result.append(" ".join(words[offset : offset + count]))
                offset += count
            return result
    return en_list


@update_metadata(("atomic_part", lambda result: result))
def split_to_atomic_part(
    dir, source_text_chunks, translated_chunks, subtitle_len=27, keep_cache=False
):
    os.makedirs("cache", exist_ok=True)
    encoding = tiktoken.encoding_for_model("gpt-4o")
    check = read_metadata(dir, ["atomic_part"]) if dir else None
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
        try_count = 0
        logsave = {
            "source_chunks": source_text_chunks,
            "translated_chunks": translated_chunks,
        }
        while True:
            chunk = source_text_chunks[i]
            translation = not_belong_this_chunk_zh + translated_chunks[i]
            not_belong_this_chunk_zh = ""
            prompt = PARAGRAPH_ALIGNMENT_TO_SENTENCE_PROMPT.format(
                PARAGRAPH_A=chunk.translate(str.maketrans(".,", "  ")),
                PARAGRAPH_B=translation.translate(str.maketrans("。，;", "   ")),
            )
            with step_progress(
                f"align chunk {i + 1}/{len(source_text_chunks)}"
            ) as advance:
                ttry_count = 0
                while True:
                    try:
                        result = get_json_completion(prompt)
                        logsave.update({"first_alignment": result})
                        for r in result["pair"]:
                            len_sena_int = len(tokenize(r["sentence_a"]))
                            len_senb_int = len(tokenize(r["sentence_b"]))
                            r["ratio"] = (
                                len_sena_int / len_senb_int if len_senb_int > 0 else 0
                            )
                        # print(json.dumps(result, ensure_ascii=False, indent=4))
                        a_sentences = [pair["sentence_a"] for pair in result["pair"]]
                        b_sentences = [pair["sentence_b"] for pair in result["pair"]]
                        break
                    except Exception as e:
                        print(e)
                        # print(prompt)
                        ttry_count += 1
                        if ttry_count == 3:
                            raise ValueError("can not get alignment in first alignment")

                nas = []
                nbs = []
                for a, b in zip(a_sentences, b_sentences):
                    if a.strip() == "":
                        nbs[-1] += " " + b.strip()
                        continue
                    nas.append(a)
                    nbs.append(b)

                # [print(s, t) for s, t in zip(nas, nbs)]

                en_texts = []
                zh_texts = []
                for idx, (source_text, translated_text) in enumerate(zip(nas, nbs)):
                    if translated_text.strip() == "":
                        en_texts[-1] += " " + source_text

                        # max_retry = 5
                        # for count in range(max_retry):
                        #     prompt = SINGLE_TRANSLATION_PROMPT_WITH_CONTEXT.format(
                        #         ORIGINAL_TEXT=source_text,
                        #         CONTEXT=str(
                        #             [(a, b) for a, b in zip(a_sentences, b_sentences)]
                        #         ),
                        #     )
                        #     res = openai_completion(prompt)
                        #     # res = get_completion(prompt)
                        #     res = re.sub(r"<[^>]*>", "", res).strip()
                        #     # print("补偿：", source_text, "->", res)
                        #     if len(extract_zh_char(res)) != 0:
                        #         zh_texts.append(res)
                        #         en_texts.append(source_text)
                        #         break
                        #     if count + 1 == max_retry:
                        #         raise ValueError("sentence can not translate")
                    else:
                        en_texts.append(source_text)
                        zh_texts.append(translated_text)
                advance()
            logsave.update(
                {"first_fix": [{"en": e, "zh": z} for e, z in zip(en_texts, zh_texts)]}
            )
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

            chunk_atomic_zhs = []
            chunk_atomic_ens = []
            for sentence_idx, (en_src, zh_tsl) in track_progress(
                enumerate(zip(en_texts, zh_texts)),
                total=len(zh_texts),
                title=f"split chunk {i + 1}/{len(source_text_chunks)}",
            ):
                if abs_uni_len(zh_tsl) > subtitle_len:
                    # print(zh_tsl)
                    if "，" in zh_tsl:
                        split_text = re.split("，|；", zh_tsl)
                        split_text = [s for s in split_text if s]
                        new_t = modify_zh_list(split_text)
                    else:
                        new_t = [zh_tsl]
                    new_t = second_split(new_t, subtitle_len)
                    new_t = second_split(new_t, subtitle_len)

                    llm_align_zh_list, llm_align_en_list = llm_align_sentences(
                        en_src, new_t
                    )
                    logsave.update(
                        {
                            f"llm_align_{sentence_idx}": [
                                {"en": e, "zh": z}
                                for e, z in zip(llm_align_en_list, llm_align_zh_list)
                            ]
                        }
                    )
                    # [print(s, t) for s, t in zip(llm_align_zh_list, llm_align_en_list)]
                    # print("--------------")
                    try:
                        zh_list, en_list = hand_repair(
                            llm_align_zh_list, llm_align_en_list
                        )
                    except Exception as e:
                        [
                            print(s, t)
                            for s, t in zip(llm_align_zh_list, llm_align_en_list)
                        ]
                        raise e
                    logsave.update(
                        {
                            f"hand_repair_{sentence_idx}": [
                                {"en": e, "zh": z} for e, z in zip(en_list, zh_list)
                            ]
                        }
                    )
                    if abs_uni_len("".join(en_list)) == 0:
                        raise ValueError(
                            f"empty translation: {[s + '|' + t for s, t in zip(llm_align_zh_list, llm_align_en_list)]}"
                        )
                    en_list = en_large_diff_ratio_repair(zh_list, en_list)
                    en_list = move_commas(en_list)

                    # [print(s, t) for s, t in zip(zh_list, en_list)]
                    # print("--------------")
                    nzh_list = []
                    for item in zh_list:
                        if len(item) > subtitle_len:
                            # print(">27", item)
                            split_t = item.split(" ")
                            fix_item = ""
                            for s in split_t:
                                if len(s) > subtitle_len:
                                    # print(">27s", s)
                                    token_integers = encoding.encode(s)
                                    parts = len(s) // subtitle_len + 1
                                    tokens_per_part = len(token_integers) // parts
                                    ffix_item = ""
                                    for ii in range(parts):
                                        start = ii * tokens_per_part
                                        end = (
                                            (ii + 1) * tokens_per_part
                                            if ii < parts - 1
                                            else None
                                        )
                                        part = encoding.decode(
                                            token_integers[start:end]
                                        )
                                        ffix_item += part + " "
                                    fix_item += ffix_item
                                else:
                                    fix_item += s + " "
                            # print(">27f", fix_item.strip())
                            nzh_list.append(fix_item.strip())
                        else:
                            nzh_list.append(item)
                    repaired_zh_list = repair_subtitle_segments_for_readability(
                        nzh_list, subtitle_len
                    )
                    repaired_en_list = rebalance_en_segments_for_subtitle_pacing(
                        repaired_zh_list, en_list
                    )
                    if len(repaired_zh_list) == len(repaired_en_list):
                        nzh_list = repaired_zh_list
                        en_list = repaired_en_list
                    logsave.update(
                        {
                            f"split_{sentence_idx}": [
                                {"en": e, "zh": z} for e, z in zip(en_list, nzh_list)
                            ]
                        }
                    )
                    chunk_atomic_zhs.extend(nzh_list)
                    chunk_atomic_ens.extend(en_list)

                else:
                    if en_src:
                        chunk_atomic_zhs.append(zh_tsl)
                        chunk_atomic_ens.append(en_src)
                    else:
                        chunk_atomic_zhs[-1] += "，" + zh_tsl
            if "" in chunk_atomic_ens or "" in chunk_atomic_zhs:
                try_count += 1
                if try_count == 3:
                    with open("log.toml", "w", encoding="utf-8") as f:
                        toml.dump(logsave, f)
                    raise ValueError("can not get alignment after 3 times")
                continue
            else:
                break
        atomic_zhs.extend(chunk_atomic_zhs)
        atomic_ens.extend(chunk_atomic_ens)
        done_idx += 1
        cache_data = {
            "done_idx": done_idx,
            "atomic_zhs": atomic_zhs,
            "atomic_ens": atomic_ens,
        }
        with open(
            Path("cache") / "split_to_atomic_part.json", "w", encoding="utf-8"
        ) as file:
            json.dump(cache_data, file, ensure_ascii=False, indent=4)

    if dir:
        with open(Path(dir) / "metadata.json", encoding="utf-8") as file:
            data = json.load(file)
        with open(Path(dir) / "metadata.json", "w", encoding="utf-8") as file:
            final_transcribe = " ".join(atomic_ens)
            data["final_transcribe"] = final_transcribe
            json.dump(data, file, ensure_ascii=False, indent=4)

    atomic_part = []
    for zh, en in zip(atomic_zhs, atomic_ens):
        atomic_part.append({"zh": zh, "en": en})
    if not keep_cache:
        shutil.rmtree("cache")
    return atomic_part


def adjust_timestamps(sentence_timestamps, start, end):
    """调整时间戳以确保连续性和一致性"""
    if sentence_timestamps and sentence_timestamps[-1]["end"] > start:
        if start > sentence_timestamps[-1]["start"]:
            sentence_timestamps[-1]["end"], start = (
                start,
                sentence_timestamps[-1]["end"],
            )
        else:
            start = sentence_timestamps[-1]["end"]
    return start, end


@update_metadata(("sentence_timestamps", lambda result: result))
def get_sentence_timestamps(dir, atomic_ens, words, atomic_zhs):
    split_atomic_ens = [s.split() for s in atomic_ens]

    if len(flatten(split_atomic_ens)) != len(words):
        # raise ValueError("Warning: The words and sentences do not match.")
        words = align_diff_words(
            words,
            "".join([word["word"] for word in words]).strip(),
            " ".join(atomic_ens).replace("  ", " ").strip(),
        )

    sentence_timestamps = []

    word_index = 0

    for sentence, zh_stc in zip(split_atomic_ens, atomic_zhs):
        # print(word_index, word_index + len(sentence) - 1)
        sentence_start = words[word_index]["start"]
        sentence_end = words[word_index + len(sentence) - 1]["end"]

        # 处理中文句子的标点符号
        zh_stc = re.sub(r"[，]", " ", zh_stc).strip()
        zh_stc = re.sub(r"[。；,*]", "", zh_stc)

        if zh_stc:
            min_duration = (abs_uni_len(zh_stc) // 6) * 1.0
            if sentence_timestamps and sentence_timestamps[-1]["end"] > sentence_start:
                sentence_start = sentence_timestamps[-1]["end"]

            sentence_timestamps.append(
                {
                    "text": zh_stc,
                    "en_text": " ".join(sentence),
                    "start": sentence_start,
                    "end": sentence_end,
                    "words": words[word_index : word_index + len(sentence)],
                }
            )

        word_index += len(sentence)

    return sentence_timestamps
