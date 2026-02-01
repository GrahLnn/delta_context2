import difflib

import regex as re
import whisper
from tqdm import tqdm

from ..infomation.llm import get_completion
from ..infomation.prompt import TRANSCRIBTION_CORECTION_PROMPT

# from pyannote.audio import Pipeline
# from pydub import AudioSegment
# from alive_progress import alive_bar, alive_it
from ..infomation.read_metadata import read_metadata
from ..text.utils import rm_repeated_sequences, split_para, split_text_into_chunks
from ..utils.decorator import show_progress, update_metadata
from ..utils.list import drop_duplicate, flatten

# def segment_audio(audio_path):
#     # 使用 pyannote.audio 进行语音分割
#     with alive_bar(
#         title="audio segmenting",
#         spinner="dots",
#         bar=None,
#         monitor=False,
#         stats=False,
#     ) as bar:
#         pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
#         diarization = pipeline(str(audio_path))
#         segments = []
#         for turn, _, speaker in diarization.itertracks(yield_label=True):
#             segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

#         # 调整分割片段的结束时间
#         for i in range(len(segments) - 1):
#             segments[i]["end"] = segments[i + 1]["start"]

#         # 获取音频文件的总时长并设置最后一个片段的结束时间
#         audio = AudioSegment.from_file(str(audio_path))
#         total_duration = len(audio) / 1000
#         segments[-1]["end"] = total_duration
#         bar()

#         # 创建保存分割片段的目录
#         chunks_dir = Path(audio_path).parent / "chunks"
#         os.makedirs(chunks_dir, exist_ok=True)

#         # 保存每个分割片段为单独的音频文件
#         for idx, segment in enumerate(segments):
#             start_time = segment["start"] * 1000  # 转换为毫秒
#             end_time = segment["end"] * 1000  # 转换为毫秒
#             chunk = audio[start_time:end_time]
#             chunk_path = chunks_dir / f"chunk_{idx + 1}_{segment['speaker']}.wav"
#             chunk.export(chunk_path, format="wav")

#     return segments


# def get_language(audio_segment, model: whisper):
#     result = model.transcribe(audio=audio_segment, task="language-detection")
#     return result["language"]


# def audio_segment_to_array(audio_path, start, end):
#     audio = AudioSegment.from_file(audio_path)
#     segment = audio[start * 1000 : end * 1000]  # pydub works in milliseconds
#     samples = (
#         np.array(segment.get_array_of_samples()).astype(np.float32) / 32768.0
#     )  # Convert to float32 and normalize
#     return samples


# def transcribe_audio(audio_path, segments, model: whisper):
#     result = []
#     for segment in alive_it(segments):
#         start = segment["start"]
#         end = segment["end"]
#         audio_array = audio_segment_to_array(audio_path, start, end)
#         language = get_language(audio_array, model)

#         # transcription = model.transcribe(
#         #     audio=audio_array,
#         #     task="transcribe",
#         #     word_timestamps=True,
#         # )
#         print(language)
#         # print(transcription["text"])
#         # result.append(
#         #     {
#         #         "speaker": segment["speaker"],
#         #         "start": start,
#         #         "end": end,
#         #         "text": transcription["text"],
#         #         "segments": transcription["segments"],
#         #     }
#         # )

#     result = {
#         "text": "".join([res["text"] for res in result]),
#         "segments": [res["segments"] for res in result],
#     }

#     return result


# @show_progress("Transcribing")
# @update_metadata(
#     ("ord_words", lambda result: result["ord_words"]),
#     ("ord_text", lambda result: result["ord_text"]),
#     ("language", lambda result: result["language"]),
# )
# def transcribe_audio(item_dir: str, audio_path: str) -> dict:
#     """
#     只负责调用 Whisper 模型进行音频转录，返回原始文本、分段、words、语言等原始信息。
#     不做任何纠正或后处理。
#     """
#     if audio_path is None:
#         raise ValueError("No audio path provided")

#     # 1. 加载模型 & 音频
#     model = whisper.load_model("turbo")
#     audio = whisper.load_audio(audio_path)

#     check = read_metadata(
#         item_dir,
#         ["ord_text", "ord_words", "language"],
#     )
#     if check:
#         return {
#             "ord_text": check.get("ord_text"),  # 原始转录文本
#             "ord_words": check.get("ord_words"),  # 原始 word 列表
#             "audio": audio,  # 已加载的音频，是否返回视需要而定
#             "language": check.get("language"),  # 语言
#         }

#     # 2. 开始转录
#     result = model.transcribe(
#         audio=audio,
#         word_timestamps=True,
#         initial_prompt="Please retain punctuation and capitalization in the transcript.",
#     )

#     # 3. 释放模型
#     del model

#     # 4. 整理并返回原始信息
#     ord_transcription = result["text"]
#     segments = result["segments"]
#     language = result["language"]
#     words = flatten([seg["words"] for seg in segments])  # 原始的 words 列表

#     return {
#         "ord_text": ord_transcription,  # 原始转录文本
#         "ord_words": words,  # 原始 word 列表
#         "audio": audio,  # 已加载的音频，是否返回视需要而定
#         "language": language,  # 语言
#     }


def _get_attr_or_key(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_qwen_words(time_stamps):
    words = []
    for stamp in time_stamps or []:
        if isinstance(stamp, dict):
            text = stamp.get("text") or stamp.get("word") or stamp.get("token")
            start = stamp.get("start") or stamp.get("start_time")
            end = stamp.get("end") or stamp.get("end_time")
        else:
            text = getattr(stamp, "text", None) or getattr(stamp, "word", None)
            start = getattr(stamp, "start", None) or getattr(stamp, "start_time", None)
            end = getattr(stamp, "end", None) or getattr(stamp, "end_time", None)
        if text is None:
            continue
        words.append({"word": text, "start": start, "end": end})
    return words


@show_progress("Transcribing")
@update_metadata(
    ("ord_words", lambda result: result["ord_words"]),
    ("ord_text", lambda result: result["ord_text"]),
    ("language", lambda result: result["language"]),
)
def transcribe_audio(item_dir: str, audio_path: str) -> dict:
    """
    使用 Qwen3-ASR (transformers 后端) + Forced Aligner 进行离线转录，返回原始信息。
    """
    if audio_path is None:
        raise ValueError("No audio path provided")

    audio = whisper.load_audio(audio_path)

    check = read_metadata(
        item_dir,
        ["ord_text", "ord_words", "language"],
    )
    if check:
        return {
            "ord_text": check.get("ord_text"),
            "ord_words": check.get("ord_words"),
            "audio": audio,
            "language": check.get("language"),
        }

    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs={
            "dtype": torch.bfloat16,
            "device_map": "cuda:0",
        },
    )

    results = model.transcribe(
        audio=audio_path,
        language=None,
        return_time_stamps=True,
    )

    del model

    if not results:
        raise ValueError("Qwen3-ASR returned no results")

    first = results[0]
    ord_transcription = _get_attr_or_key(first, "text", "")
    language = _get_attr_or_key(first, "language", None)
    time_stamps = (
        _get_attr_or_key(first, "time_stamps", None)
        or _get_attr_or_key(first, "timestamps", None)
        or _get_attr_or_key(first, "words", None)
        or []
    )
    words = _normalize_qwen_words(time_stamps)

    return {
        "ord_text": ord_transcription,
        "ord_words": words,
        "audio": audio,
        "language": language,
    }


@update_metadata(
    ("text", lambda result: result["text"]),
    ("sentences", lambda result: result["sentences"]),
    ("words", lambda result: result["words"]),
)
def correct_transcript(item_dir: str, transcribed_data: dict) -> dict:
    """
    只负责对 `transcribe_audio` 返回的原始转录结果进行纠正、分词对齐、去重处理等。
    """
    check = read_metadata(
        item_dir,
        ["text", "sentences", "language", "words"],
    )
    if check:
        return {
            "text": check.get("text"),  # 最终纠正文本
            "sentences": check.get("sentences"),  # 分好句的文本
            "words": check.get("words"),  # 对齐后的最终 words
            "audio": transcribed_data["audio"],
            "language": check.get("language"),
        }
    # 1. 取出原始转录数据
    ord_text = transcribed_data["ord_text"]
    words = transcribed_data["ord_words"]

    # 2. 初步对齐：把原始 words 与原始转录文本对齐
    words = align_diff_words(words, "".join([w["word"] for w in words]), ord_text)
    formal_words = format_words(words)

    # 3. 检查分词数是否匹配
    n = len(ord_text.split())
    m = len(formal_words)
    if n != m:
        # 只是简单打印一下对不上时的映射
        for sw, fw in zip(ord_text.split(), formal_words):
            print(sw, fw["word"])
        raise ValueError(f"Error: The words({m}) and sentences({n}) do not match.")

    # 4. 纠正文本（拼写或其他自定义修正）
    checked_transcription = corect_transcription(ord_text)

    # 5. 二次对齐纠正后结果
    trg_words = align_diff_words(words, ord_text, checked_transcription)

    # 6. 句子分割、去重、去除重复序列
    sentences = drop_duplicate(collect_sentences(trg_words))
    checked_transcription = rm_repeated_sequences(" ".join(sentences))

    # 7. 再次对齐，得到最终的对齐信息
    trg_words = align_diff_words(words, ord_text, checked_transcription)
    sentences = split_para(checked_transcription)

    # 8. 返回纠正后的文本及其他信息
    return {
        "text": checked_transcription,  # 最终纠正文本
        "sentences": sentences,  # 分好句的文本
        "words": trg_words,  # 对齐后的最终 words
        "audio": transcribed_data["audio"],  # 是否保留音频根据需求
        "language": transcribed_data["language"],
    }


def get_transcribe(item_dir: str, audio_path: str, description: str = None) -> dict:
    """
    通过先调用 transcribe_audio 做初步转录，再调用 correct_transcript 做纠正。
    最终返回纠正后的结果。
    """
    # 1. 调用转录
    raw_data = transcribe_audio(item_dir, audio_path)

    # 2. 调用纠正
    result_data = correct_transcript(item_dir, raw_data)

    # 3. 返回最终结果
    return result_data


# @show_progress("Transcribing")
# @update_metadata(
#     ("transcription", lambda result: result["text"]),
#     ("words", lambda result: result["words"]),
#     ("ord_words", lambda result: result["ord_words"]),
#     ("sentences", lambda result: result["sentences"]),
#     ("ord_text", lambda result: result["ord_text"]),
#     ("language", lambda result: result["language"]),
# )
# def get_transcribe(item_dir, audio_path, description: str = None) -> dict:
#     check = read_metadata(
#         item_dir,
#         ["transcription", "sentences", "words", "ord_text", "ord_words", "language"],
#     )
#     model, audio = (
#         (whisper.load_model("turbo"), whisper.load_audio(audio_path))
#         if audio_path
#         else (None, None)
#     )

#     if check:
#         checked_transcribtion = check["transcription"]
#         ord_transcription = check["ord_text"]
#         trg_words = check["words"]
#         sentences = check["sentences"]
#         words = check["ord_words"]
#         language = check["language"]
#     else:
#         # segments = segment_audio(audio_path)
#         # result = transcribe_audio(audio_path, segments, model)
#         if audio_path is None:
#             raise ValueError("No audio path provided")

#         result = model.transcribe(
#             audio=audio,
#             word_timestamps=True,
#         )
#         ord_transcription = result["text"]
#         segments = result["segments"]
#         language = result["language"]
#         # texts = [seg["text"] for seg in segments]
#         words = flatten([seg["words"] for seg in segments])
#         # Somtimes the transcription is not correct with segments words
#         words = align_diff_words(
#             words, "".join([word["word"] for word in words]), ord_transcription
#         )
#         formal_words = format_words(words)
#         if (n := len(ord_transcription.split())) != (m := len(formal_words)):
#             [
#                 print(sw, fw["word"])
#                 for sw, fw in zip(ord_transcription.split(), formal_words)
#             ]
#             raise ValueError(f"Error: The words({m}) and sentences({n}) do not match.")
#         checked_transcribtion = corect_transcription(ord_transcription)
#         trg_words = align_diff_words(words, ord_transcription, checked_transcribtion)
#         sentences = drop_duplicate(collect_sentences(trg_words))
#         checked_transcribtion = rm_repeated_sequences(" ".join(sentences))

#         print("diff len", len(checked_transcribtion) - len(ord_transcription))
#         trg_words = align_diff_words(words, ord_transcription, checked_transcribtion)
#         sentences = split_para(checked_transcribtion)

#     return {
#         "text": checked_transcribtion,
#         "ord_text": ord_transcription,
#         "sentences": sentences,
#         "ord_words": words,
#         "words": trg_words,
#         "audio": audio,
#         "language": language,
#     }


def merge_sentence(lst) -> list[str]:
    merged_sentences = []
    temp_sentence = ""
    for s in lst:
        if (
            temp_sentence.endswith(".")
            or temp_sentence.endswith("!")
            or temp_sentence.endswith("?")
        ):
            merged_sentences.append(temp_sentence)
            temp_sentence = ""
        temp_sentence += s
    if temp_sentence:
        merged_sentences.append(temp_sentence)
    return merged_sentences


def format_words(words):
    formal_words = []
    for word in words:
        if word["word"].startswith(" "):
            formal_words.append(word)
        else:
            formal_words[-1]["word"] += word["word"]
            formal_words[-1]["end"] = word["end"]
    return formal_words


def corect_transcription(transcription):
    chunks = split_text_into_chunks(transcription)
    texts = []
    for chunk in tqdm(chunks, desc="Correcting"):
        prompt = TRANSCRIBTION_CORECTION_PROMPT.format(TRANSCRIBED_TEXT=chunk)
        max_attempts = 3
        for attempt in range(max_attempts):
            res = get_completion(prompt=prompt)
            clean_text = re.sub(r"<[^>]*>", "", res)
            if clean_text:
                break
            if attempt == max_attempts - 1:
                raise ValueError(f"Failed to get a valid transcription.\n{res}")
        texts.append(clean_text.strip())
    return " ".join(texts)


def align_diff_words(ord_words: list, text1: str, text2: str) -> list:
    formwords = format_words(ord_words)

    words1 = text1.split()
    words2 = text2.split()

    s = difflib.SequenceMatcher(None, words1, words2)

    diff = s.get_opcodes()

    trg_words = []
    for tag, i1, i2, j1, j2 in diff:
        match tag:
            case "replace":
                replaced_words = words2[j1:j2]
                for text in replaced_words:
                    data = {
                        "word": " " + text,
                        "start": round(formwords[i2 - 1]["start"], 2),
                        "end": round(formwords[i2 - 1]["end"], 2),
                    }
                    trg_words.append(data)

            case "insert":
                inserted_words = words2[j1:j2]
                for text in inserted_words:
                    data = {
                        "word": " " + text,
                        "start": round(trg_words[-1]["start"], 2),
                        "end": round(trg_words[-1]["end"], 2),
                    }
                    trg_words.append(data)

            case "equal":
                for idx, text in enumerate(words1[i1:i2]):
                    data = {
                        "word": " " + text,
                        "start": round(formwords[i1 + idx]["start"], 2),
                        "end": round(formwords[i1 + idx]["end"], 2),
                    }

                    trg_words.append(data)

    return trg_words


def collect_sentences(words: list) -> list:
    sentences = []
    temp_sentence = ""
    last_word = None
    for word in words:
        if temp_sentence.endswith((".", "?", "!")):
            if not any([char.isupper() for char in last_word]):
                sentences.append(temp_sentence.strip())
                temp_sentence = ""
        temp_sentence += word["word"]
        last_word = word["word"]

    if temp_sentence:
        sentences.append(temp_sentence.strip())
    return sentences
