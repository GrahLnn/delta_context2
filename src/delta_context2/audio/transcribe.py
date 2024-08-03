import difflib
import re
import whisper

from ..infomation.llm import get_completion

from ..infomation.prompt import TRANSCRIBTION_CORECTION_PROMPT

# from pyannote.audio import Pipeline
# from pydub import AudioSegment
# from alive_progress import alive_bar, alive_it

from ..infomation.read_metadata import read_metadata
from ..utils.decorator import show_progress, update_metadata
from ..utils.list import flatten

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


@show_progress("Transcribing")
@update_metadata(
    ("transcription", lambda result: result["text"]),
    ("sentences", lambda result: result["sentences"]),
    ("words", lambda result: result["words"]),
)
def get_transcribe(item_dir, audio_path, description: str) -> dict:
    check = read_metadata(item_dir, ["transcription", "segments"])
    if check:
        result = {
            "text": check["transcription"],
            "segments": check["segments"],
        }
        return result
    model = whisper.load_model("large-v3")
    # segments = segment_audio(audio_path)
    # result = transcribe_audio(audio_path, segments, model)

    result = model.transcribe(
        audio=audio_path,
        word_timestamps=True,
        # prompt=description.split("\n")[0],
        initial_prompt="Prohibit the use of abbreviations.",
        language="en",
    )
    segments = result["segments"]
    # texts = [seg["text"] for seg in segments]
    words = flatten([seg["words"] for seg in segments])
    formal_words = format_words(words)
    if (n := len(result["text"].split())) != (m := len(formal_words)):
        raise ValueError(f"Error: The words({m}) and sentences({n}) do not match.")


    # sentences = merge_sentence(texts)

    transcribtion = corect_transcription(result["text"])
    trg_words = get_checked_words(words, result["text"], transcribtion)
    sentences = collect_sentences(trg_words)

    result = {
        "text": transcribtion,
        "sentences": sentences,
        "words": trg_words,
    }

    return result


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
    prompt = TRANSCRIBTION_CORECTION_PROMPT.format(TRANSCRIBED_TEXT=transcription)
    res = get_completion(prompt=prompt)
    pattern = r"<repaired_transcript>(.*?)</repaired_transcript>"
    matche = (
        re.findall(pattern, res, re.DOTALL)[0]
        .replace("\n\n", "\n")
        .replace("\n", " ")
        .replace("  ", " ")
        .strip()
    )
    return matche


def get_checked_words(ord_words: list, text1: str, text2: str) -> list:
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
                        "start": round(formwords[j1]["start"], 2),
                        "end": round(formwords[j1]["end"], 2),
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
    for word in words:
        if temp_sentence.endswith((".", "?", "!")):
            print(temp_sentence)
            sentences.append(temp_sentence)
            temp_sentence = ""
        temp_sentence += word["word"]

    if temp_sentence:
        print(temp_sentence)
        sentences.append(temp_sentence)
    return sentences
