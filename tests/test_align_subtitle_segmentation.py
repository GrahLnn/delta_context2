import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_modify_zh_list_joins_short_fragment_without_inserting_punctuation():
    from delta_context2.utils.align import modify_zh_list

    assert modify_zh_list(["完全不是一", "个量级"]) == ["完全不是一个量级"]


def test_definite_tail_repair_keeps_chinese_modifier_with_previous_segment():
    from delta_context2.utils.align import repair_subtitle_segments_for_readability

    result = repair_subtitle_segments_for_readability(
        [
            "到底是什么样子；而如果你有机会和团队里",
            "的几个人坐下来一起吃顿午饭 你能了解到的信息会多得多",
            "完全不是一个量级",
        ],
        use_llm=False,
    )

    assert result == [
        "到底是什么样子；而如果你有机会和团队里的几个人坐下来一起吃顿午饭",
        "你能了解到的信息会多得多 完全不是一个量级",
    ]


def test_rebalance_en_segments_keeps_anchor_count_after_chinese_boundary_repair():
    from delta_context2.utils.align import (
        rebalance_en_segments_for_subtitle_pacing,
        repair_subtitle_segments_for_readability,
    )

    zh_list = repair_subtitle_segments_for_readability(
        [
            "到底是什么样子；而如果你有机会和团队里",
            "的几个人坐下来一起吃顿午饭 你能了解到的信息会多得多",
            "完全不是一个量级",
        ],
        use_llm=False,
    )
    en_list = rebalance_en_segments_for_subtitle_pacing(
        zh_list,
        [
            "just by poking around online and you",
            "learn orders of magnitude more if you have a chance to sit down for lunch with",
            "a couple team members",
        ],
    )

    assert len(en_list) == len(zh_list)
    assert " ".join(en_list).split() == (
        "just by poking around online and you learn orders of magnitude more if "
        "you have a chance to sit down for lunch with a couple team members"
    ).split()


def test_uncertain_readability_issue_uses_llm_boundary_repair(monkeypatch):
    from delta_context2.utils import align

    captured = {}

    def fake_get_json_completion(prompt):
        captured["prompt"] = prompt
        return {
            "segments": [
                "而且 与记录猜测次数这件事密不可分的是",
                "他还设计了一整套方法",
            ]
        }

    monkeypatch.setattr(align, "get_json_completion", fake_get_json_completion)

    result = align.repair_subtitle_segments_for_readability(
        ["而且 与记录猜测次数这件事", "密不可分的是 他还设计了一整套方法"],
    )

    assert result == [
        "而且 与记录猜测次数这件事密不可分的是",
        "他还设计了一整套方法",
    ]
    assert "english_context" not in captured["prompt"]
    assert "And inseparably" not in captured["prompt"]


def test_llm_boundary_repair_rejects_text_changes(monkeypatch):
    from delta_context2.utils import align

    monkeypatch.setattr(
        align,
        "get_json_completion",
        lambda prompt: {"segments": ["修改过的中文"]},
    )

    original = ["而且 与记录猜测次数这件事", "密不可分的是 他还设计了一整套方法"]

    assert align.repair_subtitle_segments_for_readability(
        original,
    ) == original


def test_normal_short_chinese_segment_does_not_trigger_llm(monkeypatch):
    from delta_context2.utils import align

    def fail_if_called(prompt):
        raise AssertionError("LLM repair should not run for normal short subtitles")

    monkeypatch.setattr(align, "get_json_completion", fail_if_called)

    original = ["想象一下 你有一个机器人", "而这种传输既慢又昂贵"]

    assert align.repair_subtitle_segments_for_readability(
        original,
    ) == original


def test_short_contrast_subtitles_merge_for_readability():
    from delta_context2.utils import align

    original = ["不只是靠翻检书籍语料", "而是开始亲手设计它们了"]

    assert align.repair_subtitle_segments_for_readability(
        original,
        use_llm=False,
    ) == ["不只是靠翻检书籍语料 而是开始亲手设计它们了"]


def test_short_readability_merge_preserves_text_and_rebalances_english():
    from delta_context2.utils.align import (
        rebalance_en_segments_for_subtitle_pacing,
        repair_subtitle_segments_for_readability,
    )

    zh_list = [
        "把概率估计从",
        "不只是靠翻检书籍语料",
        "而是开始亲手设计它们了",
    ]
    en_list = [
        "moves the probability estimates from",
        "looking through books",
        "to designing them",
    ]

    repaired_zh_list = repair_subtitle_segments_for_readability(
        zh_list,
        use_llm=False,
    )
    repaired_en_list = rebalance_en_segments_for_subtitle_pacing(
        repaired_zh_list,
        en_list,
    )

    assert repaired_zh_list == [
        "把概率估计从 不只是靠翻检书籍语料 而是开始亲手设计它们了"
    ]
    assert "".join(repaired_zh_list).replace(" ", "") == "".join(zh_list)
    assert len(repaired_en_list) == len(repaired_zh_list)
    assert " ".join(repaired_en_list).split() == " ".join(en_list).split()


def test_short_readability_merge_keeps_unrelated_short_subtitles_separate():
    from delta_context2.utils import align

    original = ["想象一下 你有一个机器人", "而这种传输既慢又昂贵"]

    assert align.repair_subtitle_segments_for_readability(
        original,
        use_llm=False,
    ) == original
