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
        "you learn orders of magnitude more if you sit down for lunch with a couple team members",
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
        "you learn orders of magnitude more if you sit down for lunch with a couple team members",
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
        "And inseparably he had this whole method",
        ["而且 与记录猜测次数这件事", "密不可分的是 他还设计了一整套方法"],
    )

    assert result == [
        "而且 与记录猜测次数这件事密不可分的是",
        "他还设计了一整套方法",
    ]
    assert "English source is context only" in captured["prompt"]


def test_llm_boundary_repair_rejects_text_changes(monkeypatch):
    from delta_context2.utils import align

    monkeypatch.setattr(
        align,
        "get_json_completion",
        lambda prompt: {"segments": ["修改过的中文"]},
    )

    original = ["而且 与记录猜测次数这件事", "密不可分的是 他还设计了一整套方法"]

    assert align.repair_subtitle_segments_for_readability(
        "And inseparably he had this whole method",
        original,
    ) == original


def test_normal_short_chinese_segment_does_not_trigger_llm(monkeypatch):
    from delta_context2.utils import align

    def fail_if_called(prompt):
        raise AssertionError("LLM repair should not run for normal short subtitles")

    monkeypatch.setattr(align, "get_json_completion", fail_if_called)

    original = ["想象一下 你有一个机器人", "而这种传输既慢又昂贵"]

    assert align.repair_subtitle_segments_for_readability(
        "Imagine you have a robot and the transmission is slow and costly",
        original,
    ) == original
