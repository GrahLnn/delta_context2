from unittest.mock import patch

from delta_context2.utils.subtitle import render_video_with_subtitles


def test_subtitle_render_uses_quality_control_without_bitrate_cap(tmp_path):
    source = tmp_path / "source" / "video.mp4"
    source.parent.mkdir()
    source.touch()
    subtitles = tmp_path / "subtitle.ass"
    subtitles.touch()
    output = tmp_path / "rendered"
    output.mkdir()

    with patch("delta_context2.utils.subtitle.subprocess.run") as run:
        render_video_with_subtitles(source, subtitles, output)

    command = run.call_args.args[0]
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-vf") + 1] == f"ass={subtitles.as_posix()}"
    assert command[command.index("-rc") + 1] == "vbr"
    assert command[command.index("-cq") + 1] == "24"
    assert command[command.index("-b:v") + 1] == "0"
    assert "-maxrate" not in command
    assert command[command.index("-c:a") + 1] == "copy"
