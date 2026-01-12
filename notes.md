# Notes: Support local file paths in process ytb_url

## Sources
- Local codebase (files to be identified)

## Findings
- TBD
## Findings
- `VideoProcessor.process` in `src/delta_context2/main.py` calls `get_ytb_video_info`, builds `item_dir` from `formal_folder_name(title)`, downloads via `download_ytb_mp4`, then processes audio/subtitles.
- `download_ytb_mp4` saves to `data/videos/<formal_name>/source/<sanitized_filename>.mp4`.
- `get_ytb_video_info` is YouTube-specific and not usable for local files.
