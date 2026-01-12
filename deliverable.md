# Deliverable: Support local file paths in process ytb_url

## Summary
- Added local-path detection in `VideoProcessor.process` to bypass YouTube metadata/download when the input is a file path.
- Local files are moved into the same `data/videos/<name>/source/` location used by downloads and then processed normally.

## Files Changed
- src/delta_context2/main.py