# δ context 2

A tool to add any translation subtitles to any video. Using Whisper, translate agent translation. I strongly recommend using gemini-1.5-pro as the translation model; the results are truly amazing (chinese i tested).

## How to use

**install**

```shell
# If you are not using the uv package manager, you can simply copy the command that starts with pip.
# install pytorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install delta context 2
uv pip install git+https://github.com/GrahLnn/delta_context2.git
```

**useage**

Prepare your `.env` file in your project. Fill in according to the `.env.example` example.

Then

```python
from delta_context2 import VideoProcessor

source_lang, target_lang, country = "English", "Chinese", "China"
ytb_url = "https://www.youtube.com/watch?v=[id]"

video_processor = VideoProcessor(source_lang, target_lang, country)
video_processor.process(ytb_url)
# video in data/videos
```

## todo

- [x] suport chinese subtitle.
- [ ] other language subtitle. (maybe i won't do)
- [x] support for YouTube videos download.
- [ ] support other platform download. (e.g. bilibili, x)
- [ ] support local video laod.
- [ ] let model verify the facts in the transcription.
- [ ] generate illustrated articles correspondingly.