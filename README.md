# delta_context2

A tool to add any translation subtitles to any video. Using Whisper, translate agent translation.

## How to use

**install**

```shell
# install pytorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install delta context 2
uv pip install git+https://github.com/GrahLnn/delta_context2.git
```

**useage**

Prepare your .env file in your project. Fill in according to the .env.example example.

Import to use.

```python
from delta_context2 import VideoProcessor

source_lang, target_lang, country = "English", "Chinese", "China"
ytb_url = "https://www.youtube.com/watch?v=[id]"

video_processor = VideoProcessor(source_lang, target_lang, country)
video_processor.process(ytb_url)
# video in data/videos
```

## roadmap

- [x] chinese subtitle
- [ ] other language subtitle
- [x] support for YouTube videos download
- [ ] support other platform download (bilibili, x)