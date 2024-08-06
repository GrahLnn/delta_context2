import json
from functools import wraps
from pathlib import Path

from alive_progress import alive_bar


def update_metadata(*fields):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            data_file = Path(args[0]) / "metadata.json"

            with open(data_file, encoding="utf-8") as f:
                metadata = json.load(f)

            for field, value in fields:
                metadata[field] = value(result)

            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)

            return result

        return wrapper

    return decorator


def show_progress(title):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with alive_bar(1, bar=None, title=title, monitor=False) as bar:
                result = func(*args, **kwargs)
                bar()
            return result

        return wrapper

    return decorator
