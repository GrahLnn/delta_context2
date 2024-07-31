import json


def read_metadata(dir, keys):
    if isinstance(keys, str):
        keys = [keys]
    with open(dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    try:
        if len(keys) == 1:
            return metadata[keys[0]]
        else:
            return {key: metadata[key] for key in keys}
    except KeyError:
        return None