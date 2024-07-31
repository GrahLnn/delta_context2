from pathlib import Path


def get_data_dir(calling_file_path: Path | str) -> Path:
    """
    Get the data directory path based on the calling file path.

    Args:
        calling_file_path (Path | str): The path of the calling file.

    Returns:
        Path: The data directory path.

    """
    data_dir: Path = Path(calling_file_path).parent / "data"
    return data_dir


def get_model_dir(calling_file_path: Path | str) -> Path:
    model_dir: Path = Path(calling_file_path).parent / "assets" / "models"
    return model_dir
