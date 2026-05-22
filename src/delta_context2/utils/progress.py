from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Callable

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)


def _basic_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )


def _download_progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
    )


@contextmanager
def step_progress(title: str) -> Iterator[Callable[[], None]]:
    with _basic_progress() as progress:
        task_id = progress.add_task(title, total=1)

        def advance() -> None:
            progress.advance(task_id)

        yield advance


def track_progress(iterable: Iterable, *, title: str, total: int | None = None) -> Iterator:
    with _basic_progress() as progress:
        task_id = progress.add_task(title, total=total)
        for item in iterable:
            yield item
            progress.advance(task_id)


@contextmanager
def download_progress(title: str, total: int) -> Iterator[Callable[[int], None]]:
    progress = _download_progress() if total else _basic_progress()
    with progress:
        task_id: TaskID = progress.add_task(title, total=total or None)

        def advance(size: int) -> None:
            progress.advance(task_id, size)

        yield advance
