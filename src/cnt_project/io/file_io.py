from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def read_json(
    file_path: str | os.PathLike,
) -> dict[str, Any]:
    """
    Read a UTF-8 encoded JSON file.
    """
    with Path(file_path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def write_text(
    text: str,
    file_path: str | os.PathLike,
) -> None:
    """
    Write text to a UTF-8 encoded file.

    Parent directories are created when necessary.
    """
    output_path = Path(file_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        text,
        encoding="utf-8",
    )