from __future__ import annotations

import os
import csv
from typing import Any, Iterable, Sequence

# -----------------------------
# Small CSV helpers
# -----------------------------



def write_csv_header_if_needed(path: str, header: Sequence[str], *, rewrite: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if rewrite or (not os.path.exists(path)):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_rows_csv(path: str, rows: Iterable[Sequence[Any]]) -> None:
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        for r in rows:
            writer = csv.writer(f)
            writer.writerow(r)


def append_dict_rows_csv(path: str, rows: Iterable[dict], fieldnames: Sequence[str], *, rewrite: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = rewrite or (not os.path.exists(path))
    with open(path, "a", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


