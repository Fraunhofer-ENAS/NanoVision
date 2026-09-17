import json
from pathlib import Path
from typing import Dict, Any

def read_json(file_path: str | Path) -> Dict[str, Any]:
    """Read and return JSON content."""
    p = Path(file_path)
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)
