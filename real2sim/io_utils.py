from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _strip_comment(line: str) -> str:
    in_quote = False
    quote = ""
    out = []
    for char in line:
        if char in ("'", '"'):
            if not in_quote:
                in_quote = True
                quote = char
            elif quote == char:
                in_quote = False
        if char == "#" and not in_quote:
            break
        out.append(char)
    return "".join(out).rstrip()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    if value.startswith("[") or value.startswith("{"):
        return ast.literal_eval(value)
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load JSON or a small YAML subset used by this project.

    The YAML reader supports nested dictionaries via indentation and inline
    scalar/list values. This avoids adding PyYAML as a required dependency.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise RuntimeError(f"Expected mapping in {path}")
        return data

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise RuntimeError(f"Unsupported YAML line in {path}: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        parsed = _parse_scalar(value)
        parent[key] = parsed
        if isinstance(parsed, dict) and value.strip() == "":
            stack.append((indent, parsed))
    return root


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    return str(value)


def dump_mapping(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)

    def lines(mapping: dict[str, Any], indent: int = 0) -> list[str]:
        out = []
        prefix = " " * indent
        for key, value in mapping.items():
            if isinstance(value, dict):
                out.append(f"{prefix}{key}:")
                out.extend(lines(value, indent + 2))
            else:
                out.append(f"{prefix}{key}: {_format_scalar(value)}")
        return out

    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        path.write_text("\n".join(lines(data)) + "\n", encoding="utf-8")
    return path


def read_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_depth_meters(path: str | Path, *, depth_scale: float | None = None) -> np.ndarray:
    path = Path(path)
    depth = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Could not read depth image: {path}")
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth_scale is None:
        depth_scale = 0.001 if np.issubdtype(depth.dtype, np.integer) else 1.0
    return depth.astype(np.float32) * float(depth_scale)


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return path
