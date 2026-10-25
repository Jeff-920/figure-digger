"""Prepare one verification panel for the interactive point viewer.

The browser receives the panel image, its pixel frame, the axis ranges and the
extracted series points.  No extraction logic lives here; this only reads the
files the CLI already wrote and converts them into display coordinates.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import re
from pathlib import Path

_COLORS = ("#1d4ed8", "#dc2626", "#047857", "#b45309", "#7c3aed", "#0f766e")
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _range(value, fallback=None) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        low, high = _number(value[0]), _number(value[1])
        if low is not None and high is not None and high > low:
            return [low, high]
    if fallback is not None:
        return list(fallback)
    return None


def _frame(value) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    values = [_number(item) for item in value[:4]]
    if any(item is None for item in values):
        return None
    left, top, right, bottom = values
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _segments(path: Path) -> list[list[list[float]]]:
    """Read x/y rows; blank lines separate broken polyline segments."""
    segments: list[list[list[float]]] = []
    current: list[list[float]] = []
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.reader(handle):
                if not any(cell.strip() for cell in row):
                    if len(current) >= 2:
                        segments.append(current)
                    current = []
                    continue
                if len(row) < 2:
                    continue
                x, y = _number(row[0]), _number(row[1])
                if x is not None and y is not None:
                    current.append([x, y])
    except (OSError, csv.Error):
        return []
    if len(current) >= 2:
        segments.append(current)
    return segments


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _axis_labels(paper_root: Path, panel_id: str, record: dict) -> tuple[str, str]:
    stored_x = str(record.get("x_label") or "").strip()
    stored_y = str(record.get("y_label") or "").strip()
    fallback = (stored_x or "Time (ms)", stored_y or "Liquid Penetration (mm)")
    if stored_x and stored_y:
        return fallback
    panel_image = Path(str(record.get("panel") or "")).name
    for config_path in paper_root.glob("*_config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for panel in config.get("panels", []) if isinstance(config, dict) else []:
            if not isinstance(panel, dict):
                continue
            if panel.get("id") != panel_id and Path(str(panel.get("panel_image") or "")).name != panel_image:
                continue
            axis = panel.get("axis") or {}
            if not isinstance(axis, dict):
                continue
            x_title = str(axis.get("x_title") or "").strip()
            x_unit = str(axis.get("x_unit") or "").strip()
            y_title = str(axis.get("y_title") or "").strip()
            y_unit = str(axis.get("y_unit") or "").strip()
            return (
                stored_x or (f"{x_title} ({x_unit})" if x_title and x_unit else x_title) or fallback[0],
                stored_y or (f"{y_title} ({y_unit})" if y_title and y_unit else y_title) or fallback[1],
            )
    return fallback


def _image_path(paper_root: Path, record: dict) -> Path | None:
    name = record.get("panel")
    if not isinstance(name, str) or not name.strip():
        return None
    base = Path(name).name
    for candidate in (paper_root / "panels" / base, paper_root / "figures" / base):
        if candidate.is_file() and _inside(candidate, paper_root):
            return candidate
    return None


def _color(series: dict, index: int) -> str:
    for key in ("trace_color_hex", "color_hex", "color"):
        value = series.get(key)
        if isinstance(value, str) and _HEX.match(value.strip()):
            return value.strip()
    return _COLORS[index % len(_COLORS)]


def build_panel_chart(verify_path: Path) -> dict | None:
    """Return image + series data for a ``*_verify.png`` file, or ``None``."""

    verify_path = Path(verify_path)
    stem = verify_path.stem
    if not stem.endswith("_verify"):
        return None
    panel_id = stem[:-len("_verify")]
    paper_root = verify_path.parent.parent
    try:
        results = json.loads((paper_root / "extract_results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(results, dict):
        return None
    record = results.get(panel_id)
    if not isinstance(record, dict):
        return None

    frame = _frame(record.get("frame"))
    axis = _range(record.get("axis_range"))
    image_path = _image_path(paper_root, record)
    if frame is None or axis is None or image_path is None:
        return None
    x_range = _range((record.get("axis_range") or [])[:2])
    y_default = _range((record.get("axis_range") or [])[2:4])
    if x_range is None or y_default is None:
        return None

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
            image_format = (image.format or "PNG").upper()
        image_bytes = image_path.read_bytes()
    except (OSError, ValueError):
        return None
    mime = "image/jpeg" if image_format in {"JPEG", "JPG"} else "image/png"
    data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    series_out = []
    for index, series in enumerate(record.get("series") or []):
        if not isinstance(series, dict):
            continue
        name = series.get("file")
        if not isinstance(name, str) or not name.strip():
            continue
        csv_path = paper_root / "csv" / Path(name).name
        if not csv_path.is_file() or not _inside(csv_path, paper_root):
            continue
        segments = _segments(csv_path)
        if not segments:
            continue
        y_range = _range(series.get("y_axis_range"), y_default)
        if y_range is None:
            continue
        label = str(series.get("series_name") or Path(name).stem)
        series_out.append({
            "file": Path(name).name,
            "label": label,
            "color": _color(series, index),
            "y_range": y_range,
            "segments": segments,
        })
    if not series_out:
        return None
    x_label, y_label = _axis_labels(paper_root, panel_id, record)
    return {
        "panel_id": panel_id,
        "paper": paper_root.name,
        "image": data_uri,
        "width": width,
        "height": height,
        "frame": frame,
        "x_range": x_range,
        "y_range": y_default,
        "x_label": x_label,
        "y_label": y_label,
        "series": series_out,
    }


def _clean_segments(raw) -> list[list[list[float]]]:
    if not isinstance(raw, list):
        raise ValueError("曲线点格式无效。")
    segments: list[list[list[float]]] = []
    total = 0
    for segment in raw:
        if not isinstance(segment, list):
            continue
        points = []
        for point in segment:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            x, y = _number(point[0]), _number(point[1])
            if x is not None and y is not None:
                points.append([x, y])
        if len(points) >= 2:
            segments.append(points)
            total += len(points)
    if not segments:
        raise ValueError("曲线至少需要两个点。")
    if total > 60000:
        raise ValueError("单条曲线点数过多。")
    return segments


def _write_csv(path: Path, segments: list[list[list[float]]]) -> None:
    lines = ["x,y"]
    for index, segment in enumerate(segments):
        if index:
            lines.append("")
        lines.extend(f"{point[0]},{point[1]}" for point in segment)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _conversion_value(value, default, name, *, positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        raise ValueError(f"{name}必须是有限数字。")
    if positive and number <= 0:
        raise ValueError(f"{name}必须大于 0。")
    return number


def _convert_segments(segments, x_factor, x_offset, y_factor, y_offset):
    return [
        [[point[0] * x_factor + x_offset, point[1] * y_factor + y_offset]
         for point in segment]
        for segment in segments
    ]


def _convert_range(value, factor, offset):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        low = float(value[0]) * factor + offset
        high = float(value[1]) * factor + offset
    except (TypeError, ValueError):
        return None
    return [low, high] if high > low else [high, low]


def save_panel_curves(paper_root: Path, panel_id: str, payload, axis=None) -> dict:
    """Overwrite extracted CSVs for one panel and append manually added curves."""

    if payload is None:
        payload = []
    if not isinstance(payload, list):
        raise ValueError("曲线列表格式无效。")
    axis_payload = axis if isinstance(axis, dict) else {}
    results_path = paper_root / "extract_results.json"
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("找不到该面板的提取结果。") from exc
    record = results.get(panel_id) if isinstance(results, dict) else None
    if not isinstance(record, dict):
        raise ValueError("找不到该面板的提取结果。")

    csv_dir = paper_root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    existing = {}
    for index, series in enumerate(record.get("series") or []):
        if isinstance(series, dict) and isinstance(series.get("file"), str):
            existing[Path(series["file"]).name] = (index, series)
    record_axis = record.get("axis_range") or []
    y_fallback = [0.0, 1.0]
    if isinstance(record_axis, list) and len(record_axis) >= 4:
        try:
            if float(record_axis[3]) > float(record_axis[2]):
                y_fallback = [float(record_axis[2]), float(record_axis[3])]
        except (TypeError, ValueError):
            pass

    x_factor = _conversion_value(axis_payload.get("x_factor", 1), 1,
                                "X 轴倍率", positive=True)
    x_offset = _conversion_value(axis_payload.get("x_offset", 0), 0, "X 轴偏移")
    y_factor = _conversion_value(axis_payload.get("y_factor", 1), 1,
                                "Y 轴倍率", positive=True)
    y_offset = _conversion_value(axis_payload.get("y_offset", 0), 0, "Y 轴偏移")
    converted = (x_factor, x_offset, y_factor, y_offset) != (1.0, 0.0, 1.0, 0.0)
    x_label = str(axis_payload.get("x_label") or record.get("x_label") or "").strip()[:80]
    y_label = str(axis_payload.get("y_label") or record.get("y_label") or "").strip()[:80]
    if x_label:
        record["x_label"] = x_label
    if y_label:
        record["y_label"] = y_label
    record["unit_conversion"] = {
        "x_factor": x_factor, "x_offset": x_offset,
        "y_factor": y_factor, "y_offset": y_offset,
    }
    prepared = []
    seen = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        segments = _clean_segments(item.get("segments"))
        label = str(item.get("label") or "曲线的修正").strip()[:80] or "曲线的修正"
        color = str(item.get("color") or "").strip()
        if not _HEX.match(color):
            color = _COLORS[len(prepared) % len(_COLORS)]
        raw_file = item.get("file")
        is_new = bool(item.get("is_new")) or not raw_file
        if is_new:
            number = len(existing) + len(prepared) + 1
            while True:
                name = f"{panel_id}_manual{number}.csv"
                csv_path = csv_dir / name
                if name not in existing and name not in seen and not csv_path.exists():
                    break
                number += 1
            meta = {
                "file": name,
                "series_name": label,
                "color_hex": color,
                "trace_color_hex": color,
                "y_axis_range": y_fallback,
                "points": sum(len(segment) for segment in segments),
                "method": "manual",
                "source": "manual",
            }
            record.setdefault("series", []).append(meta)
            existing[name] = (len(record["series"]) - 1, meta)
            seen.add(name)
            prepared.append((csv_path, segments, name, label, True))
        else:
            name = Path(str(raw_file)).name
            if name not in existing:
                raise ValueError(f"曲线文件不在该面板结果中：{name}")
            index, meta = existing[name]
            meta["series_name"] = label
            meta["color_hex"] = color
            meta["trace_color_hex"] = color
            meta["points"] = sum(len(segment) for segment in segments)
            csv_path = csv_dir / name
            if not _inside(csv_path, paper_root):
                raise ValueError("曲线路径无效。")
            prepared.append((csv_path, segments, name, label, False))
            seen.add(name)

    if converted:
        # An axis conversion is panel-wide: curves the user did not touch must
        # move to the new units too, otherwise one CSV would mix two systems.
        for name, (_, meta) in existing.items():
            if name in seen:
                continue
            csv_path = csv_dir / name
            if not csv_path.is_file() or not _inside(csv_path, paper_root):
                continue
            segments = _segments(csv_path)
            if not segments:
                continue
            prepared.append((csv_path, segments, name,
                             str(meta.get("series_name") or Path(name).stem), False))
            seen.add(name)
        prepared = [
            (path, _convert_segments(segments, x_factor, x_offset, y_factor, y_offset),
             name, label, is_new)
            for path, segments, name, label, is_new in prepared
        ]
        if isinstance(record_axis, list) and len(record_axis) >= 4:
            converted_axis = [
                float(record_axis[0]) * x_factor + x_offset,
                float(record_axis[1]) * x_factor + x_offset,
                float(record_axis[2]) * y_factor + y_offset,
                float(record_axis[3]) * y_factor + y_offset,
            ]
            record["axis_range"] = converted_axis
        for series in record.get("series") or []:
            if not isinstance(series, dict):
                continue
            y_range = series.get("y_axis_range")
            converted_y = _convert_range(y_range, y_factor, y_offset)
            if converted_y:
                series["y_axis_range"] = converted_y
            for key, factor, offset in (("x_range", x_factor, x_offset),
                                        ("y_range", y_factor, y_offset)):
                converted_range = _convert_range(series.get(key), factor, offset)
                if converted_range:
                    series[key] = converted_range
    for _, segments, name, _, _ in prepared:
        meta = existing.get(name, (None, None))[1]
        if isinstance(meta, dict):
            meta["points"] = sum(len(segment) for segment in segments)
    for csv_path, segments, _, _, _ in prepared:
        _write_csv(csv_path, segments)
    temporary = results_path.with_name(results_path.name + ".tmp")
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(results_path)
    return {
        "saved": [
            {"file": name, "label": label, "is_new": is_new}
            for _, _, name, label, is_new in prepared
        ]
    }
