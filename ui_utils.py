"""Small, dependency-free helpers shared by the web interface and the CLI tests.

Keeping command construction and artifact discovery outside the web routes makes
the UI thin and lets us test the potentially error-prone filesystem/CLI boundary.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_UNSAFE_FILENAME = re.compile(r"[^\w.\-()\u4e00-\u9fff]+", re.UNICODE)
_PAGES = re.compile(r"^\s*\d+(?:\s*,\s*\d+)*\s*$")


def save_api_key(root: Path, key: str) -> None:
    """Persist an explicitly supplied key using the CLI's existing local file."""
    key = key.strip()
    if not key or any(c in key for c in "\r\n"):
        raise ValueError("API Key 不能为空或包含换行。")
    target = root / "deepseek_key.txt"
    try:
        if target.read_text(encoding="utf-8").strip() == key:
            return
    except FileNotFoundError:
        pass
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                prefix=".deepseek-key-", suffix=".tmp", delete=False) as handle:
            pending = Path(handle.name)
            handle.write(key + "\n")
        pending.chmod(0o600)
        os.replace(pending, target)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def clear_saved_api_key(root: Path) -> None:
    (root / "deepseek_key.txt").unlink(missing_ok=True)


@dataclass(frozen=True)
class RunOptions:
    """Options shared by the one-shot and batch commands."""

    dpi: int = 300
    pages: str = ""
    force: bool = True
    use_vlm: bool = False
    want: str = ""
    want_deep: bool = False
    find_series: bool = True
    verbose: bool = False
    keep_intermediates: bool = False
    jobs: int = 1
    model: str = ""
    base_url: str = ""
    template: Path | None = None


def safe_filename(name: str, fallback: str = "document.pdf") -> str:
    """Return a portable basename and discard any user-supplied path."""

    cleaned = _UNSAFE_FILENAME.sub("_", Path(name).name).strip(" ._")
    return cleaned or fallback


def valid_pages(value: str) -> bool:
    """Accept the page-list syntax understood by the existing CLI."""

    return not value.strip() or bool(_PAGES.fullmatch(value)) and all(
        int(page.strip()) > 0 for page in value.split(","))


def build_command(
    run_py: Path,
    inputs: Iterable[Path],
    output_dir: Path,
    options: RunOptions,
    *,
    analyze_only: bool = False,
) -> list[str]:
    """Translate UI state to the existing CLI without invoking a shell."""

    pdfs = list(inputs)
    if not pdfs:
        raise ValueError("至少需要一份 PDF")
    if analyze_only and len(pdfs) != 1:
        raise ValueError("仅分析模式一次只能处理一份 PDF")
    if not valid_pages(options.pages):
        raise ValueError("页码应从 1 开始，格式为 7 或 3,5,7")
    if options.template is not None and (not options.use_vlm or analyze_only):
        raise ValueError("模板导出需要自动提取模式和视觉模型")

    if len(pdfs) == 1:
        command = [sys.executable, str(run_py), "analyze" if analyze_only else "all",
                   str(pdfs[0]), "--out", str(output_dir)]
    else:
        command = [sys.executable, str(run_py), "batch", str(pdfs[0].parent),
                   "--out", str(output_dir), "--jobs", str(max(1, options.jobs))]

    command.extend(["--dpi", str(options.dpi)])
    if options.pages.strip():
        command.extend(["--pages", options.pages.replace(" ", "")])
    if options.force and not analyze_only:
        command.append("--force")
    if options.use_vlm:
        command.append("--vlm")
        if options.model.strip():
            command.extend(["--vlm-model", options.model.strip()])
        if options.base_url.strip():
            command.extend(["--vlm-base-url", options.base_url.strip()])
    if options.want.strip():
        command.extend(["--want", options.want.strip()])
    if options.use_vlm and options.want_deep:
        command.append("--want-deep")
    if options.use_vlm and not options.find_series:
        command.append("--no-find-series")
    if options.template is not None and not analyze_only:
        command.extend(["--template", str(options.template)])
    if options.verbose:
        command.append("--verbose")
    if options.keep_intermediates:
        command.append("--keep-intermediates")
    return command


def collect_artifacts(output_dir: Path) -> dict[str, list[Path]]:
    """Find the durable outputs that are useful in the result dashboard."""

    found: dict[str, list[Path]] = {
        "csv": [],
        "reports": [],
        "images": [],
        "configs": [],
        "templates": [],
    }
    if not output_dir.exists():
        return found
    # Skip the large diagnostic/render directories: they never contain a
    # dashboard artifact, and walking them dominates history-page latency on
    # papers with many panels.
    skip_dirs = {"objects", "panels", "figures"}
    for root, dirs, files in os.walk(output_dir):
        dirs[:] = [name for name in dirs if name.lower() not in skip_dirs]
        root_path = Path(root)
        for name in files:
            path = root_path / name
            suffix = path.suffix.lower()
            if "templates" in path.relative_to(output_dir).parts:
                found["templates"].append(path)
            elif suffix == ".csv" and "csv" in path.relative_to(output_dir).parts:
                found["csv"].append(path)
            elif name in {"report.md", "batch_summary.md"}:
                found["reports"].append(path)
            # Retrieval candidates include rejected figures and diagnostic crops.
            # Only the extraction overlays belong in the image review dashboard.
            elif suffix in {".png", ".jpg", ".jpeg"} and path.parent.name == "verify":
                found["images"].append(path)
            elif name.endswith("_config.json") or name == "paper_config.json":
                found["configs"].append(path)
    for values in found.values():
        values.sort(key=lambda item: item.as_posix().lower())
    return found


def result_status(metadata: dict, artifacts: dict) -> str:
    if metadata.get("code") != 0:
        return "处理失败"
    if str(metadata.get("mode", "")).startswith("仅分析"):
        return "分析完成 · 待确认坐标"
    if not artifacts["csv"] and not artifacts.get("templates"):
        return "处理结束 · 未提取到数据"
    return "提取完成 · 请核对数据"


def batch_rows(output: Path) -> list[dict]:
    """Return the per-paper rows written by the batch CLI, if present."""

    path = output / "batch_summary.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = data if isinstance(data, list) else data.get("rows", []) if isinstance(data, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def batch_failures(output: Path) -> list[str]:
    """The batch CLI can exit successfully while individual papers fail."""
    return [str(row.get("pdf", "未知文件")) for row in batch_rows(output) if row.get("error")]


def make_archive(output_dir: Path) -> bytes:
    """Create an in-memory ZIP with paths relative to the selected run."""

    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if output_dir.exists():
            for path in sorted(output_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(output_dir))
    return data.getvalue()
