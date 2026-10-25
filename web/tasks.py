"""Task storage, background execution and history state for the DIG web UI.

The extraction algorithms stay in ``run.py``/``dig``.  This module only turns
the existing CLI invocation into a queued background job and derives the
history/status view from the files that are already on disk.
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from ui_utils import (batch_rows, collect_artifacts, result_status,
                      safe_filename)

LOG_PREVIEW_CHARS = 24000
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ARTIFACT_GROUPS = ("csv", "images", "reports", "templates", "configs")
MODE_LABELS = {"auto": "自动提取", "analyze": "仅分析，稍后确认坐标"}
_FINISHED_STATES = {"succeeded", "finished", "completed"}


def now_iso() -> str:
    """Return an ISO timestamp with the local UTC offset."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_time(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def format_duration(value: object) -> str:
    if value is None:
        return "—"
    try:
        seconds = max(0.0, float(value))
    except (TypeError, ValueError):
        return "—"
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    minutes, remainder = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} 分 {remainder:02d} 秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} 小时 {minutes:02d} 分"


def format_clock(value: object) -> str:
    if not isinstance(value, datetime):
        return "未记录"
    if value.tzinfo is not None:
        value = value.astimezone()
    return value.strftime("%Y-%m-%d %H:%M")


def count_pdf_pages(data: bytes) -> int:
    """Read the page count before accepting an upload (kept patchable in tests)."""

    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as document:
        return len(document.pages)


@dataclass(frozen=True)
class KeyStatus:
    ready: bool
    saved: bool
    source: str = ""


def key_status(root: Path) -> KeyStatus:
    """Report whether a key is available without ever returning its value."""

    root = Path(root)
    file_saved = False
    try:
        lines = (root / "deepseek_key.txt").read_text(encoding="utf-8").splitlines()
        file_saved = any(line.strip() and not line.lstrip().startswith("#") for line in lines)
    except OSError:
        pass
    if os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return KeyStatus(True, file_saved, "环境变量 DEEPSEEK_API_KEY")
    if file_saved:
        return KeyStatus(True, True, "deepseek_key.txt")
    try:
        secrets = json.loads((root / "secrets.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, AttributeError):
        secrets = None
    if isinstance(secrets, dict):
        value = str(secrets.get("deepseek_api_key") or secrets.get("api_key") or "").strip()
        if value:
            return KeyStatus(True, file_saved, "secrets.json")
    return KeyStatus(False, file_saved, "")


class RunStore:
    """Owns the existing ``.ui_workspace/<run-id>`` layout."""

    def __init__(self, root: Path, workspace: Path | None = None) -> None:
        self.root = Path(root).resolve()
        raw = Path(workspace) if workspace is not None else self.root / ".ui_workspace"
        self.workspace = raw.resolve()

    def ensure_workspace(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str, *, create: bool = False) -> Path:
        value = str(run_id or "")
        if not _RUN_ID.fullmatch(value) or value in {".", ".."}:
            raise ValueError("任务 ID 无效")
        path = self.workspace / value
        if path.parent.resolve() != self.workspace:
            raise ValueError("任务目录不在工作区内")
        if create:
            (path / "input").mkdir(parents=True, exist_ok=True)
            (path / "output").mkdir(parents=True, exist_ok=True)
        return path

    def create_run(self) -> str:
        self.ensure_workspace()
        while True:
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            path = self.workspace / run_id
            if not path.exists():
                break
        self.run_dir(run_id, create=True)
        return run_id

    def metadata_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "ui-run.json"

    def output_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "output"

    def log_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "ui-run.log"

    def read_metadata(self, run_id: str) -> dict | None:
        try:
            data = json.loads(self.metadata_path(run_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def list_runs(self) -> list[tuple[str, dict]]:
        if not self.workspace.exists():
            return []
        runs: list[tuple[str, dict]] = []
        for path in self.workspace.glob("*/ui-run.json"):
            run_id = path.parent.name
            if not _RUN_ID.fullmatch(run_id):
                continue
            metadata = self.read_metadata(run_id)
            if metadata is not None:
                runs.append((run_id, metadata))
        return runs

    def write_metadata(self, run_id: str, metadata: dict) -> None:
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        pending = directory / f".ui-run-{uuid.uuid4().hex}.tmp"
        try:
            pending.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(pending, directory / "ui-run.json")
        finally:
            pending.unlink(missing_ok=True)

    def update_metadata(self, run_id: str, **changes) -> dict:
        metadata = self.read_metadata(run_id) or {}
        metadata.update(changes)
        self.write_metadata(run_id, metadata)
        return metadata

    def save_uploads(self, run_id: str, uploads: Iterable[tuple[str, bytes]]) -> list[Path]:
        directory = self.run_dir(run_id) / "input"
        directory.mkdir(parents=True, exist_ok=True)
        used: set[str] = set()
        paths: list[Path] = []
        for original, data in uploads:
            base = safe_filename(original, "document.pdf")
            name = base
            count = 2
            while name.casefold() in used:
                name = f"{Path(base).stem}_{count}{Path(base).suffix}"
                count += 1
            used.add(name.casefold())
            path = directory / name
            path.write_bytes(data)
            paths.append(path)
        return paths

    def save_template(self, run_id: str, original: str, data: bytes) -> Path:
        path = self.run_dir(run_id) / safe_filename(original, "template.txt")
        path.write_bytes(data)
        return path

    def artifacts(self, run_id: str) -> dict[str, list[Path]]:
        return collect_artifacts(self.output_dir(run_id))


def read_log_tail(path: Path, limit: int = LOG_PREVIEW_CHARS) -> str:
    """Read a bounded log tail without loading an arbitrarily large file."""

    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit * 4))
            data = handle.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    return text[-limit:]


def _command_want(command: object) -> str:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        return ""
    args = [str(item) for item in command]
    for index, item in enumerate(args):
        if item == "--want" and index + 1 < len(args):
            return args[index + 1]
        if item.startswith("--want="):
            return item.split("=", 1)[1]
    return ""


def extract_want(metadata: dict) -> str:
    if "want" in metadata:
        return str(metadata.get("want") or "")
    return _command_want(metadata.get("command"))


def _mode_key(metadata: dict) -> str:
    value = str(metadata.get("mode_key") or "").lower()
    if value in MODE_LABELS:
        return value
    label = str(metadata.get("mode") or "")
    if label.startswith("仅分析"):
        return "analyze"
    if label.startswith("自动"):
        return "auto"
    command = metadata.get("command")
    if isinstance(command, list) and len(command) > 2:
        subcommand = str(command[2]).lower()
        if subcommand == "analyze":
            return "analyze"
        if subcommand in {"all", "batch"}:
            return "auto"
    return "unknown"


def _created_at(metadata: dict, run_id: str, run_dir: Path) -> datetime:
    parsed = parse_time(metadata.get("created_at"))
    if parsed is not None:
        return parsed
    match = re.match(r"^(\d{8})-(\d{6})", run_id)
    if match:
        try:
            naive = datetime.strptime(
                f"{match.group(1)}-{match.group(2)}", "%Y%m%d-%H%M%S")
            return naive.astimezone()
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(run_dir.stat().st_mtime).astimezone()
    except OSError:
        return datetime.now().astimezone()


def _elapsed_seconds(metadata: dict, state_key: str, created: datetime) -> float | None:
    if state_key == "queued":
        return None
    value = metadata.get("elapsed")
    if value is not None:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            pass
    if state_key == "running":
        started = parse_time(metadata.get("started_at")) or created
        return max(0.0, (datetime.now().astimezone() - started).total_seconds())
    finished = parse_time(metadata.get("finished_at"))
    if finished is not None:
        return max(0.0, (finished - created).total_seconds())
    return None


def _failure_summary(failures: list[str], limit: int = 4) -> str:
    shown = "、".join(failures[:limit])
    if len(failures) > limit:
        shown += f" 等 {len(failures)} 篇"
    return f"{len(failures)} 篇论文处理失败：{shown}"


def _result_summary(counts: dict[str, int]) -> str:
    labels = (("csv", "CSV"), ("images", "核对图"), ("reports", "报告"),
              ("templates", "模板"), ("configs", "配置"))
    parts = [f"{label} {counts[key]}" for key, label in labels if counts.get(key)]
    return " · ".join(parts) if parts else "无"


@dataclass(frozen=True)
class RunState:
    key: str
    label: str
    tone: str
    icon: str
    detail: str = ""
    short_label: str = ""


def classify_state(metadata: dict, artifacts: dict, failures: list[str] | None = None) -> RunState:
    """Separate execution state from the actual extraction outcome."""

    failures = list(failures or [])
    status = str(metadata.get("status") or "").lower()
    if status == "queued":
        return RunState("queued", "排队中", "queued", "fa-hourglass-half",
                        "已进入后台队列，等待执行。", "排队中")
    if status == "running":
        return RunState("running", "运行中", "running", "fa-spinner",
                        "后台正在执行，页面和日志会持续更新。", "运行中")
    if status == "interrupted":
        return RunState("interrupted", "已中断", "warning", "fa-circle-pause",
                        "服务重启时任务仍在排队或运行；未自动重试，请重新提交。",
                        "已中断")

    code = metadata.get("code")
    try:
        numeric_code = int(code) if code is not None else None
    except (TypeError, ValueError):
        numeric_code = 1
    if status == "failed" or (numeric_code not in (None, 0)):
        detail = (f"处理程序退出码 {numeric_code}，请查看日志、报告和核对图。"
                  if numeric_code is not None else "任务未正常结束，请查看日志。")
        return RunState("failed", "处理失败", "danger", "fa-circle-xmark", detail,
                        "处理失败")

    if failures:
        return RunState("partial", "部分失败", "warning", "fa-triangle-exclamation",
                        _failure_summary(failures) + "；已有结果仍需核对。",
                        "部分失败")

    effective = dict(metadata)
    if numeric_code is None and status in _FINISHED_STATES:
        effective["code"] = 0
    base = result_status(effective, artifacts)
    if base.startswith("分析完成"):
        return RunState("analyze", "分析完成 · 待确认坐标", "info",
                        "fa-magnifying-glass-chart",
                        "仅完成分析；确认坐标轴范围后才能提取数据。",
                        "待确认坐标")
    if base.startswith("处理结束"):
        return RunState("empty", "未提取到数据", "muted", "fa-inbox",
                        "命令正常结束，但没有 CSV 或模板；请查看报告和日志。",
                        "未提取到数据")
    if base.startswith("提取完成"):
        return RunState("review", "提取完成 · 待核对", "success", "fa-circle-check",
                        "结果尚未人工核对；请先检查核对图，再使用数据。",
                        "完成 · 待核对")
    return RunState("failed", "处理失败", "danger", "fa-circle-xmark",
                    "未能确认任务结果，请查看日志。", "处理失败")


@dataclass(frozen=True)
class PaperView:
    name: str
    paper_id: str
    pages: int | None
    artifacts: dict[str, list[Path]]
    counts: dict[str, int]
    state: RunState
    scope: str = ""
    curves: int | None = None
    error: str = ""

    @property
    def result_total(self) -> int:
        return sum(self.counts.values())

    @property
    def result_summary(self) -> str:
        return _result_summary(self.counts)


def _positive_int(value: object) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _paper_entries(metadata: dict, rows: list[dict]) -> list[dict]:
    raw = metadata.get("papers")
    entries: list[dict] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name:
                entries.append({"name": name, "pages": _positive_int(item.get("pages"))})
    if entries:
        return entries
    names: list[str] = []
    for row in rows:
        name = str(row.get("pdf") or "").strip()
        if name and name not in names:
            names.append(name)
    if names:
        return [{"name": name, "pages": None} for name in names]
    label = str(metadata.get("label") or "").strip()
    if label:
        parts = [part.strip() for part in label.split("、") if part.strip()]
        if len(parts) > 1:
            return [{"name": part, "pages": None} for part in parts]
        return [{"name": label, "pages": None}]
    return [{"name": "未记录", "pages": None}]


def _paper_output(output: Path, name: str, multi: bool) -> Path:
    if not multi:
        return output
    stem = Path(name).stem
    direct = output / stem
    if direct.exists():
        return direct
    if output.exists():
        for child in output.iterdir():
            if child.is_dir() and child.name.casefold() == stem.casefold():
                return child
    return direct


def _paper_row(rows: list[dict], name: str) -> dict | None:
    target = Path(name)
    for row in rows:
        pdf = Path(str(row.get("pdf") or ""))
        if pdf.name.casefold() == target.name.casefold():
            return row
    for row in rows:
        pdf = Path(str(row.get("pdf") or ""))
        if pdf.stem.casefold() == target.stem.casefold():
            return row
    return None


def _paper_state(metadata: dict, task_state: RunState, artifacts: dict,
                 row: dict | None) -> RunState:
    if task_state.key in {"queued", "running", "interrupted"}:
        return task_state
    if row and row.get("error"):
        error = str(row.get("error"))
        return RunState("failed", "处理失败", "danger", "fa-circle-xmark",
                        error, "处理失败")
    if task_state.key == "failed" and row is None:
        return task_state
    paper_meta = {"code": 0, "mode": metadata.get("mode", ""), "status": "succeeded"}
    return classify_state(paper_meta, artifacts, [])


def build_paper_views(metadata: dict, task_state: RunState, output: Path,
                      rows: list[dict] | None = None,
                      all_artifacts: dict[str, list[Path]] | None = None) -> list[PaperView]:
    rows = list(rows or [])
    entries = _paper_entries(metadata, rows)
    multi = len(entries) > 1 or bool(rows)
    papers: list[PaperView] = []
    for entry in entries:
        name = entry["name"]
        paper_output = _paper_output(output, name, multi)
        if all_artifacts is not None:
            if multi:
                prefix = str(paper_output).casefold() + os.sep
                artifacts = {
                    key: [path for path in values if str(path).casefold().startswith(prefix)]
                    for key, values in all_artifacts.items()
                }
            else:
                artifacts = all_artifacts
        else:
            artifacts = collect_artifacts(paper_output)
        counts = {key: len(artifacts.get(key, [])) for key in _ARTIFACT_GROUPS}
        row = _paper_row(rows, name)
        scope = ""
        if multi:
            try:
                scope = paper_output.relative_to(output).as_posix()
            except ValueError:
                scope = ""
            if scope == ".":
                scope = ""
        papers.append(PaperView(
            name=name,
            paper_id=Path(name).stem or name,
            pages=entry.get("pages"),
            artifacts=artifacts,
            counts=counts,
            state=_paper_state(metadata, task_state, artifacts, row),
            scope=scope,
            curves=_positive_int((row or {}).get("curves")),
            error=str((row or {}).get("error") or ""),
        ))
    return papers


@dataclass(frozen=True)
class RunView:
    run_id: str
    metadata: dict
    state: RunState
    artifacts: dict[str, list[Path]]
    counts: dict[str, int]
    papers: list[PaperView]
    label: str
    want: str
    mode: str
    mode_key: str
    created: datetime
    elapsed: float | None
    batch_failures: list[str]
    output: Path
    log_path: Path
    command: list[str]
    active: bool

    @property
    def result_total(self) -> int:
        return sum(self.counts.values())

    @property
    def paper_count(self) -> int:
        return len(self.papers)

    @property
    def failed_paper_count(self) -> int:
        return sum(1 for paper in self.papers if paper.state.key == "failed")

    @property
    def file_summary(self) -> str:
        if not self.papers:
            return self.label
        first = self.papers[0].name
        if len(self.papers) == 1:
            return first
        return f"{first} 等 {len(self.papers)} 篇"

    @property
    def created_iso(self) -> str:
        return self.created.isoformat(timespec="seconds")

    @property
    def short_id(self) -> str:
        return f"#{self.run_id[-6:]}" if len(self.run_id) >= 6 else self.run_id

    @property
    def mode_short(self) -> str:
        return {"auto": "自动提取", "analyze": "仅分析"}.get(self.mode_key, "未记录")

    @property
    def result_summary(self) -> str:
        return _result_summary(self.counts)


def build_run_view(store: RunStore, run_id: str, metadata: dict | None = None) -> RunView | None:
    if metadata is None:
        metadata = store.read_metadata(run_id)
    if metadata is None:
        return None
    try:
        run_dir = store.run_dir(run_id)
    except ValueError:
        return None
    output = run_dir / "output"
    artifacts = collect_artifacts(output)
    counts = {key: len(artifacts.get(key, [])) for key in _ARTIFACT_GROUPS}
    rows = batch_rows(output)
    failures = [str(row.get("pdf", "未知文件")) for row in rows if row.get("error")]
    state = classify_state(metadata, artifacts, failures)
    papers = build_paper_views(metadata, state, output, rows, artifacts)
    if (papers and state.key not in {"queued", "running", "interrupted", "failed"}
            and all(paper.state.key == "failed" for paper in papers)):
        state = RunState("failed", "处理失败", "danger", "fa-circle-xmark",
                         "所有文献处理失败；请查看文献状态、日志和报告。", "处理失败")
    mode_key = _mode_key(metadata)
    mode = str(metadata.get("mode") or "").strip() or MODE_LABELS.get(mode_key, "未记录")
    want_raw = extract_want(metadata).strip()
    if want_raw:
        want = want_raw
    elif "want" in metadata or metadata.get("command"):
        want = "全部数据图"
    else:
        want = "未记录"
    command_value = metadata.get("command")
    command = [str(item) for item in command_value] if isinstance(command_value, list) else []
    created = _created_at(metadata, run_id, run_dir)
    elapsed = _elapsed_seconds(metadata, state.key, created)
    return RunView(
        run_id=run_id,
        metadata=metadata,
        state=state,
        artifacts=artifacts,
        counts=counts,
        papers=papers,
        label=str(metadata.get("label") or "未记录"),
        want=want,
        mode=mode,
        mode_key=mode_key,
        created=created,
        elapsed=elapsed,
        batch_failures=failures,
        output=output,
        log_path=store.log_path(run_id),
        command=command,
        active=state.key in {"queued", "running"},
    )


def run_command(command: Sequence[str], api_key: str, log_path: Path,
                cwd: Path) -> tuple[int, float]:
    """Run the existing CLI with an argument array and stream its log to disk."""

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    key = (api_key or "").strip()
    if key:
        env["DEEPSEEK_API_KEY"] = key
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.monotonic()
    log_path = Path(log_path)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(command), cwd=str(cwd), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1, env=env, creationflags=flags,
            )
            if process.stdout is None:
                raise RuntimeError("无法读取处理程序输出")
            for line in process.stdout:
                log.write(line)
                log.flush()
            code = process.wait()
    except OSError as exc:
        log_path.write_text(f"无法启动处理程序：{exc}\n", encoding="utf-8")
        code = 1
    return code, time.monotonic() - started


@dataclass
class _TaskJob:
    run_id: str
    command: list[str]
    api_key: str = ""


class TaskManager:
    """A single serial in-process worker; batch parallelism stays in the CLI."""

    def __init__(self, store: RunStore, runner=None) -> None:
        self.store = store
        self._runner = runner or run_command
        self._queue: queue.Queue[_TaskJob] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._work, name="dig-task-worker", daemon=True)
            self._thread.start()

    def submit(self, run_id: str, command: Sequence[str], api_key: str = "") -> None:
        self.start()
        self._queue.put(_TaskJob(run_id, list(command), api_key or ""))

    def recover_interrupted(self) -> None:
        """Mark jobs that cannot survive a server restart; never retry them."""

        for run_id, metadata in self.store.list_runs():
            if str(metadata.get("status") or "").lower() in {"queued", "running"}:
                try:
                    self.store.update_metadata(
                        run_id,
                        status="interrupted",
                        finished_at=now_iso(),
                        interrupt_reason="服务重启时任务未完成；未自动重试。",
                    )
                except OSError:
                    continue

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            try:
                self._execute(job)
            except Exception as exc:  # keep the worker alive for later jobs
                try:
                    self.store.update_metadata(
                        job.run_id, status="failed", code=1,
                        finished_at=now_iso(),
                        failure_reason=f"后台任务异常：{type(exc).__name__}",
                    )
                except Exception:
                    pass
            finally:
                job.api_key = ""
                self._queue.task_done()

    def _execute(self, job: _TaskJob) -> None:
        self.store.update_metadata(job.run_id, status="running", started_at=now_iso())
        started = time.monotonic()
        log_path = self.store.log_path(job.run_id)
        try:
            code, elapsed = self._runner(
                job.command, job.api_key, log_path, self.store.root)
        except Exception:
            code = 1
            elapsed = time.monotonic() - started
            try:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write("\n后台执行异常，请查看终端输出。\n")
            except OSError:
                pass
        artifacts = self.store.artifacts(job.run_id)
        counts = {key: len(artifacts.get(key, [])) for key in _ARTIFACT_GROUPS}
        self.store.update_metadata(
            job.run_id,
            status="succeeded" if code == 0 else "failed",
            code=int(code),
            elapsed=round(float(elapsed), 3),
            finished_at=now_iso(),
            artifact_counts=counts,
        )
