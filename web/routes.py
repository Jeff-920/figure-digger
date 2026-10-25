"""HTTP routes for the DIG web interface."""

from __future__ import annotations

import csv
import io
import json
import math
import shutil
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit

from flask import (Blueprint, abort, current_app, flash, jsonify, redirect,
                   render_template, request, send_file, url_for)

from ui_utils import (RunOptions, build_command, clear_saved_api_key,
                      load_ui_preferences, make_archive, save_api_key,
                      save_ui_preferences, valid_pages)

from .chart import build_panel_chart, save_panel_curves
from .tasks import (LOG_PREVIEW_CHARS, MODE_LABELS, RunStore, TaskManager,
                    build_run_view, count_pdf_pages, format_duration,
                    key_status, now_iso, read_log_tail)

bp = Blueprint("main", __name__)

PER_PAGE = 25
DPI_OPTIONS = (150, 200, 300, 400, 600)
TAB_DEFS = (
    ("papers", "文献核对", "fa-list-check", "papers"),
    ("images", "图像核对", "fa-images", "images"),
    ("csv", "曲线数据", "fa-table", "csv"),
    ("templates", "模板文件", "fa-file-lines", "templates"),
    ("report", "报告", "fa-file-lines", "reports"),
    ("detail", "日志与配置", "fa-terminal", "configs"),
)
STATUS_FILTERS = {
    "all": None,
    "active": {"queued", "running"},
    "review": {"review"},
    "analyze": {"analyze"},
    "empty": {"empty"},
    "partial": {"partial"},
    "failed": {"failed", "interrupted"},
}
MODE_FILTERS = {"all", "auto", "analyze", "unknown"}


def _store() -> RunStore:
    return current_app.extensions["dig_store"]


def _tasks() -> TaskManager:
    return current_app.extensions["dig_tasks"]


@bp.app_context_processor
def _inject_navigation():
    endpoint = request.endpoint or ""
    if endpoint == "main.new_run":
        section = "new"
    elif endpoint == "main.help_page":
        section = "help"
    else:
        section = "runs"
    return {"key_status": key_status(_store().root), "nav_section": section}


def _safe_next(default: str) -> str:
    target = request.form.get("next") or ""
    parsed = urlsplit(target)
    if (target.startswith("/") and not target.startswith("//")
            and not parsed.scheme and not parsed.netloc
            and "\\" not in target and "\r" not in target and "\n" not in target):
        return target
    return default


def _default_form() -> dict:
    preferences = load_ui_preferences(_store().workspace)
    return {
        "want": str(preferences.get("want") or ""),
        "figure": "",
        "mode": "auto",
        "pages": "",
        "dpi": "300",
        "jobs": "3",
        "want_deep": False,
        "find_series": True,
        "force": True,
        "keep_intermediates": False,
        "verbose": False,
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
        "remember_key": True,
    }


def _form_from_request() -> dict:
    form = request.form
    mode = form.get("mode") if form.get("mode") in {"auto", "analyze"} else "auto"
    return {
        "want": form.get("want", ""),
        "figure": form.get("figure", "").strip(),
        "mode": mode,
        "pages": form.get("pages", "").strip(),
        "dpi": form.get("dpi", "300").strip(),
        "jobs": form.get("jobs", "3").strip(),
        "want_deep": "want_deep" in form,
        "find_series": "find_series" in form,
        "force": "force" in form,
        "keep_intermediates": "keep_intermediates" in form,
        "verbose": "verbose" in form,
        "model": form.get("model", "").strip() or "deepseek-flash",
        "base_url": form.get("base_url", "").strip() or "https://api.deepseek.com",
        "remember_key": "remember_key" in form,
    }


def _render_new(form: dict, errors: list[str], status: int = 400):
    return render_template("new_run.html", form=form, errors=errors), status


def _page_number(raw: str, pages: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 1
    return max(1, min(value, pages))


def _relative(output: Path, path: Path) -> str:
    try:
        return path.relative_to(output).as_posix()
    except ValueError:
        return path.name


def _file_options(output: Path, paths: list[Path]) -> list[tuple[str, str]]:
    return [(_relative(output, path), _relative(output, path)) for path in paths]


def _paper_scope_for_path(view, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(view.output.resolve())
    except (OSError, ValueError):
        return ""
    parts = relative.parts
    if len(parts) >= 3 and parts[0] != "verify":
        paper = _select_paper(view, parts[0])
        return paper.scope if paper else ""
    return ""


def _select_paper(view, raw: str):
    key = (raw or "").strip()
    if not key:
        return None
    for paper in view.papers:
        if paper.scope and paper.scope == key:
            return paper
    for paper in view.papers:
        if paper.paper_id == key or paper.name == key:
            return paper
    return None


def _pick_file(output: Path, paths: list[Path], requested: str) -> Path | None:
    if not paths:
        return None
    requested = (requested or "").strip()
    if requested:
        try:
            target = (output / requested).resolve()
        except (OSError, ValueError):
            target = None
        if target is not None:
            for path in paths:
                try:
                    if path.resolve() == target:
                        return path
                except OSError:
                    continue
    return paths[0]


def _read_text_preview(path: Path, limit: int = LOG_PREVIEW_CHARS):
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        return "", False, f"无法读取文件：{exc}"
    truncated = len(text) > limit
    return (text[:limit] if truncated else text), truncated, ""


def _read_csv_preview(path: Path):
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                return [], [], False, "文件为空。"
            rows = list(islice(reader, 201))
            truncated = len(rows) > 200
            rows = rows[:200]
            width = len(header)
            rows = [(row + [""] * width)[:width] for row in rows]
            return header, rows, truncated, ""
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], [], False, f"预览失败：{exc}"


@bp.get("/")
def index():
    q = (request.args.get("q") or "").strip()[:200]
    status_filter = request.args.get("status", "all")
    if status_filter not in STATUS_FILTERS:
        status_filter = "all"
    mode_filter = request.args.get("mode", "all")
    if mode_filter not in MODE_FILTERS:
        mode_filter = "all"

    store = _store()
    views = []
    for run_id, metadata in store.list_runs():
        view = build_run_view(store, run_id, metadata)
        if view is not None:
            views.append(view)
    total = len(views)

    allowed_states = STATUS_FILTERS[status_filter]
    needle = q.casefold()
    filtered = []
    for view in views:
        if allowed_states is not None and view.state.key not in allowed_states:
            continue
        if mode_filter != "all" and view.mode_key != mode_filter:
            continue
        if needle and not any(
                needle in str(value).casefold()
                for value in (view.label, view.run_id, view.want)):
            continue
        filtered.append(view)
    filtered.sort(key=lambda view: (view.created, view.run_id), reverse=True)

    pages = max(1, math.ceil(len(filtered) / PER_PAGE))
    page = _page_number(request.args.get("page", "1"), pages)
    start = (page - 1) * PER_PAGE
    page_views = filtered[start:start + PER_PAGE]

    filters: dict[str, str] = {}
    if q:
        filters["q"] = q
    if status_filter != "all":
        filters["status"] = status_filter
    if mode_filter != "all":
        filters["mode"] = mode_filter

    def page_url(number: int) -> str:
        params = dict(filters)
        params["page"] = max(1, number)
        return url_for("main.index", **params)

    return render_template(
        "runs.html",
        views=page_views,
        total=total,
        matched_count=len(filtered),
        page=page,
        pages=pages,
        per_page=PER_PAGE,
        page_url=page_url,
        filters=filters,
        q=q,
        status_filter=status_filter,
        mode_filter=mode_filter,
    )


@bp.get("/new")
def new_run():
    return render_template("new_run.html", form=_default_form(), errors=[])


@bp.post("/api/pdf-info")
def pdf_info():
    """Advisory page counts for the selected files; nothing is written to disk."""

    results = []
    for storage in request.files.getlist("pdfs"):
        if storage is None or not storage.filename:
            continue
        name = storage.filename
        try:
            data = storage.read()
        except OSError:
            results.append({"name": name, "size": 0, "pages": None,
                            "error": "读取失败"})
            continue
        item = {"name": name, "size": len(data), "pages": None, "error": ""}
        if not name.lower().endswith(".pdf"):
            item["error"] = "只接受 PDF"
        elif not data:
            item["error"] = "文件为空"
        else:
            try:
                item["pages"] = count_pdf_pages(data)
            except Exception:
                item["error"] = "无法读取 PDF"
            else:
                if item["pages"] < 1:
                    item["error"] = "PDF 没有页面"
                    item["pages"] = None
        results.append(item)
    response = jsonify({"files": results})
    response.headers["Cache-Control"] = "no-store"
    return response


@bp.post("/runs")
def create_run():
    form = _form_from_request()
    errors: list[str] = []
    store = _store()

    pdf_uploads: list[dict] = []
    for storage in request.files.getlist("pdfs"):
        if storage is None or not storage.filename:
            continue
        name = storage.filename
        try:
            data = storage.read()
        except OSError:
            errors.append(f"{name}：读取上传文件失败。")
            continue
        if not name.lower().endswith(".pdf"):
            errors.append(f"{name}：只接受 PDF 文件。")
            continue
        if not data:
            errors.append(f"{name}：文件为空。")
            continue
        try:
            page_count = count_pdf_pages(data)
        except Exception:
            errors.append(f"{name}：无法读取 PDF，请检查文件是否损坏或加密。")
            continue
        if page_count < 1:
            errors.append(f"{name}：PDF 没有页面。")
            continue
        pdf_uploads.append({"name": name, "data": data, "pages": page_count})
    if not pdf_uploads and not errors:
        errors.append("请至少上传一份 PDF。")

    template_data: tuple[str, bytes] | None = None
    template_upload = request.files.get("template")
    if template_upload is not None and template_upload.filename:
        template_name = template_upload.filename
        if Path(template_name).suffix.lower() not in {".txt", ".dat", ".csv"}:
            errors.append("模板只接受 .txt、.dat 或 .csv 文件。")
        else:
            try:
                data = template_upload.read()
            except OSError:
                data = b""
                errors.append("模板文件读取失败。")
            if not data and not errors:
                errors.append("模板文件为空。")
            elif data:
                template_data = (template_name, data)

    try:
        dpi = int(form["dpi"])
    except (TypeError, ValueError):
        dpi = 300
        errors.append("渲染精度必须是 150、200、300、400 或 600 DPI。")
    if dpi not in DPI_OPTIONS:
        errors.append("渲染精度必须是 150、200、300、400 或 600 DPI。")

    try:
        jobs = int(form["jobs"])
    except (TypeError, ValueError):
        jobs = 3
        errors.append("批量并行数必须是 1 到 8 的整数。")
    if not 1 <= jobs <= 8:
        errors.append("批量并行数必须是 1 到 8 的整数。")
    # More workers than PDFs only adds spawn overhead; for one PDF extraction
    # stays in-process so the VLM client and prepared-image cache are shared.
    jobs = max(1, min(jobs, len(pdf_uploads) or 1))

    pages = form["pages"]
    if not valid_pages(pages):
        errors.append("页码从 1 开始，请填写 3,5,7 这样的页码列表。")
    elif pages.strip() and pdf_uploads:
        maximum = max(int(value.strip()) for value in pages.split(","))
        too_short = [item["name"] for item in pdf_uploads if maximum > item["pages"]]
        if too_short:
            errors.append("指定页码超出文件总页数：" + "、".join(too_short))

    if form["mode"] == "analyze" and len(pdf_uploads) > 1:
        errors.append("仅分析模式一次处理一篇 PDF。")
    if template_data and form["mode"] == "analyze":
        errors.append("模板导出需要选择自动提取；仅分析不生成模板文件。")

    want = form["want"].strip()
    if len(want) > 500:
        errors.append("提取目标不能超过 500 个字符。")
    figure = form["figure"].strip()
    if figure:
        if not figure.isdigit() or not 1 <= int(figure) <= 9999:
            errors.append("Figure 编号必须是 1 到 9999 的整数。")
        else:
            # An exact figure number is a stronger selection signal than the
            # free-text description; keep the resulting task metadata explicit.
            want = f"Figure {int(figure)}"
    api_key = (request.form.get("api_key") or "").strip()
    if "\n" in api_key or "\r" in api_key:
        errors.append("API Key 不能包含换行。")
    if not api_key and not key_status(store.root).ready:
        errors.append("视觉模型需要 API Key；请填写密钥，或先在“密钥设置”中保存。")

    if errors:
        return _render_new(form, errors)

    key_warning = ""
    if form["remember_key"] and api_key:
        try:
            save_api_key(store.root, api_key)
        except (OSError, ValueError):
            key_warning = "API Key 保存失败，请检查项目目录的写入权限；本次任务仍会使用输入的密钥。"

    try:
        run_id = store.create_run()
    except OSError:
        errors.append("无法创建任务目录，请检查 .ui_workspace 的写入权限。")
        return _render_new(form, errors, 500)

    template_path = None
    try:
        input_paths = store.save_uploads(
            run_id, ((item["name"], item["data"]) for item in pdf_uploads))
        if template_data is not None:
            template_path = store.save_template(run_id, *template_data)
        options = RunOptions(
            dpi=dpi,
            pages=pages,
            force=form["force"],
            use_vlm=True,
            want=want,
            want_deep=form["want_deep"],
            find_series=form["find_series"],
            verbose=form["verbose"],
            keep_intermediates=form["keep_intermediates"],
            jobs=jobs,
            model=form["model"],
            base_url=form["base_url"],
            template=template_path,
        )
        command = build_command(
            store.root / "run.py", input_paths, store.output_dir(run_id), options,
            analyze_only=form["mode"] == "analyze")
    except (OSError, ValueError) as exc:
        shutil.rmtree(store.run_dir(run_id), ignore_errors=True)
        errors.append(f"无法创建任务：{exc}")
        return _render_new(form, errors)

    papers = [
        {"name": path.name, "pages": upload["pages"], "size": len(upload["data"])}
        for upload, path in zip(pdf_uploads, input_paths)
    ]
    metadata = {
        "label": "、".join(path.name for path in input_paths),
        "papers": papers,
        "mode": MODE_LABELS[form["mode"]],
        "mode_key": form["mode"],
        "command": command,
        "code": None,
        "elapsed": None,
        "want": want,
        "status": "queued",
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "artifact_counts": {},
        "pages": pages.replace(" ", ""),
        "dpi": dpi,
    }
    try:
        store.write_metadata(run_id, metadata)
    except OSError:
        shutil.rmtree(store.run_dir(run_id), ignore_errors=True)
        errors.append("无法保存任务信息，请检查 .ui_workspace 的写入权限。")
        return _render_new(form, errors, 500)

    if not figure:
        try:
            save_ui_preferences(store.workspace, want=want)
        except OSError:
            pass

    try:
        _tasks().submit(run_id, command, api_key)
    except Exception:
        store.update_metadata(
            run_id, status="failed", code=1, finished_at=now_iso(),
            failure_reason="后台执行器启动失败。")
        flash("任务已创建，但后台执行器启动失败；请查看任务详情。", "danger")
        return redirect(url_for("main.run_detail", run_id=run_id), code=303)

    if key_warning:
        flash(key_warning, "warning")
    flash("任务已提交，正在后台运行。可以离开本页，稍后在“任务结果”查看。", "success")
    return redirect(url_for("main.run_detail", run_id=run_id), code=303)


@bp.get("/runs/<run_id>")
def run_detail(run_id: str):
    store = _store()
    try:
        view = build_run_view(store, run_id)
    except ValueError:
        abort(404)
    if view is None:
        abort(404)

    scope_paper = _select_paper(view, request.args.get("paper", ""))
    scope_artifacts = scope_paper.artifacts if scope_paper else view.artifacts
    scope_counts = (
        {key: len(scope_artifacts.get(key, [])) for key in view.counts}
        if scope_paper else view.counts
    )
    default_tab = "images" if scope_paper else (
        "papers" if view.paper_count > 1 else "images")
    requested_tab = request.args.get("tab", default_tab)
    valid_tabs = {item[0] for item in TAB_DEFS}
    active_tab = requested_tab if requested_tab in valid_tabs else default_tab
    tabs = [
        {"key": key, "label": label, "icon": icon,
         "count": view.paper_count if group == "papers" else scope_counts[group]}
        for key, label, icon, group in TAB_DEFS
    ]
    requested_file = request.args.get("file", "")
    context = {
        "view": view,
        "scope_paper": scope_paper,
        "scope_counts": scope_counts,
        "tabs": tabs,
        "active_tab": active_tab,
        "log_text": read_log_tail(view.log_path),
        "elapsed_display": format_duration(view.elapsed),
        "file_options": [],
        "selected_rel": None,
        "preview_text": "",
        "preview_truncated": False,
        "preview_error": "",
        "csv_header": [],
        "csv_rows": [],
        "csv_truncated": False,
        "command_json": "",
        "configs": [],
        "chart_data": None,
    }

    if active_tab == "papers":
        pass
    elif active_tab == "images":
        selected = _pick_file(view.output, scope_artifacts["images"], requested_file)
        context["file_options"] = _file_options(view.output, scope_artifacts["images"])
        if selected is not None:
            context["selected_rel"] = _relative(view.output, selected)
            context["chart_data"] = build_panel_chart(selected)
            if context["chart_data"] is not None:
                context["chart_data"]["paper_scope"] = (
                    scope_paper.scope if scope_paper
                    else _paper_scope_for_path(view, selected)
                )
    elif active_tab == "csv":
        selected = _pick_file(view.output, scope_artifacts["csv"], requested_file)
        context["file_options"] = _file_options(view.output, scope_artifacts["csv"])
        if selected is not None:
            context["selected_rel"] = _relative(view.output, selected)
            header, rows, truncated, error = _read_csv_preview(selected)
            context.update(csv_header=header, csv_rows=rows,
                           csv_truncated=truncated, preview_error=error)
    elif active_tab == "templates":
        selected = _pick_file(view.output, scope_artifacts["templates"], requested_file)
        context["file_options"] = _file_options(view.output, scope_artifacts["templates"])
        if selected is not None:
            context["selected_rel"] = _relative(view.output, selected)
            text, truncated, error = _read_text_preview(selected)
            context.update(preview_text=text, preview_truncated=truncated,
                           preview_error=error)
    elif active_tab == "report":
        selected = _pick_file(view.output, scope_artifacts["reports"], requested_file)
        context["file_options"] = _file_options(view.output, scope_artifacts["reports"])
        if selected is not None:
            context["selected_rel"] = _relative(view.output, selected)
            text, truncated, error = _read_text_preview(selected)
            context.update(preview_text=text, preview_truncated=truncated,
                           preview_error=error)
    else:
        context["command_json"] = (
            json.dumps(view.command, ensure_ascii=False, indent=2)
            if view.command else "")
        context["configs"] = [
            {"rel": _relative(view.output, path)} for path in scope_artifacts["configs"]
        ]

    return render_template("run_detail.html", **context)


@bp.get("/api/runs/<run_id>")
def run_status(run_id: str):
    store = _store()
    try:
        view = build_run_view(store, run_id)
    except ValueError:
        abort(404)
    if view is None:
        return jsonify({"error": "not_found"}), 404
    response = jsonify({
        "run_id": view.run_id,
        "status": view.state.key,
        "status_label": view.state.label,
        "detail": view.state.detail,
        "tone": view.state.tone,
        "active": view.active,
        "finished": not view.active,
        "elapsed": view.elapsed,
        "elapsed_display": format_duration(view.elapsed),
        "log": read_log_tail(view.log_path),
        "batch_failures": view.batch_failures,
        "counts": view.counts,
    })
    response.headers["Cache-Control"] = "no-store"
    return response


@bp.post("/runs/<run_id>/save-curves")
def save_curves(run_id: str):
    store = _store()
    try:
        view = build_run_view(store, run_id)
    except ValueError:
        abort(404)
    if view is None:
        abort(404)

    payload = request.get_json(silent=True) or {}
    paper_scope = str(payload.get("paper_scope") or "").strip()
    panel_id = str(payload.get("panel_id") or "").strip()
    series_payload = payload.get("series")
    axis_payload = payload.get("axis")
    if not panel_id:
        return jsonify({"error": "缺少面板标识。"}), 400
    if paper_scope:
        paper = _select_paper(view, paper_scope)
        if paper is None:
            return jsonify({"error": "找不到对应文献。"}), 400
    else:
        if view.paper_count > 1:
            return jsonify({"error": "请先指定要保存的文献。"}), 400
        paper = view.papers[0] if view.papers else None
    paper_root = view.output / paper.scope if paper and paper.scope else view.output
    try:
        paper_root.resolve().relative_to(view.output.resolve())
    except (OSError, ValueError):
        return jsonify({"error": "文献路径无效。"}), 400
    try:
        result = save_panel_curves(paper_root, panel_id, series_payload, axis_payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    response = jsonify({"ok": True, **result})
    response.headers["Cache-Control"] = "no-store"
    return response


@bp.get("/runs/<run_id>/files/<path:artifact_path>")
def download_artifact(run_id: str, artifact_path: str):
    store = _store()
    try:
        view = build_run_view(store, run_id)
    except ValueError:
        abort(404)
    if view is None:
        abort(404)

    output_root = view.output.resolve()
    allowed: dict[Path, Path] = {}
    for paths in view.artifacts.values():
        for candidate in paths:
            try:
                resolved = candidate.resolve()
                resolved.relative_to(output_root)
            except (OSError, ValueError):
                continue
            allowed[resolved] = candidate
    try:
        requested = (view.output / artifact_path).resolve()
        requested.relative_to(output_root)
    except (OSError, ValueError):
        abort(404)
    path = allowed.get(requested)
    if path is None:
        abort(404)

    as_attachment = request.args.get("download", "").lower() in {"1", "true", "yes"}
    try:
        response = send_file(path, as_attachment=as_attachment,
                             download_name=path.name, conditional=True)
    except OSError:
        abort(404)
    response.headers["Cache-Control"] = "no-store"
    return response


@bp.post("/runs/<run_id>/delete")
def delete_run(run_id: str):
    store = _store()
    try:
        view = build_run_view(store, run_id)
    except ValueError:
        abort(404)
    if view is None:
        abort(404)
    if view.state.key in {"queued", "running"}:
        flash("任务仍在运行或排队，不能删除；请先等待结束或停止服务。", "warning")
        return redirect(_safe_next(url_for("main.index")), code=303)
    try:
        shutil.rmtree(store.run_dir(run_id))
    except OSError:
        flash("删除失败，请检查 .ui_workspace 的写入权限。", "danger")
    else:
        flash(f"已删除任务 {view.short_id}。", "success")
    return redirect(_safe_next(url_for("main.index")), code=303)


@bp.get("/runs/<run_id>/archive.zip")
def download_archive(run_id: str):
    store = _store()
    try:
        view = build_run_view(store, run_id)
    except ValueError:
        abort(404)
    if view is None:
        abort(404)
    if view.result_total == 0:
        flash("这个任务还没有可下载的结果文件。", "warning")
        return redirect(url_for("main.run_detail", run_id=run_id), code=303)
    try:
        data = make_archive(view.output)
    except OSError:
        flash("打包结果时读取文件失败，请查看任务日志。", "danger")
        return redirect(url_for("main.run_detail", run_id=run_id), code=303)
    response = send_file(
        io.BytesIO(data), mimetype="application/zip", as_attachment=True,
        download_name=f"dig-{run_id}.zip")
    response.headers["Cache-Control"] = "no-store"
    return response


@bp.post("/settings/api-key")
def save_key():
    key = request.form.get("api_key", "")
    try:
        save_api_key(_store().root, key)
    except ValueError as exc:
        flash(str(exc), "danger")
    except OSError:
        flash("密钥保存失败，请检查项目目录的写入权限。", "danger")
    else:
        flash("API Key 已保存到本机 deepseek_key.txt；页面不会回显已保存密钥。", "success")
    return redirect(_safe_next(url_for("main.index")), code=303)


@bp.post("/settings/api-key/clear")
def clear_key():
    try:
        clear_saved_api_key(_store().root)
    except OSError:
        flash("清除失败，请检查项目目录的写入权限。", "danger")
    else:
        flash("已删除本机保存的 API Key 文件；环境变量中的密钥不受影响。", "success")
    return redirect(_safe_next(url_for("main.index")), code=303)


@bp.get("/help")
def help_page():
    return render_template("help.html")


@bp.app_errorhandler(404)
def not_found(error):
    return render_template(
        "error.html", code=404, title="页面不存在",
        message="请求的页面或任务不存在，可能已被移动。"), 404


@bp.app_errorhandler(500)
def server_error(error):
    return render_template(
        "error.html", code=500, title="本地服务出错了",
        message="处理请求时发生错误，请查看启动终端中的输出后重试。"), 500


@bp.after_app_request
def _security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    return response
