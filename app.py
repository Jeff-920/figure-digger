"""DIG Studio: a local workbench for extracting chart data from PDFs."""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import time
import uuid
from collections import deque
from datetime import datetime
from itertools import islice
from pathlib import Path

import streamlit as st

from ui_utils import (RunOptions, build_command, collect_artifacts, make_archive,
                      safe_filename, is_figure_number_request, valid_pages, result_status,
                      batch_failures, save_api_key, clear_saved_api_key)


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / ".ui_workspace"

st.set_page_config(page_title="DIG Studio", page_icon="▦", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    :root { --ink:#18292f; --muted:#53676b; --rule:#d9e3e4; --teal:#176b64; }
    .stApp { background:#f6f8f8; color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1220px; padding-top:2.3rem; padding-bottom:4rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
    h1 { font-size:clamp(2.15rem,3.5vw,3.4rem); line-height:1.16; max-width:820px; }
    h2 { font-size:1.35rem; }
    h1,h2,h3 { text-wrap:balance; }
    .stApp { font-family:'Segoe UI','Microsoft YaHei',sans-serif; }
    [data-testid="stMetricValue"] { font-variant-numeric:tabular-nums; }
    .stButton button { transition:background-color .15s ease, border-color .15s ease; }
    .dig-summary { border-left:3px solid var(--teal); padding:.7rem 1rem;
      background:#eaf2f0; margin:1rem 0; line-height:1.7; }
    .dig-top { display:flex; align-items:baseline; justify-content:space-between;
      border-bottom:1px solid var(--rule); padding:0 0 1rem; margin-bottom:2rem; }
    .dig-mark { font-size:1.05rem; font-weight:850; letter-spacing:-.055em; color:var(--ink); }
    .dig-mark span { color:var(--teal); font-weight:600; letter-spacing:0; }
    .dig-top small { color:var(--muted); font-size:.83rem; }
    .dig-intro { max-width:720px; color:var(--muted); font-size:1.05rem; line-height:1.7;
      margin:-.6rem 0 2rem; }
    div[data-testid="stFileUploader"] section { background:#fff; border:1.5px dashed #8faeac;
      border-radius:10px; min-height:95px; }
    div[data-testid="stVerticalBlockBorderWrapper"] > div { border-color:var(--rule); }
    .stButton button[kind="primary"] { background:var(--teal); border:1px solid var(--teal);
      color:#fff; border-radius:8px; font-weight:700; min-height:2.9rem; }
    .stButton button[kind="primary"]:hover { background:#11564f; border-color:#11564f; }
    .stButton button:focus-visible, input:focus-visible, textarea:focus-visible {
      outline:3px solid #8bc9bf; outline-offset:2px; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--rule);
      border-radius:8px; padding:.8rem 1rem; }
    .stTabs [data-baseweb="tab-list"] { border-bottom:1px solid var(--rule); gap:1rem; }
    .stTabs [data-baseweb="tab"] { font-weight:650; padding:.7rem .2rem; }
    @media (max-width:700px) { .block-container { padding-top:1rem; }
      .dig-top small { display:none; } .dig-intro { font-size:1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def new_run_dir() -> Path:
    label = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = WORKSPACE / f"{label}-{uuid.uuid4().hex[:6]}"
    (run_dir / "input").mkdir(parents=True)
    (run_dir / "output").mkdir()
    return run_dir


def save_uploads(files, directory: Path) -> list[Path]:
    paths = []
    used: set[str] = set()
    for uploaded in files:
        base = safe_filename(uploaded.name)
        name = base
        count = 2
        while name.casefold() in used:
            name = f"{Path(base).stem}_{count}{Path(base).suffix}"
            count += 1
        used.add(name.casefold())
        path = directory / name
        path.write_bytes(uploaded.getvalue())
        paths.append(path)
    return paths


def run_pipeline(command: list[str], api_key: str, log_path: Path, log_slot) -> tuple[int, float]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    if api_key.strip():
        env["DEEPSEEK_API_KEY"] = api_key.strip()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.monotonic()
    tail: deque[str] = deque(maxlen=70)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=env, creationflags=flags,
            )
            assert process.stdout is not None
            last_draw = 0.0
            for line in process.stdout:
                log.write(line)
                log.flush()
                tail.append(line.rstrip())
                if time.monotonic() - last_draw > 0.3:
                    log_slot.code("\n".join(tail), language="text")
                    last_draw = time.monotonic()
            code = process.wait()
    except OSError as exc:
        message = f"无法启动处理程序：{exc}"
        log_path.write_text(message, encoding="utf-8")
        tail.append(message)
        code = 1
    log_slot.code("\n".join(tail) or "处理程序没有输出日志。", language="text")
    return code, time.monotonic() - started


def recent_runs() -> list[tuple[str, dict]]:
    runs = []
    if not WORKSPACE.exists():
        return runs
    for path in WORKSPACE.glob("*/ui-run.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            runs.append((path.parent.name, metadata))
        except (OSError, ValueError):
            continue
    return sorted(runs, reverse=True)


def read_csv_preview(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(islice(csv.DictReader(handle), 200))


@st.cache_data(show_spinner=False, max_entries=12)
def pdf_page_count(data: bytes) -> int:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(data)) as document:
        return len(document.pages)


def has_vlm_key(entered: str) -> bool:
    if entered.strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return True
    key_file = ROOT / "deepseek_key.txt"
    try:
        if any(line.strip() and not line.lstrip().startswith("#")
               for line in key_file.read_text(encoding="utf-8").splitlines()):
            return True
    except OSError:
        pass
    try:
        secrets = json.loads((ROOT / "secrets.json").read_text(encoding="utf-8"))
        return bool(str(secrets.get("deepseek_api_key") or secrets.get("api_key") or "").strip())
    except (OSError, ValueError, AttributeError):
        return False


def forget_api_key() -> None:
    try:
        clear_saved_api_key(ROOT)
    except OSError:
        st.session_state["api_key_save_error"] = "清除失败，请检查项目目录的写入权限。"
        return
    st.session_state["api_key_input"] = ""
    st.session_state.pop("api_key_save_error", None)


def show_results(run_id: str, metadata: dict, key: str) -> None:
    output = WORKSPACE / run_id / "output"
    artifacts = collect_artifacts(output)
    state = result_status(metadata, artifacts)
    st.subheader(f"{state} · {metadata.get('label', run_id)}")
    st.caption(f"任务 {run_id}  ·  原始文件和结果保存在服务所在机器的 .ui_workspace 目录")
    failed = batch_failures(output)
    if failed:
        st.warning("部分论文处理失败：" + "、".join(failed) + "。请查看报告和日志。")
    if metadata.get("code") == 0 and not artifacts["csv"]:
        st.info("本次没有曲线 CSV。仅分析任务需确认坐标后提取；自动任务请检查报告中的选图、坐标和失败原因。")
    metrics = st.columns(4)
    metrics[0].metric("CSV 文件", len(artifacts["csv"]))
    metrics[1].metric("核对图", len(artifacts["images"]))
    metrics[2].metric("报告", len(artifacts["reports"]))
    metrics[3].metric("耗时", f"{metadata.get('elapsed', 0):.1f} 秒")

    if any(artifacts.values()):
        if st.button("准备全部结果下载", key=f"{key}-prepare-{run_id}"):
            with st.spinner("正在打包结果…"):
                st.session_state[f"archive-{run_id}"] = make_archive(output)
        if f"archive-{run_id}" in st.session_state:
            st.download_button("下载本次全部结果 (.zip)", st.session_state[f"archive-{run_id}"],
                           file_name=f"dig-{run_id}.zip", mime="application/zip",
                           key=f"{key}-archive")

    image_tab, data_tab, template_tab, report_tab, detail_tab = st.tabs(
        ["图像核对", "曲线数据", "模板文件", "报告", "日志与配置"]
    )
    with data_tab:
        if not artifacts["csv"]:
            st.info("暂无 CSV。如果选择了“仅分析”，请先确认坐标轴，再运行提取命令。")
        else:
            selected = st.selectbox("选择数据文件", artifacts["csv"],
                                    format_func=lambda p: str(p.relative_to(output)),
                                    key=f"{key}-csv")
            try:
                st.dataframe(read_csv_preview(selected), width="stretch",
                             hide_index=True)
                st.caption("最多预览前 200 行；下载文件包含全部数据。")
            except (OSError, UnicodeError, csv.Error) as exc:
                st.warning(f"预览失败：{exc}")
            st.download_button("下载当前 CSV", selected.read_bytes(), file_name=selected.name,
                               mime="text/csv", key=f"{key}-csv-download")
    with image_tab:
        if not artifacts["images"]:
            st.info("提取完成后，这里会显示曲线数据重投影到原图的核对图。")
        else:
            selected = st.selectbox("选择核对图", artifacts["images"],
                                    format_func=lambda p: str(p.relative_to(output)),
                                    key=f"{key}-image")
            st.image(str(selected), caption=str(selected.relative_to(output)),
                     width="stretch")
            st.caption("请检查提取点是否贴合原曲线，尤其是遮挡、交叉和坐标轴边缘。")
    with template_tab:
        if not artifacts["templates"]:
            st.info("本次没有模板文件。需要按指定格式导出时，在开始提取前上传模板。")
        else:
            selected = st.selectbox("选择模板结果", artifacts["templates"],
                                    format_func=lambda p: str(p.relative_to(output)), key=f"{key}-template")
            st.code(selected.read_text(encoding="utf-8", errors="replace")[:24000], language="text")
            st.download_button("下载模板结果", selected.read_bytes(), file_name=selected.name,
                               key=f"{key}-template-download")
    with report_tab:
        if not artifacts["reports"]:
            st.info("暂无报告。")
        else:
            selected = st.selectbox("选择报告", artifacts["reports"],
                                    format_func=lambda p: str(p.relative_to(output)),
                                    key=f"{key}-report")
            st.markdown(selected.read_text(encoding="utf-8", errors="replace"))
    with detail_tab:
        if metadata.get("command"):
            st.caption("本次实际执行参数（不含 API Key）")
            st.json(metadata["command"])
        for path in artifacts["configs"]:
            st.download_button(f"下载坐标配置 · {path.name}", path.read_bytes(),
                               file_name=path.name, mime="application/json",
                               key=f"{key}-config-{path}")
        log_path = output.parent / "ui-run.log"
        if log_path.exists():
            st.code(log_path.read_text(encoding="utf-8", errors="replace")[-24000:],
                    language="text")


st.markdown(
    '<div class="dig-top"><div class="dig-mark">DIG <span>/ Studio</span></div>'
    '<small>PDF 图表数据提取工作台 · 本地运行</small></div>',
    unsafe_allow_html=True,
)
st.title("论文曲线提取工作台")
st.markdown(
    '<p class="dig-intro">上传 PDF，提取折线图和散点图中的坐标点。'
    '输出 CSV、提取报告与原图核对图，方便检查后再用于分析。</p>',
    unsafe_allow_html=True,
)

new_tab, history_tab, help_tab = st.tabs(["开始提取", "任务结果", "使用说明"])

with new_tab:
    input_col, settings_col = st.columns([1.35, 1], gap="large")
    with input_col:
        st.subheader("01 / 论文与目标")
        uploads = st.file_uploader("选择一篇或多篇 PDF", type=["pdf"],
                                   accept_multiple_files=True)
        upload_errors = []
        page_counts = []
        if uploads:
            for item in uploads:
                try:
                    count = pdf_page_count(item.getvalue())
                    if count < 1:
                        raise ValueError("PDF 没有页面")
                    page_counts.append((item.name, count))
                    st.caption(f"{item.name} · {count} 页 · {item.size / 1024 / 1024:.1f} MB")
                except Exception as exc:
                    upload_errors.append(f"{item.name}：无法读取 PDF，请检查文件是否损坏或加密。")
            for message in upload_errors:
                st.error(message)
        want = st.text_area("想提取哪张图？", height=100,
                            placeholder="例如：Figure 5，或 800 bar 条件下速度随时间的曲线",
                            help="留空会处理检测到的全部数据图；按文字描述找图需要视觉模型。")
        st.caption("知道图号就填 Figure 5；没有特定目标可以留空。")
        template_upload = st.file_uploader("按模板导出（可选）", type=["txt", "dat", "csv"],
                                           help="此功能需要视觉模型。")

    with settings_col:
        st.subheader("02 / 提取设置")
        mode = st.radio("流程", ["自动提取", "仅分析，稍后确认坐标"],
                        help="自动提取会直接使用识别到的轴范围；请在结果页核对原图。")
        method = st.radio("读图方式", ["视觉模型（DeepSeek）", "本地 OCR"], horizontal=True,
                          help="与原项目 --vlm 流程对应。OCR 是独立的离线流程，选图和曲线识别能力不同。")
        use_vlm = method.startswith("视觉模型")
        api_key = st.text_input("DeepSeek API Key", type="password", disabled=not use_vlm,
                                key="api_key_input", placeholder="已保存时可留空直接使用",
                                help="勾选记住后，将密钥以明文保存在运行机器的 deepseek_key.txt；不写入任务日志和下载包。")
        remember_key = st.checkbox("记住 API Key（保存在本机）", value=True,
                                   key="remember_api_key", disabled=not use_vlm)
        if use_vlm and remember_key and api_key.strip():
            try:
                save_api_key(ROOT, api_key)
                st.session_state.pop("api_key_save_error", None)
            except (OSError, ValueError):
                st.session_state["api_key_save_error"] = "密钥保存失败，请检查输入及项目目录的写入权限。"
        if st.session_state.get("api_key_save_error"):
            st.warning(st.session_state["api_key_save_error"])
        if (ROOT / "deepseek_key.txt").exists():
            st.caption("本机已保存 API Key，后续任务和重新打开网页时自动复用。")
            st.button("清除已保存的 API Key", on_click=forget_api_key,
                      key="forget_api_key", disabled=not use_vlm)
        has_key = has_vlm_key(api_key)
        if use_vlm and not has_key:
            st.warning("视觉模型需要 API Key。填写后即可运行；本地 OCR 不需要。")
        elif use_vlm:
            st.caption("密钥已就绪 · 仅用于模型调用")
        if want.strip() and not use_vlm and not is_figure_number_request(want):
            st.warning("本地 OCR 只支持纯图号。图号后附加内容、工况或子图条件，也需要视觉模型。")

        with st.expander("更多设置"):
            pages = st.text_input("指定页码", placeholder="如 3,5,7",
                                  help="PDF 文件实际页序，从 1 开始；不是论文印刷页码或图号。")
            dpi = st.select_slider("渲染精度 (DPI)", options=[150, 200, 300, 400, 600], value=300)
            jobs = st.slider("批量并行数", 1, 8, 2,
                             disabled=not uploads or len(uploads) < 2)
            use_ocr = st.toggle("视觉模型 + OCR 交叉核对", value=False,
                                disabled=not use_vlm)
            want_deep = st.toggle("结合正文核对图意", value=False, disabled=not use_vlm)
            find_series = st.toggle("自动识别曲线", value=True, disabled=not use_vlm)
            force = st.toggle("接受未确认的自动轴范围", value=True,
                              disabled=mode.startswith("仅分析"),
                              help="对应 --force。关闭后仅提取后端已确认坐标的面板。")
            keep_intermediates = st.toggle("保留诊断图片", value=False)
            verbose = st.toggle("记录详细日志", value=False)
            model = st.text_input("模型名称", value="deepseek-flash", disabled=not use_vlm)
            base_url = st.text_input("接口地址", value="https://api.deepseek.com",
                                     disabled=not use_vlm)

        invalid_description = bool(want.strip() and not use_vlm and
                                   not is_figure_number_request(want))
        invalid_batch = bool(uploads and len(uploads) > 1 and mode.startswith("仅分析"))
        if invalid_batch:
            st.info("仅分析模式一次处理一篇 PDF。")
        if template_upload and not use_vlm:
            st.info("模板输出需要视觉模型。")
        invalid_template = bool(template_upload and mode.startswith("仅分析"))
        if invalid_template:
            st.info("模板导出需要选择自动提取；仅分析不生成模板文件。")
        page_error = ""
        if not valid_pages(pages):
            page_error = "页码从 1 开始，请填写 3,5,7 这样的页码列表。"
        elif pages.strip():
            maximum = max(int(value.strip()) for value in pages.split(","))
            too_short = [name for name, count in page_counts if maximum > count]
            if too_short:
                page_error = "指定页码超出文件总页数：" + "、".join(too_short)
        if page_error:
            st.error(page_error)
        st.markdown("**本次任务**")
        st.caption(f"{len(uploads or [])} 篇 PDF · {mode} · {method} · {dpi} DPI")
        st.caption("目标：" + (want.strip() or "检测到的全部数据图") + "；页码：" + (pages.strip() or "全部"))
        can_run = bool(uploads) and not upload_errors and not page_error and not invalid_batch and not invalid_description and (
            not use_vlm or has_key) and (not template_upload or use_vlm) and not invalid_template
        clicked = st.button("开始处理 PDF", type="primary", width="stretch",
                            disabled=not can_run)

    if clicked:
        run_dir = new_run_dir()
        input_paths = save_uploads(uploads, run_dir / "input")
        template_path = None
        if template_upload is not None:
            template_path = run_dir / safe_filename(template_upload.name, "template.txt")
            template_path.write_bytes(template_upload.getvalue())
        options = RunOptions(
            dpi=dpi, pages=pages, force=force, use_vlm=use_vlm, use_ocr=use_ocr,
            want=want, want_deep=want_deep, find_series=find_series,
            verbose=verbose, keep_intermediates=keep_intermediates,
            jobs=jobs, model=model, base_url=base_url, template=template_path,
        )
        try:
            command = build_command(ROOT / "run.py", input_paths, run_dir / "output", options,
                                    analyze_only=mode.startswith("仅分析"))
        except ValueError as exc:
            st.error(str(exc))
        else:
            with st.status("正在分析 PDF", expanded=True) as status:
                log_slot = st.empty()
                code, elapsed = run_pipeline(command, api_key, run_dir / "ui-run.log", log_slot)
                status.update(label="处理完成" if code == 0 else "处理失败",
                              state="complete" if code == 0 else "error", expanded=code != 0)
            metadata = {"code": code, "elapsed": elapsed,
                        "label": "、".join(path.name for path in input_paths),
                        "mode": mode, "method": method, "command": command}
            (run_dir / "ui-run.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            st.session_state["latest_run"] = (run_dir.name, metadata)

    if st.session_state.get("latest_run"):
        st.divider()
        run_id, metadata = st.session_state["latest_run"]
        show_results(run_id, metadata, "new")

with history_tab:
    runs = recent_runs()
    if not runs:
        st.info("完成一次处理后，结果会显示在这里；关闭浏览器后也可再次查看。")
    else:
        choices = {run_id: metadata for run_id, metadata in runs}
        selected_id = st.selectbox("本机任务", list(choices),
                                   format_func=lambda item: f"{item}  ·  {choices[item].get('label', '')}",
                                   key="history-selection")
        show_results(selected_id, choices[selected_id], "history")

with help_tab:
    st.markdown(
        """
        ### 怎么使用

        上传 PDF 后，直接点“开始处理 PDF”。建议先选少数页试跑。知道图号时填写 `Figure 5`；
        想按图中含义查找，需要启用视觉模型并提供 API Key。结果页先看“图像核对”，再下载 CSV。

        ### 什么时候选“仅分析”

        坐标轴较模糊或数值要求很严时，先生成配置文件，再用命令行确认范围：

        ```bash
        python run.py confirm <配置文件> --id <面板ID> --x 0,5 --y 0,70
        python run.py extract <配置文件>
        ```

        上传文件、日志和结果保存在本机 `.ui_workspace/`；该目录不会提交到 Git。
        完整安装说明和命令行参数见项目 README。
        """
    )
