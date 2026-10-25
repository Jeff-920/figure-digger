"""End-to-end pipeline: paper PDF -> figures -> panels -> axis ranges -> data CSVs.

Two phases, because axis ranges must be confirmed by a human before data is trusted:

  python run.py analyze paper.pdf --out out/ --vlm    # figures, panels, legend colours, axis ranges
  python run.py confirm out/paper_config.json --id fig5_p1 --x 0,5 --y 0,70
  python run.py extract out/paper_config.json         # only confirmed panels
  python run.py all paper.pdf --out out/ --vlm --force # one shot, trusting model axis ranges as-is

The visual model reads axis tick values.  Vector figures can still provide their
own calibration from the PDF's text; raster figures without --vlm have no
automatic axis reader and remain unconfirmed.

  python run.py batch ./papers --out ./out --vlm --jobs 4    # one process per paper

Outputs under --out:
  figures/     cropped figure images from the PDF
  panels/      individual panel crops (multi-panel figures)
  csv/         extracted data, one CSV per series
  verify/      reprojection overlays for visual QA
  paper_config.json   phase-1 result; edit "axis" and set "confirmed": true
  report.md           what was extracted, with the axis evidence

The CLI is concise by default. Add --verbose for detailed progress and
--keep-intermediates to persist gap/review/selection diagnostics.
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

# 库模块都在 dig/ 里，先把它加进搜索路径。要在下面的 import 之前做，
# 否则 imgio/numpy 的"环境不对"误报会先跳出来（模块之间仍是扁平 import）。
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "dig"))

# Windows consoles default to GBK; make output UTF-8 before anything prints, including
# the dependency-check message below.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

try:
    import cv2
    import imgio as iio
    import numpy as np
except ImportError as _exc:  # almost always "wrong Python environment"
    _bar = "=" * 72
    print(_bar)
    print(f"依赖导入失败：{_exc}")
    print(f"当前解释器  ：{sys.executable}")
    print()
    print("这通常不是代码问题，而是用错了 Python 环境——例如用了 conda 的 base，")
    print("而不是装好本项目依赖的那个环境。numpy「装了但导入失败」也会走到这里。")
    print()
    print("请先确认解释器，再激活正确环境：")
    print('    python -c "import sys; print(sys.executable)"')
    print("    conda activate pytorch          # 换成你装依赖的那个环境")
    print("    python run.py doctor            # 一键自检")
    print(_bar)
    raise SystemExit(2)

import axis_calibration as ac  # noqa: E402
import batch_extract as be  # noqa: E402
import extract_lines as el  # noqa: E402
import figure_index as fi  # noqa: E402
import legend_colors as lc  # noqa: E402
import split_panels as sp  # noqa: E402
import triage_pdf as tp  # noqa: E402
import verify_overlay as vo  # noqa: E402
import vector_extract as vx  # noqa: E402
import vlm_client as vlmc  # noqa: E402
import vlm_tasks as vt  # noqa: E402
import template_out as tpl  # noqa: E402

# 并行批处理时子进程不往终端打（会互相穿插），而是各写一份 <out>/<pdf>/run.log。
_LOG_FILE = None
_VERBOSE = False
_KEEP_INTERMEDIATES = False
_INTERMEDIATE_DIRS = ("gaps", "review", "debug", "select", "objects")
# Bump when analysis prompts/phases change; otherwise a cached config may be
# reused across incompatible pipeline versions.
ANALYSIS_CACHE_VERSION = 3
ANALYSIS_CACHE_DIR = ROOT / ".analysis_cache"


def _file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_cache_key(pdf_path: Path, dpi: int, pages, want, want_deep,
                        find_series: bool, vlm) -> str:
    material = {
        "version": ANALYSIS_CACHE_VERSION,
        "pdf_sha256": _file_digest(pdf_path),
        "dpi": int(dpi),
        "pages": str(pages or ""),
        "want": str(want or ""),
        "want_deep": bool(want_deep),
        "find_series": bool(find_series),
        "model": getattr(vlm, "model", "") if vlm is not None else "no-vlm",
        "base_url": getattr(vlm, "base_url", "") if vlm is not None else "",
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _portable_config(config: dict, outdir: Path, pdf_path: Path) -> dict:
    portable = json.loads(json.dumps(config, ensure_ascii=False))
    old_out, old_pdf = str(outdir), str(pdf_path)

    def walk(value):
        if isinstance(value, str):
            value = value.replace(old_out, "__OUTDIR__")
            return value.replace(old_pdf, "__PDF__")
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        return value
    return walk(portable)


def _restore_config(config: dict, outdir: Path, pdf_path: Path) -> dict:
    restored = json.loads(json.dumps(config, ensure_ascii=False))

    def walk(value):
        if isinstance(value, str):
            value = value.replace("__OUTDIR__", str(outdir))
            return value.replace("__PDF__", str(pdf_path))
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        return value
    restored = walk(restored)
    restored["outdir"] = str(outdir)
    restored["pdf"] = str(pdf_path)
    return restored


def _load_analysis_cache(cache_dir: Path, outdir: Path, pdf_path: Path,
                         cfg_path: Path) -> bool:
    cache_config = cache_dir / "config.json"
    if not cache_config.is_file() or not (cache_dir / "panels").is_dir():
        return False
    try:
        config = json.loads(cache_config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            return False
    except (OSError, ValueError):
        return False
    outdir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_dir / "panels", outdir / "panels", dirs_exist_ok=True)
    figures = cache_dir / "figures"
    if figures.is_dir():
        shutil.copytree(figures, outdir / "figures", dirs_exist_ok=True)
    cfg_path.write_text(
        json.dumps(_restore_config(config, outdir, pdf_path), indent=2, ensure_ascii=False),
        encoding="utf-8")
    return True


def _save_analysis_cache(cache_dir: Path, outdir: Path, pdf_path: Path,
                         config: dict) -> None:
    staging = cache_dir.with_name(cache_dir.name + f".tmp-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for name in ("panels", "figures"):
            source = outdir / name
            if source.is_dir():
                shutil.copytree(source, staging / name)
        (staging / "config.json").write_text(
            json.dumps(_portable_config(config, outdir, pdf_path),
                       indent=2, ensure_ascii=False), encoding="utf-8")
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        staging.rename(cache_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def log(msg, detail=False):
    """默认只打印结果和警告；--verbose 或批处理日志保留完整细节。"""
    if detail and not (_VERBOSE or _LOG_FILE is not None):
        return
    if _LOG_FILE is not None:
        _LOG_FILE.write(str(msg) + "\n")
        _LOG_FILE.flush()
    else:
        print(msg, flush=True)


def detail_log(msg):
    log(msg, detail=True)


def _clean_stale_intermediates(outdir):
    """Default runs keep only durable outputs, including after an older debug run."""
    if _KEEP_INTERMEDIATES:
        return
    base = Path(outdir)
    for name in _INTERMEDIATE_DIRS:
        path = base / name
        if path.is_dir():
            shutil.rmtree(path)


def _model_anchors_in_frame(anchors, frame, inset=2.0):
    """模型补点只能落在绘图区内部；实测轨迹仍可合法触碰坐标轴边界。"""
    left, top, right, bottom = (float(v) for v in frame)
    return [(float(x), float(y)) for x, y in (anchors or [])
            if left + inset <= float(x) <= right - inset
            and top + inset <= float(y) <= bottom - inset]


def _missing_trace_spans(trace, x0, x1, min_width):
    """Return genuinely unmeasured x spans inside a proposed repair window."""
    xs = sorted({float(x) for x, _y in (trace or []) if x0 <= float(x) <= x1})
    if not xs:
        return [(float(x0), float(x1))] if x1 - x0 >= min_width else []
    spans = []
    if xs[0] - x0 >= min_width:
        spans.append((float(x0), xs[0]))
    spans.extend((a, b) for a, b in zip(xs, xs[1:]) if b - a >= min_width)
    if x1 - xs[-1] >= min_width:
        spans.append((xs[-1], float(x1)))
    return spans


def _elapsed(t0):
    """Human-readable elapsed time since t0."""
    d = time.time() - t0
    return f"{d:.1f}s" if d < 60 else f"{d / 60:.1f}min"


def detect_legend_colors(figure_img):
    """Colours of the coloured series, read from the figure's legend (any layout)."""
    gray = cv2.cvtColor(figure_img, cv2.COLOR_BGR2GRAY)
    best = None
    for box in lc.find_boxes(gray):
        sw = lc.find_swatches(figure_img, box)
        if sw and (best is None or len(sw) > len(best[1])):
            best = (box, sw)
    if best is None:
        return []
    _box, sw = best
    colors = [s["hex"] for s in sw]
    # A photo has no legend: dozens of "swatches" are just colourful image content.
    if len(colors) > 8:
        return []
    return colors


def split_figure(figure_path, outdir, margin=None, expected_panels=None,
                 panel_geometry=None, split_meta=None):
    """Return [(panel_image_path, frame_in_panel_coords, box_in_figure)] for a figure."""
    img = iio.imread(figure_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # 轮廓框 + L 形坐标轴框：只画了左轴/下轴的子图（很常见）没有闭合轮廓，
    # 单靠轮廓法会退回"曲线围出的空洞"，把整幅子图的数据丢掉。
    boxes = sp.order_row_major(sp.merge_panel_boxes(sp.find_panel_boxes(gray),
                                                    sp.find_spine_boxes(gray)))
    expected = None
    try:
        expected = int(expected_panels) if expected_panels is not None else None
    except (TypeError, ValueError):
        expected = None
    if expected is not None and expected < 1:
        expected = None
    meta = split_meta if isinstance(split_meta, dict) else {}
    meta.update({"expected": expected, "pixel_detected": len(boxes)})

    # The Agent supplies the semantic decomposition first; code validates the count,
    # rejects overlaps, and snaps its approximate boxes to real pixel lines. This is
    # an executable plan, not advisory metadata.
    hinted = sp.model_plot_boxes(panel_geometry or {}, gray.shape, expected=expected)
    if hinted:
        refined = [sp.refine_plot_box(gray, b) for b in hinted]
        meta.update({"source": "agent-plan+pixel-refine", "used": len(refined),
                     "plot_boxes": [list(b) for b in refined],
                     "panel_labels": [str(p.get("label") or "")
                                      for p in (panel_geometry.get("panels") or [])]})
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        panels = []
        for i, (x0, y0, x1, y1) in enumerate(refined):
            bw, bh = x1 - x0, y1 - y0
            # Keep tick labels and axis titles in the crop while ensuring the
            # numerical frame remains exactly the Agent+pixel plotting rectangle.
            cx0, cy0, cx1, cy1 = sp.panel_crop_bounds(
                (x0, y0, x1, y1), refined, gray.shape, margin=margin)
            crop = img[cy0:cy1, cx0:cx1]
            name = f"{Path(figure_path).stem}_p{i + 1:02d}.png"
            path = outdir / name
            iio.imwrite(path, crop)
            panels.append((path, (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0),
                           (cx0, cy0, cx1 - cx0, cy1 - cy0)))
        return panels

    # If their counts disagree, never silently flatten a known multi-panel figure.
    if expected and expected > 1 and len(boxes) != expected:
        meta.update({"source": "count-mismatch", "used": 0})
        return []

    if not boxes:
        # 一个绘图框都没检出：整图当一个面板处理。丢弃才是更糟的选择——
        # 位图路线后面还有"非数据图"门限会把它拦下，静默丢图则连机会都没有。
        try:
            frame = el.find_frame(gray)
        except Exception:  # noqa: BLE001
            frame = (0, 0, w, h)
        log(f"      （未检出子图框，按单面板处理 frame={tuple(frame)}）", detail=True)
        meta.update({"source": "single-fallback", "used": 1})
        return [(Path(figure_path), tuple(frame), (0, 0, w, h))]

    if len(boxes) == 1:
        x, y, bw, bh = boxes[0]
        meta.update({"source": "pixel", "used": 1})
        return [(Path(figure_path), (x, y, x + bw, y + bh), (x, y, bw, bh))]

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    panels = []
    for i, (x, y, bw, bh) in enumerate(boxes):
        frames = [(a, b, a + c, b + d) for a, b, c, d in boxes]
        x0, y0, x1, y1 = sp.panel_crop_bounds((x, y, x + bw, y + bh),
                                            frames, gray.shape, margin=margin)
        crop = img[y0:y1, x0:x1]
        name = f"{Path(figure_path).stem}_p{i + 1:02d}.png"
        path = outdir / name
        iio.imwrite(path, crop)
        panels.append((path, (x - x0, y - y0, x + bw - x0, y + bh - y0),
                       (x0, y0, x1 - x0, y1 - y0)))
    meta.update({"source": "pixel", "used": len(panels)})
    return panels


def curve_content(img, frame, legend_colors, max_dist=60):
    """How curve-like the pixels matching the legend colours are.

    A line chart draws continuous strokes: most columns of the plot area contain the
    colour and each such column holds a few pixels (stroke thickness). Photo/schematic
    "matches" are scattered single-pixel edges, and figures whose series do not span the
    axis (e.g. a handful of markers) score low on coverage.

    Returns (coverage, median_thickness) - both 0 when nothing matches.
    """
    left, top, right, bottom = frame
    sub = img[top + 2:bottom - 1, left + 2:right - 1].astype(float)
    if sub.size == 0:
        return 0.0, 0.0
    best = (0.0, 0.0)
    for hexs in legend_colors:
        target = np.array(el.hex_to_bgr(hexs), dtype=float)
        mask = np.sqrt(((sub - target) ** 2).sum(axis=2)) <= max_dist
        counts = mask.sum(axis=0)
        nz = counts[counts > 0]
        if not len(nz):
            continue
        coverage = len(nz) / mask.shape[1]
        thickness = float(np.median(nz))
        if (coverage, thickness) > best:
            best = (coverage, thickness)
    return best


def _is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _overlap_frac(a, b):
    """Overlap area as a fraction of the smaller box (are these two boxes the same figure?)."""
    ox = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ox * oy / small if small > 0 else 0.0


def _vector_thumb(page, bbox, outdir, name, needed):
    """Rendered crop for VLM selection/series understanding of a vector chart.

    Vector extraction used to bypass the model entirely.  That made same-colour
    multi-instance charts impossible to name/count semantically, even though their
    geometry was exact.  A small crop lets the model provide the inventory and
    anchors while the PDF paths remain the numerical source of truth.
    """
    if not needed:
        return None
    path = Path(outdir) / "select" / "thumbs" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        page.crop(tuple(bbox), strict=False).to_image(resolution=180).original.save(path)
    except Exception:  # noqa: BLE001
        return None
    return path


def _vector_inner_text(page, axes):
    """Free text from inside a vector chart: axis titles + legend/sub-plot labels."""
    if not axes:
        return ""
    try:
        t = vx.find_figure_text(page, axes)
    except Exception:  # noqa: BLE001
        return ""
    parts = [f"x轴:{t['x_title']}" if t["x_title"] else "",
             f"y轴:{t['y_title']}" if t["y_title"] else "",
             t["inside"][:80]]
    return " ｜ ".join(p for p in parts if p)


def _dark_line_wanted(panel):
    """黑线提取默认关闭，只有面板里显式写了 allow_dark=true 才做。

    实测代价：黑线通道在这些论文里会把 "EOI"、"11 MPa"、"S ∝ t^0.5" 这些黑字串成
    一条曲线输出（2014-01-9079 / 2014-01-1413 几乎每个面板都多出 dark/dark_2 两条
    假曲线），比"黑线提不出来"更糟。所以改成显式开启：分析阶段若发现深色数据线会在
    日志里提示，用户要把该面板的 allow_dark 改成 true 才会提取。
    """
    if not panel.get("allow_dark"):
        return False
    info = panel.get("vlm_dark_line") or {}
    if not info.get("has_dark_line"):
        return False
    kind = str(info.get("kind") or "").strip().lower()
    return kind in ("", "data", "model", "data_line", "model_line", "experiment")


def _extract_panel(panel, panel_path, rng, csv_dir, vlm=None, pdf_page=None):
    """两种方式：模型给锚点 + 代码精确追踪（优先），或按颜色全局追踪（回退）。

    锚点路线的分工：模型说"有哪几条曲线、大概在哪"，代码在锚点附近取色、从锚点向
    两端连续性追踪、换算成数值。模型不再评价候选清单，所以不会出现"数据被跳过"；
    追不到的会作为 failed 明确报告，而不是静默消失。
    """
    spec = panel.get("series_spec") or []
    if spec and pdf_page is not None and panel.get("vector_panel_axes"):
        try:
            result = _extract_vector_panel(panel, panel_path, rng, csv_dir, spec, pdf_page, vlm)
            if result is not None:
                return result
        except Exception as exc:  # noqa: BLE001
            log(f"      （逐子图矢量读取不可用，继续像素提取：{type(exc).__name__}: {exc}）",
                detail=True)
    if spec:
        try:
            res = _extract_seeded(panel, panel_path, rng, csv_dir, spec, vlm=vlm)
            if res is not None:
                return res
        except Exception as exc:  # noqa: BLE001
            log(f"      （锚点路线失败，回退颜色追踪：{type(exc).__name__}: {exc}）",
                detail=True)
    try:
        return be.process_panel(
            panel_path, rng, csv_dir, sat_min=70, val_min=40, hue_tol=16,
            legend_colors=panel["legend_colors"] or None,
            name_map=panel.get("series_names") or None,
            series_y=series_y_from(panel),
            roles=panel.get("series_roles") or None,
            allow_dark=_dark_line_wanted(panel),
            frame_hint=panel.get("frame"))
    except Exception as exc:  # noqa: BLE001
        log(f"{panel['id']}: ✗ 提取失败 {type(exc).__name__}: {exc}")
        return None


def _bind_legend_styles(spec, contracts, panel_index):
    """Bind exact label matches to a separately read physical legend swatch.

    A shared legend cannot rename a series in another panel. Labels and colour
    identity must both agree before its drawing contract can override a guess.
    """
    import series_seed as ss
    def label_key(value):
        value = str(value or '').lower().replace('−','-').replace('–','-')
        return re.sub(r'[^\w+\-.]+','',value,flags=re.UNICODE)
    for s in spec:
        matches = []
        for contract in contracts:
            if contract.get('panel_index')!=panel_index and contract.get('shared') is not True:
                continue
            for item in contract.get('entries') or []:
                if not isinstance(item,dict) or not label_key(item.get('label')) \
                        or label_key(item.get('label'))!=label_key(s.get('label')):
                    continue
                try:
                    a,b = el.hex_to_bgr(item['color']),el.hex_to_bgr(s['color'])
                    ah,bh = ss._hsv_of(a),ss._hsv_of(b)
                    agree = (ah[1]<60 and bh[1]<60) or (ah[1]>=60 and bh[1]>=60
                              and ss._hue_dist(ah[0],bh[0])<=15)
                except (KeyError,ValueError,TypeError):
                    agree = False
                if agree and item.get('draw') in ('line_only','markers_only','markers_connected'):
                    matches.append(item)
        if not matches or len({q['draw'] for q in matches})>1:
            continue
        item = matches[0]
        s['draw'] = item['draw']
        s['output'] = 'line' if s['draw']=='line_only' else 'points'
        for key in ('marker_shape','marker_fill'):
            s[key] = item.get(key) if s['draw']!='line_only' else None
        s['legend_style_evidence'] = '放大图例样本，标签与颜色共同核对'
    return spec


def _read_figure_legends(vlm, panels, figure_path, outdir):
    if vlm is None:
        return []
    contracts = []
    with tempfile.TemporaryDirectory(prefix='dig_legend_') as tmp:
        dest = Path(outdir)/'objects' if _KEEP_INTERMEDIATES else Path(tmp)
        dest.mkdir(parents=True,exist_ok=True)
        for index,(path,frame,_crop) in enumerate(panels,start=1):
            img = iio.imread(path)
            for k,(x,y,w,h) in enumerate(be.legend_boxes(img,frame),start=1):
                png = dest/f'{Path(figure_path).stem}_legend_{index}_{k}.png'
                iio.imwrite(png,img[max(0,y-4):y+h+4,max(0,x-4):x+w+4])
                try:
                    contract = vt.read_legend_contract(vlm,png,figure_path,index)
                    if contract.get('entries'):
                        contract['panel_index']=index
                        contracts.append(contract)
                except Exception as exc:  # noqa: BLE001
                    log(f'      图例放大读数失败：{type(exc).__name__}: {exc}',detail=True)
    return contracts


def _validated_axis_fix(panel, fix):
    """A visual critic cannot override a coordinate fit backed by tick geometry."""
    fix = dict(fix)
    rejected = []
    reading = panel.get('vlm_axis') or {}
    for key in ('x','y'):
        value = fix.get(key)
        if not isinstance(value,(list,tuple)) or len(value)!=2:
            continue
        source = reading.get('x') if key=='x' else next(
            (a for a in reading.get('y_axes') or [] if a.get('side','left')=='left'),{})
        evidence = (source or {}).get('tick_calibration') or {}
        actual = evidence.get('frame_range')
        if evidence.get('accepted') and actual:
            try:
                error = max(abs(float(a)-float(b)) for a,b in zip(value,actual))
            except (TypeError,ValueError):
                error = float('inf')
            if error>.03*abs(actual[1]-actual[0]):
                rejected.append(key)
                fix.pop(key,None)
    return fix,rejected


def _gap_crop(img, trace, x0, x1, others, color, out_path):
    """裁出缺口那一小块，并把"已识别部分"画上去，给模型看走势。"""
    h, w = img.shape[:2]
    pad_x = max(12, int(0.03 * w))
    cx0, cx1 = max(0, int(x0 - pad_x)), min(w, int(x1 + pad_x))
    ys = [y for x, y in trace if cx0 <= x <= cx1]
    for _n, t in others:
        ys += [y for x, y in t if cx0 <= x <= cx1]
    if not ys:
        ys = [y for _x, y in trace]
    pad_y = max(14, int(0.25 * (max(ys) - min(ys) + 1)))
    cy0, cy1 = max(0, int(min(ys) - pad_y)), min(h, int(max(ys) + pad_y))
    if cx1 - cx0 < 12 or cy1 - cy0 < 12:
        return None
    crop = img[cy0:cy1, cx0:cx1].copy()
    for _n, t in others:                       # 别的曲线：灰
        pts = [(int(x - cx0), int(y - cy0)) for x, y in t if cx0 <= x < cx1]
        if len(pts) > 1:
            cv2.polylines(crop, [np.array(pts, np.int32)], False, (150, 150, 150), 1)
    pts = [(int(x - cx0), int(y - cy0)) for x, y in trace if cx0 <= x < cx1]
    if len(pts) > 1:                           # 已识别的部分：本色
        cv2.polylines(crop, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)
    iio.imwrite(out_path, crop)
    return (cx0, cy0, cx1, cy1)


def _repair_gaps(vlm, panel, panel_path, frame, picks, names, outdir, log, max_calls=3,
                 exclude=()):
    """重合处只显示一种颜色 -> 追断的那几段：让模型给锚点，代码插回去。

    只在"这条曲线明显比别人短 / 中间有长断口"时才问，每次只裁那一小块。模型读图、说锚点；
    锚点只往已有缺口里插，已经追到的部分不动（想编也编不进去）。
    """
    import series_seed as ss
    usable = [p for p in picks if p["trace"] and p["kind"] != "loop"
              and _spec_want(p["s"]) != "points"]
    if vlm is None or len(usable) < 2:
        return 0
    img = iio.imread(panel_path)
    if img is None:
        return 0
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    def evidence_mask(pick):
        if pick.get("trace_mask") is not None:
            return pick["trace_mask"]
        mask = (el.dark_mask(gray, frame, exclude_boxes=exclude, source_img=img) if pick["dark"]
                else el.color_mask(hsv, frame, pick["color"], pick["hue_tol"],
                                   exclude_boxes=exclude))
        return ss.mask_near_anchors(mask, pick["anchors"], frame) if pick.get("guided") else mask
    frame_left, frame_top, frame_right, frame_bottom = (float(v) for v in frame)
    fw = max(1.0, frame_right - frame_left)
    allx = [x for p in usable for x, _y in p["trace"]]
    lo, hi = min(allx), max(allx)
    if hi - lo < 0.25 * fw:
        return 0
    calls, added = 0, 0
    gap_tmp = None
    if _KEEP_INTERMEDIATES:
        shot_dir = Path(outdir) / "gaps"
        shot_dir.mkdir(parents=True, exist_ok=True)
    else:
        gap_tmp = tempfile.TemporaryDirectory(prefix="dig_gaps_")
        shot_dir = Path(gap_tmp.name)
    # Shorter traces are more likely to be the curve hidden underneath another one;
    # give the limited repair budget to them first.
    for p in sorted(usable, key=lambda q: len(q["trace"])):
        if calls >= max_calls:
            break
        xs = [x for x, _y in p["trace"]]
        windows = []
        expected = p["s"].get("expected_x_extent") or []
        expected_valid = False
        try:
            expected_valid = (len(expected) == 2
                              and 0.0 <= float(expected[0]) < float(expected[1]) <= 1.0)
        except (TypeError, ValueError):
            expected_valid = False
        # Different physical series often have genuinely different endpoints.  The
        # union-of-all-traces heuristic is only a fallback when the semantic planner
        # did not provide an expected extent; otherwise it would extend every shorter
        # curve to the longest curve in the chart.
        if not expected_valid:
            if min(xs) - lo > 0.05 * fw:
                windows.append((lo, min(xs)))
            if hi - max(xs) > 0.05 * fw:
                windows.append((max(xs), hi))
        for gap in ((p["completion"] or {}).get("open_gaps") or []):
            if gap["cols"] > 0.04 * fw:
                windows.append((gap["x0"], gap["x1"]))
        # The planner sees the whole curve, including the portion whose own colour is
        # hidden.  Its expected extent/ranges are a much stronger trigger than merely
        # comparing this trace with the union of whatever the pixel code happened to find.
        try:
            if expected_valid:
                ex0 = frame_left + float(expected[0]) * fw
                ex1 = frame_left + float(expected[1]) * fw
                if min(xs) - ex0 > 0.035 * fw:
                    windows.append((ex0, min(xs)))
                if ex1 - max(xs) > 0.035 * fw:
                    windows.append((max(xs), ex1))
        except (TypeError, ValueError):
            pass
        for rg in (p["s"].get("occluded_ranges") or []):
            try:
                rx0 = frame_left + float(rg[0]) * fw
                rx1 = frame_left + float(rg[1]) * fw
            except (TypeError, ValueError, IndexError):
                continue
            if rx1 > rx0:
                windows.append((rx0, rx1))
        for a in (p["s"].get("occluded") or []):
            try:
                cx = frame_left + float(a[0]) * fw
            except (TypeError, ValueError, IndexError):
                continue
            if not any(abs(x - cx) <= 0.025 * fw for x in xs):
                windows.append((cx - 0.06 * fw, cx + 0.06 * fw))

        # Merge overlapping planner/geometric windows so one physical gap costs one call.
        merged_windows = []
        for a, b in sorted((max(frame_left, min(a, b)), min(frame_right, max(a, b)))
                           for a, b in windows):
            if merged_windows and a <= merged_windows[-1][1] + 0.015 * fw:
                merged_windows[-1] = (merged_windows[-1][0], max(b, merged_windows[-1][1]))
            else:
                merged_windows.append((a, b))
        for x0, x1 in merged_windows[:3]:
            if calls >= max_calls or x1 - x0 < 0.04 * fw:
                continue
            # Planner occlusion is semantic context, not proof that pixels are missing.
            # Ask the local Agent only where the measured trace has a real long hole or
            # a genuinely absent endpoint; otherwise a correct continuous start can be
            # "repaired" into an axis-border spike.
            missing_spans = _missing_trace_spans(
                p["trace"], x0, x1, min_width=max(6.0, 0.018 * fw))
            if not missing_spans:
                continue
            label = str(p["s"].get("label") or "曲线")
            color = (p["color"] or (0, 0, 0)) if not p["dark"] else (60, 60, 60)
            others = [(names[j], q["trace"]) for j, q in enumerate(picks) if q is not p
                      and q["trace"] and q["kind"] != "loop"
                      and not _spec_has_markers(q["s"])
                      and names[j] in (p["s"].get("overlaps") or [])]
            safe = ss.safe_tag(label)
            crop_path = shot_dir / f"{panel['id']}_{safe}_{int(x0)}_{int(x1)}.png"
            box = _gap_crop(img, p["trace"], x0, x1, others, color, crop_path)
            if box is None:
                continue
            calls += 1
            try:
                data = vt.gap_anchors(
                    vlm, crop_path, label,
                    "#%02x%02x%02x" % (color[2], color[1], color[0]),
                    info=(f"缺口在整张图的 x={int(x0)}~{int(x1)} 像素（图宽 {w}）"))
            except Exception as exc:  # noqa: BLE001
                log(f"      （补缺口调用失败：{type(exc).__name__}: {exc}）")
                continue
            cx0, cy0, cx1, cy1 = box
            got = []
            for a in (data.get("anchors") or []):
                try:
                    ax, ay = float(a[0]), float(a[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if not (0.0 <= ax <= 1.0 and 0.0 <= ay <= 1.0):
                    continue
                px = cx0 + ax * (cx1 - cx0)
                py = cy0 + ay * (cy1 - cy0)
                if (x0 - 4 <= px <= x1 + 4
                        and any(a - 2 <= px <= b + 2 for a, b in missing_spans)):
                    got.append((px, py))
            got = _model_anchors_in_frame(got, frame, inset=2.0)
            # Reject gross hallucinations while keeping the gate deliberately broad:
            # an anchor should agree with endpoint continuation or lie near an actual
            # occluding trace.  The VLM chooses the shape; this only catches points in
            # labels/legend/background that cannot connect to either side.
            known = sorted((float(x), float(y)) for x, y in p["trace"])
            other_dense = [ss.dense_trace(t) for _n, t in others]
            plausible = []
            for px, py in sorted(got):
                prev_pt = next((q for q in reversed(known) if q[0] <= px), None)
                next_pt = next((q for q in known if q[0] >= px), None)
                ok = False
                if prev_pt and next_pt and next_pt[0] > prev_pt[0]:
                    pred = (prev_pt[1] + (next_pt[1] - prev_pt[1])
                            * (px - prev_pt[0]) / (next_pt[0] - prev_pt[0]))
                    ok = abs(py - pred) <= max(18.0, 0.18 * h)
                if not ok:
                    near = []
                    for d in other_dense:
                        cy = d.get(int(round(px)))
                        if cy is not None:
                            near.append(abs(cy - py))
                    ok = bool(near) and min(near) <= max(14.0, 0.08 * h)
                if not ok and (prev_pt or next_pt):
                    endpoint = prev_pt or next_pt
                    ok = abs(py - endpoint[1]) <= max(24.0, 0.16 * (frame_bottom - frame_top))
                if ok:
                    plausible.append((px, py))
            got = plausible
            if not got:
                continue
            observed = [(names[j], ss.measured_trace(q["trace"], evidence_mask(q)))
                        for j, q in enumerate(picks) if q is not p and q["trace"]]
            merged, repair_info = ss.retrace_from_anchors(
                p["trace"], got, evidence_mask(p), frame, observed)
            n_add = len(merged) - len(p["trace"])
            if n_add > 0:
                added += n_add
                p["completion"] = dict(p["completion"] or {})
                p["completion"]["model_anchors"] = {
                    "cols": n_add, "x0": int(x0), "x1": int(x1),
                    "by": f"模型指路后从像素重追（{len(got)} 个锚点，x={int(x0)}~{int(x1)}）"}
                p["completion"].setdefault("spans", []).append(
                    {"x0": int(x0), "x1": int(x1), "cols": n_add,
                     "by": "模型读图补缺口"})
                p["trace"] = merged
                log(f"      模型补缺口：{label} x={int(x0)}~{int(x1)}，"
                    f"用 {len(got)} 个锚点重新搜像素（+{n_add} 列）", detail=True)
    if gap_tmp is not None:
        gap_tmp.cleanup()
    return added


def _spec_dark(s, color):
    """模型说这条线是黑色/深灰（或它给的颜色本身就是深灰）。

    黑白论文图（实测 2006-01-1387 整本都是）里模型会说 "color=#000000"，旧实现
    直接判"颜色定不下来"把整本图的曲线全丢了——其实是代码不认黑白，不是图上没有
    数据。模型给了 dark=true，代码就改用"深色笔画 + 锚点"这条路线。
    """
    flag = s.get("dark")
    if isinstance(flag, str):
        flag = flag.strip().lower() in ("true", "1", "yes", "y")
    if flag:
        return True
    # 模型给的颜色本身是黑/深灰（黑白论文图整本都这样）。注意 resolve_color 会拒绝
    # 灰白色，所以这里必须看**模型给的原始十六进制**，不能只看解析结果。
    for candidate in (color, s.get("line_color"), s.get("color")):
        if candidate is None:
            continue
        try:
            bgr = (el.hex_to_bgr(str(candidate)) if isinstance(candidate, str)
                   else candidate)
            _h, sat, val = el.bgr_to_hsv(bgr)
        except Exception:  # noqa: BLE001
            continue
        if val < 110 and sat < 70:
            return True
    return False


def _spec_want(s):
    """draw -> 要出什么。一个图例条目 = 一条曲线：

    markers_connected（散点用折线连起来）只出**点**——线就是这些点的连线，
    不再另出一个 `_line.csv`，否则用户看到的是"一条曲线变成了两条"。
    markers_plus_line（散点 + 独立的拟合/模型线）才是真的两条。
    """
    draw = str(s.get("draw") or "").strip().lower()
    if draw in ("markers_only", "markers_connected"):
        return "points"
    if draw == "line_only":
        return "line"
    if draw == "markers_plus_line":
        return "both"
    return str(s.get("output") or "line").strip().lower()


def _spec_has_markers(s):
    if s.get("has_markers") is True:
        return True
    if str(s.get("draw") or "").lower().startswith("markers"):
        return True
    ls = str(s.get("linestyle") or "").lower()
    return ls in ("marker", "markers", "scatter")


def _markers_cover_curve(markers, trace, min_n=5, cover=0.6):
    """这条曲线是不是"就是这些标记点连起来的"——标记够多、且铺满曲线的横向范围。

    实测（2023-01-1635 图 4 那种散点图）：每个圆点宽十几像素，逐列追踪会在它上面走出
    一条水平段，画出来就是明显的**阶梯**。标记中心才是真正的数据点，一个点一个值。
    """
    if not markers or len(markers) < min_n or len(trace) < 20:
        return False
    mx = [float(m[0]) for m in markers]
    tx = [float(q[0]) for q in trace]
    span_t, span_m = max(tx) - min(tx), max(mx) - min(mx)
    return span_t > 0 and span_m >= cover * span_t


_NON_DATA_ROLES = {"control", "reference", "annotation", "legend", "inset", "other"}


def _planner_series(found):
    """Normalise the VLM's semantic plan into extraction-ready series specs.

    The planner may describe a black connector with coloured identity markers.  Code
    traces ``line_color`` while retaining ``marker_color`` for identity/visualisation.
    Non-data roles stay auditable in ``ignore`` but never reach the pixel extractor.
    """
    series, ignored = [], []
    for raw in found.get("series") or []:
        if not isinstance(raw, dict):
            continue
        s = dict(raw)
        role = str(s.get("role") or "data").strip().lower()
        if role in _NON_DATA_ROLES:
            ignored.append({"what": s.get("label") or role,
                            "why": f"planner role={role}，不是数值数据系列"})
            continue
        draw = str(s.get("draw") or "").strip().lower()
        line_color = s.get("line_color")
        marker_color = s.get("marker_color")
        trace_color = line_color or (marker_color if draw == "markers_only" else None) \
            or s.get("color")
        s["role"] = role
        s["line_color"] = line_color
        s["marker_color"] = marker_color
        s["display_color"] = marker_color or trace_color
        s["color"] = trace_color
        series.append(s)
    ignored.extend(x for x in (found.get("ignore") or []) if isinstance(x, dict))
    return series, ignored


def _series_plan_valid(rows):
    """Reject mixed data/normalised coordinates instead of silently dropping anchors."""
    if not isinstance(rows, list) or not rows:
        return False
    for s in rows:
        if not isinstance(s, dict):
            return False
        for a in s.get("anchors") or []:
            try:
                x, y = float(a[0]), float(a[1])
            except (TypeError, ValueError, IndexError):
                return False
            if not (0 <= x <= 1 and 0 <= y <= 1):
                return False
    return True


def _planner_exclude_boxes(panel, frame, img=None):
    """Validated VLM-declared legend/inset boxes -> pixel exclusion rectangles.

    This is a coarse semantic hint, not pixel measurement. It complements local
    rectangle detection when a legend has no visible border. Validation keeps a bad
    model box from erasing most of the plotting area.
    """
    left, top, right, bottom = (float(v) for v in frame)
    fw, fh = max(1.0, right - left), max(1.0, bottom - top)
    out = []
    regions = ((panel.get("extraction_plan") or {}).get("exclude_regions") or [])
    for item in regions:
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").strip().lower() not in ("legend", "inset"):
            continue
        box = item.get("box") or []
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            continue
        if (x1 - x0) * (y1 - y0) > 0.35:
            continue
        px0, px1 = left + x0 * fw, left + x1 * fw
        py0, py1 = bottom - y1 * fh, bottom - y0 * fh
        pad = 3.0
        xx0, yy0 = max(left, px0 - pad), max(top, py0 - pad)
        xx1, yy1 = min(right, px1 + pad), min(bottom, py1 + pad)
        if xx1 - xx0 >= 12 and yy1 - yy0 >= 12:
            candidate = (int(xx0), int(yy0), int(xx1 - xx0), int(yy1 - yy0))
            if img is not None:
                if item.get('role')=='inset':
                    local = be.inner_boxes(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),frame,img=img)
                    nearby = [b for b in local if sp._inter(candidate,b)/max(1,min(sp._area(candidate),sp._area(b)))>=.4]
                    if nearby:
                        out.extend(nearby)
                    continue  # Coarse inset hints alone cannot erase data pixels.
                local = be.legend_boxes(img, frame)
                def overlap(a, b):
                    area = sp._inter(a, b)
                    return area / max(1, min(sp._area(a), sp._area(b)))
                if item.get("role") == "legend" and local:
                    # Boxed legends have exact pixel geometry. A model may confuse
                    # top-left image y with Cartesian plot y; neither hint may erase
                    # a real curve when it disagrees with the actual legend frame.
                    nearby = [b for b in local if overlap(candidate, b) >= .4]
                    if nearby:
                        out.extend(nearby)
                        continue
                xx, yy, ww, hh = candidate
                patch = img[yy:yy + hh, xx:xx + ww]
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                sat = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1]
                dark_text = ((gray < 150) & (sat < 60)).astype(np.uint8)
                count, _labels, stats, _cents = cv2.connectedComponentsWithStats(dark_text)
                letters = [s for s in stats[1:count] if 4 <= s[cv2.CC_STAT_HEIGHT] <= .15 * fh
                           and 2 <= s[cv2.CC_STAT_WIDTH] <= .15 * fw
                           and s[cv2.CC_STAT_AREA] >= 6]
                if len(letters) < 6:
                    # A coloured line/error band contains no legend text. Semantic
                    # rectangles alone are insufficient evidence to mask its pixels.
                    continue
            out.append(candidate)
    return out


REVIEW_COLORS = [(0, 0, 255), (0, 170, 0), (255, 0, 0), (0, 150, 255),
                 (255, 0, 190), (110, 60, 0), (0, 90, 255), (170, 0, 170)]


def revise_panel(vlm, panel_path, panel, spec, entries, failed, aliases, csv_dir=None):
    """复盘：把追出来的轨迹**编号画回原图**，让模型核对数量与归属。

    只做三件事的判定：哪两个编号是同一条（merge）、哪条追错了（drop）、漏了哪条
    （add）。模型像素上不准，但"这两条是不是同一条"这种判断是它的强项——这正是
    "算法在模型指导下收尾"。只在有问题的面板上调用，结果落盘可审计。
    """
    img = iio.imread(panel_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    rows = []
    for i, e in enumerate(entries, start=1):
        p, s = e["p"], e["p"]["s"]
        color = REVIEW_COLORS[(i - 1) % len(REVIEW_COLORS)]
        points_only = e.get("want") == "points"
        pixel_trace = e.get("point_trace",p["trace"]) if points_only else p["trace"]
        pts = [(int(round(x)), int(round(y))) for x, y in pixel_trace]
        if points_only:
            for point in pts:
                cv2.circle(img,point,4,color,2,cv2.LINE_AA)
        elif len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, color, 3, cv2.LINE_AA)
        if pts:
            mx, my = pts[len(pts) // 2]
            mx, my = min(max(mx + 4, 4), w - 60), min(max(my - 6, 16), h - 6)
            cv2.putText(img, f"#{i}", (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(img, f"#{i}", (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        color, 2, cv2.LINE_AA)
        data = e["point_data"] if points_only else (e["line_data"] or e["point_data"])
        if data:
            xs = [q[0] for q in data]
            ys = [q[1] for q in data]
            rows.append(f"#{i}: {s.get('label')} | {len(data)} 点 | "
                        f"x=[{min(xs):.3g},{max(xs):.3g}] y=[{min(ys):.3g},{max(ys):.3g}] | "
                        f"颜色={'黑/灰' if p['dark'] else s.get('color')} | "
                        f"线型={s.get('linestyle')}")
        else:
            rows.append(f"#{i}: {s.get('label')} | （没有可用点）")
    for f in failed or []:
        a = next((q for q in (f.get("anchors") or []) if len(q) == 2), None)
        if not a:
            continue
        ax, ay = int(round(float(a[0]) * w)), int(round(float(a[1]) * h))
        if not (0 <= ax < w and 0 <= ay < h):
            continue
        cv2.drawMarker(img, (ax, ay), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 3)
        cv2.putText(img, f"✗{str(f.get('label'))[:12]}", (ax + 10, ay + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    spec_rows = []
    for s in spec:
        extent = s.get("expected_x_extent")
        spec_rows.append(
            f"{s.get('label')}(draw={s.get('draw') or s.get('output')}"
            f"{', 黑线' if s.get('dark') else ''}{', 闭合' if s.get('closed') else ''}"
            f"{', 期望x范围=' + str(extent) if extent else ''})")
    spec_text = f"期望物理曲线总数={len(spec)}；" + ("；".join(spec_rows) or "（无）")
    base = Path(csv_dir).parent if csv_dir else panel_path.parent.parent
    review_tmp = None
    if _KEEP_INTERMEDIATES:
        outdir = base / "review"
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        review_tmp = tempfile.TemporaryDirectory(prefix="dig_review_")
        outdir = Path(review_tmp.name)
    png = outdir / f"{panel['id']}_review.png"
    iio.imwrite(png, img)
    ax = panel.get("axis") or {}
    axis_text = (f"x={ax.get('x')} y={ax.get('y')}"
                 + (f" 数量级乘数={ax.get('multiplier')}" if ax.get("multiplier") else "")
                 + (f"（来源 {panel.get('axis_src_x')}/{panel.get('axis_src_y')}）"
                    if panel.get("axis_src_x") else ""))
    try:
        data = vt.revise_traces(vlm, png, spec_text, "\n".join(rows), axis_text=axis_text,
                                original_path=panel_path)
    except Exception as exc:  # noqa: BLE001
        if review_tmp is not None:
            review_tmp.cleanup()
        log(f"      （复盘调用失败，按原样输出：{type(exc).__name__}: {exc}）")
        return None
    if _KEEP_INTERMEDIATES:
        (outdir / f"{panel['id']}_review.json").write_text(
            json.dumps({"spec": spec_text, "found": rows, "answer": data},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"      复盘：merge={data.get('merge') or []} drop={data.get('drop') or []} "
        f"add={len(data.get('add') or [])} extend={len(data.get('extend') or [])} "
        f"axis_fix={'有' if data.get('axis_fix') else '无'}"
        f" —— {str(data.get('note'))[:60]}", detail=True)
    if review_tmp is not None:
        review_tmp.cleanup()
    return data


def _apply_review(review, entries, failed, aliases):
    """按复盘结论合并/丢弃（add 由调用方重追）。编号都是 1 起。"""
    n = len(entries)
    drop = {int(i) for i in (review.get("drop") or [])
            if str(i).strip().lstrip("-").isdigit() and 1 <= int(i) <= n}
    merged_into = {}
    for pair in (review.get("merge") or []):
        try:
            a, b = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 1 <= a <= n and 1 <= b <= n and a != b:
            sa = entries[a - 1]["p"]["s"]
            sb = entries[b - 1]["p"]["s"]
            role_a = str(sa.get("role") or "data").strip().lower()
            role_b = str(sb.get("role") or "data").strip().lower()
            draw_a = str(sa.get("draw") or "").strip().lower()
            draw_b = str(sb.get("draw") or "").strip().lower()
            label_a = str(sa.get("label") or "").strip().lower()
            label_b = str(sb.get("label") or "").strip().lower()
            # The full-figure planner sees captions, panel identity and legends. The
            # local critic sees only an overlay and frequently calls near-coincident
            # experimental markers + simulation line "duplicates". It may merge two
            # geometric traces only when their semantic contracts are compatible.
            semantically_distinct = (
                {role_a, role_b} == {"data", "model"}
                or {draw_a, draw_b} == {"markers_only", "line_only"}
                or (label_a and label_b and label_a != label_b
                    and ("exp" in label_a or "exp" in label_b)
                    and ("sim" in label_a or "sim" in label_b)))
            if semantically_distinct:
                aliases.append({
                    "label": sa.get("label") or f"series{a}",
                    "alias_of": None, "overlap": None,
                    "by": "语义计划保护",
                    "why": (f"忽略局部复盘 merge={pair}：{role_a}/{draw_a} 与 "
                            f"{role_b}/{draw_b} 是计划中不同物理系列"),
                })
                continue
            merged_into[max(a, b)] = min(a, b)
    out = []
    for idx, e in enumerate(entries, start=1):
        if idx in drop or idx in merged_into:
            keep = merged_into.get(idx)
            aliases.append({
                "label": e["p"]["s"].get("label") or f"series{idx}",
                "alias_of": (entries[keep - 1]["p"]["s"].get("label") if keep else None),
                "overlap": 1.0, "by": "模型复盘",
                "why": "drop" if idx in drop else "同一条曲线被重复提取"})
            continue
        out.append(e)
    return out, failed, aliases


def _extract_vector_panel(panel, panel_path, rng, csv_dir, spec, page, vlm=None):
    """Read physical PDF paths in the independently verified panel coordinate frame.

    Rendering for semantic recognition does not discard the original geometry.
    Every planned series must have a matching physical instance, otherwise this
    route returns None and the raster extractor reports its own partial results.
    """
    import series_seed as ss
    if len(panel.get('y_axes') or {})>1:
        return None  # Per-series dual-axis calibration remains in the raster route.
    frame = list(panel['frame'])
    axes = vx.snap_axes(page,panel['vector_panel_axes'])
    left,top,right,bottom = axes
    if right<=left or bottom<=top:
        return None
    img = iio.imread(panel_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    exclusions = be.inner_boxes(gray,frame,img=img)
    exclusions.extend(_planner_exclude_boxes(panel,frame,img=img))
    sx,sy = (right-left)/(frame[2]-frame[0]),(bottom-top)/(frame[3]-frame[1])
    pdf_exclusions = [[left+(x-frame[0])*sx,top+(y-frame[1])*sy,
                       left+(x+w-frame[0])*sx,top+(y+h-frame[1])*sy]
                      for x,y,w,h in exclusions]
    pdf_exclusions.extend(vx.inner_exclusions(page,axes))
    xmin,xmax,ymin,ymax = map(float,rng)
    xcal = {'slope':(xmax-xmin)/(right-left)}
    xcal['intercept'] = xmin-xcal['slope']*left
    ycal = {'slope':(ymin-ymax)/(bottom-top)}
    ycal['intercept'] = ymax-ycal['slope']*top
    rows = vx.extract_series(page,axes,xcal,ycal,series_spec=spec,exclude_boxes=pdf_exclusions)
    if {s['spec_index'] for s in rows} != set(range(len(spec))):
        log(f"      逐子图矢量：匹配 {len(rows)}/{len(spec)} 个系列，继续像素提取",detail=True)
        return None
    series,entries = [],[]
    for row in rows:
        s = spec[row['spec_index']]
        points_only = row['source'].startswith('markers:')
        data = row['data']
        trace = ss.data_anchors_to_pixels(data,frame,rng)
        if not trace:
            return None
        warning = []
        if (row.get('anchor_score') or 0)>.12:
            warning.append('PDF 实测实例与模型粗锚点偏离较大，需核对系列身份')
        extent = s.get('expected_x_extent')
        if isinstance(extent,(list,tuple)) and len(extent)==2:
            start,end = (row['x_range'][0]-xmin)/(xmax-xmin),(row['x_range'][1]-xmin)/(xmax-xmin)
            if start-float(extent[0])>.08 or float(extent[1])-end>.08:
                warning.append('PDF 实测横向范围未覆盖模型预期端点，未向空白区补点')
        color = row['color']
        name = ss.unique_path(csv_dir,f"{panel_path.stem}_{ss.safe_tag(s.get('label'))}.csv")
        ss.write_csv(name,data)
        series.append({'file':name.name,'source':f"vector-panel:{row['source']}",
                       'series_name':s.get('label'),'kind':'points' if points_only else 'line',
                       'draw':s.get('draw'),'color_hex':color,'points':len(data),
                       'anchor_score':row.get('anchor_score'),'y_axis_range':[ymin,ymax],
                       'x_range':row['x_range'],'y_range':row['y_range'],
                       'measurement_status':'needs_review' if warning else 'measured',
                       'warnings':warning})
        entries.append({'p':{'s':s,'trace':trace,'dark':color.lower()=='#000000'},
                        'want':'points' if points_only else 'line','point_trace':trace,
                        'point_data':data if points_only else [],'line_data':[] if points_only else data})
    review = revise_panel(vlm,panel_path,panel,spec,entries,[],[],csv_dir) if vlm else None
    if review and (review.get('ok') is False or any(review.get(k) for k in
                                                 ('drop','merge','add','extend','axis_fix','issues'))):
        # Model criticism is quality evidence, not permission to manufacture or
        # delete exact PDF samples. Keep the physical readings auditable.
        for row in series:
            row['measurement_status'] = 'needs_review'
            row['warnings'].append('模型复盘仍有异议，需结合原图核对 PDF 实例/坐标映射')
    return {'panel':panel_path.name,'frame':frame,'pdf_axes':axes,'series':series,
            'failed':[],'aliases':[],'review':review,'method':'vector-panel','_bridged_px':[]}


def _extract_seeded(panel, panel_path, rng, csv_dir, spec, vlm=None):
    """"模型指路 + 代码追踪 + 模型复盘"；返回 None 表示这条路线不可用（回退）。

    分工（这一版把模型的理解用足，代码只做像素）：
      * 模型说：有哪几条曲线、一条还是两条（draw）、是不是闭合回线（closed）、
        是不是黑线（dark）、跟谁重合（overlaps）、哪几段看不见（occluded）+
        每条线的颜色与锚点；
      * 代码做：按模型给的颜色/深色掩膜做精确追踪、遮挡补全、坐标换算；
      * 追完之后再问一次模型（见 revise_panel）：把追出来的轨迹**编号画回原图**
        让它核对——哪两个编号是同一条、哪条追错了、漏了哪条。只在有问题的面板
        上触发，正常面板不多花 token。
    """
    import series_seed as ss
    img = iio.imread(panel_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame = be._frame_of(gray, panel.get("frame"))
    exclude = be.inner_boxes(gray, frame, img=img)
    exclude.extend(_planner_exclude_boxes(panel, frame, img=img))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    xmin, xmax, ymin, ymax = rng
    palette = ss.pixel_palette(hsv, frame, exclude)
    series, failed = [], []
    used_keys, accepted = {}, []
    width, height = img.shape[1], img.shape[0]

    def data_range_for(s):
        """Axis range for one semantic series, including dual-Y multipliers."""
        symin, symax = ymin, ymax
        side = str(s.get("y_axis") or "left").lower()
        detail = (panel.get("y_axes") or {}).get(side) or {}
        yr = detail.get("range")
        if isinstance(yr, (list, tuple)) and len(yr) == 2:
            mult = _as_float(detail.get("multiplier"), 1.0)
            symin, symax = float(yr[0]) * mult, float(yr[1]) * mult
        return [xmin, xmax, symin, symax]

    def trace_one(s, hue_tol):
        """一条 spec -> (pick, 失败说明)。颜色/黑线两条路线在这里合流。"""
        anchors = ss.anchors_to_pixels(s.get("anchors"), frame, (width, height))
        color, how = ss.resolve_color(img, anchors, s.get("color"), palette)
        dark = _spec_dark(s, color)
        claims = used_keys.setdefault((str(s.get("role")), str(s.get("draw"))), set())
        # closed 字段由模型给；老配置没有这个字段时退回几何判据（见 candidates_in_mask）
        allow_loop = bool(s["closed"]) if "closed" in s else True
        colour_key = str(s.get("line_color") or s.get("marker_color") or s.get("color") or "").lower()
        shared_colour = sum(str(q.get("line_color") or q.get("marker_color") or q.get("color") or "").lower()
                            == colour_key for q in spec) > 1
        try_guide = shared_colour and len(anchors) >= 3 and not allow_loop
        guided = False
        line_mask = None
        if str(s.get("draw")) == "line_only" and not allow_loop:
            line_mask = (el.dark_mask(gray, frame, exclude_boxes=exclude, source_img=img)
                         if dark else el.color_mask(hsv, frame, color, hue_tol, exclude_boxes=exclude)
                         if color is not None else None)
            competing = []
            for other in picks:
                if not other.get("pre_markers"):
                    continue
                same_ink = (dark and other["dark"]) or (
                    not dark and not other["dark"] and color is not None
                    and ss._hue_dist(ss._hsv_of(color)[0],ss._hsv_of(other["color"])[0]) <= hue_tol)
                if same_ink:
                    competing.extend(other["pre_markers"])
            if competing and line_mask is not None:
                line_mask = ss.without_marker_glyphs(line_mask, competing)
                shared_colour = True
                try_guide = len(anchors) >= 3
            elif not competing:
                line_mask = None  # Preserve colour-resolution fallback for ordinary lines.
        if line_mask is not None:
            trace, kind, info = ss.trace_in_mask(line_mask, frame, anchors,
                avoid=claims, allow_loop=False, reject_rules=dark)
            if dark:
                dmask, how = line_mask, "dark"
        elif dark:
            dmask = el.dark_mask(gray, frame, exclude_boxes=exclude, source_img=img)
            trace, kind, info = ss.trace_in_mask(dmask, frame, anchors,
                                                 avoid=claims, allow_loop=allow_loop,
                                                 reject_rules=True)
            info["dark_px"] = int((dmask > 0).sum())
            how = "dark"
        elif color is not None:
            trace, kind, info = ss.trace_series(hsv, frame, color, anchors, exclude,
                                            hue_tol=hue_tol, avoid=claims,
                                            allow_loop=allow_loop)
        else:
            return None, {"label": s.get("label"), "anchors": s.get("anchors"),
                          "reason": ("颜色定不下来（模型没给颜色、锚点也没落在线上、"
                                     "图例里也没有）")}
        if try_guide and trace:
            raw_mask = line_mask if line_mask is not None else (dmask if dark else el.color_mask(hsv, frame, color, hue_tol, exclude_boxes=exclude))
            guide_mask = ss.mask_near_anchors(raw_mask, anchors, frame)
            gt, gk, gi = ss.trace_in_mask(guide_mask, frame, anchors, avoid=claims,
                                        allow_loop=False, reject_rules=dark)
            raw_hits, raw_dist = ss._anchor_stats(trace, anchors)
            guide_hits, guide_dist = ss._anchor_stats(gt, anchors)
            raw_span = max(q[0] for q in trace)-min(q[0] for q in trace)
            guide_span = max(q[0] for q in gt)-min(q[0] for q in gt) if gt else 0
            if (gt and guide_span >= .85*raw_span and len(gt) >= .65*len(trace)
                    and guide_hits >= raw_hits and guide_dist <= (raw_dist or 0)+5):
                trace, kind, info, guided = gt, gk, gi, True
        if shared_colour and str(s.get("draw")) == "line_only" and trace:
            full_mask = line_mask if line_mask is not None else (dmask if dark else el.color_mask(hsv,frame,color,hue_tol,exclude_boxes=exclude))
            global_trace = ss.soft_guided_trace(full_mask,anchors,frame)
            hits, distance = ss._anchor_stats(trace,anchors)
            ghits, gdist = ss._anchor_stats(global_trace,anchors)
            old_span = max(q[0] for q in trace)-min(q[0] for q in trace)
            new_span = max(q[0] for q in global_trace)-min(q[0] for q in global_trace) if global_trace else 0
            if (global_trace and new_span >= .85*old_span and len(global_trace) >= .3*len(trace)
                    and (ghits > hits or (ghits >= hits and gdist < .85*(distance or 0)))):
                trace, kind, guided = global_trace, "line", False
                info.update({"hits":ghits,"med_dist":gdist,"tracking":"global-pixel-path"})
        info["hue_tol"] = hue_tol
        span = ((max(p[0] for p in trace) - min(p[0] for p in trace))
                / max(1.0, frame[2] - frame[0])) if trace else 0.0
        pick = {"s": s, "anchors": anchors, "color": color, "how": how,
                "hue_tol": hue_tol, "trace": trace, "kind": kind, "info": info,
                "span": span, "completion": {}, "dark": dark, "guided": guided}
        if line_mask is not None:
            pick["trace_mask"] = line_mask
        if _spec_has_markers(s):
            marker_color, _ = ss.resolve_color(img, anchors,
                s.get("marker_color") or s.get("color"), palette)
            mmask = (el.dark_mask(gray, frame, exclude_boxes=exclude, source_img=img,
                                  clean_ring=False) if dark
                     else el.color_mask(hsv, frame, marker_color or color,
                                        hue_tol, exclude_boxes=exclude))
            candidates = ss.marker_centers(mmask)
            fill = s.get("marker_fill")
            hollow = True if fill == "hollow" else False if fill == "filled" else None
            if hollow is False and not dark and hue_tol>=14:
                # A filled glyph can have a navy outline and a lighter blue core.
                # Snapping its colour to the dominant model stroke can remove the
                # entire core. Preserve compatible declared-colour pixels too,
                # only when no nearby competing series has tightened hue tolerance.
                declared = ss._hex_to_bgr(s.get('marker_color') or s.get('color'))
                if declared and ss._hsv_of(declared)[1]>60 and marker_color \
                        and ss._hue_dist(ss._hsv_of(declared)[0],ss._hsv_of(marker_color)[0])<=14:
                    mmask |= el.color_mask(hsv,frame,declared,14,exclude_boxes=exclude)
            if hollow is None and "空心" in str(s.get("note") or ""):
                hollow = True
            occluding_ink = None
            if hollow is True or (hollow is False and s.get('marker_shape')=='square'):
                occluding_ink = np.zeros_like(mmask)
                for other_hex in palette:
                    other_color = el.hex_to_bgr(other_hex)
                    if marker_color and ss._hue_dist(ss._hsv_of(other_color)[0],ss._hsv_of(marker_color)[0]) > 14:
                        occluding_ink |= el.color_mask(hsv,frame,other_color,12,exclude_boxes=exclude)
                if not np.any(occluding_ink):
                    occluding_ink = None
            shape_centres = ss.shape_marker_centers(mmask, s.get("marker_shape"), frame,
                                                    hollow=hollow, threshold=.80 if hollow is False else .70,
                                                    occlusion_mask=occluding_ink)
            if len(shape_centres) >= 3:
                # Glyph fitting avoids interpreting individual short dashes as data
                # markers. Filled/error-bar glyphs retain distance-peak detections.
                if hollow is True:
                    # Preserve closed-glyph measurements, supplement with partly
                    # occluded template fits. Filled/error-bar glyphs use peaks.
                    candidates = list(candidates)
                    for x, y in shape_centres:
                        if not any(np.hypot(x-a, y-b) < max(5., .015*(frame[3]-frame[1]))
                                   for a,b in candidates):
                            candidates.append((x,y))
                elif hollow is False:
                    # Partial overlap can spoil a silhouette while its distance
                    # maximum still measures a real glyph core. Retain both sources.
                    candidates = ss.filled_marker_cores(mmask)
                    for x,y in shape_centres:
                        if not any(np.hypot(x-a,y-b) < 4 for a,b in candidates):
                            candidates.append((x,y))
            marker_key = str(s.get("marker_color") or s.get("color") or "").lower()
            shared_colour = sum(str(q.get("line_color") or q.get("marker_color") or q.get("color") or "").lower()
                                == marker_key for q in spec) > 1
            # Different model hex strings can resolve to the same physical ink.
            # Apply identity guidance to every scatter series, not only exact hex
            # matches; constrain its declared horizontal extent before path search.
            extent = s.get("expected_x_extent")
            if isinstance(extent,(list,tuple)) and len(extent)==2:
                try:
                    lo,hi = sorted(float(v) for v in extent)
                    if 0 <= lo < hi <= 1:
                        pad=max(8.,.06*(frame[2]-frame[0]))
                        candidates=[(x,y) for x,y in candidates
                            if frame[0]+lo*(frame[2]-frame[0])-pad <= x
                            <= frame[0]+hi*(frame[2]-frame[0])+pad]
                except (TypeError,ValueError):
                    pass
            identity_candidates = ss.markers_near_anchors(candidates, anchors, frame,
                                                         tolerance=.10)
            centres = ss.best_marker_track(identity_candidates, frame,
                                           min_points=3, min_extent_frac=.12,
                                           anchors=anchors)
            if str(s.get("draw") or "").lower() == "markers_only":
                centres = ss.extend_vertical_markers(centres, candidates, frame)
            if centres:
                pick["pre_markers"] = centres
                if str(s.get("draw") or "").lower() in ("markers_only", "markers_connected"):
                    # Marker identity does not depend on a connected colour component.
                    # Keep centres as measurements and a dense guide only for tracking.
                    pick["trace"] = sorted(ss.dense_trace(centres).items())
                    pick["span"] = (centres[-1][0] - centres[0][0]) / max(1, frame[2] - frame[0])
        if trace and info.get("key") is not None:
            claims.add(info["key"])       # 同一语义类型里的同色实例不能被重复认领
        return pick, None

    # 先给每条曲线定颜色，再按"颜色之间离得多远"给每条独立的色相容差：橙和金
    # 只差 12 个色相时，统一的容差会让两条线互相串色（实测 E30 因此追成了 E00）。
    resolved = []
    for s in spec:
        anchors = ss.anchors_to_pixels(s.get("anchors"), frame, (width, height))
        color, how = ss.resolve_color(img, anchors, s.get("color"), palette)
        resolved.append((s, anchors, color, how))
    tols = ss.hue_tolerances([c for _s, _a, c, _h in resolved])
    # ---- 第一遍：每条曲线按自己的颜色精确追踪 ----
    picks = []
    for (s, anchors, color, how), hue_tol in sorted(zip(resolved, tols),
            key=lambda item: not _spec_has_markers(item[0][0])):
        # 太短的先留着：它可能正是"大半段被压在另一条线下面"的那条，补全后再定去留。
        pick, why = trace_one(s, hue_tol)
        if pick is None:
            failed.append(why)
            continue
        picks.append(pick)
    order = {id(s): i for i,s in enumerate(spec)}
    picks.sort(key=lambda p: order[id(p["s"])])

    # ---- 第二遍：遮挡补全（"重合部分颜色只显示一个"）----
    # 两条线画在一起时，压在下面的那条在重合段里一个像素都没有：不是追踪失败，是
    # 图上没有东西可追。这里拿"另一条线真的画在那里"当证据，把断掉的那几段沿它补
    # 回来；没有证据的长断口一律不补（曲线真到头了的情况必须保持原样）。
    names = {i: str(p["s"].get("label") or f"series{i + 1}")
             for i, p in enumerate(picks)}
    def mask_for(p):
        mask = p.get("trace_mask")
        if mask is None:
            mask = (el.dark_mask(gray, frame, exclude_boxes=exclude, source_img=img) if p["dark"]
                else el.color_mask(hsv, frame, p["color"], p["hue_tol"], exclude_boxes=exclude))
        return ss.mask_near_anchors(mask, p["anchors"], frame) if p.get("guided") else mask

    occl = {i: ss.measured_trace(p["trace"], mask_for(p)) for i, p in enumerate(picks)
            if p["kind"] != "loop" and p["trace"] and not _spec_has_markers(p["s"])}
    for idx, p in sorted(enumerate(picks), key=lambda t: -len(t[1]["trace"])):
        if p["kind"] == "loop" or _spec_want(p["s"]) == "points":
            # 闭合回线的补全要先把散成十几块的弧串起来（见 series_seed 说明），
            # 这一版还没做，先不在这里假装补上。
            continue
        # A nearby series is not evidence that this series was occluded. Require
        # explicit overlap identity; dense interpolation of scatter is never ink.
        declared = set(p["s"].get("overlaps") or [])
        others = [(names[j], d) for j, d in occl.items()
                  if j != idx and names[j] in declared]
        if not others:
            continue
        # 只补"本来就追到了大半条"的曲线。碎片（几像素的杂点、网格线的残段）不补：
        # 补全是"把被压住的一段接回来"，不是"凭一小段造出一条曲线"。
        if p["span"] < 0.08 or len(p["trace"]) < 30:
            continue
        mask = mask_for(p)
        filled, cinfo = ss.complete_trace(p["trace"], mask, frame, others)
        # 补出来的部分不能超过整条曲线的 70%：否则说明原始轨迹只是个碎片，
        # 那更可能是网格线/杂点（实测被补成一条横线报了出去）。
        if filled and len(filled) > len(p["trace"]) \
                and len(p["trace"]) >= 0.3 * len(filled):
            p["raw_trace"], p["trace"], p["completion"] = p["trace"], filled, cinfo
        elif cinfo:
            p["completion"] = cinfo

    # Marker centres are stronger evidence than an inferred hidden curve.  When a
    # colour belongs to exactly one semantic series, detect its glyphs across the
    # whole plotting frame (legend boxes already excluded) instead of restricting
    # detection to the possibly truncated trace corridor.  Same-colour multi-series
    # charts stay on the corridor/anchor path to preserve instance identity.
    color_counts = {}
    for s in spec:
        key = str(s.get("color") or "").strip().lower()
        if key:
            color_counts[key] = color_counts.get(key, 0) + 1
    for p in picks:
        if str(p["s"].get("draw") or "").lower() == "markers_only":
            continue  # Do not replace shape-tested markers by a second broad scan.
        skey = str(p["s"].get("color") or "").strip().lower()
        if p["dark"] or not _spec_has_markers(p["s"]) or color_counts.get(skey) != 1:
            continue
        mmask = el.color_mask(hsv, frame, p["color"], 12, exclude_boxes=exclude)
        centres = ss.best_marker_track(
            ss.marker_centers(mmask), frame, min_points=3, min_extent_frac=0.12)
        marker_trace = ss.line_through_markers(centres)
        if len(marker_trace) < 3:
            continue
        mspan = (marker_trace[-1][0] - marker_trace[0][0]) / max(1.0, frame[2] - frame[0])
        if mspan < 0.12:
            continue
        p["pre_markers"] = centres
        # Keep exact line pixels outside the first/last fully visible marker.  Endpoint
        # glyphs are often clipped by the axes border and disappear during erosion
        # (e.g. a data point exactly at the origin); the connected line still reaches
        # that boundary and is valid measured evidence.
        combined = {int(round(x)): (float(x), float(y)) for x, y in p["trace"]}
        combined.update({int(round(x)): (float(x), float(y)) for x, y in marker_trace})
        p["trace"] = [combined[x] for x in sorted(combined)]
        p["span"] = max(p.get("span") or 0.0, mspan)

    # 还有没补上的缺口（重合处只显示一种颜色、压在下面的追断了）-> 让模型读图给锚点
    if vlm is not None:
        _repair_gaps(vlm, panel, panel_path, frame, picks, names,
                     Path(csv_dir).parent, log, exclude=exclude)

    def build_entry(p):
        """一条 pick -> 待落盘的 entry（数据先在内存里，复盘后统一写文件）。"""
        s, trace, kind, info = p["s"], p["trace"], p["kind"], p["info"]
        span = ((max(q[0] for q in trace) - min(q[0] for q in trace))
                / max(1.0, frame[2] - frame[0])) if trace else 0.0
        if (len(trace) < 20 or span < 0.12) and len(p.get("pre_markers") or []) < 3:
            return None, {"label": s.get("label"), "anchors": s.get("anchors"),
                          "reason": (f"没追上这条线（颜色取自{p['how']}，图上候选 "
                                     f"{info.get('candidates', 0)} 条，最长只追到 "
                                     f"{len(trace)} 点；补全后仍不足）")}
        want = _spec_want(s)
        markers = []
        if _spec_has_markers(s):
            markers = list(p.get("pre_markers") or [])
            if not markers:
                mmask = (el.dark_mask(gray, frame, exclude_boxes=exclude, source_img=img) if p["dark"]
                         else el.color_mask(hsv, frame, p["color"], 12,
                                            exclude_boxes=exclude))
                markers = ss.markers_in_mask(mmask, trace)
        note_draw = None
        marker_line = False
        point_trace = list(markers)
        if markers:
            trace = ss.snap_line_to_markers(trace, markers)      # 标记处别走平台
        if _markers_cover_curve(markers, trace):
            # 散点图：标记中心就是数据。用"标记中心连线"替掉逐列追出来的线，
            # 阶梯消失，每个点对应一个值（线是点连出来的，不额外损失信息）
            centers = ss.line_through_markers(markers)
            if len(centers) >= 3:
                # 模型/遮挡补全产生的点必须保留。除了标记范围两端，还要保留补全信息
                # 明确指出的中间区间；否则 markers_connected 分支会再次把修好的缺口抹掉。
                trace = centers
                point_trace = list(markers)
                # 一个标记一个点：散点图上十几个点就是完整数据，不能再按"线要 20 点"砍
                marker_line = True
                note_draw = f"按标记中心出数据（{len(centers)} 个点），消除标记造成的阶梯"
        elif _spec_has_markers(s):
            # 标记检测不全（点太密/连成串）时退一步：把逐列追踪留下的"平台"压成一个点，
            # 平台正是每个标记的宽度造成的阶梯
            flat = ss.collapse_plateaus(trace)
            if len(flat) < len(trace):
                note_draw = f"标记处的平台已压缩（{len(trace)} -> {len(flat)} 点），消除阶梯"
                trace = flat
        if str(s.get("draw") or "").lower() == "markers_connected" and len(markers) >= 3:
            want = "points"
            trace = list(markers)
            point_trace = list(markers)
            note_draw = note_draw or "散点用折线连起来：点即曲线（不再另出线文件）"
        elif str(s.get("draw") or "").lower() == "markers_only" and len(markers) < 3:
            return None, {"label": s.get("label"), "reason": "独立实验散点未检出足够标记，保留为未完成；不以同色模型线替代"}
        elif want == "points" and len(markers) < 3:
            want = "line"
            note_draw = "模型说这条带标记，但没检出标记符号，改出线轨迹"
        p["trace"] = trace
        _sx0, _sx1, symin, symax = data_range_for(s)
        line_data = el.to_data(trace, frame, xmin, xmax, symin, symax)
        point_data = el.to_data(point_trace, frame, xmin, xmax, symin, symax)
        return {"p": p, "want": want, "markers": markers, "line_data": line_data,
                "point_data": point_data, "point_trace": point_trace, "note_dup": None, "note_draw": note_draw,
                "marker_line": marker_line, "span": span}, None

    # ---- 第三遍：定去留 + 合并重复轨迹 ----
    # 同一根线被模型拆成两条（甚至四条）上报是实测最常见的错法：63 个面板里出现
    # 30 条完全相同的轨迹。≥0.97 直接合并；0.85~0.97 保留但标注，交给复盘轮定夺。
    entries, aliases = [], []
    for pidx, p in enumerate(picks, start=1):
        entry, why = build_entry(p)
        if entry is None:
            failed.append(why)
            continue
        trace = p["trace"]
        dup_of, ov, same_inst = None, 0.0, False
        other_specs = {}
        for name, other, other_key, other_spec in accepted:
            other_specs[name] = other_spec
            o = ss.trace_overlap(trace, other)
            k = p["info"].get("key")
            if k is not None and other_key is not None and k == other_key and o >= 0.8:
                dup_of, ov, same_inst = name, max(o, 0.97), True
                continue
            if o > ov:
                dup_of, ov = name, o
        # 模型明说"这两条有重合"时不能合并：那是两条不同的曲线画在一起（图 10 那种
        # 两条相距几像素的虚线，中位差只有 4.5px，只按像素去重会把它们合掉）。
        overlaps_hi = {str(x).strip().lower() for x in (p["s"].get("overlaps") or [])}
        veto = str(dup_of).strip().lower() in overlaps_hi if dup_of else False
        prior = other_specs.get(dup_of) or {}
        if (prior.get("role") != p["s"].get("role")
                or prior.get("draw") != p["s"].get("draw")):
            veto = True  # Experimental markers and a model line remain separate series.
        if (same_inst or ov >= 0.97) and not veto:
            aliases.append({"label": p["s"].get("label") or f"series{pidx}",
                            "alias_of": dup_of,
                            "overlap": round(ov, 2)})
            continue
        if ov >= 0.80:
            entry["note_dup"] = (f"与「{dup_of}」重合 {int(round(ov * 100))}%"
                                 "（可能是同一条，已请模型复核）")
        accepted.append((p["s"].get("label") or f"series{pidx}", trace,
                         p["info"].get("key"), p["s"]))
        entries.append(entry)

    # ---- 第四遍：模型复盘（把追出来的轨迹编号画回原图，让模型核对）----
    review = None
    # Critic is a mandatory agent stage when VLM is enabled.  Conditional review
    # missed the most dangerous case: a plausible-looking but truncated curve with
    # the correct count.  One full-panel call is cheap and can request only targeted
    # extend/add/drop actions; local repair calls happen solely when it finds a defect.
    if vlm is not None and (entries or failed or spec):
        review = revise_panel(vlm, panel_path, panel, spec, entries, failed, aliases,
                              csv_dir=csv_dir)
        if review:
            for ext in (review.get("extend") or []):
                try:
                    idx = int(ext.get("id")) - 1
                except (TypeError, ValueError, AttributeError):
                    continue
                if not (0 <= idx < len(entries)):
                    continue
                entry0 = entries[idx]
                p0 = entry0["p"]
                if _spec_want(p0["s"]) == "points":
                    continue  # Point measurements can only be added by glyph detection.
                coord_space = str(ext.get("coord_space") or "data").lower()
                if coord_space in ("plot_norm", "normalized", "norm"):
                    anchors_px = ss.anchors_to_pixels(
                        ext.get("anchors"), frame, (width, height))
                else:
                    anchors_px = ss.data_anchors_to_pixels(
                        ext.get("anchors"), frame, data_range_for(p0["s"]))
                anchors_px = _model_anchors_in_frame(anchors_px, frame, inset=2.0)
                if not anchors_px:
                    continue
                measured = sorted((float(x), float(y)) for x, y in p0["trace"])
                gap_min = 0.05 * max(1.0, frame[2] - frame[0])
                long_gaps = [(a[0], b[0]) for a, b in zip(measured, measured[1:])
                             if b[0] - a[0] >= gap_min]
                # `extend` may add missing evidence, never rewrite an already observed
                # span.  This protects exact marker/path measurements from a critic
                # hallucinating a visual defect in the review rendering.
                anchors_px = [(x, y) for x, y in anchors_px
                              if x < measured[0][0] - 1.0 or x > measured[-1][0] + 1.0
                              or any(a < x < b for a, b in long_gaps)]
                if not anchors_px:
                    log(f"      复盘补范围已忽略：#{idx + 1} 的锚点均落在已有观测区间")
                    continue
                observed_others = [(str(e["p"]["s"].get("label")),
                                   ss.measured_trace(e["p"]["trace"], mask_for(e["p"])))
                                  for e in entries if e is not entry0
                                  and not _spec_has_markers(e["p"]["s"])
                                  and e["p"]["s"].get("label") in (p0["s"].get("overlaps") or [])]
                merged, repair_info = ss.retrace_from_anchors(
                    p0["trace"], anchors_px, mask_for(p0), frame, observed_others)
                if len(merged) <= len(p0["trace"]):
                    continue
                old_n = len(p0["trace"])
                p0["raw_trace"] = p0.get("raw_trace") or list(p0["trace"])
                p0["trace"] = merged
                p0["completion"] = dict(p0.get("completion") or {})
                p0["completion"]["model_anchors"] = {
                    "cols": len(merged) - old_n,
                    "by": f"模型复盘指路后从像素重追（{len(anchors_px)} 个锚点）"}
                p0["completion"].setdefault("spans", []).append({
                    "x0": int(min(x for x, _y in anchors_px)),
                    "x1": int(max(x for x, _y in anchors_px)),
                    "cols": len(merged) - old_n,
                    "by": "模型复盘补范围"})
                rebuilt, why_ext = build_entry(p0)
                if rebuilt is not None:
                    entries[idx] = rebuilt
                    log(f"      复盘补范围：#{idx + 1} {p0['s'].get('label')} "
                        f"+{len(merged) - old_n} 列（{str(ext.get('why') or '')[:42]}）")
                elif why_ext:
                    failed.append(why_ext)
            entries, failed, aliases = _apply_review(review, entries, failed, aliases)
            for s in (review.get("add") or []):
                pick, why = trace_one(s, 8)
                if pick is None or len(pick["trace"]) < 20:
                    failed.append({"label": s.get("label"), "anchors": s.get("anchors"),
                                   "reason": f"模型复盘补的这条没追上（{why or '点太少'}）"})
                    continue
                others = [(str(e["p"]["s"].get("label")), ss.measured_trace(e["p"]["trace"], mask_for(e["p"])))
                          for e in entries if e["p"]["kind"] != "loop" and e["p"]["trace"]]
                mmask = mask_for(pick)
                if others and _spec_want(pick["s"]) != "points":
                    filled, cinfo = ss.complete_trace(pick["trace"], mmask, frame, others)
                    if filled and len(filled) > len(pick["trace"]):
                        pick["raw_trace"], pick["trace"] = pick["trace"], filled
                        pick["completion"] = cinfo
                entry, why2 = build_entry(pick)
                if entry is None:
                    failed.append(why2)
                    continue
                entry["from_review"] = True
                entries.append(entry)

    # ---- 第五遍：落盘 ----
    bridged_px = []
    for entry in entries:
        p = entry["p"]; s = p["s"]
        trace = p["trace"]
        label = ss.safe_tag(s.get("label") or f"series{len(series) + 1}")
        want = entry["want"]
        emitted = []
        if want in ("points", "both") and len(entry["point_data"]) >= 3:
            name = ss.unique_path(csv_dir, f"{panel_path.stem}_{label}.csv")
            ss.write_csv(name, entry["point_data"])
            emitted.append(("points", name))
        min_line = 3 if entry.get("marker_line") else 20   # 散点图：一个点一个值
        if want in ("line", "both") and len(entry["line_data"]) >= min_line:
            suffix = "_line" if emitted else ""
            name = ss.unique_path(csv_dir, f"{panel_path.stem}_{label}{suffix}.csv")
            ss.write_csv(name, entry["line_data"])
            emitted.append(("line", name))
        if not emitted:
            failed.append({"label": s.get("label"), "anchors": s.get("anchors"),
                           "reason": "追踪出的点太少，未写出"})
            continue
        color = p["color"]
        color_hex = ("#{:02x}{:02x}{:02x}".format(int(color[2]), int(color[1]), int(color[0]))
                     if color is not None else str(s.get("color") or "#000000"))
        display_hex = str(s.get("display_color") or color_hex)
        if not display_hex.startswith("#"):
            display_hex = "#" + display_hex
        cinfo = p["completion"] or {}
        raw = p.get("raw_trace")
        if raw and cinfo:
            rx = {int(round(x)) for x, _y in raw}
            bridged_px.extend((x, y) for x, y in trace if int(round(x)) not in rx)
        completion = jsonable_completion(cinfo)
        for kind_out, name in emitted:
            data = entry["point_data"] if kind_out == "points" else entry["line_data"]
            measured_pixels = entry["point_trace"] if kind_out == "points" else trace
            warning = []
            visible_count = len(measured_pixels)
            if kind_out == 'line' and p['kind'] != 'loop':
                visible = ss.measured_trace(measured_pixels,mask_for(p))
                visible_count = sum(int(round(x)) in visible for x,y in measured_pixels)
            inferred_count = len(measured_pixels)-visible_count
            if inferred_count:
                warning.append(f"轨迹含 {inferred_count} 个内插或遮挡推断点；可见笔画支持 {visible_count} 点，不能视为逐点实测")
            extent = s.get("expected_x_extent")
            if isinstance(extent, (list, tuple)) and len(extent) == 2:
                try:
                    start = (min(x for x,y in measured_pixels) - frame[0]) / max(1,frame[2]-frame[0])
                    end = (max(x for x,y in measured_pixels) - frame[0]) / max(1,frame[2]-frame[0])
                    if start-float(extent[0]) > .08 or float(extent[1])-end > .08:
                        warning.append("实测横向范围未覆盖模型预期端点；当前数据为部分识别，未向空白区补点")
                except (TypeError, ValueError):
                    pass
            hits, distance = ss._anchor_stats(measured_pixels, p["anchors"])
            if distance is not None and distance > .12*min(frame[2]-frame[0],frame[3]-frame[1]):
                warning.append("实测轨迹与粗锚点偏离较大，需要核对系列身份")
            if cinfo.get("open_gaps"):
                warning.append("仍有缺少笔画证据的长缺口")
            if review and (review.get('ok') is False or review.get('issues')):
                warning.append("模型复盘仍有异议，需核对原图中的形状、标记数量与系列身份")
            series.append({
                "file": name.name, "source": "seeded", "kind": kind_out,
                "geometry": p["kind"], "series_name": s.get("label"), "output": want,
                "draw": s.get("draw"), "dark": bool(p["dark"]),
                "color_hex": display_hex, "trace_color_hex": color_hex,
                "color_src": p["how"], "y_axis": s.get("y_axis"),
                "y_axis_range": data_range_for(s)[2:],
                "anchors": s.get("anchors"), "note": s.get("note"),
                "anchors_hit": hits,
                "anchor_med_dist": round(distance,1) if distance is not None else None,
                "coverage": round(entry["span"], 3), "points": len(data),
                "visible_points": visible_count, "inferred_points": inferred_count,
                "duplicate_note": entry.get("note_dup"),
                "draw_note": entry.get("note_draw"),
                "completion": completion,
                "measurement_status": "needs_review" if warning else "measured",
                "warnings": warning,
                "x_range": [round(min(q[0] for q in data), 4), round(max(q[0] for q in data), 4)],
                "y_range": [round(min(q[1] for q in data), 4), round(max(q[1] for q in data), 4)],
            })
    emitted_names = {str(s.get("series_name")) for s in series if s.get("series_name")}
    failed = [f for f in failed if str(f.get("label")) not in emitted_names]
    if not series and not failed:
        return None
    traces_px = [(str(e["p"]["s"].get("label") or f"series{i + 1}"), e["p"]["trace"])
                 for i, e in enumerate(entries)]
    return {"panel": panel_path.name, "frame": list(frame), "series": series,
            "failed": failed, "aliases": aliases, "review": review,
            "_traces": traces_px, "_bridged_px": bridged_px, "method": "seeded"}


def jsonable_completion(cinfo):
    """补全信息进 config/report 前瘦身（只留计数与前几段，避免 JSON 膨胀）。"""
    if not cinfo:
        return None
    out = {"bridged_cols": int(cinfo.get("bridged_cols") or 0),
            "filled": int(cinfo.get("filled") or 0),
            "spans": [{"x0": int(s["x0"]), "x1": int(s["x1"]), "cols": int(s["cols"]),
                       "by": str(s.get("by"))} for s in (cinfo.get("spans") or [])[:8]],
            "open_gaps": [{"x0": int(s["x0"]), "x1": int(s["x1"]), "cols": int(s["cols"])}
                          for s in (cinfo.get("open_gaps") or [])[:8]]}
    if cinfo.get("model_anchors"):
        ma = cinfo["model_anchors"]
        out["model_anchors"] = {"cols": int(ma.get("cols") or 0), "by": str(ma.get("by"))}
    return out


def series_y_from(panel):
    """{颜色: 纵轴范围} —— 双 Y 轴图里每条曲线读自己那条轴。"""
    out = {}
    for col, side in (panel.get("series_axes") or {}).items():
        detail = (panel.get("y_axes") or {}).get(side) or {}
        if detail.get("range"):
            out[col.upper()] = detail["range"]
    return out or None


def _body_digest(pdf_path, want_pages=None, max_chars=12000):
    """正文里所有提到图的段落（不是全文）——按需才读。

    图注只说这张图"是什么"，正文才会说它"画的是什么、在什么工况下"：
    "Figure 5 presents the penetration at Pi = 800 bar"。这些句子只占全文的 3-12%，
    所以比灌全文便宜一个量级，而且是选图真正需要的那部分信息。
    """
    lines = []
    with tp.pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            if want_pages is not None and i not in want_pages:
                continue
            lines += [f"[p{i}] {t}" for t in tp.figure_mentions(page)]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…（正文摘要已截断）"
    return text


def doctor():
    """Environment self-check: interpreter, dependencies (really importing them), key."""
    import importlib
    import importlib.util

    print("环境自检")
    print(f"  解释器      : {sys.executable}")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  工作目录    : {Path.cwd()}")
    print()
    checks = [("numpy", "数组运算", True), ("cv2", "图像处理", True),
              ("pdfplumber", "PDF 解析", True), ("PIL", "图片编码", True),
              ("pypdfium2", "PDF 渲染", True), ("pdfminer", "PDF 文本", True)]
    bad = 0
    for mod, why, required in checks:
        ok = importlib.util.find_spec(mod) is not None
        detail = ""
        if ok:
            try:
                importlib.import_module(mod)
            except Exception as exc:  # noqa: BLE001
                ok = False
                detail = f"  导入失败: {type(exc).__name__}: {exc}"
        if not ok:
            bad += 1 if required else 0
        tag = "✅" if ok else ("❌" if required else "⚠️ ")
        print(f"  {mod:<22} {tag} {why}{detail}")
    key = vlmc.load_api_key()
    print(f"  {'VLM key':<22} {'✅' if key else '⚠️ '} "
          + ("已配置（内容不会被打印）" if key else "未配置：位图轴读数不可用，矢量流程仍可运行"))
    print()
    if bad:
        print(f"❌ 有 {bad} 个必需依赖不可用。多半是环境不对——请激活装了依赖的环境"
              "（例如 conda activate pytorch）后重试。")
    else:
        print("✅ 必需依赖齐备，可以运行：python run.py all <pdf> --out out_dir --vlm")


def _norm_axis(value):
    """Accept {'min','max'} (VLM) or [lo, hi] (manual) -> [lo, hi] or None."""
    if isinstance(value, dict):
        if str(value.get('scale') or 'linear').lower() != 'linear':
            return None  # This pipeline has a linear pixel-to-value transform.
        lo, hi = value.get("min"), value.get("max")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = value
    else:
        return None
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return None
    return [lo, hi] if np.isfinite([lo,hi]).all() and hi > lo else None


def _as_float(value, default=1.0):
    """宽容地把模型给的数字（可能是 "1e-4" / "×10^-4" / None）变成 float。"""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace("×", "").replace("x", "").replace("*", "")
    text = text.replace("^", "e").replace("10e", "1e")
    try:
        return float(text)
    except ValueError:
        return default


def _axis_hint(want):
    """把用户需求交给轴读数那一步顺带核对——不额外调用，也不靠"模型觉得像"。

    实测：`--want "贯穿距随时间变化"` 会把"贯穿距 vs 曲轴转角"也算命中。这里让模型在
    读刻度时就回答"这张图的横轴/纵轴是不是用户要的"，命中靠标题核对。
    """
    if not want:
        return None
    return (f"用户要的是：{want}。请额外读出 x 轴与各纵轴的标题/单位，并判断这张图的"
            f"横轴与纵轴是否就是用户要的那两个量：符合填 matches_request=true，不符合填 "
            f"false（例如用户要时间轴而这里是曲轴转角），看不清填 null。")


def _record_axis_check(entry, va, want, log):
    """把轴标题与"是否合需求"记进 config；明确不符合的图直接标跳过（报告里给理由）。"""
    x = va.get("x") if isinstance(va.get("x"), dict) else {}
    ylist = [d for d in (va.get("y_axes") or []) if isinstance(d, dict)]
    y0 = ylist[0] if ylist else {}
    entry["axis"]["x_title"] = x.get("title")
    entry["axis"]["x_unit"] = x.get("unit")
    entry["axis"]["y_title"] = y0.get("title")
    titles = (f"x={x.get('title')}"
              + (f" [{x.get('unit')}]" if x.get("unit") else "")
              + f" / y={y0.get('title')}"
              + (f" [{y0.get('unit')}]" if y0.get("unit") else ""))
    if not want:
        return
    match, note = va.get("matches_request"), va.get("match_note")
    note = str(note or "")
    # "读不出标题"不等于"不符合"：模型有时因为图里没有轴标题就填 false，上一版据此把
    # 面板整个跳过（实测 2026-01-0340 图 10 左图：横轴标题被裁掉，但确实是时间轴，
    # 结果整张图一条数据都没出）。只有明确"轴是别的东西"才跳过。
    if match is False and any(w in note for w in
                              ("无法确认", "无法判断", "无法判定", "看不清", "不确定",
                               "无标题", "没有标题", "未标")):
        match = None
        note += "（标题读不出，按「保留待复核」处理）"
    entry["axis_check"] = {"want": want, "matches": match, "note": note,
                           "x_title": x.get("title"), "y_title": y0.get("title")}
    verdict = "符合" if match is True else ("不符合" if match is False else "看不清")
    log(f"      轴标题核对（{verdict}）: {titles}"
        + (f" —— {str(note)[:60]}" if note else ""))
    if match is False:
        entry["skip"] = True
        entry["skip_reason"] = (f"轴标题与需求不符（x={x.get('title')} / "
                                f"y={y0.get('title')}）：{str(note or '')[:80]}")
        log("      ⏭ 按需求跳过这张图（轴标题核对没通过）")


def _ensure_panel_request(vlm, panel_path, want, va, panel_plan, context=""):
    """A planned axis cannot bypass semantic filtering; fallback checks only this panel."""
    check = (panel_plan or {}).get("request_check") or {}
    if not isinstance(check, dict):
        check = {}
    complete = all(isinstance(check.get(k), bool)
                   for k in ("matches_request", "axes_match", "conditions_match"))
    if not complete:
        try:
            check = vt.check_panel_request(vlm, panel_path, want, context=context)
        except Exception as exc:  # noqa: BLE001
            check = {"matches_request": None,
                     "note": f"需求核对失败：{type(exc).__name__}: {exc}"}
    checked = dict(va or {})
    match = check.get("matches_request")
    if isinstance(va, dict) and va.get("matches_request") is False:
        # A close-up axis contradiction cannot be overridden by a whole-figure plan.
        if not any(word in str(va.get("match_note") or "") for word in
                   ("看不清", "无法确认", "无标题", "没有标题", "未标", "不确定", "被裁")):
            check = dict(check, axes_match=False, note=va.get("match_note"))
    if check.get("axes_match") is False or check.get("conditions_match") is False:
        match = False
    elif not isinstance(match, bool):
        match = None
    checked["matches_request"] = match
    checked["match_note"] = check.get("note")
    return checked, {k: v for k, v in check.items() if k != "_raw"}


def _norm_y_axes(vlm_result):
    """VLM may return a list of y axes (left/right) or a single 'y'.

    -> {side: {"range": [lo, hi], "labels": [...]}}
    """
    out = {}
    duplicated = set()
    for item in (vlm_result.get("y_axes") or []):
        if not isinstance(item, dict):
            continue
        side = str(item.get("side") or "left").strip().lower()
        rng = _norm_axis(item)
        if rng:
            if side in out or side in duplicated:
                out.pop(side, None)
                duplicated.add(side)
                continue  # Different panels' left axes are not a dual-Y axis.
            out[side] = {"range": rng, "labels": item.get("labels"),
                         "multiplier": _as_float(item.get("multiplier"), 1.0),
                         "unit": item.get("unit"), "title": item.get("title")}
    if not out and not duplicated:
        single = _norm_axis(vlm_result.get("y"))
        if single:
            out["left"] = {"range": single, "labels": None}
    return out


def resolve_vlm_axis(vlm_axis):
    """Return the model-read axis range as (values, confirmed, source, note)."""

    values = _norm_axis(vlm_axis)
    if values:
        return values, True, "vlm", ""
    return None, False, "none", "未读出轴范围（位图任务需要视觉模型）"


def print_vlm_usage(client, phase, before=None):
    """默认一行汇总；--verbose/并行日志里保留逐项明细。"""
    if client is None:
        return
    u = client.usage
    d = {k: u.get(k, 0) - (before or {}).get(k, 0) for k in u}
    p = vlmc.PRICES.get(client.model)
    cost = ""
    if p:
        hit = d["cache_hit_tokens"]
        miss = d["cache_miss_tokens"] or max(0, d["prompt_tokens"] - hit)
        lo = (hit / 1e6 * p["in_hit"][0] + miss / 1e6 * p["in_miss"][0]
              + d["completion_tokens"] / 1e6 * p["out"][0])
        hi = (hit / 1e6 * p["in_hit"][1] + miss / 1e6 * p["in_miss"][1]
              + d["completion_tokens"] / 1e6 * p["out"][1])
        cost = f", 约 ${lo:.4f}–${hi:.4f}"
    if not (_VERBOSE or _LOG_FILE is not None):
        log(f"VLM {phase}: {d['calls']} 次调用, {d['cache_hits']} 次本地缓存, "
            f"{d['prompt_tokens']:,} in / {d['completion_tokens']:,} out{cost}")
        return
    bar = "─" * 66
    title = (f"VLM 用量（{phase}）" if phase.endswith(("合计", "总计"))
             else f"VLM 用量（{phase} 阶段）")
    log("\n" + bar)
    log(title)
    log(f"  调用次数     : {d['calls']} 次   （本地缓存命中 {d['cache_hits']} 次，不消耗 token）")
    line = f"  输入 tokens  : {d['prompt_tokens']:,}"
    if d["cache_hit_tokens"] or d["cache_miss_tokens"]:
        line += (f"   （上下文缓存命中 {d['cache_hit_tokens']:,} / "
                 f"未命中 {d['cache_miss_tokens']:,}）")
    log(line)
    log(f"  输出 tokens  : {d['completion_tokens']:,}")
    if d["calls"]:
        log(f"  每次平均     : 输入 {d['prompt_tokens'] / d['calls']:,.0f} / "
            f"输出 {d['completion_tokens'] / d['calls']:,.0f} tokens")
    if p:
        log(f"  预估费用     : ${lo:.4f} ~ ${hi:.4f}   （{client.model}，闲时~高峰，官方价）")
    log(f"  逐次明细     : {vlmc.LOG_PATH}")
    log(bar)


def analyze(pdf_path, outdir, dpi=300, vlm=None, pages=None, want=None,
            want_deep=False, find_series=True, report_usage=True):
    """Read figures, panel crops and axis ranges; the visual model reads raster axes.

    Vector figures carry their own calibration from the PDF text.  Without a VLM,
    raster panels remain unconfirmed instead of being guessed.
    """
    t0 = time.time()
    pdf_path = Path(pdf_path).resolve()
    outdir = Path(outdir).resolve()
    _clean_stale_intermediates(outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "panels").mkdir(parents=True, exist_ok=True)
    cfg_path = outdir / f"{pdf_path.stem}_config.json"
    analysis_cache_enabled = getattr(vlm, "use_cache", True) if vlm is not None else True
    cache_dir = None
    if analysis_cache_enabled:
        try:
            cache_key = _analysis_cache_key(
                pdf_path, dpi, pages, want, want_deep, find_series, vlm)
            cache_dir = ANALYSIS_CACHE_DIR / cache_key
            if _load_analysis_cache(cache_dir, outdir, pdf_path, cfg_path):
                log(f"分析缓存命中：{pdf_path.name}（跳过选图/面板/轴识别）")
                return cfg_path
        except Exception as exc:  # noqa: BLE001 - cache must never block analysis
            log(f"（分析缓存读取失败，继续正常分析：{type(exc).__name__}: {exc}）")
            cache_dir = None

    with tp.pdfplumber.open(pdf_path) as pdf:
        want_pages = None
        if pages:
            want_pages = {int(v) for v in str(pages).replace(" ", "").split(",") if v}
        report = {
            "pdf": str(pdf_path),
            "pdf_stem": pdf_path.stem,
            "pages": [tp.analyse_page(p, i + 1) for i, p in enumerate(pdf.pages)
                      if want_pages is None or (i + 1) in want_pages],
        }
        report["exported"] = tp.export_figures(pdf, report, outdir, dpi)

        # Vector figures contain their own data: no rasterising needed.
        vector_found = []
        vector_fallback = []   # 矢量区域读不出刻度 -> 渲染成位图，交给后面的位图路线
        exported_by_page = {}
        for e in report.get("exported", []):
            if e.get("file") and e.get("bbox"):
                exported_by_page.setdefault(e["page"], []).append(tuple(e["bbox"]))
        for pno, page in enumerate(pdf.pages, start=1):
            if want_pages is not None and pno not in want_pages:
                continue
            for j, reg in enumerate(vx.find_vector_regions(page), start=1):
                vid = f"{pdf_path.stem}_vec_p{pno}_{j}"
                res = vx.extract_vector_figure(page, reg["bbox"])
                if res.get("x_calibration") and res.get("y_calibration"):
                    vector_found.append({"page": pno, "reg": reg, "res": res, "id": vid,
                                         "inner": _vector_inner_text(page, res.get("axes")),
                                         "thumb": _vector_thumb(
                                             page, reg["bbox"], outdir, vid,
                                             bool(want or (vlm is not None and find_series)))})
                    continue
                # 区域里没有 PDF 自带的可读刻度文字。两种常见原因：刻度是路径而非文字；
                # 或者页面边框/分栏线与图并成了一个区域。前者渲染成位图还能救回来，后者不是图。
                bx0, by0, bx1, by1 = reg["bbox"]
                if (bx1 - bx0) > 0.9 * page.width and (by1 - by0) > 0.9 * page.height:
                    log(f"[矢量] p{pno} bbox={reg['bbox']} 跳过：区域覆盖整页"
                        "（页面边框/分栏线被并了进来，不是数据图）", detail=True)
                    continue
                if not res.get("axes") and not res.get("axes_boxes"):
                    # 连坐标框都没有，只是一堆页面线条/表格线：回退成位图也读不出坐标轴
                    log(f"[矢量] p{pno} bbox={reg['bbox']} 跳过：无坐标框"
                        "（页面线条/表格，非数据图）", detail=True)
                    continue
                # 留出边距：区域边界常常正好压在坐标框上，紧贴裁剪会让后面的找框
                # 逻辑把框线当成图片边界而丢弃；另外刻度标签也需要这点空间。
                m = 24.0
                crop_box = (max(0.0, bx0 - m), max(0.0, by0 - m),
                            min(page.width, bx1 + m), min(page.height, by1 + m))
                if any(_overlap_frac(crop_box, eb) > 0.5
                       for eb in exported_by_page.get(pno, [])):
                    log(f"[矢量] p{pno} bbox={reg['bbox']} 跳过：与已导出的位图图重叠，"
                        "同一张图不重复提取", detail=True)
                    continue
                fb_path = outdir / "figures" / f"{vid}.png"
                try:
                    page.crop(crop_box, strict=False).to_image(
                        resolution=dpi).original.save(fb_path)
                except Exception as exc:  # noqa: BLE001
                    log(f"[矢量] p{pno} bbox={reg['bbox']} 跳过：无刻度文字且渲染失败"
                        f"（{type(exc).__name__}: {exc}）", detail=True)
                    continue
                # 坐标框在 PDF 里已经找到了，直接换算成像素给位图路线用：
                # 比自己再去图里猜一次框更可靠（刻度线贴边时猜框会失手）。
                scale = dpi / 72.0
                ax = res.get("axes")
                frame = ([int(round((ax[0] - crop_box[0]) * scale)),
                         int(round((ax[1] - crop_box[1]) * scale)),
                         int(round((ax[2] - crop_box[0]) * scale)),
                         int(round((ax[3] - crop_box[1]) * scale))] if ax else None)
                vector_fallback.append({
                    "path": fb_path, "page": pno, "frame": frame,
                    "note": f"矢量回退 bbox={[round(v, 1) for v in crop_box]}",
                    "id": vid, "bbox": reg["bbox"], "thumb": fb_path,
                    "pdf_crop_bbox": list(crop_box), "render_dpi": dpi,
                })
                log(f"[矢量] p{pno} bbox={reg['bbox']} 无刻度文字 → "
                    f"渲染为位图改走图像路线：{fb_path.name}", detail=True)

    captions_by_page = {}
    for p in report["pages"]:
        caps = [c for c in p["captions"] if c.get("kind") != "table"]
        if caps:
            captions_by_page[p["page"]] = caps

    config = {"pdf": str(pdf_path), "outdir": str(outdir), "panels": []}
    vlm_usage0 = dict(vlm.usage) if vlm is not None else None

    # ---- 候选图清单：先把"每张图 + 它自己的图注"配好，再按需求筛 ----
    cards = [{"id": it["id"], "page": it["page"], "bbox": list(it["reg"]["bbox"]),
              "kind": "vector", "thumb": it["thumb"], "inner_text": it.get("inner", "")}
             for it in vector_found]
    cards += [{"id": it["id"], "page": it["page"], "bbox": list(it["bbox"]),
               "kind": "vector-fallback", "thumb": it["thumb"]} for it in vector_fallback]
    exported = []
    for exp in report["exported"]:
        if not exp.get("file"):
            # 不要静默跳过：渲染失败通常意味着缺依赖（如 pypdfium2），
            # 而症状只是"后面一个位图面板都没有"，极难排查。
            log(f"  ⚠ 第 {exp.get('page')} 页的图导出失败：{exp.get('error')}")
            if exp.get("error") and "pypdfium2" in str(exp["error"]):
                log("     ↑ 缺 pypdfium2（pdfplumber 渲染 PDF 用）。"
                    "修：pip install -r requirements.txt")
            continue
        exported.append({"path": Path(exp["file"]), "page": exp["page"],
                         "frame": None, "note": None})
        cards.append({"id": Path(exp["file"]).stem, "page": exp["page"],
                      "bbox": list(exp["bbox"]), "kind": "raster",
                      "thumb": Path(exp["file"])})
    for it in vector_fallback:
        exported.insert(0, {"path": it["path"], "page": it["page"], "frame": it["frame"],
                            "note": it["note"],"pdf_crop_bbox":it.get("pdf_crop_bbox"),
                            "render_dpi":it.get("render_dpi")})
    cards.sort(key=lambda c: (c["page"], round(c["bbox"][1]), round(c["bbox"][0])))
    fi.attach_captions(cards, captions_by_page, log=detail_log)
    caption_of = {c["id"]: c["caption"] for c in cards}

    keep = None
    if want:
        select_tmp = None
        if _KEEP_INTERMEDIATES:
            select_dir = outdir / "select"
        else:
            select_tmp = tempfile.TemporaryDirectory(prefix="dig_select_")
            select_dir = Path(select_tmp.name)
        try:
            with tp.pdfplumber.open(pdf_path) as selection_pdf:
                fi.render_selection_images(selection_pdf, cards, select_dir)
            decision = fi.select(want, cards, vlm=vlm, out_dir=select_dir,
                                 log=detail_log,
                                 body_text=lambda: _body_digest(pdf_path),
                                 deep=want_deep)
        finally:
            if select_tmp is not None:
                select_tmp.cleanup()
        keep = set(decision["ids"])
        config["selection"] = decision
        if not keep:
            log("\n" + "!" * 66)
            verdict = "没有匹配" if decision.get("missing") is True else "暂时无法确认匹配"
            log(f"这篇里{verdict}「{want}」的图 —— 没有提取任何数据。")
            if decision.get("available"):
                log(f"  这篇实际有：{decision['available']}")
            if decision.get("reason"):
                log(f"  判断依据：{decision['reason']}")
            log("  候选清单（可用 --want 换个说法，或直接写图号）：")
            for c in decision["candidates"]:
                log(f"    {c['n']:>2}. {c['label'] or '(无编号)':<12} p{c['page']:<3} "
                    f"{c['caption'][:78]}")
            log("!" * 66)

    for it in vector_found:
        pno, reg, res = it["page"], it["reg"], it["res"]
        if keep is not None and it["id"] not in keep:
            continue
        # 只有"坐标框 + 刻度文字都读到了"的区域才会进 vector_found
        xcal, ycal = res["x_calibration"], res["y_calibration"]
        entry = {
            "id": it["id"],
            "page": pno,
            "method": "vector",
            "vector_bbox": reg["bbox"],
            "vector_objects": reg["objects"],
            "frame": res["axes"],
            "caption": caption_of.get(it["id"], ""),
            "legend_colors": [],
            "axis": {
                "x": vx.axis_range_from_calibration(xcal, res["axes"], "x"),
                "y": vx.axis_range_from_calibration(ycal, res["axes"], "y"),
                # values come from the PDF's own text -> trust them
                "confirmed": True,
                "source": "vector-text",
            },
            "vector": {
                "x_labels": res["x_labels"], "y_labels": res["y_labels"],
                "x_r2": xcal["r2"], "y_r2": ycal["r2"],
                "x_positions": xcal.get("positions", "label-centres"),
                "y_positions": ycal.get("positions", "label-centres"),
                "series_count": len(res.get("series", [])),
            },
        }
        if vlm is not None and find_series and it.get("thumb"):
            try:
                found = vt.find_series(vlm, it["thumb"], it.get("inner") or "（未读到图例文字）")
                spec, ignored = _planner_series(found)
                if spec:
                    entry["series_spec"] = spec
                    entry["extraction_plan"] = {
                        "chart_family": found.get("chart_family"),
                        "series_count": len(spec),
                        "reported_series_count": found.get("series_count"),
                        "exclude_regions": found.get("exclude_regions") or [],
                    }
                    entry["ignore"] = ignored
                    entry["legend_colors"] = list(dict.fromkeys(
                        str(s.get("display_color") or s.get("color"))
                        for s in spec if s.get("display_color") or s.get("color")))
                    log(f"      VLM 矢量曲线清单: {len(spec)} 条（几何路径将逐实例匹配，不按颜色合并）",
                        detail=True)
                    for s in spec:
                        log(f"        - {str(s.get('label'))[:28]} 颜色={s.get('color')} "
                            f"锚点={len(s.get('anchors') or [])}", detail=True)
            except vlmc.VLMError as exc:
                log(f"      VLM 矢量曲线定位失败: {exc}（仍按 PDF 物理路径分实例）")
            except Exception as exc:  # noqa: BLE001
                log(f"      VLM 矢量曲线定位异常: {type(exc).__name__}: {exc}")
        log(f"\n[矢量图] p{pno} bbox={reg['bbox']}  "
            f"x={entry['axis']['x']} y={entry['axis']['y']}  "
            f"{len(res.get('series', []))} 条路径序列（直接从 PDF 读取）", detail=True)
        config["panels"].append(entry)

    for fig in exported:                     # 矢量回退图排在最前，其余是导出的位图图
        fig_path, page, note = fig["path"], fig["page"], fig["note"]
        if keep is not None and fig_path.stem not in keep:
            continue
        fig_img = iio.imread(fig_path)
        if fig_img is None:
            continue
        caption = caption_of.get(fig_path.stem, "")
        log(f"\n[图] {fig_path.name}  (p{page})" + (f"  ← {note}" if note else ""),
            detail=True)

        # ---- 第一步就交给 VLM 判断：不是数据图就整图跳过，省掉切分与读轴的开销 ----
        vlm_cls, name_map = None, {}
        if vlm is not None:
                try:
                    vlm_cls = vt.classify_chart(vlm, fig_path, want=want, caption=caption)
                    log(f"      VLM 分类: {vlm_cls.get('chart_type')}  "
                        f"子图={vlm_cls.get('panel_count')}  双Y轴={vlm_cls.get('dual_y_axis')}  "
                        f"可提取={vlm_cls.get('is_line_chart')}"
                        f"  （{str(vlm_cls.get('reason', ''))[:40]}）", detail=True)
                    if vlm_cls.get("is_line_chart") is False:
                        log("      ⏭ 跳过：VLM 判定为非数据图（照片/示意图/表格），不再切分与读轴",
                            detail=True)
                        continue
                    multi_y = bool(vlm_cls.get("dual_y_axis"))
                    if multi_y:
                        log("      ⚠ 检出双 Y 轴：将为每条曲线单独指派纵轴", detail=True)
                except vlmc.VLMError as exc:
                    log(f"      VLM 分类失败: {exc}（继续走图像路线）")

        legend_colors = detect_legend_colors(fig_img)
        log(f"      图例色: {legend_colors}", detail=True)
        series_sides = {}
        series_roles = {}
        dark_info = {}
        # The semantic planner below already reads labels, roles, axis side,
        # line/marker identity and exclusions in one response. Keep the older naming
        # call only when that planner is explicitly disabled.
        if vlm is not None and legend_colors and not find_series:
            try:
                nm = vt.name_series(vlm, fig_path, legend_colors)
                for s in nm.get("series", []):
                    col = (s.get("color") or "").upper()
                    if col and s.get("label"):
                        name_map[col] = s["label"]
                    if col and s.get("y_axis"):
                        series_sides[col] = str(s["y_axis"]).strip().lower()
                    if col:
                        # 模型说"这不是数据线"（切线/拟合线/标注/放大子图）就别提取
                        role = str(s.get("role") or "").strip().lower()
                        if s.get("is_data") is False and not role:
                            role = "not-data"
                        if role:
                            series_roles[col] = role
                log("      VLM 系列命名: " + ", ".join(f"{k}->{v}" for k, v in name_map.items()),
                    detail=True)
                if series_sides:
                    log("      VLM 轴归属: " + ", ".join(f"{k}->{v}轴" for k, v in series_sides.items()),
                        detail=True)
                drop = {k: v for k, v in series_roles.items()
                        if v in ("tangent", "fit", "annotation", "legend", "inset",
                                 "other", "not-data")}
                if drop:
                    log("      VLM 判定不提取: " + ", ".join(f"{k}({v})" for k, v in drop.items()),
                        detail=True)
                dark_info = nm.get("dark_series") or {}
                if dark_info.get("has_dark_line"):
                    log(f"      图中有深色线：{str(dark_info.get('kind') or '?')}"
                        f"（{str(dark_info.get('desc') or '')[:50]}）"
                        " → 默认不提取；要提就在 config 里把该面板的 allow_dark 设为 true",
                        detail=True)
            except vlmc.VLMError as exc:
                log(f"      VLM 命名失败: {exc}")

        if fig.get("frame") and int((vlm_cls or {}).get("panel_count") or 1) == 1:
            # 矢量回退：坐标框已知，不必（也不该）再去图里猜子图
            panels = [(fig_path, tuple(fig["frame"]), (0, 0, fig_img.shape[1], fig_img.shape[0]))]
            split_info = {"source": "vector-known-frame", "expected": 1,
                          "pixel_detected": 1, "used": 1}
        else:
            expected_panels = (vlm_cls or {}).get("panel_count")
            split_info = {}
            panels = split_figure(fig_path, outdir / "panels",
                                  expected_panels=expected_panels,
                                  panel_geometry=vlm_cls,
                                  split_meta=split_info)
            expected_n = split_info.get("expected")
            if expected_n and expected_n > 1 and len(panels) != expected_n and vlm is not None:
                log(f"      像素面板检测={split_info.get('pixel_detected', 0)}，"
                    f"但模型识别={expected_n}；调用版面定位工具，不再整图降级",
                    detail=True)
                try:
                    geometry = vt.locate_panels(vlm, fig_path, expected_n)
                    panels = split_figure(fig_path, outdir / "panels",
                                          expected_panels=expected_n,
                                          panel_geometry=geometry,
                                          split_meta=split_info)
                    if panels:
                        log(f"      模型定位 + 像素校准：得到 {len(panels)} 个独立面板",
                            detail=True)
                except vlmc.VLMError as exc:
                    log(f"      面板定位失败: {exc}")
                except Exception as exc:  # noqa: BLE001
                    log(f"      面板定位异常: {type(exc).__name__}: {exc}")
            if expected_n and expected_n > 1 and len(panels) != expected_n:
                log(f"      ⏭ 跳过该图：已确认有 {expected_n} 个面板，但只能可靠定位"
                    f" {len(panels)} 个；禁止把整图当一个坐标系生成错误数据")
                continue
        legend_contracts = _read_figure_legends(vlm,panels,fig_path,outdir)
        for idx, (panel_path, frame, box) in enumerate(panels, start=1):
            pid = f"{fig_path.stem}_p{idx}"
            log(f"  面板 {idx}/{len(panels)}: {panel_path.name} frame={frame}", detail=True)
            panel_img = iio.imread(panel_path)
            if panel_img is None:
                log(f"      ⏭ 面板图读取失败：{panel_path}")
                continue
            panel_legend_colors = (detect_legend_colors(panel_img)
                                   or legend_colors)
            plan_rows = (vlm_cls or {}).get("panels") or []
            panel_plan = plan_rows[idx - 1] if idx <= len(plan_rows) \
                and isinstance(plan_rows[idx - 1], dict) else {}
            entry = {
                "id": pid,
                "page": page,
                "figure": str(fig_path.relative_to(outdir)) if fig_path.is_relative_to(outdir) else str(fig_path),
                "panel_image": str(panel_path.relative_to(outdir)) if panel_path.is_relative_to(outdir) else str(panel_path),
                "frame": list(frame),
                "box_in_figure": list(box),
                "caption": caption,
                "legend_colors": panel_legend_colors,
                "series_names": name_map,
                "series_roles": series_roles,
                "vlm_dark_line": dark_info,
                "vlm_classification": vlm_cls,
                "panel_split": {k: v for k, v in split_info.items()
                                if k != "panel_labels"},
                "axis": {"x": None, "y": None, "confirmed": False},
            }
            if fig.get("pdf_crop_bbox"):
                scale = float(fig["render_dpi"])/72.
                fx,fy = fig["pdf_crop_bbox"][:2]
                cx,cy = box[:2]
                entry["vector_panel_axes"] = [fx+(cx+frame[0])/scale,fy+(cy+frame[1])/scale,
                                               fx+(cx+frame[2])/scale,fy+(cy+frame[3])/scale]
            if panel_plan:
                entry["agent_plan"] = {
                    "label": panel_plan.get("label"),
                    "plot_box": panel_plan.get("plot_box"),
                    "axis_planned": bool(panel_plan.get("axis")),
                    "series_planned": len(panel_plan.get("series") or []),
                    "exclude_regions": panel_plan.get("exclude_regions") or [],
                }
            labels = split_info.get("panel_labels") or []
            if idx <= len(labels) and labels[idx - 1]:
                entry["panel_label"] = labels[idx - 1]
            # ---- 轴读数：视觉模型读取刻度与范围，再把刻度值校准到实际刻线 ----
            planned_axis = panel_plan.get("axis") if isinstance(panel_plan, dict) else None
            if not planned_axis and isinstance(vlm_cls, dict):
                planned_axis = vlm_cls.get("shared_axis")
            va = planned_axis if isinstance(planned_axis, dict) \
                and planned_axis.get("x") and (planned_axis.get("y_axes")
                                                or planned_axis.get("y")) else None
            if vlm is not None:
                try:
                    va = vt.read_axis_ranges(vlm, panel_path, hint=_axis_hint(want),
                        figure_path=fig_path, panel_label=entry.get("panel_label"),frame=frame)
                    va = ac.calibrate_vlm_ticks(panel_img, frame, va)
                    entry["axis_read_source"] = "full-resolution-panel"
                    for key,source in [('x',va.get('x'))]+[
                            (a.get('side','left')+' y',a) for a in va.get('y_axes') or []]:
                        if source and str(source.get('scale') or 'linear').lower()!='linear':
                            entry.setdefault('notes',[]).append(
                                f"{key}: 非线性坐标尚不支持此线性标定路线，拒绝自动套用最小/最大值")
                        evidence = (source or {}).get('tick_calibration') or {}
                        if evidence:
                            note = f"{key}: 刻度位置拟合"+(
                                f"通过 R²={evidence['r2']}，边界范围={evidence.get('frame_range')}"
                                if evidence.get('accepted') else '证据不足，需核对轴范围')
                            entry.setdefault('notes',[]).append(note)
                except vlmc.VLMError as exc:
                    va = None
                    log(f"      VLM 子图轴读数失败: {exc}（不使用未经子图核对的整图轴值）")
                except Exception as exc:  # noqa: BLE001
                    va = None
                    log(f"      VLM 轴读数异常: {type(exc).__name__}: {exc}")
            if want and vlm is not None:
                va, entry["request_check"] = _ensure_panel_request(
                    vlm, panel_path, want, va, panel_plan,
                    context=caption + "\n选图依据：" +
                    str(config.get("selection", {}).get("reason") or ""))
            if va is not None:
                y_axes = _norm_y_axes(va)
                entry["vlm_axis"] = {"x": va.get("x"), "y_axes": va.get("y_axes"),
                                     "y": va.get("y"),
                                     "confidence": va.get("confidence"),
                                     "note": va.get("note")}
                if y_axes:
                    entry["y_axes"] = y_axes
                _record_axis_check(entry, va, want, log)
                log("      VLM 轴读数: x=" + str(_norm_axis(va.get("x")))
                    + "  y=" + (", ".join(
                        f"{k}轴{[round(d['range'][0], 4), round(d['range'][1], 4)]}"
                        for k, d in y_axes.items()) or "None")
                    + f"  置信度={va.get('confidence')}", detail=True)

            # x 轴全面板共用；y 轴以"左轴"作为面板默认值，逐条曲线再按 VLM 的归属覆盖
            for axis_key in ("x", "y"):
                if axis_key == "x":
                    v_val = va.get("x") if va else None
                    v_mult = 1.0
                    if isinstance(v_val, dict):          # 新格式：{"min","max","multiplier"}
                        v_mult = _as_float(v_val.get("multiplier"), 1.0)
                    elif isinstance(va, dict):
                        v_mult = _as_float(va.get("multiplier"), 1.0)
                else:
                    y_axes = (entry.get("y_axes") or {})
                    first = y_axes.get("left") or (list(y_axes.values())[0] if y_axes else None)
                    v_val = first["range"] if first else None
                    v_mult = (first or {}).get("multiplier", 1.0)
                vals, conf, src, note = resolve_vlm_axis(v_val)
                if vals:
                    entry["axis"][axis_key] = vals
                if abs(_as_float(v_mult, 1.0) - 1.0) > 1e-12:
                    entry["axis"].setdefault("multiplier", {})[axis_key] = \
                        _as_float(v_mult, 1.0)
                    entry.setdefault("notes", []).append(
                        f"{axis_key}: 图上标注了 ×{_as_float(v_mult, 1.0):g}，"
                        f"数据已按此换算")
                entry[f"axis_src_{axis_key}"] = src
                if note:
                    entry.setdefault("notes", []).append(f"{axis_key}: {note}")
                if conf:
                    entry["axis"]["confirmed"] = True
            log(f"      最终轴范围: x={entry['axis'].get('x')} y={entry['axis'].get('y')}"
                f"  来源={entry.get('axis_src_x')}/{entry.get('axis_src_y')}"
                f"  已确认={entry['axis'].get('confirmed')}", detail=True)
            if entry.get("y_axes") and len(entry["y_axes"]) > 1:
                pairs = []
                for col, side in (series_sides or {}).items():
                    detail = entry["y_axes"].get(side) or {}
                    pairs.append(f"{col}→{side}轴{detail.get('range')}")
                log("      逐曲线轴指派: "
                    + ("; ".join(pairs) if pairs else "（VLM 未给出归属，统一用左轴）"),
                    detail=True)
            entry["series_axes"] = series_sides

            if entry.get("skip"):
                config["panels"].append(entry)
                continue

            # ---- 物件级判定：让模型看着编号裁图说清"哪根是哪根" ----
            # 让模型**自己指出**数据曲线和它们的大致位置（锚点），代码再从锚点精确追踪。
            # 关键在于模型输出的是"要提取什么"（正向清单），不再是"我给的候选里哪个不是"
            # ——后者必然出现"数据被 skip 掉、该 skip 的却留下"。
            if vlm is not None and find_series:
                planned_series = panel_plan.get("series") if isinstance(panel_plan, dict) else None
                try:
                    legend_text = "、".join(
                        f"{v}（{k}）" for k, v in (name_map or {}).items()
                    ) or "、".join(panel_legend_colors)
                    context = caption + "\n整图中该子图的候选身份（需要按当前图重新核对）：" + json.dumps(
                        [{k: s.get(k) for k in ("label", "role", "draw", "marker_shape")}
                         for s in (planned_series or []) if isinstance(s, dict)], ensure_ascii=False)
                    if legend_contracts:
                        context += '\n单独放大图例样本得到的画法（同名同色时以真实样本为准）：'+json.dumps(
                            legend_contracts,ensure_ascii=False)
                    found = vt.find_series(vlm, panel_path, legend_text, context=context,
                                           figure_path=fig_path, panel_label=entry.get("panel_label"), frame=frame)
                    entry["series_read_source"] = "panel+shared-legend"
                    spec, ignored = _planner_series(found)
                    spec = _bind_legend_styles(spec,legend_contracts,idx)
                    entry['legend_contracts'] = legend_contracts
                    if spec:
                        entry["series_spec"] = spec
                        entry["extraction_plan"] = {
                            "chart_family": found.get("chart_family"),
                            "series_count": len(spec),
                            "reported_series_count": found.get("series_count"),
                            "exclude_regions": found.get("exclude_regions") or [],
                        }
                        entry["ignore"] = ignored
                        for s in spec:
                            a = s.get("anchors") or []
                            log(f"      模型指出曲线: {str(s.get('label'))[:26]:<26}"
                                f" 线色={s.get('line_color') or s.get('color')}"
                                f" 标记色={s.get('marker_color')} 角色={s.get('role')}"
                                f" 画法={s.get('draw') or s.get('linestyle')}"
                                f" 锚点={len(a)}", detail=True)
                        for ig in entry["ignore"]:
                            log(f"      模型指出忽略: {str(ig.get('what'))[:44]}"
                                f"（{str(ig.get('why'))[:40]}）", detail=True)
                except vlmc.VLMError as exc:
                    log(f"      曲线定位失败: {exc}")
                except Exception as exc:  # noqa: BLE001
                    log(f"      曲线定位异常: {type(exc).__name__}: {exc}")

            # Not-a-line-chart filter: needs both no readable ticks AND no curve-like
            # pixel content. Photos/schematics fail on both counts; a line chart only
            # has to pass one of them.
            coverage, thickness = curve_content(panel_img, frame, panel_legend_colors)
            entry["curve_content"] = {"coverage": round(coverage, 3),
                                      "median_thickness": round(thickness, 2)}
            curve_ok = coverage >= 0.75 and thickness >= 2
            # 第二重证据 = 两轴范围确实由模型读出来了。模型读数已经过子图轴核对；
            # "是不是数据图"由最前面的分类把关。
            vlm_ok = vlm is not None and all(
                s == "vlm"
                for s in (entry.get("axis_src_x"), entry.get("axis_src_y")))
            if not (curve_ok or vlm_ok):
                entry["skip"] = True
                entry["skip_reason"] = (
                    f"非折线图（曲线覆盖率 {coverage:.2f}/厚度 {thickness:.1f} 未达标，"
                    f"且没有可信的轴读数；"
                    f"疑似照片/示意图/表格）")
                log(f"      跳过：{entry['skip_reason']}")
            elif not curve_ok:
                entry["needs_review"] = True
                log(f"      注意：曲线覆盖率 {coverage:.2f}、厚度 {thickness:.1f}，"
                    f"系列可能不连续或不是折线图，建议人工确认")
            config["panels"].append(entry)

    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if cache_dir is not None:
        try:
            _save_analysis_cache(cache_dir, outdir, pdf_path, config)
            log(f"分析结果已缓存：{cache_dir.name[:12]}（下次同 PDF/同需求可直接复用）")
        except Exception as exc:  # noqa: BLE001 - cache must never block analysis
            log(f"（分析缓存写入失败，不影响本次结果：{type(exc).__name__}: {exc}）")
    write_report(config, outdir / "report.md", phase="analyze")
    log(f"\n共 {len(config['panels'])} 个面板，耗时 {_elapsed(t0)} -> {cfg_path}")
    if want and not keep:
        log(f"没提取任何数据：请查看 report.md 中的检索状态和候选清单（需求「{want}」）。")
    elif want:
        log(f"按需求筛出 {len(keep)} 张图（其余图未处理）。可核对 report.md 里的选择依据。")
    elif report_usage:
        log("下一步：核对 JSON 里的 axis 数值，把 confirmed 改成 true（或用 run.py confirm），再跑 extract")
    if report_usage:
        print_vlm_usage(vlm, "analyze", vlm_usage0)
    return cfg_path


def write_report(config, path, phase, results=None):
    lines = [f"# 提取报告 ({phase})", "", f"PDF: `{config['pdf']}`", ""]
    sel = config.get("selection")
    if sel:
        lines += ["## 本次需求", "",
                  f"- 需求: {sel.get('want')}",
                  f"- 判定方式: {sel.get('method')}",
                  f"- 命中: {len(sel.get('ids') or [])} 张图"]
        status = "有匹配" if sel.get("ids") else "没有匹配（全部候选明确不符）" \
            if sel.get("missing") is True else "暂时无法确认（证据不足或调用失败）"
        lines += [f"- 检索状态: {status}",
                  f"- 原图核对: {sel.get('inspected_n', 0)} / {len(sel.get('candidates') or [])} 张",
                  f"- 待复核候选: {len(sel.get('unresolved_ids') or [])} 张"]
        if sel.get("reason"):
            lines.append(f"- 依据: {sel['reason']}")
        if sel.get("body_used"):
            lines.append(f"- 正文通道: 查了 {sel.get('body_chars', 0)} 字符的"
                         f"「提到图的段落」" + (f"，依据: {sel['body_reason']}"
                                              if sel.get("body_reason") else ""))
        if sel.get("available"):
            lines.append(f"- 这篇实际有: {sel['available']}")
        lines.append("")
        if sel.get("candidates"):
            head = ["#", "图号", "页", "图注", "选中", "核对状态", "轴/工况证据"]
            lines += ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
            for c in sel["candidates"]:
                row = [str(c["n"]), c["label"] or "-", str(c["page"]),
                       c["caption"][:70].replace("|", "/"), "✅" if c["selected"] else ""]
                inspection = c.get("inspection") or {}
                evidence = []
                for panel in inspection.get("panels") or []:
                    evidence.append(f"{panel.get('label') or ''}: x={panel.get('x_title')}; "
                                    f"y={panel.get('y_titles')}; {panel.get('note') or ''}")
                evidence.append(str(inspection.get("error") or inspection.get("note") or ""))
                row.extend([str(c.get("status") or "本地图号定位"),
                            " / ".join(evidence).replace("|", "/").replace("\n", " ")[:600]])
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    for p in config["panels"]:
        ax = p["axis"]
        if p.get("skip"):
            conf = "⏭️ 已跳过"
        else:
            conf = "✅ 已确认" if ax.get("confirmed") else "⚠️ 未确认"
        lines += [
            f"## {p['id']}  (p{p['page']}, {conf})",
            "",
            f"- 提取方式: {'矢量（直接从 PDF 路径读取）' if p.get('method') == 'vector' else '位图（图例取色 + 图像处理）'}",
            f"- 图注: {p['caption'][:120] or '(未匹配到图注)'}",
            (f"- 矢量区域: {p['vector_bbox']}（第 {p['page']} 页）" if p.get("method") == "vector"
             else f"- 面板图: `{p.get('panel_image', '')}`"),
            f"- 坐标框: {p['frame']}",
            f"- 图例颜色: {', '.join(p['legend_colors']) or '(未检出)'}",
            f"- 轴范围: x={ax.get('x')}  y={ax.get('y')}"
            + ("  数量级标注: "
               + "、".join(f"{k}×{float(v):g}" for k, v in (ax.get("multiplier") or {}).items())
               + "（CSV 里的数值已按此换算）" if ax.get("multiplier") else ""),
        ]
        if p.get("panel_label"):
            lines.append(f"- 面板语义: {p['panel_label']}")
        split = p.get("panel_split") or {}
        if split:
            lines.append(f"- 面板定位: {split.get('source') or '?'}"
                         f"（模型预期 {split.get('expected') or '?'}，"
                         f"像素初检 {split.get('pixel_detected', '?')}）")
        if ax.get("x_title") or ax.get("y_title"):
            chk = p.get("axis_check") or {}
            verdict = ("✅ 符合" if chk.get("matches") is True else
                       "❌ 不符合（已跳过）" if chk.get("matches") is False else
                       "⚠️ 看不清（保留）" if chk else "")
            lines.append(f"- 轴标题: x={ax.get('x_title') or '?'}"
                         + (f" [{ax.get('x_unit')}]" if ax.get("x_unit") else "")
                         + f"  y={ax.get('y_title') or '?'}"
                         + (f"  需求核对: {verdict} — {str(chk.get('note') or '')[:80]}"
                            if verdict else ""))
        if p.get("skip") and p.get("skip_reason"):
            lines.append(f"- ⏭ 跳过原因: {p['skip_reason']}")
        spec = p.get("series_spec")
        if spec:
            lines.append("- 模型指出的曲线（锚点只是种子，数值由代码从锚点追踪得出）:")
            for s in spec:
                lines.append(f"  - {s.get('label') or '(无图例条目)'} ｜ 颜色={s.get('color')} "
                             f"｜ 线型={s.get('linestyle')} ｜ 标记={s.get('has_markers')} "
                             f"｜ 输出={s.get('output')} ｜ 锚点 {len(s.get('anchors') or [])} 个"
                             + (f" ｜ {s['note']}" if s.get("note") else ""))
        for ig in (p.get("ignore") or []):
            lines.append(f"  - 忽略：{ig.get('what')}（{ig.get('why')}）")
        if results and p["id"] in results:
            r = results[p["id"]]
            lines.append(f"- 提取结果: {len(r['series'])} 条曲线")
            for s in r["series"]:
                extra = []
                if s.get("geometry") == "loop":
                    extra.append("闭合回线（沿轮廓取点，不是按 x 排序）")
                if s.get("dark"):
                    extra.append("黑/灰线（按深色笔画追）")
                if s.get("color_src"):
                    extra.append(f"颜色取自{s['color_src']}")
                if s.get("anchors_hit") is not None:
                    extra.append(f"锚点命中 {s['anchors_hit']} 个"
                                 + (f"，中位偏差 {s['anchor_med_dist']}px"
                                    if s.get("anchor_med_dist") is not None else ""))
                if s.get("duplicate_note"):
                    extra.append("⚠️ " + s["duplicate_note"])
                if s.get("draw_note"):
                    extra.append(s["draw_note"])
                extra.extend("⚠️ " + str(warning) for warning in s.get("warnings") or [])
                comp = s.get("completion") or {}
                if comp.get("bridged_cols"):
                    bys = sorted({str(x.get("by")) for x in (comp.get("spans") or [])})
                    extra.append(f"补全 {comp['bridged_cols']} 列"
                                 + (f"（{('、'.join(bys[:3]))}）" if bys else ""))
                if comp.get("open_gaps"):
                    cols = sum(int(x["cols"]) for x in comp["open_gaps"])
                    extra.append(f"⚠️ {len(comp['open_gaps'])} 段长断口未补"
                                 f"（共 {cols} 列，没有别的笔画经过，保持断开）")
                if comp.get("model_anchors"):
                    extra.append(f"🧠 {comp['model_anchors'].get('by')}"
                                 f"（+{comp['model_anchors'].get('cols')} 列）")
                lines.append(f"  - `{s['file']}`  {s['points']} 点, "
                             f"y=[{s['y_range'][0]}, {s['y_range'][1]}]"
                             + ("（" + "；".join(extra) + "）" if extra else ""))
            for f in (r.get("failed") or []):
                lines.append(f"  - ✗ 未追到 `{f.get('label')}`：{f.get('reason')}")
            for a in (r.get("aliases") or []):
                why = a.get("why") or "轨迹完全相同"
                lines.append(f"  - ⇄ 合并：`{a.get('label')}` 与 `{a.get('alias_of')}` "
                             f"是同一条曲线（{why}，重合度 {a.get('overlap')}）——只出一份数据")
            rv = r.get("review") or {}
            if rv:
                lines.append(f"  - 模型复盘：merge={rv.get('merge') or []} "
                             f"drop={rv.get('drop') or []} "
                             f"add={len(rv.get('add') or [])} ｜ {str(rv.get('note'))[:70]}")
            params = r.get("template_params") or {}
            if params:
                mode = r.get("template_param_mode") or "legacy"
                lines.append(f"- 模板参数 Agent（{r.get('template_param_calls', 0)} 次调用，"
                             f"模式={mode}）:")
                fallback_reason = r.get("template_param_fallback_reason")
                if fallback_reason:
                    lines.append(f"  - ⚠️ 论文知识链回退到旧参数读取: "
                                 f"{str(fallback_reason)[:240]}")
                binding = r.get("template_param_binding") or {}
                if binding:
                    global_conditions = binding.get("global_condition_ids") or []
                    series_conditions = {
                        name: value.get("condition_ids") or []
                        for name, value in (binding.get("series") or {}).items()
                        if isinstance(value, dict) and value.get("condition_ids")}
                    lines.append("  - 工况绑定: global=" + str(global_conditions)
                                 + (f"；series={series_conditions}" if series_conditions else ""))
                added_facts = r.get("template_param_panel_facts_added") or []
                if added_facts:
                    lines.append("  - 当前图明确事实已写入论文知识库: "
                                 + ", ".join(str(item) for item in added_facts))
                evidence = r.get("template_param_evidence") or {}
                basis = r.get("template_param_basis") or {}
                fact_ids = r.get("template_param_fact_ids") or {}
                raw_values = r.get("template_param_raw") or {}
                conflicts = r.get("template_param_conflicts") or {}
                for name, kv in params.items():
                    values = "；".join(f"{k}={v}" for k, v in kv.items() if v not in (None, ""))
                    lines.append(f"  - {name}: {values or '未找到可唯一归属的参数'}")
                    for key, quote in (evidence.get(name) or {}).items():
                        how = (basis.get(name) or {}).get(key)
                        fact_id = (fact_ids.get(name) or {}).get(key)
                        raw = (raw_values.get(name) or {}).get(key) or {}
                        raw_text = " ".join(str(raw.get(part) or "").strip()
                                            for part in ("value", "unit")).strip()
                        tags = [item for item in (how,
                                                  f"fact={fact_id}" if fact_id else "",
                                                  f"raw={raw_text}" if raw_text else "") if item]
                        tag_text = f" [{', '.join(tags)}]" if tags else ""
                        lines.append(f"    - {key}{tag_text}: {str(quote)[:180]}")
                    for key, competing in (conflicts.get(name) or {}).items():
                        lines.append(f"    - ⚠️ {key} 换算后仍有冲突，已留空: "
                                     + ", ".join(str(item) for item in competing))
            lines.append(f"- 质检图: `verify/{p['id']}_verify.png`")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def make_verify_image(panel_path, frame, axis, legend_colors, series_files, out_path,
                      failed=None, anchors=None, bridged=None, series_meta=None):
    """Thin wrapper so callers do not have to build the series tuples themselves."""
    axis = dict(axis)
    multiplier = axis.get("multiplier") or {}
    for key in ("x", "y"):
        axis[key] = [float(v) * float(multiplier.get(key, 1)) for v in axis[key]]
    series,styles = [],{}
    meta_by_file = {s.get("file"):s for s in series_meta or []}
    for csv_path, color_hex in zip(series_files, legend_colors):
        xs, ys = vo.read_csv(csv_path)
        meta = meta_by_file.get(Path(csv_path).name) or {}
        label = str(meta.get("series_name") or Path(csv_path).stem)
        series.append((label, color_hex, xs, ys))
        styles[label] = meta.get("kind")
    # 模型指出了但代码没追到的曲线：把它的锚点圈出来，肉眼即可判断是模型错还是代码错
    marks = []
    if failed:
        left, top, right, bottom = (float(v) for v in frame)
        for f in failed:
            for a in (f.get("anchors") or []):
                try:
                    x = left + float(a[0]) * (right - left)
                    y = bottom - float(a[1]) * (bottom - top)
                except (TypeError, ValueError, IndexError):
                    continue
                marks.append(([int(x) - 10, int(y) - 10, int(x) + 10, int(y) + 10],
                              f"missed:{str(f.get('label'))[:12]}"))
    vo.compose_overlay(panel_path, frame, tuple(axis["x"]), tuple(axis["y"]),
                       series, out_path, annotations=marks, anchors=anchors,
                       bridged=bridged, styles=styles)


def _write_template_output(fmt, res, panel, panel_path, csv_dir, outdir, vlm, log,
                           paper_path=None, param_state=None):
    """把这一面板的曲线按用户模板再输出一份（模板代码由模型写一次，之后复用）。"""
    # 同一个系列可能有两个文件（标记点 + 线）：模板里一条曲线只该出现一次，
    # 取**线**（它包含被遮挡段补出来的点）；只出点的那种取点。
    picked = {}
    for s in res["series"]:
        xs, ys = vo.read_csv(csv_dir / s["file"])
        if not xs:
            continue
        name = str(s.get("series_name") or Path(s["file"]).stem)
        rank = 1 if s.get("kind") == "line" else 0
        old = picked.get(name)
        if old is None or rank > old[0] or (rank == old[0] and len(xs) > len(old[1]["x"])):
            picked[name] = (rank, {"name": name, "file": s["file"], "x": xs, "y": ys,
                                   "params": {}})
    series = [v[1] for v in picked.values()]
    if not series:
        return []
    context = {"paper": panel_path.parent.parent.name, "panel": panel["id"],
               "ext": fmt.ext, "template": fmt.name}
    # 模板声明的参数 + 代码实际用到的参数（模型有时只列了"按曲线"的那部分，
    # 模板里的工况块就被漏掉 -> 渲染 KeyError。让代码自己说它要什么）
    keys = list(dict.fromkeys(fmt.params + tpl.probe_params(fmt, series, context)))
    vals = {}
    if keys:
        evidence = tpl.paper_param_context(
            paper_path, page=panel.get("page"), caption=panel.get("caption") or "", keys=keys)
        param_answer = tpl.ask_params_with_knowledge(
            vlm, panel_path, panel.get("caption"), series, keys,
            hints=tpl.params_hints(fmt, keys), evidence=evidence,
            paper_path=paper_path,
            knowledge_cache=Path(outdir) / "condition_knowledge.json",
            state=param_state, return_meta=True)
        # Keep the rendering path tolerant of third-party/older parameter agents
        # that still return only the values mapping.  Metadata is additive and must
        # never prevent a template file from being produced.
        if (isinstance(param_answer, tuple) and len(param_answer) == 2
                and isinstance(param_answer[0], dict)):
            vals, param_meta = param_answer
        else:
            vals = param_answer if isinstance(param_answer, dict) else {}
            param_meta = {"evidence": {}, "basis": {}, "calls": 0}
        res["template_params"] = vals
        res["template_param_evidence"] = param_meta.get("evidence") or {}
        res["template_param_basis"] = param_meta.get("basis") or {}
        res["template_param_calls"] = int(param_meta.get("calls") or 0)
        res["template_param_mode"] = param_meta.get("mode") or "legacy"
        res["template_param_binding"] = param_meta.get("binding") or {}
        res["template_param_fact_ids"] = param_meta.get("fact_ids") or {}
        res["template_param_raw"] = param_meta.get("raw") or {}
        res["template_param_conflicts"] = param_meta.get("conflicts") or {}
        res["template_param_panel_facts_added"] = param_meta.get("panel_facts_added") or []
        res["template_param_fallback_reason"] = param_meta.get("fallback_reason") or ""
        n_filled = sum(1 for by_series in vals.values() for v in by_series.values()
                       if v not in (None, ""))
        log(f"      参数 Agent：填出 {n_filled}/{len(series) * len(keys)} 项"
            f"（{res['template_param_calls']} 次调用）")
    for item in series:
        item["params"] = {k: str((vals.get(item["name"]) or {}).get(k, "")) for k in keys}
    try:
        files = fmt.render(series, context) or {}
    except Exception as exc:  # noqa: BLE001 - 模板坏了不该拖垮提取
        # 模板里写的是 %.E 这类数值格式，而模型读回来的是字符串 -> 能转数字的转数字，
        # 没读到的填空串会报错，就退成 0，再试一次（绝不因为参数格式把整份输出丢掉）
        coerced, blanked = 0, 0
        for item in series:
            fixed = {}
            for k, v in item["params"].items():
                if v == "":
                    fixed[k], blanked = 0.0, blanked + 1
                    continue
                try:
                    fixed[k] = float(str(v).strip())
                    coerced += 1
                except (TypeError, ValueError):
                    fixed[k] = v
            item["params"] = fixed
        try:
            files = fmt.render(series, context) or {}
            log(f"      （模板要数值：{coerced} 个参数转成数字"
                + (f"，{blanked} 个没读到按 0 填" if blanked else "")
                + "，渲染继续）")
        except Exception as exc2:  # noqa: BLE001
            log(f"      （模板渲染失败，跳过：{type(exc2).__name__}: {exc2}）")
            return []
    dest = Path(outdir) / "templates" / Path(str(fmt.name)).stem
    for fname, text in files.items():
        p = dest / Path(str(fname)).name               # 只取文件名，防目录穿越
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(text), encoding="utf-8")
    log(f"      模板输出 {len(files)} 个文件 -> {dest}")
    return [Path(str(fname)).name for fname in files]


def extract(cfg_path, force=False, only=None, vlm=None, template=None, report_usage=True):
    """template: 文件路径 = 用这个模板；None = 沿用上次的模板；False = 不用模板（只出 CSV）。"""
    t0 = time.time()
    cfg_path = Path(cfg_path).resolve()
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    vlm_usage0 = dict(vlm.usage) if vlm is not None else None
    outdir = Path(config.get("outdir") or cfg_path.parent)
    _clean_stale_intermediates(outdir)
    csv_dir = outdir / "csv"
    verify_dir = outdir / "verify"
    csv_dir.mkdir(parents=True, exist_ok=True)
    verify_dir.mkdir(parents=True, exist_ok=True)
    old_results = {}
    old_results_path = outdir / "extract_results.json"
    if old_results_path.exists():
        try:
            old_results = json.loads(old_results_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old_results = {}
    fmt = None
    param_state = {}
    if template is not False:
        try:
            fmt = tpl.load(template, vlm=vlm, log=detail_log)
        except Exception as exc:  # noqa: BLE001 - 模板不可用不该拖垮提取
            log(f"      （模板不可用，只出 CSV：{type(exc).__name__}: {exc}）")

    if fmt:
        template_dir = outdir / "templates" / Path(str(fmt.name)).stem
        if not only:
            for f in template_dir.glob("*") if template_dir.exists() else []:
                if f.is_file():
                    f.unlink()
        else:
            for panel_id in only:
                for name in (old_results.get(panel_id) or {}).get("template_files") or []:
                    target = template_dir / Path(str(name)).name
                    if target.is_file():
                        target.unlink()

    # 重跑时先清掉这次要覆盖的旧产物：否则新旧结果混在一个目录里，
    # 分不清哪个 CSV 是这一次的（实测把这批论文重跑时踩到过）
    if not only:
        for d in (csv_dir, verify_dir):
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink()
    else:
        for p in config["panels"]:
            if p["id"] not in only:
                continue
            # Bitmap panels are keyed by their image stem; vector panels have no
            # panel_image and write files with the panel id directly.  Falling back
            # to the id is essential here, otherwise a targeted vector rerun leaves
            # obsolete CSVs from the previous extraction beside the new result.
            stem = Path(p.get("panel_image") or "").stem or str(p.get("id") or "")
            for d in (csv_dir, verify_dir):
                for f in list(d.glob(f"{stem}*")) if stem else []:
                    f.unlink()

    results = {}
    pdf_doc = None
    for p in config["panels"]:
        if only and p["id"] not in only:
            continue
        ax = p["axis"]
        if p.get("skip"):
            log(f"跳过 {p['id']}（{p.get('skip_reason', '已标记跳过')}）")
            continue

        # ---- vector figure: read the polylines straight out of the PDF ----
        if p.get("method") == "vector":
            if pdf_doc is None:
                pdf_doc = tp.pdfplumber.open(config["pdf"])
            page = pdf_doc.pages[p["page"] - 1]
            vres = vx.extract_vector_figure(page, tuple(p["vector_bbox"]), csv_dir, p["id"],
                                            series_spec=p.get("series_spec"))
            series = [{
                "file": s["file"], "points": s["n_points"], "color": s["color"],
                "series_name": s.get("series_name"), "source": s.get("source"),
                "x_range": [round(v, 6) for v in s["x_range"]],
                "y_range": [round(v, 6) for v in s["y_range"]],
            } for s in vres.get("series", [])]
            results[p["id"]] = {"series": series, "method": "vector",
                                "frame": vres.get("axes"), "axis_range": ax}
            log(f"{p['id']} [矢量·直接读取]: {len(series)} 条曲线")
            crop_path = outdir / "figures" / f"{p['id']}.png"
            try:
                scale = 300 / 72.0
                bbox = p["vector_bbox"]
                page.crop(tuple(bbox), strict=False).to_image(resolution=300).original.save(crop_path)
                frame_c = [(p["frame"][0] - bbox[0]) * scale, (p["frame"][1] - bbox[1]) * scale,
                           (p["frame"][2] - bbox[0]) * scale, (p["frame"][3] - bbox[1]) * scale]
                results[p["id"]].update({"panel": crop_path.name, "frame": frame_c,
                                         "frame_space": "pixels"})
                make_verify_image(crop_path, frame_c, ax,
                                  [s["color"] for s in series],
                                  [csv_dir / s["file"] for s in series],
                                  verify_dir / f"{p['id']}_verify.png")
            except Exception as exc:  # noqa: BLE001
                log(f"      （质检图渲染失败：{type(exc).__name__}: {exc}）")
            if fmt and series and crop_path.exists():
                results[p["id"]]["template_files"] = _write_template_output(
                    fmt, results[p["id"]], p, crop_path, csv_dir, outdir,
                    vlm, log, paper_path=config.get("pdf"), param_state=param_state)
            continue

        if not (ax.get("confirmed") or force):
            log(f"跳过 {p['id']}（轴范围未确认；核对后把 confirmed 设为 true，或用 --force）")
            continue
        if ax.get("x") is None or ax.get("y") is None:
            log(f"跳过 {p['id']}（缺少轴范围）")
            continue
        xr, yr = ax["x"], ax["y"]
        # 数量级标注（轴旁边的 ×10^-4 之类）：模型读出来、代码乘上去。缩放两端点等于
        # 缩放整条映射，所以在这里做一次就够了（y = ymin + t*(ymax-ymin) 是线性的）。
        mult = ax.get("multiplier") or {}
        kx = _as_float(mult.get("x"), 1.0)
        ky = _as_float(mult.get("y"), 1.0)
        xr = [xr[0] * kx, xr[1] * kx]
        yr = [yr[0] * ky, yr[1] * ky]
        rng = [xr[0], xr[1], yr[0], yr[1]]
        # dual-axis figures: give each series the y range of the axis it reads
        series_y = {}
        tol_by_range = {}
        for col, side in (p.get("series_axes") or {}).items():
            detail = (p.get("y_axes") or {}).get(side) or {}
            r = detail.get("range")
            if r:
                k = _as_float(detail.get("multiplier"), 1.0)   # 该纵轴自己的数量级
                r = [r[0] * k, r[1] * k]
                series_y[col.upper()] = r
                step = None
                labels = detail.get("labels")
                if labels and len(labels) > 1:
                    vals = sorted(float(v) for v in labels if _is_number(v))
                    if len(vals) > 1:
                        step = min(abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1))
                tol_by_range[tuple(r)] = vt.axis_tolerance(r, step)
        panel_path = Path(p["panel_image"])
        if not panel_path.is_absolute():
            panel_path = outdir / panel_path
        pdf_page = None
        if p.get('vector_panel_axes'):
            if pdf_doc is None:
                pdf_doc = tp.pdfplumber.open(config['pdf'])
            pdf_page = pdf_doc.pages[p['page']-1]
        res = _extract_panel(p, panel_path, rng, csv_dir, vlm=vlm,pdf_page=pdf_page)
        if res is None:
            results[p["id"]] = {"series": [], "error": "提取失败", "axis_range": rng}
            continue
        res["axis_range"] = rng
        # 复盘里模型如果指出"轴的映射不对"（曲线整体偏移/压平/漏了数量级），用它给的
        # 范围重跑这个面板一次：CSV 先清掉，避免留下上一轮的旧文件。
        fix = (res.get("review") or {}).get("axis_fix")
        if isinstance(fix,dict):
            fix,rejected = _validated_axis_fix(p,fix)
            if rejected:
                res['axis_fix_rejected'] = rejected
                log(f"      复盘轴修正与刻度拟合矛盾（{','.join(rejected)}），保留刻度证据映射",detail=True)
        if isinstance(fix, dict) and any(fix.get(k) for k in ("x", "y", "multiplier")):
            try:
                new_rng = list(rng)
                if isinstance(fix.get("x"), (list, tuple)) and len(fix["x"]) == 2:
                    new_rng[0], new_rng[1] = float(fix["x"][0]), float(fix["x"][1])
                if isinstance(fix.get("y"), (list, tuple)) and len(fix["y"]) == 2:
                    new_rng[2], new_rng[3] = float(fix["y"][0]), float(fix["y"][1])
                kfix = _as_float(fix.get("multiplier"), 1.0)
                if abs(kfix - 1.0) > 1e-12:
                    new_rng = [new_rng[0], new_rng[1], new_rng[2] * kfix, new_rng[3] * kfix]
                for f in csv_dir.glob(f"{panel_path.stem}*"):
                    f.unlink()
                res2 = _extract_panel(p, panel_path, new_rng, csv_dir,pdf_page=pdf_page)
                if res2:
                    log(f"      轴映射按模型复盘修正：{rng} -> {new_rng}，已重跑该面板",
                        detail=True)
                    res2["axis_range"] = new_rng
                    res2["axis_fix"] = fix
                    res = res2
            except Exception as exc:  # noqa: BLE001
                log(f"      （轴映射修正失败，保留原结果：{type(exc).__name__}: {exc}）")
        results[p["id"]] = res
        log(f"{p['id']}: {len(res.get('series', []))} 条曲线"
            + (f"  方式={ {'seeded':'锚点+追踪','vector-panel':'逐子图矢量'}.get(res.get('method'),'颜色追踪')}"
               if res.get("series") else ""), detail=True)
        for f in res.get("failed") or []:
            log(f"      ✗ 未追到：{str(f.get('label'))[:30]}（{str(f.get('reason'))[:60]}）",
                detail=True)
        if fmt and res.get("series"):
            res["template_files"] = _write_template_output(
                fmt, res, p, panel_path, csv_dir, outdir, vlm, log,
                paper_path=config.get("pdf"), param_state=param_state)
        if res.get("series"):
            files = [csv_dir / s["file"] for s in res["series"]]
            colors = []
            for s in res["series"]:
                hexs = s.get("color_hex")
                if not hexs and s.get("target_bgr"):
                    hexs = "#{:02x}{:02x}{:02x}".format(
                        s["target_bgr"][2], s["target_bgr"][1], s["target_bgr"][0])
                colors.append(hexs or "#ff00ff")
            make_verify_image(panel_path, res["frame"], ax, colors, files,
                              verify_dir / f"{p['id']}_verify.png",
                              failed=res.get("failed"), anchors=None,
                              bridged=res.get("_bridged_px"), series_meta=res.get("series"))

        # ---- VLM spot check: independent re-reading of a few points ----
        if vlm is not None and res.get("series"):
            try:
                # Positions are described visually (not as our computed x values) so the
                # model's reading is a genuinely independent cross-check of the mapping.
                spots = [0.0, 0.5, 1.0]
                queries, expected = [], {}
                qid = 0
                for s in res["series"][:2]:
                    xs, ys = vo.read_csv(csv_dir / s["file"])
                    if not xs:
                        continue
                    rng = s.get("y_axis_range")
                    s_tol = tol_by_range.get(tuple(rng)) if rng else None
                    for frac in spots:
                        qid += 1
                        target = xs[0] + frac * (xs[-1] - xs[0])
                        i = min(range(len(xs)), key=lambda k: abs(xs[k] - target))
                        bgr = s.get("target_bgr")
                        hexs = ("#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])).upper() \
                            if bgr else None
                        side = (p.get("series_axes") or {}).get(hexs) if hexs else None
                        queries.append({"id": qid, "x": target, "tol": s_tol,
                                        "axis": side if len(p.get("y_axes") or {}) > 1 else None,
                                        "color":s.get('trace_color_hex') or s.get('color_hex') or hexs,
                                        "draw":s.get('draw') or s.get('kind'),
                                        "series": s.get("series_name") or s.get("label")})
                        expected[qid] = ys[i]
                if queries:
                    figure_path = Path(p.get('figure') or '')
                    if not figure_path.is_absolute():
                        figure_path = outdir/figure_path
                    raw = vt.spot_check(vlm, panel_path, queries,
                                        figure_path=figure_path if figure_path.is_file() else None)
                    # tolerance from the chart's own tick spacing, not a % of the range
                    step = None
                    axis_sources = [(p.get("vlm_axis") or {}).get("y")]
                    axis_sources.extend((p.get("y_axes") or {}).values())
                    for src in axis_sources:
                        if isinstance(src, dict) and src.get("step"):
                            step = abs(src["step"]) if step is None else min(step, abs(src["step"]))
                        labels = (src or {}).get("labels") if isinstance(src, dict) else None
                        if labels and len(labels) > 1:
                            vals = sorted(float(v) for v in labels if _is_number(v))
                            if len(vals) > 1:
                                gaps = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
                                g = min(gaps)
                                step = g if step is None else min(step, g)
                    tol = vt.axis_tolerance(ax["y"], step)
                    tol_by_id = {q["id"]: q.get("tol") for q in queries}
                    cmp = vt.compare_spot_check(raw, expected, tol, tol_by_id)
                    res["spot_check"] = cmp
                    if cmp['n_compared'] and cmp['n_agree']<cmp['n_compared']:
                        for s in res['series']:
                            s['measurement_status']='needs_review'
                            s.setdefault('warnings',[]).append('独立图像抽查存在不一致，需核对数值映射与系列身份')
                    tols = sorted({c.get("tol") for c in cmp["checks"] if c.get("tol")})
                    log(f"      VLM 抽查: {cmp['n_agree']}/{cmp['n_compared']} 一致"
                        f"（容差 ±{'/'.join(f'{t:g}' for t in tols) or f'{tol:g}'}）"
                        + (f"  ⚠ 存在不一致" if cmp["n_compared"] and
                           cmp["n_agree"] < cmp["n_compared"] else ""), detail=True)
            except vlmc.VLMError as exc:
                log(f"      VLM 抽查失败: {exc}")
            except Exception as exc:  # noqa: BLE001
                log(f"      VLM 抽查异常: {type(exc).__name__}: {exc}")

        failed_n = len(res.get("failed") or [])
        spot = res.get("spot_check") or {}
        spot_text = (f"，抽查 {spot.get('n_agree', 0)}/{spot.get('n_compared', 0)}"
                     if spot else "")
        log(f"{p['id']}: {len(res.get('series', []))} 条曲线，"
            f"{failed_n} 条未追到{spot_text}")

    for r in results.values():                 # 私有字段（像素轨迹）不进 JSON
        for k in [k for k in r if k.startswith("_")]:
            r.pop(k, None)
    (outdir / "extract_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if pdf_doc is not None:
        pdf_doc.close()
    write_report(config, outdir / "report.md", phase="extract", results=results)
    total = sum(len(r.get("series", [])) for r in results.values())
    log(f"完成：{total} 条曲线，耗时 {_elapsed(t0)}；CSV={csv_dir}；"
        f"质检图={verify_dir}；报告={outdir / 'report.md'}")
    if report_usage:
        print_vlm_usage(vlm, "extract", vlm_usage0)
    return results


def _run_one(pdf, sub, row, dpi, pages, force, vlm, want=None, want_deep=False,
             find_series=True, template=None):
    """One PDF end-to-end (analyze + extract). Shared by serial and parallel batch."""
    cfg = analyze(pdf, sub, dpi=dpi, vlm=vlm, pages=pages, want=want,
                  want_deep=want_deep, find_series=find_series, report_usage=False)
    config = json.loads(Path(cfg).read_text(encoding="utf-8"))
    if config.get("selection"):
        row["matched"] = len(config["selection"].get("ids") or [])
        row["retrieval_missing"] = config["selection"].get("missing")
    row["panels"] = len([p for p in config["panels"] if not p.get("skip")])
    row["panels_skipped"] = len([p for p in config["panels"] if p.get("skip")])
    results = extract(cfg, force=force, vlm=vlm, template=template, report_usage=False) or {}
    row["curves"] = sum(len(r.get("series", [])) for r in results.values())
    for r in results.values():
        sc = r.get("spot_check") or {}
        row["spot_ok"] += sc.get("n_agree", 0)
        row["spot_n"] += sc.get("n_compared", 0)
    row["panels_extracted"] = len(results)
    return row


def new_row(pdf, sub):
    return {"pdf": Path(pdf).name, "outdir": str(sub), "curves": 0, "panels": 0,
            "panels_skipped": 0, "spot_ok": 0, "spot_n": 0}


def _log_row(row, with_vlm):
    u = row.get("usage") or {}
    log(f"  本文件小结: {row.get('curves', 0)} 条曲线 | 抽查 "
        f"{row['spot_ok']}/{row['spot_n']} | "
        + (f"VLM {u.get('calls', 0)} 次调用/{u.get('prompt_tokens', 0)} tokens"
           if with_vlm else "未启用 VLM")
        + f" | 耗时 {row.get('elapsed', '?')}s"
        + (f" | ✗ {row['error']}" if row.get("error") else ""))


def _batch_one(pdf_str, sub_str, dpi, pages, force, vlm_spec, want=None,
               want_deep=False, find_series=True, template=None,
               verbose=False, keep_intermediates=False):
    """Worker body for --jobs > 1: one PDF per process.

    Output goes to that paper's own run.log - several processes printing to one terminal
    interleaves into unreadable text. VLM usage is returned with the row and summed by
    the parent, because a client built in a child cannot report back through pickling.
    """
    global _LOG_FILE, _VERBOSE, _KEEP_INTERMEDIATES
    _VERBOSE = bool(verbose)
    _KEEP_INTERMEDIATES = bool(keep_intermediates)
    pdf, sub = Path(pdf_str), Path(sub_str)
    sub.mkdir(parents=True, exist_ok=True)
    row = new_row(pdf, sub)
    t_file = time.time()
    client = None
    try:
        if vlm_spec:
            client = vlmc.DeepSeekVLM(model=vlm_spec["model"],
                                      base_url=vlm_spec["base_url"])
            if not client.api_key:
                client = None
        _LOG_FILE = ((sub / "run.log").open("w", encoding="utf-8")
                     if _VERBOSE else io.StringIO())
        _run_one(pdf, sub, row, dpi, pages, force, client, want, want_deep,
                 find_series, template)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        log(f"  ✗ 失败: {row['error']}")
        if isinstance(exc, ImportError):
            log("     ↑ 缺依赖（常见于用 --no-deps 装的包）。修：pip install -r requirements.txt")
    finally:
        if _LOG_FILE is not None:
            _LOG_FILE.close()
            _LOG_FILE = None
    if client is not None:
        row["usage"] = dict(client.usage)
    row["elapsed"] = round(time.time() - t_file, 1)
    return row


def _batch_serial(pdfs, outdir, dpi, pages, force, vlm, want=None,
                  want_deep=False, find_series=True, template=None):
    rows = []
    for i, pdf in enumerate(pdfs, start=1):
        t_file = time.time()
        sub = outdir / pdf.stem
        log(f"\n[{i}/{len(pdfs)}] {pdf.name}")
        before = dict(vlm.usage) if vlm is not None else None
        row = new_row(pdf, sub)
        try:
            _run_one(pdf, sub, row, dpi, pages, force, vlm, want, want_deep,
                     find_series, template)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            log(f"  ✗ 失败: {row['error']}")
            if isinstance(exc, ImportError):
                log("     ↑ 缺依赖（常见于用 --no-deps 装的包）。"
                    "修：pip install -r requirements.txt")
        if before is not None:
            row["usage"] = {k: vlm.usage.get(k, 0) - before.get(k, 0) for k in vlm.usage}
        row["elapsed"] = round(time.time() - t_file, 1)
        rows.append(row)
        _log_row(row, before is not None)
    return rows


def _batch_parallel(pdfs, outdir, jobs, dpi, pages, force, vlm, want=None,
                    want_deep=False, find_series=True, template=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    spec = {"model": vlm.model, "base_url": vlm.base_url} if vlm is not None else None
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(_batch_one, str(pdf), str(outdir / pdf.stem), dpi, pages,
                             force, spec, want, want_deep,
                             find_series, template, _VERBOSE,
                             _KEEP_INTERMEDIATES): pdf for pdf in pdfs}
        for fut in as_completed(futures):
            pdf = futures[fut]
            done += 1
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001 (spawn/pickle failure)
                row = new_row(pdf, outdir / pdf.stem)
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            log(f"[{done}/{len(pdfs)}] {pdf.name}")
            _log_row(row, spec is not None)
    rows.sort(key=lambda r: r["pdf"])
    return rows


def batch(dir_path, outdir, dpi=300, vlm=None, pages=None, force=False,
          pattern="*.pdf", recursive=False, jobs=1, want=None,
          want_deep=False, find_series=True, template=None):
    """Process every PDF in a folder, then write a combined summary.

    jobs=1 keeps everything in one process (shared VLM client and caches).
    jobs>1 runs one paper per process: the VLM calls are network-bound, so this is where
    the wall-clock speedup comes from. Each paper logs to <out>/<pdf>/run.log.
    """
    src = Path(dir_path).resolve()
    t_batch = time.time()
    pdfs = sorted(src.rglob(pattern) if recursive else src.glob(pattern))
    pdfs = [p for p in pdfs if p.is_file()]
    if not pdfs:
        log(f"目录里没有匹配 {pattern} 的文件: {src}")
        return
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = max(1, int(jobs or 1))
    log(f"批量处理 {len(pdfs)} 个 PDF -> {outdir}"
        + (f"   并行 {jobs} 进程，每篇日志见 <out>/<pdf>/run.log" if jobs > 1 else "")
        + (f"\n需求：{want}" if want else "")
        + "\n" + "=" * 66)

    if jobs > 1:
        rows = _batch_parallel(pdfs, outdir, jobs, dpi, pages, force, vlm, want,
                               want_deep, find_series, template)
        # 用量在子进程里累计，父进程按行加总后再打印
        if vlm is not None:
            for k in vlm.usage:
                vlm.usage[k] = sum(r.get("usage", {}).get(k, 0) for r in rows)
    else:
        rows = _batch_serial(pdfs, outdir, dpi, pages, force, vlm, want,
                             want_deep, find_series, template)

    md = ["# 批量提取汇总", "",
          f"- 来源目录: `{src}`", f"- 处理文件: {len(pdfs)} 个", ""]
    if vlm is not None:
        md += [f"- VLM 合计: {vlm.usage['calls']} 次调用（本地缓存命中 "
               f"{vlm.usage['cache_hits']} 次），输入 {vlm.usage['prompt_tokens']:,} / "
               f"输出 {vlm.usage['completion_tokens']:,} tokens", ""]
    extra_col = ["命中图"] if want else []
    md += ["| PDF | " + " | ".join(extra_col + ["面板", "跳过", "提取面板", "曲线",
                                                "抽查一致", "VLM调用", "输入tokens"]) + " |",
           "|---" * (7 + len(extra_col)) + "|"]
    for r in rows:
        u = r.get("usage") or {}
        name = r["pdf"] + (f" ⚠ {r['error']}" if r.get("error") else "")
        mid = f"{r.get('matched', '-')} | " if want else ""
        md.append(f"| {name} | {mid}{r['panels']} | {r['panels_skipped']} | "
                  f"{r.get('panels_extracted', 0)} | {r['curves']} | "
                  f"{r['spot_ok']}/{r['spot_n']} | {u.get('calls', '-')} | "
                  f"{u.get('prompt_tokens', '-')} |")
    tot_c = sum(r["curves"] for r in rows)
    tot_ok = sum(r["spot_ok"] for r in rows)
    tot_n = sum(r["spot_n"] for r in rows)
    md += ["", f"**合计**: {tot_c} 条曲线，抽查一致 {tot_ok}/{tot_n}"
           + (f"（{100 * tot_ok / tot_n:.0f}%）" if tot_n else "")]
    (outdir / "batch_summary.md").write_text("\n".join(md), encoding="utf-8")
    (outdir / "batch_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    log("\n" + "=" * 66)
    log(f"批量完成: {len(pdfs)} 个文件, 共 {tot_c} 条曲线, "
        f"抽查一致 {tot_ok}/{tot_n}，总耗时 {_elapsed(t_batch)}")
    log(f"汇总 -> {outdir / 'batch_summary.md'}")
    if tot_c == 0 and vlm is None and not force:
        log("提示：一条曲线都没提取。轴范围未经确认时 extract 会跳过所有面板——"
            "位图任务请加 --vlm 读取轴范围；矢量图自带坐标；也可用 --force 信任已有轴范围。")
    elif tot_c == 0 and want:
        hit = [r["pdf"] for r in rows if r.get("matched")]
        log(f"提示：没有提取到曲线。需求「{want}」在"
            + (f" {len(hit)} 篇里命中（{', '.join(hit[:3])}…），但没追出曲线；"
               if hit else "检索未得到确认匹配，请区分明确不符与待复核候选；")
            + "各篇的候选清单见 <out>/<PDF>/report.md 的「本次需求」一节。")
    if vlm is not None:
        print_vlm_usage(vlm, "批量合计")


def confirm(cfg_path, panel_id, x, y, skip=False):
    cfg_path = Path(cfg_path).resolve()
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    hit = False
    for p in config["panels"]:
        if p["id"] != panel_id:
            continue
        hit = True
        if skip:
            p["skip"] = True
            p["skip_reason"] = "人工标记跳过"
        if x:
            p["axis"]["x"] = [float(v) for v in x.split(",")]
        if y:
            p["axis"]["y"] = [float(v) for v in y.split(",")]
        if p["axis"].get("x") and p["axis"].get("y"):
            p["axis"]["confirmed"] = True
            # an explicit confirmation overrides an automatic skip
            p.pop("skip", None)
            p.pop("skip_reason", None)
    if not hit:
        raise SystemExit(f"panel id not found: {panel_id}")
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"已更新 {panel_id}: x={x} y={y}" + ("  [已标记跳过]" if skip else ""))


def main():
    global _VERBOSE, _KEEP_INTERMEDIATES
    ap = argparse.ArgumentParser(description="PDF -> 图表数据 端到端流水线")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_runtime_flags(parser):
        parser.add_argument("--verbose", action="store_true",
                            help="打印逐图、逐曲线与 VLM 用量明细")
        parser.add_argument("--keep-intermediates", action="store_true",
                            help="保留 gaps/review/select 等诊断中间文件")

    a = sub.add_parser("analyze", help="PDF -> 图片/面板/图例色/轴范围 + 配置")
    a.add_argument("pdf")
    a.add_argument("--out", required=True)
    a.add_argument("--dpi", type=int, default=300)
    a.add_argument("--pages", default=None, help="只处理指定页，如 7 或 3,5,7")
    a.add_argument("--vlm", action="store_true",
                   help="启用 DeepSeek VLM（分类/轴读数/命名），需配置 key")
    a.add_argument("--vlm-model", default=vlmc.DEFAULT_MODEL)
    a.add_argument("--vlm-base-url", default=vlmc.DEFAULT_BASE_URL)
    a.add_argument("--want", default=None,
                   help="用一句话说要哪些图，如 --want \"800 bar 下速度随时间的折线图\"")
    a.add_argument("--want-deep", action="store_true",
                   help="再用正文里提到图的段落判定一次（更准，每篇约 ¥0.02）")
    a.add_argument("--no-find-series", action="store_true",
                   help="不让模型找曲线（省掉每面板一次调用，但就不知道该提哪几条线）")
    add_runtime_flags(a)

    e = sub.add_parser("extract", help="按配置提取数据（只跑已确认的面板）")
    e.add_argument("config")
    e.add_argument("--force", action="store_true", help="不检查 confirmed")
    e.add_argument("--only", nargs="*", default=None, help="只跑指定面板 id")
    e.add_argument("--vlm", action="store_true", help="提取后用 VLM 抽查校验")
    e.add_argument("--vlm-model", default=vlmc.DEFAULT_MODEL)
    e.add_argument("--vlm-base-url", default=vlmc.DEFAULT_BASE_URL)
    e.add_argument("--template", default=None,
                   help="按这个模板文件（txt/dat 等）再输出一份；模型只读一次即缓存")
    e.add_argument("--no-template", action="store_true", help="不用模板（只出 CSV）")
    add_runtime_flags(e)

    c = sub.add_parser("confirm", help="确认/修正某个面板的轴范围")
    c.add_argument("config")
    c.add_argument("--id", required=True)
    c.add_argument("--x", default=None)
    c.add_argument("--y", default=None)
    c.add_argument("--skip", action="store_true", help="标记该面板为跳过")

    al = sub.add_parser("all", help="analyze + extract（--force 时不再要求人工确认轴范围）")
    al.add_argument("pdf")
    al.add_argument("--out", required=True)
    al.add_argument("--dpi", type=int, default=300)
    al.add_argument("--pages", default=None)
    al.add_argument("--force", action="store_true")
    al.add_argument("--vlm", action="store_true")
    al.add_argument("--vlm-model", default=vlmc.DEFAULT_MODEL)
    al.add_argument("--vlm-base-url", default=vlmc.DEFAULT_BASE_URL)
    al.add_argument("--want", default=None, help="用一句话说要哪些图（见 README）")
    al.add_argument("--want-deep", action="store_true", help="再用正文段落判定一次")
    al.add_argument("--no-find-series", action="store_true", help="不让模型找曲线")
    al.add_argument("--template", default=None,
                    help="按这个模板文件（txt/dat 等）再输出一份；模型只读一次即缓存")
    al.add_argument("--no-template", action="store_true", help="不用模板（只出 CSV）")
    add_runtime_flags(al)

    b = sub.add_parser("batch", help="批量处理一个目录下的所有 PDF，并输出汇总")
    b.add_argument("dir")
    b.add_argument("--out", required=True)
    b.add_argument("--dpi", type=int, default=300)
    b.add_argument("--pages", default=None, help="只处理每篇的这些页（调试用）")
    b.add_argument("--pattern", default="*.pdf", help="文件名通配，默认 *.pdf")
    b.add_argument("--recursive", action="store_true", help="递归子目录")
    b.add_argument("--force", action="store_true")
    b.add_argument("--vlm", action="store_true")
    b.add_argument("--vlm-model", default=vlmc.DEFAULT_MODEL)
    b.add_argument("--vlm-base-url", default=vlmc.DEFAULT_BASE_URL)
    b.add_argument("--jobs", type=int, default=1,
                   help="并行进程数，每篇一个进程（默认 1=串行）")
    b.add_argument("--want", default=None,
                   help="用一句话说要哪些图；每篇都按同一句需求筛")
    b.add_argument("--want-deep", action="store_true", help="再用正文段落判定一次")
    b.add_argument("--no-find-series", action="store_true", help="不让模型找曲线")
    b.add_argument("--template", default=None,
                   help="按这个模板文件（txt/dat 等）再输出一份；模型只读一次即缓存")
    b.add_argument("--no-template", action="store_true", help="不用模板（只出 CSV）")
    add_runtime_flags(b)

    sub.add_parser("doctor", help="环境自检：解释器、依赖、API key")

    args = ap.parse_args()
    _VERBOSE = bool(getattr(args, "verbose", False))
    _KEEP_INTERMEDIATES = bool(getattr(args, "keep_intermediates", False))

    def make_vlm():
        if not getattr(args, "vlm", False):
            return None
        client = vlmc.DeepSeekVLM(model=args.vlm_model, base_url=args.vlm_base_url,
                                  verbose=_VERBOSE)
        if not client.api_key:
            log("⚠ 指定了 --vlm 但没找到 API key："
                "请把 key 写入项目根目录的 deepseek_key.txt 或设置 DEEPSEEK_API_KEY。"
                "本次只做矢量/图像处理；位图坐标轴无法自动读取。")
            return None
        log(f"VLM 已启用：model={client.model}  base_url={client.base_url}")
        return client

    if getattr(args, "want", None) and not getattr(args, "vlm", False):
        log("⚠ 指定了 --want 但没启用 --vlm：只能按图号匹配（--want \"Figure 5\"）；"
            "按图的内容找图需要模型（--vlm）。")

    if args.cmd == "analyze":
        analyze(args.pdf, args.out, args.dpi, vlm=make_vlm(), pages=args.pages,
                want=args.want, want_deep=args.want_deep,
                find_series=not args.no_find_series)
    elif args.cmd == "extract":
        extract(args.config, force=args.force,
                only=set(args.only) if args.only else None, vlm=make_vlm(),
                template=False if args.no_template else args.template)
    elif args.cmd == "confirm":
        confirm(args.config, args.id, args.x, args.y, args.skip)
    elif args.cmd == "all":
        client = make_vlm()
        cfg = analyze(args.pdf, args.out, args.dpi, vlm=client, pages=args.pages,
                      want=args.want, want_deep=args.want_deep,
                      find_series=not args.no_find_series, report_usage=False)
        extract(cfg, force=args.force, vlm=client,
                template=False if args.no_template else args.template,
                report_usage=False)
        if client is not None:
            print_vlm_usage(client, "本次合计")
    elif args.cmd == "batch":
        batch(args.dir, args.out, dpi=args.dpi, vlm=make_vlm(), pages=args.pages,
              force=args.force, pattern=args.pattern, recursive=args.recursive,
              jobs=args.jobs, want=args.want,
              want_deep=args.want_deep, find_series=not args.no_find_series,
              template=False if args.no_template else args.template)
    elif args.cmd == "doctor":
        doctor()


if __name__ == "__main__":
    main()
