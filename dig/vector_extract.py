"""Extract data straight from a PDF's vector graphics - no rasterising, no OCR.

When a figure is drawn with vector operators, the PDF already contains:
  * the exact polyline coordinates of every series (with its stroke colour), and
  * the tick labels as real text with their positions.

So the data can be recovered essentially losslessly. This module finds vector figure
regions, calibrates the axes from the real tick text, and maps path points to data.

Usage:
  python vector_extract.py paper.pdf --page 3 --out out/vec
  python vector_extract.py paper.pdf --list         # just report vector regions
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import pdfplumber
import series_seed as ss

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _bbox_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap(a, b, tol=6.0):
    return not (a[2] < b[0] - tol or b[2] < a[0] - tol
                or a[3] < b[1] - tol or b[3] < a[1] - tol)


def find_vector_regions(page, min_w=120, min_h=80, min_objects=12, min_path_pts=8):
    """Cluster the page's vector paths into figure-sized regions.

    Tick marks and axis rules are tiny; a real figure region accumulates many path
    objects (the series polylines plus the axes box plus ticks).
    """
    items = []
    for c in page.curves:
        pts = c.get("pts") or []
        if len(pts) < min_path_pts and not (c.get("stroke") or c.get("fill")):
            continue
        items.append((c["x0"], c["top"], c["x1"], c["bottom"]))
    for ln in page.lines:
        items.append((ln["x0"], ln["top"], ln["x1"], ln["bottom"]))
    for rc in page.rects:
        items.append((rc["x0"], rc["top"], rc["x1"], rc["bottom"]))
    if len(items) < min_objects:
        return []

    regions = []
    for box in items:
        merged = False
        for r in regions:
            if _overlap(r, box, tol=10.0):
                r[0] = min(r[0], box[0])
                r[1] = min(r[1], box[1])
                r[2] = max(r[2], box[2])
                r[3] = max(r[3], box[3])
                r[4] += 1
                merged = True
                break
        if not merged:
            regions.append([box[0], box[1], box[2], box[3], 1])

    # merge regions that ended up overlapping after growing
    changed = True
    while changed:
        changed = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if _overlap(regions[i], regions[j], tol=10.0):
                    a, b = regions[i], regions[j]
                    regions[i] = [min(a[0], b[0]), min(a[1], b[1]),
                                  max(a[2], b[2]), max(a[3], b[3]), a[4] + b[4]]
                    regions.pop(j)
                    changed = True
                    break
            if changed:
                break

    out = []
    for r in regions:
        w, h = r[2] - r[0], r[3] - r[1]
        if w >= min_w and h >= min_h and r[4] >= min_objects:
            out.append({"bbox": [round(v, 1) for v in r[:4]], "objects": r[4]})
    out.sort(key=lambda r: (-_bbox_area(r["bbox"]), r["bbox"][1]))
    return out


def _inside(b, x, y, tol=0.0):
    return b[0] - tol <= x <= b[2] + tol and b[1] - tol <= y <= b[3] + tol


def find_axes_boxes(page, bbox):
    """Independent PDF plot frames; never calibrate their union as one coordinate system."""
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    boxes = []
    for rc in page.rects:
        b = (rc["x0"], rc["top"], rc["x1"], rc["bottom"])
        if (_inside(bbox, b[0], b[1], 1) and _inside(bbox, b[2], b[3], 1)
                and b[2] - b[0] > .2 * bw and b[3] - b[1] > .15 * bh):
            boxes.append(b)
    horizontal, vertical = [], []
    for ln in page.lines:
        b = (ln["x0"], ln["top"], ln["x1"], ln["bottom"])
        if not (_inside(bbox, b[0], b[1], 1) and _inside(bbox, b[2], b[3], 1)):
            continue
        if abs(b[3] - b[1]) <= .5 and b[2] - b[0] > .2 * bw:
            horizontal.append(b)
        if abs(b[2] - b[0]) <= .5 and b[3] - b[1] > .15 * bh:
            vertical.append(b)
    for l in vertical:
        for h in horizontal:
            if abs(l[0] - h[0]) <= 1 and abs(l[3] - h[1]) <= 1:
                boxes.append((l[0], l[1], h[2], h[1]))
    unique = []
    for b in sorted(boxes, key=_bbox_area):
        if any(max(abs(x - y) for x, y in zip(b, k)) < 2 for k in unique):
            continue
        inside = [k for k in unique if _inside(b, k[0], k[1]) and _inside(b, k[2], k[3])]
        if len(inside) >= 2:
            continue
        if any(_bbox_area((max(b[0], k[0]), max(b[1], k[1]), min(b[2], k[2]), min(b[3], k[3])))
               > .2 * min(_bbox_area(b), _bbox_area(k)) for k in unique):
            continue
        unique.append(b)
    return sorted(unique, key=lambda b: (b[1], b[0]))


def find_axes_box(page, bbox, min_frac=0.45):
    """The largest axis-aligned rectangle inside the region is the plot frame."""
    best = None
    for rc in page.rects:
        b = (rc["x0"], rc["top"], rc["x1"], rc["bottom"])
        if not _inside(bbox, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2):
            continue
        if (b[2] - b[0]) >= min_frac * (bbox[2] - bbox[0]) and \
           (b[3] - b[1]) >= min_frac * (bbox[3] - bbox[1]):
            if best is None or _bbox_area(b) > _bbox_area(best):
                best = b
    if best is not None:
        return [round(v, 2) for v in best]
    # fall back to the extent of the longest straight lines (two-axis charts do this
    # when the frame is drawn as four separate strokes rather than one rectangle)
    xs, ys = [], []
    for ln in page.lines:
        if _inside(bbox, ln["x0"], ln["top"]) and _inside(bbox, ln["x1"], ln["bottom"]):
            if abs(ln["x1"] - ln["x0"]) > 0.6 * (bbox[2] - bbox[0]):
                xs.append(ln["x0"]); xs.append(ln["x1"]); ys.append(ln["top"])
            if abs(ln["bottom"] - ln["top"]) > 0.6 * (bbox[3] - bbox[1]):
                ys.append(ln["top"]); ys.append(ln["bottom"]); xs.append(ln["x0"])
    if len(xs) >= 2 and len(ys) >= 2:
        return [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]
    return None


def _parse_number(text):
    t = text.strip().replace("−", "-").replace("–", "-").replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def find_tick_labels(page, axes, x_band=26, y_band=46):
    """Real text objects just outside the frame -> (position, value) pairs.

    No OCR involved: these are the PDF's own glyphs, so the values are exact.
    """
    left, top, right, bottom = axes
    xs, ys = [], []
    for w in page.extract_words():
        val = _parse_number(w["text"])
        if val is None:
            continue
        cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
        if bottom - 2 <= w["top"] <= bottom + x_band and left - 6 <= cx <= right + 6:
            xs.append((cx, val, w["text"]))
        elif left - y_band <= w["x1"] <= left - 2 and top - 4 <= cy <= bottom + 4:
            ys.append((cy, val, w["text"]))
    return xs, ys


def find_figure_text(page, axes, band=42.0, cap=60):
    """Axis titles and legend words around the frame - the PDF's own text, no OCR.

    Tick numbers are already handled by the calibration; what is left over is exactly
    the vocabulary a reader uses to describe a chart ("spray tip penetration" versus
    "time after start of injection"), so it makes the text search far more useful than
    the caption alone. Numbers are kept when they come with letters ("Pi=800 bar"),
    dropped when they are tick labels ("0.5").
    """
    left, top, right, bottom = axes
    x_words, y_words, inside = [], [], []
    for w in page.extract_words():
        t = (w["text"] or "").strip()
        if not t or not any(ch.isalpha() for ch in t):
            continue
        cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
        if bottom - 2 <= w["top"] <= bottom + band and left - 40 <= cx <= right + 40:
            x_words.append((w["x0"], t))
        elif left - band <= w["x1"] <= left - 2 and top - 8 <= cy <= bottom + 8:
            y_words.append((w["bottom"], t))     # 竖排标题按纵向读数排序
        elif left < cx < right and top < cy < bottom:
            inside.append((w["top"], w["x0"], t))
    join = lambda ws, key: " ".join(t for _, t in sorted(ws, key=key))[:cap]
    return {"x_title": join(x_words, lambda p: p[0]),
            "y_title": join(y_words, lambda p: p[0]),
            "inside": join([(a * 10000 + b, t) for a, b, t in inside], lambda p: p[0])}


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None, None, 0.0
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return slope, intercept, (1 - ss_res / ss_tot if ss_tot else 0.0)


def calibrate(labels, min_labels=3, min_r2=0.999):
    pts = sorted(labels, key=lambda t: t[0])
    if len(pts) < min_labels:
        return None
    slope, intercept, r2 = linfit([p[0] for p in pts], [p[1] for p in pts])
    if slope is None or r2 < min_r2:
        return None
    return {"slope": slope, "intercept": intercept, "r2": r2,
            "labels": [p[2] for p in pts], "n": len(pts)}


def find_tick_marks(page, axes, max_len=16.0, gap=14.0):
    """Positions of the tick strokes just outside the frame.

    Tick marks are vector geometry, so their positions are exact - unlike a text
    bounding box centre, which carries a fraction-of-a-point bias from font metrics.
    Using tick positions and letting the labels supply only the *values* removes that
    systematic offset.
    """
    left, top, right, bottom = axes
    xs, ys = [], []
    for ln in page.lines:
        w = abs(ln["x1"] - ln["x0"])
        h = abs(ln["bottom"] - ln["top"])
        cx = (ln["x0"] + ln["x1"]) / 2.0
        cy = (ln["top"] + ln["bottom"]) / 2.0
        if h > w and h <= max_len and left - 3 <= cx <= right + 3:
            if bottom - 1 <= cy <= bottom + gap:
                xs.append(cx)
        elif w > h and w <= max_len and top - 3 <= cy <= bottom + 3:
            if left - gap <= cx <= left - 1:
                ys.append(cy)
    return sorted(xs), sorted(ys)


def calibrate_from_ticks(page, axes, labels, axis="x", min_labels=3, min_r2=0.999,
                         max_match=10.0):
    """Pair each tick stroke with the nearest numeric label, then fit."""
    xs, ys = find_tick_marks(page, axes)
    positions = xs if axis == "x" else ys
    if len(positions) < min_labels:
        return None
    pts = []
    for pos in positions:
        nearest = None
        for lp, val, text in labels:
            d = abs(lp - pos)
            if d <= max_match and (nearest is None or d < nearest[0]):
                nearest = (d, val, text)
        if nearest:
            pts.append((pos, nearest[1], nearest[2]))
    # a label may not sit next to every tick; require most ticks matched
    if len(pts) < min_labels or len(pts) < 0.6 * len(labels):
        return None
    slope, intercept, r2 = linfit([p[0] for p in pts], [p[1] for p in pts])
    if slope is None or r2 < min_r2:
        return None
    return {"slope": slope, "intercept": intercept, "r2": r2,
            "labels": [p[2] for p in pts], "n": len(pts), "positions": "tick-marks"}


def _color_key(c):
    stroke_col = c.get("stroking_color")
    fill_col = c.get("non_stroking_color")

    def chroma(col):
        return (max(col) - min(col)) if isinstance(col, (list, tuple)) and len(col) == 3 else 0

    # Marker glyphs are often encoded as a coloured fill with either no stroke or a
    # black outline.  Using the dormant/outline stroking colour turns every red marker
    # into a fake black series, so choose the visible chromatic fill in that case.
    if c.get("fill") and isinstance(fill_col, (list, tuple)) and len(fill_col) == 3 \
            and (not c.get("stroke") or chroma(fill_col) > chroma(stroke_col)):
        col = fill_col
    else:
        col = stroke_col if stroke_col is not None else fill_col
    if isinstance(col,(int,float)):
        return (round(float(col),2),)*3
    if isinstance(col,(list,tuple)) and len(col)==1:
        return (round(float(col[0]),2),)*3
    if isinstance(col, (list, tuple)) and len(col) == 3:
        return tuple(round(float(v), 2) for v in col)
    return None


def _color_hex(color):
    if not isinstance(color, tuple) or len(color) != 3:
        return "unknown"
    vals = (int(round(max(0.0, min(1.0, v)) * 255)) for v in color)
    return "#{:02x}{:02x}{:02x}".format(*vals)


def _hex_key(text):
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(text or "").strip())
    if not m:
        return None
    s = m.group(1)
    return tuple(round(int(s[i:i + 2], 16) / 255.0, 2) for i in (0, 2, 4))


def _dedupe_points(points, tol=1e-6):
    out = []
    for x, y in points:
        p = (float(x), float(y))
        if not out or abs(p[0] - out[-1][0]) > tol or abs(p[1] - out[-1][1]) > tol:
            out.append(p)
    return out


def _vector_candidates(page, axes, min_pts=6, min_extent_frac=0.15,
                       retain_markers=False, exclude_boxes=()):
    """Return physical vector instances instead of merging everything by colour.

    A single colour may represent several experimental holes/conditions.  Each long
    PDF path is therefore a separate candidate.  If a series is marker-only, its
    small circle/square glyphs are reconstructed as a monotone marker track.
    """
    left, top, right, bottom = axes
    fw, fh = max(1.0, right - left), max(1.0, bottom - top)
    candidates = []
    long_colors = set()
    def excluded(x,y):
        return any(_inside(b,x,y) for b in exclude_boxes)
    for object_index, c in enumerate(list(page.curves) + list(page.lines)):
        pts = c.get("pts") or [(c["x0"], c["top"]), (c["x1"], c["bottom"])]
        inside = _dedupe_points((x, y) for (x, y) in pts if _inside(axes, x, y) and not excluded(x,y))
        if len(inside) < min_pts:
            continue
        xs = [p[0] for p in inside]
        if (max(xs) - min(xs)) < min_extent_frac * (right - left):
            continue
        key = _color_key(c) or ("?",)
        candidates.append({"color_key": key, "points": inside, "source": "path",
                           "object_index": object_index})
        long_colors.add(key)

    # A semantic inventory can require both marker data and a same-colour model
    # path. Without an inventory, retain the legacy avoidance of duplicate samples.
    marker_groups = defaultdict(list)
    for c in page.curves:
        key = _color_key(c) or ("?",)
        if key in long_colors and not retain_markers:
            continue
        cx, cy = (c["x0"] + c["x1"]) / 2.0, (c["top"] + c["bottom"]) / 2.0
        w, h = c["x1"] - c["x0"], c["bottom"] - c["top"]
        if not _inside(axes, cx, cy) or excluded(cx,cy) or not (0.3 <= w <= 0.05 * fw and 0.3 <= h <= 0.08 * fh):
            continue
        path = c.get("path") or []
        commands = [cmd[0] for cmd in path if cmd]
        if retain_markers:
            # Outlined fonts are also PDF curves. Marker primitives have simple
            # one-subpath geometry; a many-Bezier letter is not a data glyph.
            if commands.count('m') > 1 or commands.count('c') not in (0,4):
                continue
            if not (.45 <= w/max(h,1e-6) <= 2.2):
                continue
            vertices = _dedupe_points(c.get('pts') or [])
            if vertices and vertices[0] == vertices[-1]:
                vertices = vertices[:-1]
            if commands.count('c') == 4:
                shape = 'circle'
            elif len(vertices) == 3:
                shape = 'triangle'
                cx,cy = sum(x for x,y in vertices)/3.,sum(y for x,y in vertices)/3.
            elif len(vertices) == 4:
                shape = 'diamond' if all(abs(x-cx)<.2*w or abs(y-cy)<.2*h
                                         for x,y in vertices) else 'rect'
            elif len(vertices) == 10:
                shape = 'star'
                cx,cy = sum(x for x,y in vertices)/10.,sum(y for x,y in vertices)/10.
            else:
                continue
        else:
            shape = "circle" if any(cmd and cmd[0] == "c" for cmd in path) else "glyph"
        marker_groups[(key, shape)].append((cx, cy))
    for c in page.rects:
        key = _color_key(c) or ("?",)
        if key in long_colors and not retain_markers:
            continue
        cx, cy = (c["x0"] + c["x1"]) / 2.0, (c["top"] + c["bottom"]) / 2.0
        w, h = c["x1"] - c["x0"], c["bottom"] - c["top"]
        if not _inside(axes, cx, cy) or excluded(cx,cy) or not (0.3 <= w <= 0.05 * fw and 0.3 <= h <= 0.08 * fh):
            continue
        marker_groups[(key, "rect")].append((cx, cy))
    shaped_colors = {key for (key, shape), pts in marker_groups.items()
                     if shape in ("circle", "rect") and len(pts) >= 4}
    for (key, shape), centres in marker_groups.items():
        # Generic polygon/dash glyphs commonly duplicate the same sampled series
        # already represented by circles or squares.  Prefer the explicit marker
        # geometry; keep generic glyphs only when no such shape exists.
        if shape == "glyph" and key in shaped_colors:
            continue
        track = ss.best_marker_track(centres, axes, min_extent_frac=min_extent_frac)
        if track:
            candidates.append({"color_key": key, "points": track,
                               "source": f"markers:{shape}", "object_index": 0})

    return candidates


def snap_axes(page, axes):
    """Snap an independently calibrated panel frame to nearby exact PDF spines.

    The panel identity has already been established in the rendered image. A
    smaller legend rectangle must never replace it through global box ranking.
    """
    left,top,right,bottom = map(float,axes)
    fw,fh = right-left,bottom-top
    out = [left,top,right,bottom]
    for index,target in enumerate(out):
        choices = []
        for line in page.lines:
            x0,x1,y0,y1 = line['x0'],line['x1'],line['top'],line['bottom']
            if index in (0,2):
                if abs(x1-x0)>.05 or min(bottom,y1)-max(top,y0)<.85*fh:
                    continue
                pos,tol = (x0+x1)/2.,.025*fw
            else:
                if abs(y1-y0)>.05 or min(right,x1)-max(left,x0)<.85*fw:
                    continue
                pos,tol = (y0+y1)/2.,.025*fh
            if abs(pos-target)<=tol:
                choices.append((abs(pos-target),pos))
        if choices:
            out[index] = min(choices)[1]
    return out


def inner_exclusions(page, axes):
    """Exact stroked legend/inset rectangles wholly inside one panel's axes."""
    left,top,right,bottom = axes
    fw,fh = right-left,bottom-top
    boxes = []
    for rect in page.rects:
        box = [rect['x0'],rect['top'],rect['x1'],rect['bottom']]
        w,h = box[2]-box[0],box[3]-box[1]
        key = _color_key(rect)
        if (rect.get('stroke') and key and len(key)==3 and max(key)-min(key)<.1
                and max(key)<.5 and left<box[0]<box[2]<right
                and top<box[1]<box[3]<bottom
                and .1*fw<w<.92*fw and .07*fh<h<.92*fh):
            boxes.append(box)
    return boxes


def _spec_distance(candidate, spec, image_bbox):
    """Distance from VLM anchors to one exact vector candidate (normalised units)."""
    anchors = spec.get("anchors") or []
    if not anchors:
        return 1.0
    bx0, by0, bx1, by1 = image_bbox
    bw, bh = max(1.0, bx1 - bx0), max(1.0, by1 - by0)
    # Planner anchors use Cartesian plot-normalised coordinates (bottom-left origin),
    # while PDF/image y coordinates grow downward.
    pts = [((x - bx0) / bw, (by1 - y) / bh) for x, y in candidate["points"]]
    ds = []
    for a in anchors:
        try:
            ax, ay = float(a[0]), float(a[1])
        except (TypeError, ValueError, IndexError):
            continue
        ds.append(min(math.hypot(px - ax, py - ay) for px, py in pts))
    return sum(ds) / len(ds) if ds else 1.0


def _assign_specs(candidates, series_spec, image_bbox):
    """Use the model's semantic inventory to name/count exact vector instances."""
    unused = set(range(len(candidates)))
    ordered = []
    for spec_index, spec in enumerate(series_spec or []):
        skey = _hex_key(spec.get("color"))
        ranked = []
        for i in unused:
            source = candidates[i]['source']
            draw = str(spec.get('draw') or '')
            if (draw in ('markers_only','markers_connected') and not source.startswith('markers:')) \
                    or (draw == 'line_only' and source.startswith('markers:')):
                continue
            shape = str(spec.get('marker_shape') or '')
            required_shape = {'square':'rect','diamond':'diamond','triangle':'triangle','circle':'circle','star':'star'}.get(shape)
            if source.startswith('markers:') and required_shape and source.split(':')[1] != required_shape:
                continue
            ckey = candidates[i]["color_key"]
            color_penalty = 0.0
            if skey is not None and ckey != skey:
                if len(ckey) == 3:
                    sh = ss._hsv_of(tuple(round(255*v) for v in skey[::-1]))
                    ch = ss._hsv_of(tuple(round(255*v) for v in ckey[::-1]))
                    color_penalty = .03 if sh[1]>40 and ch[1]>40 and ss._hue_dist(sh[0],ch[0])<=32 else .35
                else:
                    color_penalty = .35
            ranked.append((_spec_distance(candidates[i], spec, image_bbox) + color_penalty, i))
        if not ranked:
            continue
        score, best = min(ranked)
        # A wildly distant anchor is not evidence for this path.  Keep the candidate
        # generic rather than giving it a confident but wrong legend name.
        if score > 0.32:
            continue
        unused.remove(best)
        candidates[best]["series_name"] = spec.get("label")
        candidates[best]["spec_index"] = spec_index
        candidates[best]["anchor_score"] = round(score, 4)
        ordered.append(candidates[best])
    if not series_spec:
        ordered.extend(candidates[i] for i in sorted(unused))
    return ordered


def extract_series(page, axes, xcal, ycal, min_pts=6, min_extent_frac=0.15,
                   series_spec=None, image_bbox=None, exclude_boxes=()):
    """Map every physical vector path/marker track into data coordinates."""
    candidates = _vector_candidates(page, axes, min_pts=min_pts,
                                    min_extent_frac=min_extent_frac,
                                    retain_markers=bool(series_spec),exclude_boxes=exclude_boxes)
    candidates = _assign_specs(candidates, series_spec or [], axes)

    out = []
    for cand in candidates:
        color = cand["color_key"]
        data = []
        for x, y in cand["points"]:
            # calibration was fitted as value = slope * position + intercept
            xv = xcal["slope"] * x + xcal["intercept"]
            yv = ycal["slope"] * y + ycal["intercept"]
            data.append((round(xv, 8), round(yv, 8)))
        if len(data) < min(4, min_pts):
            continue
        out.append({
            "color": _color_hex(color),
            "rgb": list(color) if len(color) == 3 else None,
            "n_points": len(data),
            "x_range": [min(d[0] for d in data), max(d[0] for d in data)],
            "y_range": [min(d[1] for d in data), max(d[1] for d in data)],
            "series_name": cand.get("series_name"),
            "spec_index": cand.get("spec_index"),
            "source": cand.get("source"),
            "anchor_score": cand.get("anchor_score"),
            "data": data,
        })
    return out


def _safe_tag(text, limit=48):
    s = re.sub(r"[^0-9A-Za-z._ -]+", "_", str(text or "")).strip(" ._")
    return (s or "series")[:limit]


def extract_vector_figure(page, bbox, outdir=None, name="vec", series_spec=None):
    frames = find_axes_boxes(page, bbox)
    if len(frames) > 1:
        return {"error": "multiple independent plot frames", "bbox": bbox,
                "axes_boxes": frames, "series": []}
    axes = find_axes_box(page, bbox)
    if axes is None:
        return {"error": "no axes frame found", "bbox": bbox}
    xl, yl = find_tick_labels(page, axes)
    # prefer tick-mark geometry for positions; fall back to label centres
    xcal = calibrate_from_ticks(page, axes, xl, "x") or calibrate(xl)
    ycal = calibrate_from_ticks(page, axes, yl, "y") or calibrate(yl)
    result = {
        "bbox": bbox, "axes": axes,
        "x_labels": [t[2] for t in xl], "y_labels": [t[2] for t in yl],
        "x_calibration": xcal, "y_calibration": ycal,
        "series": [],
    }
    if xcal and ycal:
        result["series"] = extract_series(page, axes, xcal, ycal,
                                          series_spec=series_spec, image_bbox=bbox)
        if outdir:
            out = Path(outdir)
            out.mkdir(parents=True, exist_ok=True)
            for i, s in enumerate(result["series"], start=1):
                label = _safe_tag(s.get("series_name")) if s.get("series_name") else \
                    f"s{i}_{s['color'].lstrip('#')}"
                p = out / f"{name}_{label}.csv"
                with p.open("w", newline="", encoding="utf-8") as fh:
                    wr = csv.writer(fh)
                    wr.writerow(["x", "y"])
                    for xv, yv in s["data"]:
                        wr.writerow([f"{xv:.6g}", f"{yv:.6g}"])
                s["file"] = p.name
    return result


def axis_range_from_calibration(cal, axes, which="x"):
    """Value at both frame edges, straight from the calibration."""
    if not cal:
        return None
    left, top, right, bottom = axes
    if which == "x":
        a = cal["slope"] * left + cal["intercept"]
        b = cal["slope"] * right + cal["intercept"]
    else:
        a = cal["slope"] * top + cal["intercept"]
        b = cal["slope"] * bottom + cal["intercept"]
    return [round(min(a, b), 10), round(max(a, b), 10)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--page", type=int, default=None, help="1-based page number")
    ap.add_argument("--list", action="store_true", help="only list vector regions")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    with pdfplumber.open(args.pdf) as pdf:
        pages = [pdf.pages[args.page - 1]] if args.page else pdf.pages
        for pno, page in enumerate(pages, start=args.page or 1):
            regions = find_vector_regions(page)
            print(f"p{pno}: {len(regions)} 个矢量区域")
            for r in regions:
                print(f"   bbox={r['bbox']}  路径对象={r['objects']}")
            if args.list:
                continue
            for i, r in enumerate(regions, start=1):
                res = extract_vector_figure(page, r["bbox"], args.out,
                                            name=f"p{pno}_r{i}")
                cal = res.get("x_calibration"), res.get("y_calibration")
                if not all(cal):
                    print(f"   区域 {i}: 无法标定（x标签={res.get('x_labels')}, "
                          f"y标签={res.get('y_labels')}）")
                    continue
                print(f"   区域 {i}: 轴框={res['axes']} "
                      f"x标签={res['x_labels']} y标签={res['y_labels']}")
                for s in res["series"]:
                    print(f"      {s.get('file', '')}  颜色={s['color']} "
                          f"{s['n_points']} 点  x=[{s['x_range'][0]:.3f},{s['x_range'][1]:.3f}] "
                          f"y=[{s['y_range'][0]:.3f},{s['y_range'][1]:.3f}]")
                if args.json:
                    Path(args.json).write_text(
                        json.dumps(res, indent=2, ensure_ascii=False)[:200000],
                        encoding="utf-8")


if __name__ == "__main__":
    main()
