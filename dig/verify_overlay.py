"""QA tool: project extracted CSVs back onto the source figure, and re-plot the data.

Deliberately uses only OpenCV + NumPy (no matplotlib) because matplotlib is broken
in some conda environments and crashes at the first draw with 0xC06D007F.

Usage:
  python verify_overlay.py fig.png --frame 119,57,907,737 --xrange 0,5 --yrange 0,70 \
      --csv "out/fig5_series1.csv=#ff8c00:S_tail" --csv "out/fig5_series2.csv=#008080:S_tip" \
      --xlabel "Time (ms)" --ylabel "Axial penetration (mm)" --out out/fig5_verify.png

Each --csv is  path=color:label  (color as #rrggbb or a name; color/label optional).
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import imgio as iio
import numpy as np

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

NAMED = {
    "teal": (128, 128, 0), "brown": (42, 42, 165), "orange": (0, 140, 255),
    "red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 160, 0),
    "black": (0, 0, 0), "magenta": (255, 0, 255), "cyan": (255, 255, 0),
}


def parse_color(text):
    if not text:
        return None
    t = text.strip().lower()
    if t in NAMED:
        return NAMED[t]
    if t.startswith("#") and len(t) == 7:
        r, g, b = int(t[1:3], 16), int(t[3:5], 16), int(t[5:7], 16)
        return (b, g, r)
    return None


def read_csv(path):
    xs, ys = [], []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    for row in rows[1:]:
        if len(row) < 2 or not row[0].strip():
            continue
        try:
            xs.append(float(row[0]))
            ys.append(float(row[1]))
        except ValueError:
            continue
    return xs, ys


def nice_ticks(lo, hi, count=5):
    step = (hi - lo) / (count - 1)
    return [lo + i * step for i in range(count)]


def draw_plot(canvas, box, series, xr, yr, xlabel, ylabel, styles=None):
    left, top, right, bottom = box
    xmin, xmax = xr
    ymin, ymax = yr
    cv2.rectangle(canvas, (left, top), (right, bottom), (255, 255, 255), -1)
    for xv in nice_ticks(xmin, xmax):
        px = int(left + (xv - xmin) / (xmax - xmin) * (right - left))
        cv2.line(canvas, (px, top), (px, bottom), (230, 230, 230), 1)
        cv2.putText(canvas, f"{xv:.3g}", (px - 14, bottom + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for yv in nice_ticks(ymin, ymax):
        py = int(bottom - (yv - ymin) / (ymax - ymin) * (bottom - top))
        cv2.line(canvas, (left, py), (right, py), (230, 230, 230), 1)
        cv2.putText(canvas, f"{yv:.4g}", (8, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 0), 1)
    cv2.putText(canvas, xlabel, ((left + right) // 2 - 40, bottom + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, ylabel, (8, top - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    legend_y = top + 16
    for label, color, xs, ys in series:
        pts = []
        for x, y in zip(xs, ys):
            px = int(round(left + (x - xmin) / (xmax - xmin) * (right - left)))
            py = int(round(bottom - (y - ymin) / (ymax - ymin) * (bottom - top)))
            pts.append((px, py))
        if (styles or {}).get(label) == "points":
            for point in pts:
                cv2.circle(canvas,point,3,color,1,cv2.LINE_AA)
        elif len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, color, 1, cv2.LINE_AA)
        cv2.line(canvas, (right - 210, legend_y), (right - 180, legend_y), color, 2)
        cv2.putText(canvas, label[:32], (right - 175, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        legend_y += 20


def compose_overlay(image_path, frame, xrange, yrange, series, out_path,
                    xlabel="x", ylabel="y", annotations=None, anchors=None,
                    bridged=None, styles=None):
    """Build the two-panel QA image: left = points reprojected on the figure,
    right = the extracted data replotted.

    `series` is a list of (label, colour, xs, ys); colour may be a name, #rrggbb or a BGR
    tuple. `annotations` is a list of (bbox, label) drawn as dashed grey boxes - used for
    the objects the model judged "not data", so a wrong verdict is visible at a glance.
    `anchors` is a list of (x, y, label) in image pixels: the model's "this curve passes
    about here" hints, drawn as yellow crosses so a reader can tell at a glance whether a
    miss was the model's fault (cross off the curve) or the tracer's (cross on the curve).
    `bridged` is a list of (x, y) in image pixels where the curve was hidden behind
    another one: drawn as orange rings, because those values are interpolated along the
    occluding stroke rather than measured on this curve's own pixels.
    This is the single implementation used by both the CLI and run.py.
    """
    left, top, right, bottom = frame
    xmin, xmax = xrange
    ymin, ymax = yrange
    img = iio.imread(Path(image_path).resolve())
    if img is None:
        raise SystemExit(f"cannot read image: {image_path}")

    loaded = []
    for label, color, xs, ys in series:
        if not xs:
            continue
        if isinstance(color, str):
            color = parse_color(color) or (255, 0, 255)
        loaded.append((label, color, xs, ys))
        for x, y in zip(xs, ys):
            px = int(round(left + (x - xmin) / (xmax - xmin) * (right - left)))
            py = int(round(bottom - (y - ymin) / (ymax - ymin) * (bottom - top)))
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                is_point = (styles or {}).get(label) == "points"
                cv2.circle(img, (px, py), 3 if is_point else 1, color, 1 if is_point else -1)

    for x, y, label in (anchors or []):
        x, y = int(round(x)), int(round(y))
        if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
            continue
        cv2.drawMarker(img, (x, y), (0, 200, 255), cv2.MARKER_CROSS, 14, 1,
                       cv2.LINE_AA)
        cv2.putText(img, str(label)[:14], (x + 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 140, 190), 1, cv2.LINE_AA)

    for x, y in (bridged or []):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 2, (0, 140, 255), 1, cv2.LINE_AA)

    for item in (annotations or []):
        bbox, label = item if isinstance(item, tuple) else (item["bbox"], item.get("what"))
        if not bbox:
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        cv2.rectangle(img, (x0, y0), (x1, y1), (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, str(label)[:18], (x0 + 2, max(12, y0 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

    panel_w = 900
    h, w = img.shape[:2]
    canvas_h = max(h + 40, 700)
    canvas = np.full((canvas_h, w + 60 + panel_w, 3), 255, np.uint8)
    canvas[10:10 + h, 10:10 + w] = img
    box = (w + 90, 70, w + 60 + panel_w, min(canvas_h - 80, 600))
    draw_plot(canvas, box, loaded, (xmin, xmax), (ymin, ymax), xlabel, ylabel, styles=styles)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, canvas)
    return out


def main():
    ap = argparse.ArgumentParser(description="生成提取结果的质检叠加图")
    ap.add_argument("image")
    ap.add_argument("--frame", required=True, help="left,top,right,bottom in image pixels")
    ap.add_argument("--xrange", required=True)
    ap.add_argument("--yrange", required=True)
    ap.add_argument("--csv", action="append", required=True, help="path=color:label")
    ap.add_argument("--xlabel", default="x")
    ap.add_argument("--ylabel", default="y")
    ap.add_argument("--out", default="verify.png")
    args = ap.parse_args()

    left, top, right, bottom = (float(v) for v in args.frame.split(","))
    xmin, xmax = (float(v) for v in args.xrange.split(","))
    ymin, ymax = (float(v) for v in args.yrange.split(","))

    series = []
    for spec in args.csv:
        path, _, opts = spec.partition("=")
        color_txt, _, label = opts.partition(":")
        xs, ys = read_csv(path)
        label = label or Path(path).stem
        if not xs:
            print(f"  {label}: no rows")
            continue
        series.append((label, color_txt, xs, ys))
        print(f"  {label}: n={len(xs)} x=[{min(xs):.3f},{max(xs):.3f}] "
              f"y=[{min(ys):.3f},{max(ys):.3f}]")

    out = compose_overlay(args.image, (left, top, right, bottom), (xmin, xmax),
                          (ymin, ymax), series, args.out, args.xlabel, args.ylabel)
    print(f"verify png -> {out}")


if __name__ == "__main__":
    main()
