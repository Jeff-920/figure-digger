"""Reusable axis-range reading: OCR the tick labels, fit, snap to the tick grid.

Wraps the pure helpers in ocr_ticks (linfit / fit_axis / snap_to_step / parse_number /
run_ocr) with the band-cropping and multi-band orchestration, so run.py can call it
without shelling out to the ocr_ticks CLI.
"""

from pathlib import Path
import math

import cv2
import imgio as iio
import numpy as np

import ocr_ticks as ot

_ENGINE = None


def tick_positions(img, frame, axis='y', side='left'):
    """Locate short low-saturation tick strokes without reading any numbers.

    Positions use Cartesian plot fractions, independent of the labels' values.
    The continuous spine is excluded from the strip; colour curves cannot become
    dark ticks. Grid intersections and frame endpoints are valid tick locations.
    """
    left,top,right,bottom = map(int,frame)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,1]
    ink = (gray<235)&(sat<60)
    depth = max(8,min(20,round(.03*min(right-left,bottom-top))))
    h,w = gray.shape
    if axis=='y':
        spine = right if side=='right' else left
        a,b = (max(0,spine-depth),max(0,spine-2)) if side=='right' else \
              (min(w,spine+3),min(w,spine+depth+1))
        strip = ink[max(0,top):min(h,bottom+1),a:b].copy()
        profile = strip.sum(axis=1)
        lo,span = top,max(1,bottom-top)
    else:
        a,b = max(0,bottom-depth),max(0,bottom-2)
        strip = ink[a:b,max(0,left):min(w,right+1)].copy()
        profile = strip.sum(axis=0)
        lo,span = left,max(1,right-left)
    flags = profile>=max(3,int(.25*depth))
    runs = []
    start = None
    for i,flag in enumerate(list(flags)+[False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i-start<=max(8,.025*span):
                p = (start+i-1)/2.
                runs.append(1.-p/span if axis=='y' else p/span)
            start = None
    return sorted(set([0.,1.]+runs))


def calibrate_vlm_ticks(img, frame, reading):
    """Fit model-read tick values to measured strokes; infer frame endpoints.

    A highest visible label can be below the top edge (100 at 5/6 height implies
    a 120 upper frame limit). Never silently identify label extrema with edges.
    Invalid or insufficient evidence leaves the original reading with diagnostics.
    """
    sources = [('x',reading.get('x'),'left')]
    sources.extend(('y',r,r.get('side','left')) for r in reading.get('y_axes') or []
                   if isinstance(r,dict))
    for axis,source,side in sources:
        if not isinstance(source,dict) or source.get('scale','linear')!='linear':
            continue
        ticks = source.get('ticks') or []
        if not ticks:
            continue  # Older configs/mocks remain readable.
        positions = tick_positions(img,frame,axis,side)
        pairs,snapped = [],0
        for tick in ticks:
            try:
                value,position = float(tick['value']),float(tick['position'])
            except (KeyError,TypeError,ValueError):
                continue
            if not (math.isfinite(value) and 0<=position<=1):
                continue
            nearest = min(positions,key=lambda q:abs(q-position))
            if abs(nearest-position)<=.04:
                position = nearest
                snapped += 1
            if not any(abs(position-p)<.01 or abs(value-v)<1e-9 for p,v in pairs):
                pairs.append((position,value))
        if len(pairs)<2 or max(p for p,v in pairs)-min(p for p,v in pairs)<.3:
            source['tick_calibration']={'accepted':False,'reason':'刻度位置证据不足'}
            continue
        px = sum(p for p,v in pairs)/len(pairs)
        vy = sum(v for p,v in pairs)/len(pairs)
        variance = sum((p-px)**2 for p,v in pairs)
        slope = sum((p-px)*(v-vy) for p,v in pairs)/variance
        intercept = vy-slope*px
        error = sum((v-(slope*p+intercept))**2 for p,v in pairs)
        total = sum((v-vy)**2 for p,v in pairs)
        r2 = 1.-error/max(total,1e-20)
        accepted = slope>0 and r2>=.995 and snapped>=min(3,len(pairs))
        evidence = {'accepted':accepted,'r2':round(r2,6),'pairs':pairs,
                    'snapped':snapped,'original_range':[source.get('min'),source.get('max')]}
        if accepted:
            # Coarse positioning may leave <1% residual error. Snap only values
            # close to the label step's quarter grid, never force a whole step.
            values = sorted(v for p,v in pairs)
            step = min(b-a for a,b in zip(values,values[1:]) if b>a)
            quantum = step/4.
            def clean(value,prior):
                if isinstance(prior,(int,float)) and abs(prior-value)<.015*abs(slope):
                    return float(prior)
                rounded = round(value/quantum)*quantum
                return round(rounded if abs(rounded-value)<.015*abs(slope) else value,8)
            source['min'],source['max'] = clean(intercept,source.get('min')),clean(intercept+slope,source.get('max'))
            evidence['frame_range'] = [source['min'],source['max']]
        else:
            evidence['reason']='刻度位置与数值不一致，未覆盖原范围'
        source['tick_calibration'] = evidence
    return reading


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def _attempt(img, frame, engine, x_band, y_band):
    left, top, right, bottom = frame
    h, w = img.shape[:2]
    x_pad = 25
    x_x0, x_x1 = max(0, left - x_pad), min(w, right + x_pad)
    x_y0, x_y1 = min(h, bottom + 2), min(h, bottom + 2 + x_band)
    y_x0, y_x1 = max(0, left - y_band), max(1, left - 2)
    x_crop = img[x_y0:x_y1, x_x0:x_x1]
    y_crop = img[top:bottom, y_x0:y_x1]

    x_items = []
    for it in ot.run_ocr(engine, x_crop):
        val = ot.parse_number(it["text"])
        if val is not None:
            x_items.append((x_x0 + it["cx"], val, it["text"], it["score"]))
    y_items = []
    for it in ot.run_ocr(engine, y_crop):
        val = ot.parse_number(it["text"])
        if val is not None:
            y_items.append((top + it["cy"], val, it["text"], it["score"]))

    fx, fy = ot.fit_axis(x_items, None, "x"), ot.fit_axis(y_items, None, "y")
    score = ((fx["n_labels"] if fx else 0) + (fy["n_labels"] if fy else 0),
             -sum((1.0 - min(1.0, f["r2"])) + 0.02 * len(f["outliers"])
                  for f in (fx, fy) if f))
    return score, fx, fy, (x_band, y_band), (x_crop, y_crop)


def _edges(fit, lo_px, hi_px):
    raw_lo = fit["slope"] * lo_px + fit["intercept"]
    raw_hi = fit["slope"] * hi_px + fit["intercept"]
    v_lo, s_lo = ot.snap_to_step(raw_lo, fit["step"])
    v_hi, s_hi = ot.snap_to_step(raw_hi, fit["step"])
    return {
        "range": [min(v_lo, v_hi), max(v_lo, v_hi)],
        "raw_range": [min(raw_lo, raw_hi), max(raw_lo, raw_hi)],
        "labels": [l["text"] for l in fit["labels"]],
        "n_labels": fit["n_labels"],
        "r2": round(fit["r2"], 6),
        "step": fit["step"],
        "outliers": fit["outliers"],
        "repaired": [f"{t}->{c:g}" for t, _, c in fit["repaired"]],
        "snapped": bool(s_lo or s_hi),
        "looks_log": fit["looks_log"],
    }


def read_axis_ranges(image_path, frame, x_band=None, y_band=None, engine=None,
                     log=None, debug_crops=None, min_r2=0.999):
    """OCR both tick-label bands and return the axis ranges implied by the frame edges.

    Returns {'x': {...}, 'y': {...}, 'bands': (x_band, y_band)} where each axis dict has
    'range', the OCR evidence and quality flags. A missing axis means OCR could not read
    enough labels - the caller should ask a human rather than guess.
    """
    img = iio.imread(Path(image_path).resolve())
    if img is None:
        raise RuntimeError(f"cannot read {image_path}")
    engine = engine or get_engine()
    left, top, right, bottom = frame
    h, w = img.shape[:2]

    base_x = x_band or max(30, int(0.10 * (bottom - top)))
    base_y = y_band or max(40, int(0.12 * (right - left)))
    # The y band can never be wider than the space left of the axes box; a too-narrow
    # band cuts each label in half ("2.5" becomes "5" or splits into "2" and "5").
    y_max = max(30, left - 2)
    trials = [(base_x, base_y)]
    if not x_band and not y_band:
        trials += [(int(base_x * 1.6), int(min(y_max, base_y * 1.6))),
                   (int(base_x * 2.2), int(min(y_max, base_y * 2.2))),
                   (int(base_x * 1.6), y_max)]

    # Per axis, keep only fits that are essentially perfectly linear: real tick labels
    # are collinear by construction, so a low R^2 means OCR mixed up values.
    best_x = best_y = None
    best_bands = best_crops = None
    for xb, yb in trials:
        _, fx, fy, bands, crops = _attempt(img, frame, engine, xb, yb)
        if log:
            log(f"      x_band={xb} y_band={yb}: "
                f"x {fx['n_labels'] if fx else 0} 标签"
                + (f" R²={fx['r2']:.5f}" if fx else "")
                + f", y {fy['n_labels'] if fy else 0} 标签"
                + (f" R²={fy['r2']:.5f}" if fy else ""))
        for axis, fit in (("x", fx), ("y", fy)):
            if fit is None:
                continue
            cand = (fit["r2"] >= min_r2, fit["n_labels"], fit["r2"], fit, bands, crops)
            if axis == "x":
                if best_x is None or cand[:3] > best_x[:3]:
                    best_x = cand
            else:
                if best_y is None or cand[:3] > best_y[:3]:
                    best_y = cand

    fx = best_x[3] if best_x and best_x[0] else None
    fy = best_y[3] if best_y and best_y[0] else None
    bands = (best_x or best_y)[4] if (best_x or best_y) else (base_x, base_y)
    crops = (best_x or best_y)[5] if (best_x or best_y) else (None, None)
    if log and best_y and not best_y[0]:
        log(f"      y 轴拟合质量不足（最佳 R²={best_y[2]:.5f} < {min_r2}），"
            f"不输出轴范围，请人工确认")
    out = {"bands": list(bands), "image": str(image_path)}
    if fx:
        out["x"] = _edges(fx, left, right)
    if fy:
        out["y"] = _edges(fy, bottom, top)
    if debug_crops and crops[0] is not None:
        d = Path(debug_crops)
        d.mkdir(parents=True, exist_ok=True)
        iio.imwrite(d / f"{Path(image_path).stem}_xlabels.png", crops[0])
        iio.imwrite(d / f"{Path(image_path).stem}_ylabels.png", crops[1])
    return out
