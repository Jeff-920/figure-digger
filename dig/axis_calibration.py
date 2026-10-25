"""Calibrate model-read tick values against measured tick strokes.

The model reads the axis labels; this module fits those values to the tick marks
that are actually visible in the panel.  It only uses image geometry and plain
least-squares arithmetic, with no text recognition engine.
"""

import math

import cv2
import numpy as np


def tick_positions(img, frame, axis='y', side='left'):
    """Locate short low-saturation tick strokes without reading any numbers.

    Positions use Cartesian plot fractions, independent of the labels' values.
    The continuous spine is excluded from the strip; colour curves cannot become
    dark ticks. Grid intersections and frame endpoints are valid tick locations.
    """
    left, top, right, bottom = map(int, frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    ink = (gray < 235) & (sat < 60)
    depth = max(8, min(20, round(.03 * min(right - left, bottom - top))))
    h, w = gray.shape
    if axis == 'y':
        spine = right if side == 'right' else left
        a, b = ((max(0, spine - depth), max(0, spine - 2)) if side == 'right' else
                (min(w, spine + 3), min(w, spine + depth + 1)))
        strip = ink[max(0, top):min(h, bottom + 1), a:b].copy()
        profile = strip.sum(axis=1)
        span = max(1, bottom - top)
    else:
        a, b = max(0, bottom - depth), max(0, bottom - 2)
        strip = ink[a:b, max(0, left):min(w, right + 1)].copy()
        profile = strip.sum(axis=0)
        span = max(1, right - left)
    flags = profile >= max(3, int(.25 * depth))
    runs = []
    start = None
    for i, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start <= max(8, .025 * span):
                p = (start + i - 1) / 2.
                runs.append(1. - p / span if axis == 'y' else p / span)
            start = None
    return sorted(set([0., 1.] + runs))


def calibrate_vlm_ticks(img, frame, reading):
    """Fit model-read tick values to measured strokes; infer frame endpoints.

    A highest visible label can be below the top edge (100 at 5/6 height implies
    a 120 upper frame limit). Never silently identify label extrema with edges.
    Invalid or insufficient evidence leaves the original reading with diagnostics.
    """
    sources = [('x', reading.get('x'), 'left')]
    sources.extend(('y', r, r.get('side', 'left')) for r in reading.get('y_axes') or []
                   if isinstance(r, dict))
    for axis, source, side in sources:
        if not isinstance(source, dict) or source.get('scale', 'linear') != 'linear':
            continue
        ticks = source.get('ticks') or []
        if not ticks:
            continue  # Older configs/mocks remain readable.
        positions = tick_positions(img, frame, axis, side)
        pairs, snapped = [], 0
        for tick in ticks:
            try:
                value, position = float(tick['value']), float(tick['position'])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(value) and 0 <= position <= 1):
                continue
            nearest = min(positions, key=lambda q: abs(q - position))
            if abs(nearest - position) <= .04:
                position = nearest
                snapped += 1
            if not any(abs(position - p) < .01 or abs(value - v) < 1e-9 for p, v in pairs):
                pairs.append((position, value))
        if len(pairs) < 2 or max(p for p, v in pairs) - min(p for p, v in pairs) < .3:
            source['tick_calibration'] = {'accepted': False, 'reason': '刻度位置证据不足'}
            continue
        px = sum(p for p, v in pairs) / len(pairs)
        vy = sum(v for p, v in pairs) / len(pairs)
        variance = sum((p - px) ** 2 for p, v in pairs)
        slope = sum((p - px) * (v - vy) for p, v in pairs) / variance
        intercept = vy - slope * px
        error = sum((v - (slope * p + intercept)) ** 2 for p, v in pairs)
        total = sum((v - vy) ** 2 for p, v in pairs)
        r2 = 1. - error / max(total, 1e-20)
        accepted = slope > 0 and r2 >= .995 and snapped >= min(3, len(pairs))
        evidence = {'accepted': accepted, 'r2': round(r2, 6), 'pairs': pairs,
                    'snapped': snapped,
                    'original_range': [source.get('min'), source.get('max')]}
        if accepted:
            # Coarse positioning may leave <1% residual error. Snap only values
            # close to the label step's quarter grid, never force a whole step.
            values = sorted(v for p, v in pairs)
            step = min(b - a for a, b in zip(values, values[1:]) if b > a)
            quantum = step / 4.

            def clean(value, prior):
                if isinstance(prior, (int, float)) and abs(prior - value) < .015 * abs(slope):
                    return float(prior)
                rounded = round(value / quantum) * quantum
                return round(rounded if abs(rounded - value) < .015 * abs(slope) else value, 8)

            source['min'], source['max'] = (
                clean(intercept, source.get('min')),
                clean(intercept + slope, source.get('max')))
            evidence['frame_range'] = [source['min'], source['max']]
        else:
            evidence['reason'] = '刻度位置与数值不一致，未覆盖原范围'
        source['tick_calibration'] = evidence
    return reading
