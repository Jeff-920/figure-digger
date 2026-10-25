"""Pair every figure with *its own* caption, then pick the figures a request asks for.

Two jobs:

1. Caption pairing (local, no API cost).
   A journal page often carries several figures and several "Figure N" text lines -
   real captions plus references in the body ("Figure 3 shows that …"). Taking "the
   first caption on the page" - the old behaviour - labels the wrong figure. Here each
   figure is paired with the caption that overlaps it horizontally and sits closest to
   its bottom (or top) edge; body references lose on both distance and phrasing.

2. Figure selection from one sentence.
   Pure figure-number requests are matched locally. Semantic requests inspect every
   original figure with full captions, then reconcile all axis/condition evidence,
   consulting paper-wide figure mentions when needed. Failed or unclear inspections
   remain unresolved; thumbnails and shortlists never establish absence.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imgio as iio
import numpy as np

import vlm_tasks as vt

SHEET_MAX = 12          # 仅用于诊断对照图，检索不再按此截断候选
TILE_W, TILE_H = 430, 300

LABEL_NUM_RE = re.compile(r"(?:fig(?:ure)?s?\.?|图|圖)\s*(\d+)", re.IGNORECASE)
# 正文里的引用句：Figure 3 shows… / Figure 2) is used… / Figure 15 and Figure 16 gives…
BODY_REF_RE = re.compile(
    r"^\s*fig(?:ure)?s?\.?\s*\d+[a-z]?\s*"
    r"(?:and\s+fig(?:ure)?s?\.?\s*\d+[a-z]?\s*)?[)\]]?\s*"
    r"(?:shows?|compares?|presents?|depicts?|indicates?|illustrates?|gives?|"
    r"is|are|was|were|can|will|has|have|also|and)\b", re.IGNORECASE)
TITLE_RE = re.compile(r"^\s*fig(?:ure)?s?\.?\s*\d+[a-z]?\s*[.:\-–—)]?\s+\S", re.IGNORECASE)


def looks_like_title(text):
    """True for "Figure 5. Spray penetration …", False for "Figure 5 shows …"."""
    t = (text or "").strip()
    if not t or BODY_REF_RE.match(t):
        return False
    return bool(TITLE_RE.match(t))


def label_number(label):
    m = re.search(r"(\d+)", label or "")
    return int(m.group(1)) if m else None


def _overlap_x(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def _touching(a, b, gap=8.0, stack_gap=60.0):
    """Two boxes that belong to one figure split into stacked/side-by-side pieces.

    Vertically stacked chart panels often have a real gutter (tick labels and
    the upper panel's x title) while still sharing one caption below the lower
    panel.  Side-by-side panels are only merged when truly adjacent, otherwise
    two columns from different figures could inherit each other's caption.
    """
    dx = max(a[0], b[0]) - min(a[2], b[2])      # >0 when separated horizontally
    dy = max(a[1], b[1]) - min(a[3], b[3])      # >0 when separated vertically
    overlap_x = _overlap_x(a, b)
    overlap_y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    stacked = overlap_x >= 0.6 * min(a[2] - a[0], b[2] - b[0]) and dy <= stack_gap
    side_by_side = overlap_y >= 0.6 * min(a[3] - a[1], b[3] - b[1]) and dx <= gap
    return stacked or side_by_side


def attach_captions(cards, captions_by_page, log=None):
    """Give each card the caption that belongs to it (in place).

    cards:     [{"id", "page", "bbox", ...}] - bbox in PDF points
    captions:  {page: [{"label", "text", "bbox"}, ...]} from triage_pdf
    """
    for card in cards:
        card.setdefault("caption", "")
        card.setdefault("label", "")
        card.pop("caption_kind", None)

    for page, group in _by_page(cards).items():
        caps = captions_by_page.get(page, []) or []
        if not caps:
            continue
        scored = []
        for card in group:
            fig = card["bbox"]
            for cap in caps:
                box = cap["bbox"]
                if _overlap_x(fig, box) < 0.5 * min(fig[2] - fig[0], box[2] - box[0]):
                    continue
                below, above = box[1] - fig[3], fig[1] - box[3]
                if 0 <= below <= 80:
                    gap, side = below, "below"
                elif 0 <= above <= 40:
                    gap, side = above, "above"
                else:
                    continue
                score = gap
                if not looks_like_title(cap["text"]):
                    score += 45          # 正文引用句，能不选就不选
                if side == "above":
                    score += 20
                scored.append((score, card["id"], cap))
        scored.sort(key=lambda t: t[0])
        taken, used_caps, leftovers, direct = {}, set(), set(c["id"] for c in group), set()
        for _, cid, cap in scored:      # 第一轮：一条图注只给一张图
            if cid not in leftovers or id(cap) in used_caps:
                continue
            taken[cid] = cap
            used_caps.add(id(cap))
            leftovers.discard(cid)
            direct.add(cid)
        # 第二轮：一张图常由好几块紧挨着的子图组成（上下拼接的照片阵列），图注只写在
        # 整块下面——离得远，但"贴在一起"这件事是可靠的，按这个把图注传过去。
        for card in group:
            if card["id"] not in leftovers:
                continue
            for other in group:
                if other["id"] == card["id"] or other["id"] not in taken:
                    continue
                if _touching(card["bbox"], other["bbox"]):
                    taken[card["id"]] = taken[other["id"]]
                    leftovers.discard(card["id"])
                    break
        for card in group:
            cap = taken.get(card["id"])
            if not cap:
                continue
            card["caption"] = cap["text"]
            card["caption_bbox"] = cap["bbox"]
            card["label"] = cap["label"]
            card["caption_kind"] = "title" if looks_like_title(cap["text"]) else "body-ref"
            card["caption_shared"] = card["id"] not in direct
    if log:
        paired = sum(1 for c in cards if c["caption"])
        log(f"      图注配对: {paired}/{len(cards)} 张图找到文字")
    return cards


def _by_page(cards):
    out = {}
    for c in cards:
        out.setdefault(c["page"], []).append(c)
    return out


def index_text(cards, max_caption=4000, numbers=None):
    """The plain-text figure list handed to the model (numbers = candidate numbers)."""
    numbers = numbers or list(range(1, len(cards) + 1))
    lines = []
    for i, c in zip(numbers, cards):
        parts = [f"{i}. {c.get('label') or '(无编号)'} (p{c['page']})"]
        if c.get("caption"):
            parts.append(f"图注: {c['caption'][:max_caption]}")
        if c.get("inner_text"):
            parts.append(f"图内文字: {c['inner_text'][:4000]}")
        if not c.get("caption") and not c.get("inner_text"):
            parts.append("（没有图注文字，只能看缩略图）")
        lines.append(" ｜ ".join(parts))
    return "\n".join(lines)


def build_contact_sheet(cards, out_dir, numbers=None, cols=3,
                        tile_w=TILE_W, tile_h=TILE_H):
    """Thumbnails of the candidates in a numbered grid - one image for one API call."""
    tiles = []
    numbers = numbers or list(range(1, len(cards) + 1))
    for card, num in zip(cards, numbers):
        img = iio.imread(card["thumb"]) if card.get("thumb") else None
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(tile_w / float(w), tile_h / float(h))
        resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        tile = np.full((tile_h, tile_w, 3), 255, dtype="uint8")
        y0 = (tile_h - resized.shape[0]) // 2
        x0 = (tile_w - resized.shape[1]) // 2
        tile[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (180, 180, 180), 1)
        cv2.rectangle(tile, (2, 2), (78, 34), (255, 255, 255), -1)
        cv2.putText(tile, str(num), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None
    rows = []
    for i in range(0, len(tiles), cols):
        chunk = tiles[i:i + cols]
        while len(chunk) < cols:
            chunk.append(np.full((tile_h, tile_w, 3), 255, dtype="uint8"))
        rows.append(cv2.hconcat(chunk))
    sheet = cv2.vconcat(rows)
    out_path = Path(out_dir) / "contact_sheet.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, sheet)
    return out_path


def local_match(want, cards):
    """A figure number in the request ("Figure 5" / "图5") -> no model call at all."""
    wanted = {int(n) for n in LABEL_NUM_RE.findall(want or "")}
    if not wanted:
        return [], ""
    remainder = LABEL_NUM_RE.sub("", want or "")
    remainder = re.sub(r"\b(?:and|or)\b|[\s,，、;；与和及]", "", remainder, flags=re.I)
    if remainder:
        return [], ""    # “Figure 5 中的贯穿距”仍需要语义和子图核对
    ids = [c["id"] for c in cards if label_number(c.get("label")) in wanted]
    if ids:
        return ids, f"需求里的编号直接命中图 {sorted(wanted)}"
    return [], ""


def _candidates(cards, selected=(), screened=None):
    """One row per candidate figure. `screened` marks the ones the model actually saw."""
    out = []
    for n, card in enumerate(cards, start=1):
        out.append({"n": n, "id": card["id"], "label": card.get("label") or "",
                    "page": card["page"], "caption": (card.get("caption") or "")[:200],
                    "selected": card["id"] in selected,
                    "screened": (n in screened) if screened is not None else True})
    return out


def _pick(data, cards):
    """Model reply -> list of card ids (numbers are 1-based positions in the index)."""
    picked = []
    values = data.get("selected") or []
    if not isinstance(values, list):
        return []
    for n in values:
        if isinstance(n, bool) or not re.fullmatch(r"[0-9]+", str(n)):
            continue
        n = int(n)
        if 1 <= n <= len(cards) and cards[n - 1]["id"] not in picked:
            picked.append(cards[n - 1]["id"])
    return picked


def render_selection_images(pdf, cards, out_dir, dpi=220, margin=36):
    """Render individual figures with axis/title context missing from PDF image objects."""
    folder = Path(out_dir) / "originals"
    folder.mkdir(parents=True, exist_ok=True)
    for i, card in enumerate(cards, 1):
        page = pdf.pages[card["page"] - 1]
        a = card["bbox"]
        box = [max(0, a[0] - margin), max(0, a[1] - margin),
               min(page.width, a[2] + margin), min(page.height, a[3] + margin)]
        # Clip expanded margins at the midpoints between neighbouring figures.
        for other in cards:
            if other is card or other["page"] != card["page"]:
                continue
            b = other["bbox"]
            if _overlap_x(a, b) > 0:
                if b[3] <= a[1]:
                    box[1] = max(box[1], (b[3] + a[1]) / 2)
                elif b[1] >= a[3]:
                    box[3] = min(box[3], (a[3] + b[1]) / 2)
            if min(a[3], b[3]) > max(a[1], b[1]):
                if b[2] <= a[0]:
                    box[0] = max(box[0], (b[2] + a[0]) / 2)
                elif b[0] >= a[2]:
                    box[2] = min(box[2], (a[2] + b[0]) / 2)
        try:
            path = folder / f"{i:03d}.png"
            page.crop(tuple(box), strict=False).to_image(resolution=dpi).original.save(path)
            card["selection_image"] = path
        except Exception:  # Keep the native export when page rendering fails.
            card["selection_image"] = card.get("thumb")


def _verdict(inspection):
    """False requires explicit contradictory evidence; missing fields stay unknown."""
    if inspection.get("is_data_chart") is False:
        return False
    rows = [p for p in (inspection.get("panels") or []) if isinstance(p, dict)]
    if not rows:
        return None
    values = []
    for p in rows:
        if p.get("axes_match") is False or p.get("conditions_match") is False:
            values.append(False)
        elif (p.get("matches_request") is True and p.get("axes_match") is True
              and p.get("conditions_match") is True):
            values.append(True)
        else:
            values.append(None)
    if True in values:
        return True
    return False if all(v is False for v in values) else None


def select(want, cards, vlm=None, out_dir=None, log=None, body_text=None, deep=False):
    """Read every candidate at original resolution, then resolve missing body evidence.

    No text shortlist or thumbnail confidence can hide a candidate from vision.
    A missing/failed model response is uncertainty, never proof of absence.
    """
    decision = {"want": want, "method": "none", "ids": [], "missing": True,
                "reason": "这篇里没找到任何候选图", "available": "",
                "body_used": False, "body_chars": 0, "body_reason": "", "body_tokens": 0}
    if not cards:
        decision["candidates"] = []
        return decision
    ids, how = local_match(want, cards)
    if ids:
        decision.update(method="label", ids=ids, missing=False, reason=how,
                        candidates=_candidates(cards, selected=ids, screened=set()))
        return decision
    if vlm is None:
        decision.update(missing=None, reason="未启用 VLM：只能按图号找图，无法按内容判断是否存在",
                        candidates=_candidates(cards, screened=set()))
        return decision

    inspections, screened = {}, set()
    fatal_error = ""
    def inspect_one(n, card, client):
        path = card.get("selection_image") or card.get("thumb")
        try:
            data = vt.inspect_figure(client, want, index_text([card], numbers=[n]), path)
            # A missing image is visible in the audit and cannot justify a confident rejection.
            if path is None:
                data["image_missing"] = True
            return {k: v for k, v in data.items() if k != "_raw"}, path is not None
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "panels": []}, False

    def record(n, card, result):
        nonlocal fatal_error
        inspections[n], seen = result
        if seen:
            screened.add(n)
        error = inspections[n].get("error", "")
        if re.search(r"\b(?:401|402)\b", error):
            fatal_error = error
        if log:
            verdict = _verdict(inspections[n])
            log(f"      原图语义核对 {n}/{len(cards)} {card.get('label') or card['id']}: "
                + ("符合" if verdict is True else "不符" if verdict is False else "待核对"))

    try:
        workers = max(1, min(3, int(os.environ.get("DIG_SELECT_WORKERS", "2"))))
    except ValueError:
        workers = 2
    # Custom/mock clients without isolated request state retain their serial contract.
    if not callable(getattr(vlm, "fork", None)):
        workers = 1
    if workers == 1:
        for n, card in enumerate(cards, 1):
            result = ({"error": fatal_error, "panels": []}, False) if fatal_error else inspect_one(n, card, vlm)
            record(n, card, result)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Bounded batches: fatal authentication errors cannot enqueue the entire paper.
            for start in range(0, len(cards), workers):
                batch = list(enumerate(cards[start:start + workers], start + 1))
                if fatal_error:
                    for n, card in batch:
                        record(n, card, ({"error": fatal_error, "panels": []}, False))
                    continue
                jobs = []
                for n, card in batch:
                    child = vlm.fork()
                    jobs.append((n, card, child, pool.submit(inspect_one, n, card, child)))
                for n, card, child, future in jobs:
                    result = future.result()
                    for key, value in child.usage.items():
                        vlm.usage[key] = vlm.usage.get(key, 0) + value
                    record(n, card, result)

    # Always carry the FULL candidate inventory, even if most figures were rejected.
    evidence_index = "\n".join(
        index_text([c], numbers=[n]) + "\n原图读图记录: "
        + json.dumps(inspections[n], ensure_ascii=False)
        for n, c in enumerate(cards, 1))
    unresolved = {n for n, d in inspections.items() if _verdict(d) is None}
    positives = [n for n, d in inspections.items() if _verdict(d) is True]
    need_body = deep or bool(unresolved) or not positives or any(
        d.get("needs_body") for d in inspections.values())
    digest = ""
    if need_body and body_text is not None:
        try:
            digest = body_text() or ""
        except Exception as exc:
            decision["body_reason"] = f"正文提取失败：{type(exc).__name__}: {exc}"
    decision.update(body_used=bool(digest), body_chars=len(digest))
    usage0 = dict(getattr(vlm, "usage", {}))
    try:
        if fatal_error:
            raise RuntimeError(fatal_error)
        data = vt.select_figures_by_evidence(vlm, want, evidence_index, digest)
        requested_labels = {int(x) for x in LABEL_NUM_RE.findall(want or "")}
        picked = _pick(data, cards)
        # Explicit visible axis/condition contradictions cannot be overturned by prose.
        forbidden = {cards[n - 1]["id"] for n, d in inspections.items() if _verdict(d) is False}
        picked = [cid for cid in picked if cid not in forbidden and (
            not requested_labels or label_number(next(c for c in cards if c["id"] == cid).get("label"))
            in requested_labels)]
        model_unresolved = set(_pick({"selected": data.get("unresolved") or []}, cards))
        # Missing/failed images remain unresolved even if the summarizer claims otherwise.
        uncertain_ids = model_unresolved | {cards[n - 1]["id"] for n, d in inspections.items()
                                           if _verdict(d) is None or d.get("error")
                                           or d.get("image_missing")}
        uncertain_ids -= set(picked)
        decision.update(method="vlm-originals", ids=picked,
                        missing=False if picked else None if uncertain_ids else True,
                        unresolved_ids=sorted(uncertain_ids),
                        reason=str(data.get("reason") or ""),
                        available=str(data.get("available") or ""),
                        confidence=data.get("confidence"), evidence=data.get("evidence") or [])
        if digest:
            decision["body_reason"] = decision["reason"]
    except Exception as exc:
        # Retain individually established matches, but report incomplete search honestly.
        picked = [cards[n - 1]["id"] for n in positives]
        uncertain_ids = {cards[n - 1]["id"] for n in unresolved}
        decision.update(method="error", ids=picked, missing=False if picked else None,
                        unresolved_ids=sorted(uncertain_ids),
                        reason=f"选图汇总失败：{type(exc).__name__}: {exc}")
    usage = getattr(vlm, "usage", {})
    if digest:
        decision["body_tokens"] = sum(usage.get(k, 0) - usage0.get(k, 0)
                                      for k in ("prompt_tokens", "completion_tokens"))
    rows = _candidates(cards, selected=decision["ids"], screened=screened)
    for row in rows:
        d = inspections[row["n"]]
        row["inspection"] = d
        row["status"] = ("selected" if row["selected"] else "unresolved"
                         if row["id"] in decision.get("unresolved_ids", []) else "rejected")
    decision.update(candidates=rows, screened_n=len(screened), inspected_n=len(inspections))
    if out_dir is not None:
        # Optional diagnostics only; models receive individual images above.
        for start in range(0, len(cards), SHEET_MAX):
            folder = Path(out_dir) if start == 0 else Path(out_dir) / f"batch_{start // SHEET_MAX + 1}"
            build_contact_sheet(cards[start:start + SHEET_MAX], folder,
                                numbers=list(range(start + 1, min(start + SHEET_MAX, len(cards)) + 1)))
    return decision
