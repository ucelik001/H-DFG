from __future__ import annotations

import colorsys
import html
import math
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
from pipeline import agg_level_index, norm_code
from route_engine import (
    _OPP,
    _arr,
    _ideal_coord,
    _inf,
    _orthogonalize,
    _path_cost,
    _port_xy,
    _side_candidates,
    _simplify_points,
    _stub_xy,
    manhattan_fallback,
    r4_route_between,
    r4_separate,
    r4_spread,
)

# Depends on EventLogConfig/HierLevel in section1_cfg_pipeline

# ============================================================================
# VISUAL / LAYOUT SETTINGS — independent of data schema (cfg), only determines
# how the visualization looks. If you want to add sliders in Streamlit later,
# you would connect them to the fields of this dataclass; each user session
# maintains their own RenderSettings() instance (no module-level global).
# ============================================================================
@dataclass
class RenderSettings:
    order_by: str = "code"                 # "degree" | "flow" | "volume" | "code"
    rankdir: str = "LR"                    # "LR" | "TB"
    top_child_rows: int = 0                # subrow limit in box; 0 = all
    edge_min_count: int = 1
    label_top_n: int = 40
    edge_min_w: float = 1.0
    edge_max_w: float = 2.0
    font: str = "Arial, Helvetica, sans-serif"
    target_aspect_top: float = 1.8
    iso_aspect: float = 10.0
    pad: float = 14
    gap_leaf: float = 60
    gap_group: float = 60
    gap_iso: float = 70
    margin: float = 50
    header_h: float = 78
    char_w: float = 0.60
    strip_w: float = 15
    activity_label: str = "Operation"


# ============================================================================
# Box — layout tree node (root / group / leaf / flow)
# ============================================================================
class Box:
    def __init__(self, kind, path, depth):
        self.kind = kind
        self.path = path
        self.depth = depth
        self.base = None 
        self.level_label = ""
        self.code = ""
        self.name = ""
        self.n = 0
        self.dur_mean = None
        self.dur_min = None
        self.dur_max = None
        self.child_rows = []
        self.grid_cols = 1
        self.grid_col_w = []
        self.short = ""
        self.tip = ""
        self.lines = []
        self.children = []
        self.flows = []
        self.flows_before = []
        self.flows_after = []
        self.score = 0.5
        self.side_before: bool | None = None
        self.iso_gap_y: float | None = None
        self.rx = self.ry = 0.0
        self.x = self.y = self.w = self.h = 0.0
        self.core_w = self.core_h = 0.0
        self.core_dx = self.core_dy = 0.0
        self.id: str | None = None


# ============================================================================
# node_id classification — no longer uses SEP-split. Section 1 generates node_ids
# with fixed prefixes ("start_", "finish_", "flow_L", or bare base) so we use simple
# prefix checks to distinguish them; the hierarchy path is read from
# pipeline["base_paths"] (no string parsing).
# ============================================================================
def _classify_node_id(nid: str):
    if nid.startswith("flow_L"):
        return "flow", nid
    if nid.startswith("start_"):
        return "start", nid[len("start_"):]
    if nid.startswith("finish_"):
        return "finish", nid[len("finish_"):]
    return "point", nid   # bare "lvl..." -> point event, node_id itself is the base


def fmt_dur(sec):
    sec = abs(float(sec))
    if sec < 60:
        return f"{sec:.0f} sec"
    if sec < 3600:
        return f"{sec / 60:.0f} min"
    if sec < 86400:
        return f"{sec / 3600:.1f} hr"
    return f"{sec / 86400:.1f} day"


def duration_to_color(value, vmin, vmax):
    if vmax <= vmin:
        t = 0.5
    else:
        t = min(max((value - vmin) / (vmax - vmin), 0.0), 1.0)
    green, yellow, red = (46, 204, 113), (241, 196, 15), (231, 76, 60)
    if t < 0.5:
        t2 = t / 0.5
        r = int(green[0] + (yellow[0] - green[0]) * t2)
        g = int(green[1] + (yellow[1] - green[1]) * t2)
        b = int(green[2] + (yellow[2] - green[2]) * t2)
    else:
        t2 = (t - 0.5) / 0.5
        r = int(yellow[0] + (red[0] - yellow[0]) * t2)
        g = int(yellow[1] + (red[1] - yellow[1]) * t2)
        b = int(yellow[2] + (red[2] - yellow[2]) * t2)
    return f"#{r:02x}{g:02x}{b:02x}"


# ============================================================================
# Build box tree from pipeline output
# ============================================================================
def _collect_ends(b: Box, cfg) -> list:
    """Returns the actual node_ids of a leaf/group box (start_.../finish_.../bare base).
    No longer uses string split — the b.base written during build_view is used."""
    if b.kind == "leaf":
        base = b.base
        return [f"start_{base}", f"finish_{base}"] if cfg.show_start_finish_box else [base]
    if b.kind == "group":
        ends = []
        for c in b.children:
            ends += _collect_ends(c, cfg)
        return ends
    return []


def _flow_side_before(nid, target, dfg_full: dict, scores: dict, cfg) -> bool:
    """It determines whether a flow event (Place Order, etc.) occurs *before* or *after* 
    a specific box in the process by examining the *actual* DFG weights of the flows entering or leaving that box."""
    ends = _collect_ends(target, cfg)
    into_t = sum(dfg_full.get((nid, e), 0) for e in ends)
    out_t = sum(dfg_full.get((e, nid), 0) for e in ends)
    if into_t == out_t:
        return scores.get(nid, 0.5) < 0.5
    return into_t > out_t


def total_degree(b: Box, deg: dict, cfg) -> int:
    """Returns the total degree of a box (and its descendants) in the filtered DFG.
    A degree of 0 is considered 'isolated', and layout() separates such nodes."""
    d = sum(deg.get(f.path[0], 0) for f in b.flows)
    if b.kind == "leaf":
        d += sum(deg.get(e, 0) for e in _collect_ends(b, cfg))
    elif b.kind == "group":
        for c in b.children:
            d += total_degree(c, deg, cfg)
    return d


def build_view(pipeline: dict, cfg, agg_level, settings: RenderSettings, only_codes=None):
    """pipeline: dictionary returned by section1_cfg_pipeline.build_pipeline().
    only_codes: filter based on the outermost (0th level) hierarchy code
    (e.g., {"009", "017"}); None = all."""

    ac_f = pipeline["activities_filtered"]
    base_paths = pipeline["base_paths"]
    node_stats = pipeline["node_stats"]
    names = pipeline["names"]
    child_rows_all = pipeline["child_rows"]
    scores = pipeline["scores"]

    root = Box("root", (), 0)
    index = {(): root}
    leaf_box, flow_box, by_last = {}, {}, {}

    def get_group(path):
        if path in index:
            return index[path]
        parent = get_group(path[:-1])
        b = Box("group", path, len(path))
        lvl = cfg.hierarchy_levels[len(path) - 1]
        b.level_label = lvl.label
        b.code = path[-1]
        b.name = names.get(path, "")
        b.short = f"{lvl.label} {path[-1]}"
        parent.children.append(b)
        index[path] = b
        by_last[(len(path), path[-1])] = path
        return b

    idx = agg_level_index(cfg, agg_level)

    bases = set()
    for nid in ac_f:
        kind, key = _classify_node_id(nid)
        if kind != "flow":
            bases.add(key)
    bases = sorted(bases)

    for base in bases:
        parts = base_paths.get(base)
        if parts is None:
            continue
        if only_codes and parts[0] not in only_codes:
            continue
        parent = get_group(parts[:-1]) if len(parts) > 1 else root
        stt = node_stats.get(base, {})
        n = int(stt.get("n", 1))
        code = parts[-1]
        nm = names.get(parts, "")
        b = Box("leaf", parts, len(parts))
        b.base = base
        b.n = n
        b.code = code
        b.name = nm
        b.level_label = (settings.activity_label if agg_level == "activity"
                         else cfg.hierarchy_levels[idx].label)
        b.dur_mean = stt.get("dur_mean")
        b.dur_min = stt.get("dur_min")
        b.dur_max = stt.get("dur_max")
        rows = child_rows_all.get(base, [])
        b.child_rows = rows[: settings.top_child_rows] if settings.top_child_rows else rows
        b.short = f"{code} · {nm}" if nm else code
        b.tip = b.short + f"\nn={n:,}" + (f"\navg={fmt_dur(b.dur_mean)}" if b.dur_mean is not None else "")
        sc = [scores[k] for k in (f"start_{base}", f"finish_{base}", base) if k in scores]
        b.score = float(np.mean(sc)) if sc else 0.5
        parent.children.append(b)
        leaf_box[base] = b
        index[parts] = b
        by_last[(len(parts), parts[-1])] = parts

    for nid in ac_f:
        kind, _ = _classify_node_id(nid)
        if kind != "flow":
            continue
        attach = pipeline["flow_attach_info"].get(nid)
        if attach is None:
            continue
        lvl, code = attach
        if only_codes and lvl == 0 and norm_code(code) not in {norm_code(c) for c in only_codes}:
            continue
        path = by_last.get((lvl + 1, str(code)))
        if path is None:   # Matching that is tolerant of format differences, such as '9' / '009' / '9.0'
            path = next((p for (ln, c), p in by_last.items()
                         if ln == lvl + 1 and norm_code(c) == norm_code(code)), None)
        if path is None:
            continue
        target = index[path]
        label = pipeline["node_label_lookup"].get(nid, nid)
        b = Box("flow", (nid,), target.depth + 1)
        b.short = str(label)
        b.lines = [(str(label), 9, True, "#6E2C00")]
        b.tip = f"{label}  ({target.short})"
        b.score = scores.get(nid, 0.5)
        b.side_before = _flow_side_before(nid, target, pipeline["dfg_full"], scores, cfg)
        target.flows.append(b)
        flow_box[nid] = b

    def agg(b):
        if b.kind == "group":
            for c in b.children:
                agg(c)
            b.n = sum(c.n for c in b.children)
            b.score = (sum(c.score * c.n for c in b.children) / b.n) if b.n else 0.5
            means = [(c.dur_mean, c.n) for c in b.children if c.dur_mean is not None and c.n]
            b.dur_mean = (sum(d * n for d, n in means) / sum(n for _, n in means)) if means else None
            mins_ = [c.dur_min for c in b.children if c.dur_min is not None]
            maxs_ = [c.dur_max for c in b.children if c.dur_max is not None]
            b.dur_min = min(mins_) if mins_ else None
            b.dur_max = max(maxs_) if maxs_ else None
        elif b.kind == "root":
            for c in b.children:
                agg(c)
    agg(root)

    def prune(b):
        b.children = [c for c in b.children if c.kind != "group" or (prune(c) or c.children or c.flows)]
    prune(root)
    return root, leaf_box, flow_box


# ============================================================================
# Layout engine compeletely geometric
# ============================================================================
def tw(text, size, settings: RenderSettings, bold=False):
    return len(text) * size * (settings.char_w + (0.06 if bold else 0.0))


def shelf_pack(sizes, gap, target_aspect=1.6, max_width=None):
    if not sizes:
        return [], 0.0, 0.0

    def run(W):
        pos, x, y, rowh, usedw = [], 0.0, 0.0, 0.0, 0.0
        for w, h in sizes:
            if x > 0 and x + w > W + 1e-6:
                y += rowh + gap
                x, rowh = 0.0, 0.0
            pos.append((x, y))
            x += w + gap
            rowh = max(rowh, h)
            usedw = max(usedw, x - gap)
        return pos, usedw, y + rowh

    if max_width is not None:
        return run(max_width)
    maxw = max(w for w, _ in sizes)
    total = sum(w for w, _ in sizes) + gap * (len(sizes) - 1)
    best = None
    for W in np.linspace(maxw, max(total, maxw), 48):
        pos, uw, uh = run(W)
        score = uw * uh * (1 + abs(math.log((uw / uh) / target_aspect)))
        if best is None or score < best[0]:
            best = (score, pos, uw, uh)
    if best is None:
        pos, uw, uh = run(maxw)
        return pos, uw, uh
    return best[1], best[2], best[3]


def shelf_pack_by_columns(sizes, gap, n_cols):
    if not sizes:
        return [], 0.0, 0.0
    n_cols = max(1, min(n_cols, len(sizes)))
    col_w = [0.0] * n_cols
    for i, (w, h) in enumerate(sizes):
        c = i % n_cols
        col_w[c] = max(col_w[c], w)
    col_x = [0.0] * n_cols
    for c in range(1, n_cols):
        col_x[c] = col_x[c - 1] + col_w[c - 1] + gap
    total_w = col_x[-1] + col_w[-1]
    pos, row_y, row_h = [], 0.0, 0.0
    for i, (w, h) in enumerate(sizes):
        c = i % n_cols
        if c == 0 and i > 0:
            row_y += row_h + gap
            row_h = 0.0
        pos.append((col_x[c], row_y))
        row_h = max(row_h, h)
    total_h = row_y + row_h
    return pos, total_w, total_h


def _order(items, settings: RenderSettings):
    if settings.order_by == "volume":
        return sorted(items, key=lambda b: (-b.n, b.path))
    if settings.order_by == "code":
        return sorted(items, key=lambda b: b.path)
    return sorted(items, key=lambda b: (b.score, b.path))


def _hdr(depth):
    return {1: 13, 2: 11}.get(depth, 10)


def _hdr2(depth):
    return max(8, _hdr(depth) - 3)


def _hdr3(depth):
    return max(7, _hdr(depth) - 5)


def _fmt_metric_line(n, dur_mean, dur_min, dur_max, mode):
    if mode == "perf":
        if dur_mean is None:
            return "(no data)"
        dmin = fmt_dur(dur_min) if dur_min is not None else "-"
        dmax = fmt_dur(dur_max) if dur_max is not None else "-"
        return f"(avg={fmt_dur(dur_mean)}, min={dmin}, max={dmax})"
    return f"(n={n:,})"


def _header_lines(level_label, code, name, n, dur_mean, dur_min, dur_max, mode, fs1, fs2, fs3):
    lines = [(f"{level_label} Code: {code}", fs1, True)]
    if name:
        lines.append((name, fs2, False))
    lines.append((_fmt_metric_line(n, dur_mean, dur_min, dur_max, mode), fs3, False))
    return lines


def _header_height(lines):
    return sum(fsz + 3.5 for _, fsz, _ in lines)


def _grid_pack_rows(n):
    if n <= 0:
        return 1, 1
    n_rows = math.ceil(math.sqrt(n))
    n_cols = math.ceil(n / n_rows)
    return n_rows, n_cols


def _child_row_text(row: dict, mode: str) -> str:
    metric = _fmt_metric_line(row["n"], row["dur_mean"], row["dur_min"], row["dur_max"], mode)
    return f'{row["icon"]} {row["label_base"]} {metric}'


def stack_side(items, gap, vertical, max_extent=None):
    if not items:
        return [], 0.0, 0.0
    sizes = [(it.w, it.h) for it in items]
    if max_extent is None:
        max_extent = (sum(h for _, h in sizes) if vertical else sum(w for w, _ in sizes)) \
            + gap * (len(sizes) - 1)
    pos = []
    if vertical:
        cx = cy = col_w = total_w = total_h = 0.0
        for w, h in sizes:
            if cy > 0 and cy + h > max_extent + 1e-6:
                cx += col_w + gap
                cy, col_w = 0.0, 0.0
            pos.append((cx, cy))
            cy += h + gap
            col_w = max(col_w, w)
            total_w = max(total_w, cx + col_w)
            total_h = max(total_h, cy - gap)
    else:
        cx = cy = row_h = total_w = total_h = 0.0
        for w, h in sizes:
            if cx > 0 and cx + w > max_extent + 1e-6:
                cy += row_h + gap
                cx, row_h = 0.0, 0.0
            pos.append((cx, cy))
            cx += w + gap
            row_h = max(row_h, h)
            total_w = max(total_w, cx - gap)
            total_h = max(total_h, cy + row_h)
    return pos, total_w, total_h


def _finalize_box(b: Box, mode, core_w, core_h, cfg, settings: RenderSettings):
    b.core_w, b.core_h = core_w, core_h
    fl = _order(b.flows, settings)
    if not fl:
        b.w, b.h = core_w, core_h
        b.core_dx = b.core_dy = 0.0
        b.flows_before, b.flows_after = [], []
        return
    for f in fl:
        layout(f, mode, cfg, settings, None)
    before = [f for f in fl if (f.side_before if f.side_before is not None else f.score < 0.5)]
    after = [f for f in fl if not (f.side_before if f.side_before is not None else f.score < 0.5)]
    vertical = (settings.rankdir != "TB")
    gap = 6.0
    max_extent = max(core_h if vertical else core_w, 1.0) * 1.4
    bpos, bw, bh = stack_side(before, gap, vertical, max_extent)
    apos, aw, ah = stack_side(after, gap, vertical, max_extent)
    for f, (x, y) in zip(before, bpos):
        f.rx, f.ry = x, y
    for f, (x, y) in zip(after, apos):
        f.rx, f.ry = x, y
    if vertical:
        core_dx = (bw + gap) if before else 0.0
        for f in before:
            f.ry += max(0.0, (max(core_h, bh) - bh) / 2)
        for f in after:
            f.rx += core_dx + core_w + gap
            f.ry += max(0.0, (max(core_h, ah) - ah) / 2)
        b.core_dx, b.core_dy = core_dx, 0.0
        b.w = core_dx + core_w + ((gap + aw) if after else 0.0)
        b.h = max(core_h, bh, ah)
    else:
        core_dy = (bh + gap) if before else 0.0
        for f in before:
            f.rx += max(0.0, (max(core_w, bw) - bw) / 2)
        for f in after:
            f.ry += core_dy + core_h + gap
            f.rx += max(0.0, (max(core_w, aw) - aw) / 2)
        b.core_dx, b.core_dy = 0.0, core_dy
        b.w = max(core_w, bw, aw)
        b.h = core_dy + core_h + ((gap + ah) if after else 0.0)
    b.flows_before, b.flows_after = before, after


def layout(b: Box, mode, cfg, settings: RenderSettings, deg: dict | None):
    if b.kind == "leaf":
        fs1, fs2, fs3 = _hdr(b.depth), _hdr2(b.depth), _hdr3(b.depth)
        header_dummy = _header_lines(b.level_label, b.code, b.name, b.n, b.dur_mean, b.dur_min, b.dur_max,
                                     mode, fs1, fs2, fs3)
        wtxt = max(tw(txt, fsz, settings, bold) for txt, fsz, bold in header_dummy)
        header_h = _header_height(header_dummy)

        n_rows, n_cols = _grid_pack_rows(len(b.child_rows))
        ch_child = 0.0
        fsc = 8.5
        if b.child_rows:
            col_w = [0.0] * n_cols
            for i, row in enumerate(b.child_rows):
                c = i % n_cols
                col_w[c] = max(col_w[c], tw(_child_row_text(row, mode), fsc, settings, False))
            b.grid_cols, b.grid_col_w = n_cols, col_w
            grid_w = sum(col_w) + 8 * (n_cols - 1)
            ch_child = n_rows * (fsc + 4) + 6
            wtxt = max(wtxt, grid_w)

        core_w = max(96.0, wtxt + 14 + (2 * settings.strip_w if cfg.show_start_finish_box else 0))
        core_h = 8 + header_h + ch_child
        _finalize_box(b, mode, core_w, core_h, cfg, settings)
        return
    if b.kind == "flow":
        b.w, b.h = tw(b.lines[0][0], 9, settings, True) + 18, 22.0
        b.core_w, b.core_h = b.w, b.h
        b.core_dx = b.core_dy = 0.0
        return

    isolated = []
    if settings.order_by == "degree" and b.children:
        deg = deg or {}
        deg_of = {k: total_degree(k, deg, cfg) for k in b.children}
        connected = sorted([k for k in b.children if deg_of[k] > 0],
                           key=lambda k: (-deg_of[k], k.path))
        isolated = sorted([k for k in b.children if deg_of[k] == 0], key=lambda k: k.path)
        centered, left = deque(), True
        for k in connected:
            (centered.appendleft if left else centered.append)(k)
            left = not left
        kids = list(centered)
    else:
        kids = _order(b.children, settings)

    for k in kids + isolated:
        layout(k, mode, cfg, settings, deg)

    b.iso_gap_y = None
    cw = ch = 0.0
    if kids:
        gap = settings.gap_group if any(k.kind == "group" for k in kids) else settings.gap_leaf
        if b.depth == 0:
            pos, cw, ch = shelf_pack([(k.w, k.h) for k in kids], gap, settings.target_aspect_top)
        else:
            n_rows = math.ceil(math.sqrt(len(kids)))
            n_cols = math.ceil(len(kids) / n_rows)
            pos, cw, ch = shelf_pack_by_columns([(k.w, k.h) for k in kids], gap, n_cols)
        for k, (x, y) in zip(kids, pos):
            k.rx, k.ry = x, y
    if isolated:
        igap = settings.gap_group if any(k.kind == "group" for k in isolated) else settings.gap_leaf
        ipos, icw, ich = shelf_pack([(k.w, k.h) for k in isolated], igap,
                                    settings.iso_aspect, max_width=(cw if kids else None))
        y_off = ch + (settings.gap_iso if kids else 0.0)
        for k, (x, y) in zip(isolated, ipos):
            k.rx, k.ry = x, y + y_off
        b.iso_gap_y = ch + settings.gap_iso / 2 if kids else None
        cw, ch = max(cw, icw), y_off + ich

    pad = settings.pad if b.depth > 0 else 0
    if b.depth > 0:
        fs1, fs2, fs3 = _hdr(b.depth), _hdr2(b.depth), _hdr3(b.depth)
        header_dummy = _header_lines(b.level_label, b.code, b.name, b.n, b.dur_mean, b.dur_min, b.dur_max,
                                     mode, fs1, fs2, fs3)
        hdr = _header_height(header_dummy) + 8
        title_w = max(tw(txt, fsz, settings, bold) for txt, fsz, bold in header_dummy)
    else:
        title_w = 0.0
        hdr = 0
    for k in kids + isolated:
        k.rx += pad
        k.ry += hdr + pad
    if b.iso_gap_y is not None:
        b.iso_gap_y += hdr + pad
    core_w = max(cw, title_w) + 2 * pad
    core_h = hdr + pad + ch + pad
    b.children = kids + isolated
    _finalize_box(b, mode, core_w, core_h, cfg, settings)


def place(b: Box, ox, oy):
    ext_x, ext_y = ox + b.rx, oy + b.ry
    b.x, b.y = ext_x + b.core_dx, ext_y + b.core_dy
    for c in b.children:
        place(c, b.x, b.y)
    for f in b.flows_before + b.flows_after:
        f.x, f.y = ext_x + f.rx, ext_y + f.ry


def _walk(b: Box):
    yield b
    for c in list(b.children) + list(b.flows_before) + list(b.flows_after):
        yield from _walk(c)


def _hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _shorten_last_segment(points, L):
    if len(points) < 2:
        return list(points)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    seg_len = math.hypot(x1 - x0, y1 - y0) or 1.0
    t = 0.0 if seg_len <= L else (seg_len - L) / seg_len
    nx, ny = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
    return list(points[:-1]) + [(nx, ny)]


_JS = """
(function() {
    var svg = document.querySelector('svg'),
        layer = svg.getElementById('edges'),
        hl = svg.getElementById('hl');
    var edges = [].slice.call(svg.querySelectorAll('.e'));
    [].forEach.call(svg.querySelectorAll('.n'), function(n) {
        n.addEventListener('mouseenter', function() {
            var id = n.getAttribute('data-id');
            edges.forEach(function(e) {
                if (e.getAttribute('data-s') === id || e.getAttribute('data-t') === id) {
                    e.classList.add('hl');
                    hl.appendChild(e);
                } else e.classList.add('dim');
            });
        });
        n.addEventListener('mouseleave', function() {
            edges.forEach(function(e) {
                e.classList.remove('hl');
                e.classList.remove('dim');
                layer.appendChild(e);
            });
        });
    });
    var vb = svg.viewBox.baseVal,
        X0 = vb.x,
        Y0 = vb.y,
        W0 = vb.width,
        H0 = vb.height;

    function pt(ev) {
        var p = svg.createSVGPoint();
        p.x = ev.clientX;
        p.y = ev.clientY;
        return p.matrixTransform(svg.getScreenCTM().inverse());
    }
    svg.addEventListener('wheel', function(ev) {
        ev.preventDefault();
        var k = ev.deltaY < 0 ? 0.85 : 1.18,
            p = pt(ev);
        vb.x = p.x - (p.x - vb.x) * k;
        vb.y = p.y - (p.y - vb.y) * k;
        vb.width *= k;
        vb.height *= k;
    }, {
        passive: false
    });
    var drag = false,
        lx = 0,
        ly = 0;
    svg.addEventListener('mousedown', function(ev) {
        drag = true;
        lx = ev.clientX;
        ly = ev.clientY;
    });
    window.addEventListener('mouseup', function() {
        drag = false;
    });
    window.addEventListener('mousemove', function(ev) {
        if (!drag) return;
        var r = svg.getBoundingClientRect(),
            s = Math.max(vb.width / r.width, vb.height / r.height);
        vb.x -= (ev.clientX - lx) * s;
        vb.y -= (ev.clientY - ly) * s;
        lx = ev.clientX;
        ly = ev.clientY;
    });
    svg.addEventListener('dblclick', function() {
        vb.x = X0;
        vb.y = Y0;
        vb.width = W0;
        vb.height = H0;
    });
})();
"""

# ============================================================================
# Hiyerarşik harita için kat planı yönlendirme motoru (A2'yi yeniden kullanır)
# ============================================================================
def _hier_label_pos(pts):
    i = max(range(len(pts) - 1), key=lambda k: math.dist(pts[k], pts[k + 1]))
    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    return (x0 + x1) / 2, (y0 + y1) / 2


def route_hier_edges(nl, boxes, mode, settings):
    """nl: [(edge_index, source_box, target_box)] (döngü olmayan kenarlar).
    Döner: (route_of {ei: points}, dashed {ei}, info dict)"""
    by_id = {b.id: b for b in boxes}

    # --- ancestor groups: an edge can only pass through the ancestor groups of its terminal boxes
    ancs = {}

    def _anc(b, chain):
        ancs[b.id] = chain
        sub = (chain | {b.id}) if b.kind == "group" else chain
        for c in b.children:
            _anc(c, sub)
        for f in b.flows:  # Order-event box: the target group is also considered a "ancestor"
            _anc(f, chain | {b.id})

    _anc(boxes[0], frozenset())

    def rect(b):
        return (b.x, b.y, b.core_w, b.core_h)

    def ctr(b):
        return (b.x + b.core_w / 2, b.y + b.core_h / 2)

    leaves = [b for b in boxes if b.kind in ("leaf", "flow")]
    groups = [b for b in boxes if b.kind == "group" and b.depth > 0]

    def hdr_h(g):
        fs1, fs2, fs3 = _hdr(g.depth), _hdr2(g.depth), _hdr3(g.depth)
        return _header_height(_header_lines(g.level_label, g.code, g.name, g.n, g.dur_mean,
                                            g.dur_min, g.dur_max, mode, fs1, fs2, fs3)) + 8

    box_raw = _arr([rect(b) for b in leaves])
    hdr_raw = _arr([(g.x, g.y, g.core_w, hdr_h(g)) for g in groups])   # header text obstacle
    cont_raw = _arr([rect(g) for g in groups])
    cont_keys = [g.id for g in groups]

    w_max = settings.edge_max_w
    arrow_len = 5 + 1.1 * w_max
    margins = (6.0, 3.0, 1.5)
    corner = 4.0
    gap = max(2.5 * w_max + 2.0, 6.0)
    min_end = arrow_len * 1.1

    cache = {}

    def obs(mg, allowed):
        k = (mg, allowed)
        if k not in cache:
            parts = [_inf(box_raw, mg), _inf(hdr_raw, 1.0)]
            if allowed is not None and len(cont_raw):
                keep = np.array([c not in allowed for c in cont_keys])
                parts.append(_inf(cont_raw[keep], mg))
            parts = [p for p in parts if len(p)]
            cache[k] = np.vstack(parts) if parts else np.zeros((0, 4))
        return cache[k]

    def route_edge(a, sa, ta, b, sb, tb, hanan, use_forb):
        pa, pb = _port_xy(rect(a), sa, ta), _port_xy(rect(b), sb, tb)
        allowed = frozenset(ancs[a.id] | ancs[b.id]) if use_forb else None
        for mg in margins:
            arr = obs(mg, allowed)
            L = max(arrow_len * 1.2, mg + 3.0)
            A, B = _stub_xy(pa, sa, L), _stub_xy(pb, sb, L)
            mid = r4_route_between(A, B, sa, _OPP[sb], arr, hanan)
            if mid:
                full = _simplify_points([pa] + mid + [pb])
                return full, _path_cost(full, sa, _OPP[sb])[0]
        return None

    # ---- 1st pass: edge selection using cheap (direct) candidates
    items = sorted(nl, key=lambda t: math.dist(ctr(t[1]), ctr(t[2])))
    plan = []
    for ei, a, b in items:
        srcs, dsts = _side_candidates(ctr(a), ctr(b))
        pref = [(s, t) for s in srcs for t in dsts]
        rest = [(s, t) for s in "NSWE" for t in "NSWE" if (s, t) not in pref]
        best = None
        for grp in (pref, rest):
            for use_forb in (True, False):
                for s, t in grp:
                    ta = _ideal_coord(s, rect(a), rect(b), corner)
                    tb = _ideal_coord(t, rect(b), rect(a), corner)
                    res = route_edge(a, s, ta, b, t, tb, False, use_forb)
                    if res and (best is None or res[1] < best["cost"]):
                        best = {"cost": res[1], "s": s, "t": t, "ta": ta, "tb": tb}
                if best:
                    break
            if best:
                break
        if best is None:
            s, t = srcs[0], dsts[0]
            best = {"s": s, "t": t,
                    "ta": _ideal_coord(s, rect(a), rect(b), corner),
                    "tb": _ideal_coord(t, rect(b), rect(a), corner)}
        plan.append({"ei": ei, "a": a, "b": b, **best})

    # ---- port distribution (lines that share the same edge)
    groups_p = defaultdict(list)
    for i, p in enumerate(plan):
        groups_p[(p["a"].id, p["s"])].append((i, "ta", p["b"]))
        groups_p[(p["b"].id, p["t"])].append((i, "tb", p["a"]))
    for (bid, side), lst in groups_p.items():
        x, y, w, h = rect(by_id[bid])
        lo, hi = ((x + corner, x + w - corner) if side in "NS" else (y + corner, y + h - corner))
        ax = 0 if side in "NS" else 1
        vals = [plan[i][k] + 1e-4 * ctr(partner)[ax] for i, k, partner in lst]
        for (i, k, _), t in zip(lst, r4_spread(vals, lo, max(lo, hi), gap)):
            plan[i][k] = t

    # ---- 2nd pass: final routing (Hanan, if necessary)
    for p in plan:
        a, b = p["a"], p["b"]
        res = None
        for use_forb in (True, False):
            res = route_edge(a, p["s"], p["ta"], b, p["t"], p["tb"], True, use_forb)
            if res:
                break
        if res is None:
            for s in "NSWE":
                for t in "NSWE":
                    ta = _ideal_coord(s, rect(a), rect(b), corner)
                    tb = _ideal_coord(t, rect(b), rect(a), corner)
                    r2 = route_edge(a, s, ta, b, t, tb, True, False)
                    if r2 and (res is None or r2[1] < res[1]):
                        res = r2
                        p.update({"s": s, "t": t, "ta": ta, "tb": tb})
        p["pts"] = res[0] if res else None

    # ---- parallel lane separation
    ok = [p for p in plan if p["pts"] is not None]
    routes = [p["pts"] for p in ok]
    hcache = {}

    def hard_for(ri):
        key = frozenset(ancs[ok[ri]["a"].id] | ancs[ok[ri]["b"].id])
        if key not in hcache:
            parts = [_inf(box_raw, 1.0), _inf(hdr_raw, 1.0)]
            if len(cont_raw):
                keep = np.array([c not in key for c in cont_keys])
                parts.append(_inf(cont_raw[keep], 1.0))
            parts = [q for q in parts if len(q)]
            hcache[key] = np.vstack(parts) if parts else np.zeros((0, 4))
        return hcache[key]

    moved = 0
    if routes:
        moved, _ = r4_separate(routes, hard_for, _arr([]), gap, min_end)
        for p, r in zip(ok, routes):
            p["pts"] = r

    route_of, dashed = {}, set()
    for p in plan:
        if p["pts"] is None:  # unresolved: dashed L
            pa = _port_xy(rect(p["a"]), p["s"], p["ta"])
            pb = _port_xy(rect(p["b"]), p["t"], p["tb"])
            route_of[p["ei"]] = _simplify_points(_orthogonalize(manhattan_fallback(pa, pb)))
            dashed.add(p["ei"])
        else:
            route_of[p["ei"]] = p["pts"]
    return route_of, dashed, {"edges": len(plan), "failed": len(dashed), "moved": moved}

# ============================================================================
# SVG producer
# ============================================================================
def render_svg(view, mode, title, out_path, pipeline: dict, cfg, settings: RenderSettings):
    root, leaf_box, flow_box = view
    dfg_f, perf = pipeline["dfg_filtered"], pipeline["perf_full"]
    boxes = list(_walk(root))
    for i, b in enumerate(boxes):
        b.id = f"b{i}"
    top = sorted([g for g in root.children if g.kind == "group"], key=lambda g: g.path)
    hue_of = {g.path: i / max(1, len(top)) for i, g in enumerate(top)}

    def hue_for(b):
        return hue_of.get(b.path[:1], 0.6) if b.kind == "group" else 0.6

    def _ancestor_hue(b):
        return hue_of.get(b.path[:1], 0.6)

    E = []
    for (s, t), c in dfg_f.items():
        if c < settings.edge_min_count:
            continue
        ks, base_s = _classify_node_id(s)
        kt, base_t = _classify_node_id(t)
        sb = leaf_box.get(base_s) if ks in ("start", "finish", "point") else flow_box.get(s)
        tb = leaf_box.get(base_t) if kt in ("start", "finish", "point") else flow_box.get(t)
        if sb is None or tb is None:
            continue
        if sb is tb and not ((ks == "finish" and kt == "start") or (ks == "point" and kt == "point")):
            continue
        E.append({"s": s, "t": t, "sb": sb, "tb": tb, "ks": ks, "kt": kt, "c": c, "loop": (sb is tb),
                      "p": perf.get((s, t), {})})
    if not E:
        raise ValueError("No edges to render (filter may be too strict).")
    E.sort(key=lambda e: e["c"])
    for i, e in enumerate(sorted(E, key=lambda e: -e["c"])[:settings.label_top_n]):
        e["top"] = True
    cmax = max(e["c"] for e in E)
    lmax = math.log1p(cmax)
    vmin = vmax = 0
    if mode == "perf":
        means = np.array([e["p"].get("mean", 0.0) for e in E])
        vmin, vmax = np.percentile(means, [5, 95])

    canvas_w = max(root.core_w + 2 * settings.margin, 760.0)
    canvas_h = root.core_h + settings.header_h + settings.margin
    nl = [(ei, e["sb"], e["tb"]) for ei, e in enumerate(E) if not e["loop"]]
    route_of, dashed, _rinfo = route_hier_edges(nl, boxes, mode, settings)

    out = []
    W, H = canvas_w, canvas_h
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
               f'font-family="{settings.font}">')
    out.append("<style>.e .lbl{display:none}.e.top .lbl,.e.hl .lbl{display:inline}"
               ".e.dim{opacity:.05}.e.hl{opacity:1}.n{cursor:pointer}.n:hover rect.box{stroke-width:2.4}"
               ".lbl{font-size:9px;font-weight:bold;fill:#1B2631;text-anchor:middle;"
               "paint-order:stroke;stroke:#fff;stroke-width:3px}</style>")
    out.append('<defs><linearGradient id="gperf" x1="0" x2="1" y1="0" y2="0">'
               '<stop offset="0" stop-color="#2ecc71"/><stop offset=".5" stop-color="#f1c40f"/>'
               '<stop offset="1" stop-color="#e74c3c"/></linearGradient></defs>')
    out.append(f'<rect x="0" y="0" width="{W:.0f}" height="{H:.0f}" fill="#FFFFFF"/>')
    out.append(f'<text x="{settings.margin}" y="26" font-size="17" font-weight="bold" fill="#1B2631">'
               f'{html.escape(title)}</text>')
    ly = 50
    if cfg.show_start_finish_box:
        out.append(f'<polygon points="{settings.margin},{ly - 9} {settings.margin + 10},{ly - 4} {settings.margin},{ly + 1}" fill="#229954"/>'
                   f'<text x="{settings.margin + 15}" y="{ly}" font-size="10">Start</text>'
                   f'<rect x="{settings.margin + 55}" y="{ly - 9}" width="9" height="9" fill="#CB4335"/>'
                   f'<text x="{settings.margin + 69}" y="{ly}" font-size="10">Finish</text>')
    out.append(f'<rect x="{settings.margin + 115}" y="{ly - 10}" width="26" height="11" rx="5" fill="#F5CBA7" stroke="#CA6F1E"/>'
               f'<text x="{settings.margin + 146}" y="{ly}" font-size="10">Order event</text>')
    if mode == "perf":
        out.append(f'<text x="{settings.margin + 225}" y="{ly}" font-size="10">Line color = avg wait:</text>'
                   f'<rect x="{settings.margin + 365}" y="{ly - 9}" width="90" height="9" fill="url(#gperf)"/>'
                   f'<text x="{settings.margin + 365}" y="{ly + 11}" font-size="9">{fmt_dur(vmin)}</text>'
                   f'<text x="{settings.margin + 455}" y="{ly + 11}" font-size="9" text-anchor="end">{fmt_dur(vmax)}</text>'
                   f'<text x="{settings.margin + 470}" y="{ly}" font-size="10">· Width = frequency (log)</text>')
    else:
        out.append(f'<text x="{settings.margin + 225}" y="{ly}" font-size="10">'
                   f'Line width/color = transition frequency (log scale) · hover: highlight links · '
                   f'scroll: zoom · drag: pan · double-click: reset</text>')

    titles = []
    for b in boxes:
        if b.kind != "group" or b.depth == 0:
            continue
        iso_gap_y = getattr(b, "iso_gap_y", None)
        if iso_gap_y is not None:
            yy = float(b.y) + float(iso_gap_y)
            out.append(f'<line x1="{b.x+6:.1f}" y1="{yy:.1f}" x2="{b.x+b.core_w-6:.1f}" y2="{yy:.1f}" '
                       f'stroke="#B2BEC3" stroke-width="1" stroke-dasharray="4 4"/>')
            out.append(f'<text x="{b.x+6:.1f}" y="{yy-3:.1f}" font-size="7" fill="#909497">'
                       f'low connectivity</text>')
        h = hue_for(b)
        if b.depth == 1:
            fill, stroke, sw, dash = _hsv(h, 0.10, 1.0), _hsv(h, 0.55, 0.75), 2.0, ""
        elif b.depth == 2:
            fill, stroke, sw, dash = _hsv(h, 0.04, 1.0), _hsv(h, 0.40, 0.85), 1.2, ' stroke-dasharray="5 3"'
        else:
            fill, stroke, sw, dash = "#FFFFFF", "#B2BEC3", 1.0, ' stroke-dasharray="3 3"'
        out.append(f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.core_w:.1f}" height="{b.core_h:.1f}" rx="8" '
                   f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>')
        fs1, fs2, fs3 = _hdr(b.depth), _hdr2(b.depth), _hdr3(b.depth)
        header = _header_lines(b.level_label, b.code, b.name, b.n, b.dur_mean, b.dur_min, b.dur_max,
                               mode, fs1, fs2, fs3)
        dark = _hsv(h, 0.7, 0.42)
        cur = b.y + 5
        for txt, fsz, bold in header:
            cur += fsz
            fw = ' font-weight="bold"' if bold else ""
            titles.append(f'<text x="{b.x+8:.1f}" y="{cur:.1f}" font-size="{fsz}"{fw} fill="{fill}" '
                          f'stroke="{fill}" stroke-width="4" stroke-linejoin="round">{html.escape(txt)}</text>')
            titles.append(f'<text x="{b.x+8:.1f}" y="{cur:.1f}" font-size="{fsz}"{fw} '
                          f'fill="{dark}">{html.escape(txt)}</text>')
            cur += 3.5

    out.append('<g id="edges">')
    for ei, e in enumerate(E):
        c, sb, tb = e["c"], e["sb"], e["tb"]
        w = settings.edge_min_w + (math.log1p(c) / lmax) * (settings.edge_max_w - settings.edge_min_w)
        if mode == "perf":
            mean = e["p"].get("mean", 0.0)
            col = duration_to_color(mean, vmin, vmax)
            label = fmt_dur(mean)
            tip = (f"{sb.short} → {tb.short}\n{c:,} transition | avg {fmt_dur(mean)} | "
                   f"min {fmt_dur(e['p'].get('min', 0))} | max {fmt_dur(e['p'].get('max', 0))}")
        else:
            col = _hsv(_ancestor_hue(sb), 0.60, 0.55)
            label = f"{c:,}"
            tip = (f"{sb.short} → {tb.short}\n{c:,} transitions | avg wait "
                   f"{fmt_dur(e['p'].get('mean', 0))}")
        L, hw = 5 + 1.1 * w, 2.5 + 0.9 * w

        if e["loop"]:
            xr, xl, ym = sb.x + sb.core_w, sb.x, sb.y + sb.core_h / 2
            bul = sb.core_h / 2 + 38
            p0, p1, p2, p3 = (xr, ym), (xr + 45, ym + bul), (xl - 45, ym + bul), (xl - L, ym)
            tip_x, tip_y, dirx = xl, ym, 1
            mx = (p0[0] + 3 * p1[0] + 3 * p2[0] + p3[0]) / 8
            my = (p0[1] + 3 * p1[1] + 3 * p2[1] + p3[1]) / 8
            bx_ = tip_x - dirx * L
            poly = f"{tip_x:.1f},{tip_y:.1f} {bx_:.1f},{tip_y - hw:.1f} {bx_:.1f},{tip_y + hw:.1f}"
            d = (f"M{p0[0]:.1f},{p0[1]:.1f} C{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f} "
                 f"{p3[0]:.1f},{p3[1]:.1f}")
        else:
            path_world = route_of[ei]
            tip_x, tip_y = path_world[-1]
            trimmed = _shorten_last_segment(path_world, L)
            (lx0, ly0) = trimmed[-2] if len(trimmed) > 1 else trimmed[-1]
            (bx0, by0) = trimmed[-1]
            seg_dx, seg_dy = bx0 - lx0, by0 - ly0
            seg_len = math.hypot(seg_dx, seg_dy) or 1.0
            ux, uy = seg_dx / seg_len, seg_dy / seg_len
            perp_x, perp_y = -uy, ux
            poly = (f"{tip_x:.1f},{tip_y:.1f} "
                    f"{bx0 + perp_x * hw:.1f},{by0 + perp_y * hw:.1f} "
                    f"{bx0 - perp_x * hw:.1f},{by0 - perp_y * hw:.1f}")
            d = f"M{trimmed[0][0]:.1f},{trimmed[0][1]:.1f} " + " ".join(
                f"L{x:.1f},{y:.1f}" for x, y in trimmed[1:]
            )
            mx, my = _hier_label_pos(path_world)

        dash = ' stroke-dasharray="6 4"' if ei in dashed else ""
        out.append(
            f'<g class="e{" top" if e.get("top") else ""}" data-s="{sb.id}" data-t="{tb.id}">'
            f'<title>{html.escape(tip)}</title>'
            f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{w:.2f}" '
            f'stroke-linecap="butt" stroke-linejoin="miter" opacity=".8"{dash}/>'
            f'<polygon points="{poly}" fill="{col}" opacity=".9"/>'
            f'<text class="lbl" x="{mx:.1f}" y="{my + 3:.1f}">{html.escape(label)}</text></g>'
        )
    out.append("</g>")
    out.extend(titles)

    for b in boxes:
        if b.kind == "flow":
            out.append(f'<g class="n" data-id="{b.id}"><title>{html.escape(b.tip)}</title>'
                       f'<rect class="box" x="{b.x:.1f}" y="{b.y:.1f}" width="{b.core_w:.1f}" height="{b.core_h:.1f}" rx="11" '
                       f'fill="#F5CBA7" stroke="#CA6F1E" stroke-width="1.2"/>'
                       f'<text x="{b.x + b.core_w / 2:.1f}" y="{b.y + 15:.1f}" font-size="9" font-weight="bold" '
                       f'text-anchor="middle" fill="#6E2C00">{html.escape(b.lines[0][0])}</text></g>')
        elif b.kind == "leaf":
            fs1, fs2, fs3 = _hdr(b.depth), _hdr2(b.depth), _hdr3(b.depth)
            header = _header_lines(b.level_label, b.code, b.name, b.n, b.dur_mean, b.dur_min, b.dur_max,
                                   mode, fs1, fs2, fs3)
            g = [f'<g class="n" data-id="{b.id}"><title>{html.escape(b.tip)}</title>',
                 (f'<rect class="box" x="{b.x:.1f}" y="{b.y:.1f}" width="{b.core_w:.1f}" height="{b.core_h:.1f}" rx="5" '
                 f'fill="#FFFFFF" stroke="#2C3E50" stroke-width="1.2"/>')]
            if cfg.show_start_finish_box:
                cy = b.y + b.core_h / 2
                g.append(f'<rect x="{b.x + 0.6:.1f}" y="{b.y + 0.6:.1f}" width="{settings.strip_w}" height="{b.core_h - 1.2:.1f}" '
                         f'rx="4" fill="#D5F5E3"/>'
                         f'<polygon points="{b.x + 4:.1f},{cy - 5:.1f} {b.x + 12:.1f},{cy:.1f} {b.x + 4:.1f},{cy + 5:.1f}" fill="#229954"/>')
                g.append(f'<rect x="{b.x + b.core_w - settings.strip_w - 0.6:.1f}" y="{b.y + 0.6:.1f}" width="{settings.strip_w}" '
                         f'height="{b.core_h - 1.2:.1f}" rx="4" fill="#FADBD8"/>'
                         f'<rect x="{b.x + b.core_w - settings.strip_w + 3:.1f}" y="{cy - 4:.1f}" width="8" height="8" fill="#CB4335"/>')
            cur = b.y + 5
            for txt, fsz, bold in header:
                cur += fsz
                fw = ' font-weight="bold"' if bold else ""
                g.append(f'<text x="{b.x + b.core_w / 2:.1f}" y="{cur:.1f}" font-size="{fsz}"{fw} '
                         f'text-anchor="middle" fill="#1B2631">{html.escape(txt)}</text>')
                cur += 3.5
            if b.child_rows:
                cur += 3
                fsc = 8.5
                n_cols = max(1, b.grid_cols)
                col_w = b.grid_col_w or [0.0]
                col_x = [0.0] * n_cols
                for c in range(1, n_cols):
                    col_x[c] = col_x[c - 1] + col_w[c - 1] + 10
                total_w = col_x[-1] + col_w[-1] if col_w else 0.0
                start_x = b.x + (b.core_w - total_w) / 2
                base_y = cur
                for i, row in enumerate(b.child_rows):
                    c, rr = i % n_cols, i // n_cols
                    txt = _child_row_text(row, mode)
                    yy = base_y + rr * (fsc + 4) + fsc
                    g.append(f'<text x="{start_x + col_x[c]:.1f}" y="{yy:.1f}" font-size="{fsc}" '
                             f'text-anchor="start" fill="#566573">{html.escape(txt)}</text>')
            g.append("</g>")
            out.append("".join(g))
    out.append('<g id="hl"></g>')
    out.append("<script><![CDATA[" + _JS + "]]></script>")
    out.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return out_path, W, H, len(E)


def render_process_map(pipeline: dict, cfg, agg_level, settings: RenderSettings | None = None,
                       only_codes=None, mode: str = "freq", title: str | None = None,
                       out_path: str = "process.svg"):
    settings = settings or RenderSettings()
    view = build_view(pipeline, cfg, agg_level, settings, only_codes)
    layout(view[0], mode, cfg, settings, pipeline["deg"])
    place(view[0], settings.margin, settings.header_h)
    default_title = "Factory Process Map (Frequency)" if mode == "freq" else "Factory Bottleneck Map (Duration)"
    return render_svg(view, mode, title or default_title, out_path, pipeline, cfg, settings)
