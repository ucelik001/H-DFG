from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict

import numpy as np

EPS = 1e-6
BEND_PEN = 60.0  # each 90° turn = 60 px length penalty
DOOR_BALANCE_PX = 0.0 # >0: door usage penalty (distributes congestion to doors)
_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIRV = {"E": (1, 0), "W": (-1, 0), "S": (0, 1), "N": (0, -1)} # SVG: y down, x right
_DNAMES = "EWSN" # indices: E0 W1 S2 N3  (reverse direction = idx ^ 1)

def manhattan_fallback(exit_a, exit_b):
    _, y0 = exit_a
    x1, _ = exit_b
    mid = (x1, y0)
    if mid == exit_a or mid == exit_b:
        return [exit_a, exit_b]
    return [exit_a, mid, exit_b]


def _orthogonalize(points):
    out = [points[0]]
    for p in points[1:]:
        q = out[-1]
        if abs(p[0] - q[0]) > 1e-6 and abs(p[1] - q[1]) > 1e-6:
            out.append((p[0], q[1]))
        out.append(p)
    return out


def _simplify_points(pts, eps=1e-6):
    """Removes duplicate and collinear intermediate points; leaves endpoints untouched."""
    dedup = []
    for p in pts:
        if dedup and abs(p[0] - dedup[-1][0]) < eps and abs(p[1] - dedup[-1][1]) < eps:
            continue
        dedup.append(p)
    if len(dedup) < 3:
        return dedup
    res = [dedup[0]]
    for i in range(1, len(dedup) - 1):
        a, b, c = res[-1], dedup[i], dedup[i + 1]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
        if abs(cross) < eps and dot > 0:
            continue
        res.append(b)
    res.append(dedup[-1])
    return res


def _sdir(p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    if abs(dx) > EPS and abs(dy) < EPS:
        return "E" if dx > 0 else "W"
    if abs(dy) > EPS and abs(dx) < EPS:
        return "S" if dy > 0 else "N"
    return None


def _path_cost(pts, d_start, d_end):
    """length + BEND_PEN*bends. Start/end directions must be perpendicular to the sides; backtracking is forbidden."""
    dirs = [d for d in (_sdir(p, q) for p, q in itertools.pairwise(pts)) if d]
    if not dirs:
        return math.inf, 0
    L = sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) for p, q in itertools.pairwise(pts))
    bends, prev = 0, d_start
    for d in dirs:
        if d != prev:
            bends += 1
            prev = d
    if dirs[-1] != d_end:
        bends += 1
    pen = (1e6 if dirs[0] == _OPP[d_start] else 0) + (1e6 if dirs[-1] == _OPP[d_end] else 0)
    return L + BEND_PEN * bends + pen, bends


def _direct_candidates(A, B):
    (x0, y0), (x1, y1) = A, B
    c = [[A, B]] if (abs(x0 - x1) < EPS or abs(y0 - y1) < EPS) else []
    c += [[A, (x1, y0), B], [A, (x0, y1), B]]
    for f in (0.5, 0.25, 0.75):
        mx, my = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        c += [[A, (mx, y0), (mx, y1), B], [A, (x0, my), (x1, my), B]]
    return c


def _port_xy(bb, side, t):
    x, y, w, h = bb
    return {"N": (t, y), "S": (t, y + h), "W": (x, t), "E": (x + w, t)}[side]


def _stub_xy(p, side, L):
    dx, dy = _DIRV[side]
    return (p[0] + dx * L, p[1] + dy * L)


def _side_candidates(Ca, Cb, tol=2.0):
    """Source sides based on the quadrant of the target center; target sides are their counterparts."""
    dx, dy = Cb[0] - Ca[0], Cb[1] - Ca[1]
    s = []
    if abs(dx) > tol:
        s.append("E" if dx > 0 else "W")
    if abs(dy) > tol:
        s.append("S" if dy > 0 else "N")
    if not s:
        s = ["E"]
    s.sort(key=lambda k: (k in "NS") if abs(dx) >= abs(dy) else (k in "EW"))
    return s, [_OPP[k] for k in s]


def _ideal_coord(side, bb, other, corner):
    """Ideal position along the side: center of overlapping regions if any, otherwise center of the side."""
    x, y, w, h = bb
    X, Y, W, H = other
    if side in "NS":
        lo, hi, a0, a1 = max(x, X), min(x + w, X + W), x, x + w
    else:
        lo, hi, a0, a1 = max(y, Y), min(y + h, Y + H), y, y + h
    c = (lo + hi) / 2 if hi - lo > 2 * corner else (a0 + a1) / 2
    return min(max(c, a0 + corner), a1 - corner)


# ---- numpy collision
def _arr(rects):
    return np.array(list(rects), float).reshape(-1, 4)


def _inf(a, m):
    return a + np.array([-m, -m, 2 * m, 2 * m]) if len(a) else a


def a_hit(p, q, arr, m=0.0):
    """if segment is parallel to an axis, checks if it intersects any rectangle in the array (tangent not counted)."""
    if len(arr) == 0:
        return False
    x0, x1 = (p[0], q[0]) if p[0] <= q[0] else (q[0], p[0])
    y0, y1 = (p[1], q[1]) if p[1] <= q[1] else (q[1], p[1])
    return bool(((x0 < arr[:, 0] + arr[:, 2] + m - EPS) & (x1 > arr[:, 0] - m + EPS) &
                 (y0 < arr[:, 1] + arr[:, 3] + m - EPS) & (y1 > arr[:, 1] - m + EPS)).any())


def a_path_hit(pts, arr, m=0.0):
    return any(a_hit(p, q, arr, m) for p, q in itertools.pairwise(pts))


def a_window(arr, x0, y0, x1, y1):
    if len(arr) == 0:
        return arr
    mk = (arr[:, 0] < x1) & (arr[:, 0] + arr[:, 2] > x0) & (arr[:, 1] < y1) & (arr[:, 1] + arr[:, 3] > y0)
    return arr[mk]


# ---- Hanan grid + bending-penalized Dijkstra (NumPy masked)
def r4_hanan(A, B, d_start, d_end, arr):
    rects = arr.tolist()
    xs = sorted({A[0], B[0]} | {v for r in rects for v in (r[0], r[0] + r[2])})
    ys = sorted({A[1], B[1]} | {v for r in rects for v in (r[1], r[1] + r[3])})
    nx, ny = len(xs), len(ys)
    xa, ya = np.array(xs), np.array(ys)
    vb = np.zeros((ny, nx), bool)
    hb = np.zeros((ny, max(nx - 1, 1)), bool)
    wb = np.zeros((max(ny - 1, 1), nx), bool)
    for rx, ry, rw, rh in rects:
        i0, i1 = int(np.searchsorted(xa, rx)), int(np.searchsorted(xa, rx + rw))
        j0, j1 = int(np.searchsorted(ya, ry)), int(np.searchsorted(ya, ry + rh))
        vb[j0 + 1:j1, i0 + 1:i1] = True
        hb[j0 + 1:j1, i0:i1] = True
        wb[j0:j1, i0 + 1:i1] = True
    vbl, hbl, wbl = vb.tolist(), hb.tolist(), wb.tolist()
    xi, yi = {x: i for i, x in enumerate(xs)}, {y: j for j, y in enumerate(ys)}
    si, sj, gi, gj = xi[A[0]], yi[A[1]], xi[B[0]], yi[B[1]]
    if vbl[sj][si] or vbl[gj][gi]:
        return None
    d0 = _DNAMES.index(d_start)
    de = _DNAMES.index(d_end)
    _h = lambda i, j: abs(xs[i] - xs[gi]) + abs(ys[j] - ys[gj]) 
    start = (si, sj, d0)
    best, prev, pq, cnt = {start: 0.0}, {}, [(_h(si, sj), 0, 0.0, start)], 0
    while pq:
        _, _, c, cur = heapq.heappop(pq)
        if best.get(cur, math.inf) < c - 1e-9:
            continue
        i, j, d = cur
        if (i, j) == (gi, gj):
            pts, key = [], cur
            while True:
                pts.append((xs[key[0]], ys[key[1]]))
                if key == start:
                    break
                key = prev[key]
            return _simplify_points(pts[::-1])
        for nd in range(4):
            if nd == (d ^ 1):
                continue
            if nd == 0:
                ok = i + 1 < nx and not hbl[j][i] and not vbl[j][i + 1]
                ni, nj = i + 1, j
            elif nd == 1:
                ok = i - 1 >= 0 and not hbl[j][i - 1] and not vbl[j][i - 1]
                ni, nj = i - 1, j
            elif nd == 2:
                ok = j + 1 < ny and not wbl[j][i] and not vbl[j + 1][i]
                ni, nj = i, j + 1
            else:
                ok = j - 1 >= 0 and not wbl[j - 1][i] and not vbl[j - 1][i]
                ni, nj = i, j - 1
            if not ok:
                continue
            nc = c + abs(xs[ni] - xs[i]) + abs(ys[nj] - ys[j]) + (BEND_PEN if nd != d else 0)
            if (ni, nj) == (gi, gj) and nd != de:
                nc += BEND_PEN
            key = (ni, nj, nd)
            if nc < best.get(key, math.inf) - 1e-9:
                best[key] = nc
                prev[key] = cur
                cnt += 1
                heapq.heappush(pq, (nc + _h(ni, nj), cnt, nc, key))
    return None

def r4_route_between(A, B, ds, de, arr, allow_hanan=True):
    if math.dist(A, B) < EPS:
        return [A]
    best = None
    for c in _direct_candidates(A, B):
        if not a_path_hit(c, arr):
            cost = _path_cost(c, ds, de)[0]
            if best is None or cost < best[0]:
                best = (cost, c)
    if best:
        return _simplify_points(best[1])
    if not allow_hanan:
        return None
    for pad in (200.0, 600.0, None):
        sub = arr if pad is None else a_window(arr, min(A[0], B[0]) - pad, min(A[1], B[1]) - pad,
                                               max(A[0], B[0]) + pad, max(A[1], B[1]) + pad)
        pts = r4_hanan(A, B, ds, de, sub)
        if pts and not a_path_hit(pts, arr):
            return pts
    return None


def r4_spread(ts, lo, hi, gap):
    n = len(ts)
    if n == 1:
        return [min(max(ts[0], lo), hi)]
    order = sorted(range(n), key=lambda i: ts[i])
    v = [ts[i] for i in order]
    for k in range(1, n):
        v[k] = max(v[k], v[k - 1] + gap)
    sh = (sum(ts) - sum(v)) / n
    v = [x + sh for x in v]
    if v[-1] > hi:
        d = v[-1] - hi
        v = [x - d for x in v]
    if v[0] < lo:
        d = lo - v[0]
        v = [x + d for x in v]
    if v[-1] > hi:
        v = [lo + (hi - lo) * k / (n - 1) for k in range(n)]
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = v[pos]
    return out


# ---- parallel strip separation
def r4_orient(p, q, eps=0.6):
    if abs(p[1] - q[1]) < eps and abs(p[0] - q[0]) > eps:
        return "H"
    if abs(p[0] - q[0]) < eps and abs(p[1] - q[1]) > eps:
        return "V"
    return None


def r4_try_move(pts, k, o, c, hard, min_end, apply=True):
    n = len(pts) - 1
    v0, v1 = pts[k], pts[k + 1]
    nv0 = (v0[0], c) if o == "H" else (c, v0[1])
    nv1 = (v1[0], c) if o == "H" else (c, v1[1])
    if a_hit(nv0, nv1, hard):
        return False
    for kk, pv, nv, far in ((k - 1, v0, nv0, pts[k - 1]), (k + 1, v1, nv1, pts[k + 2])):
        old = abs(far[0] - pv[0]) + abs(far[1] - pv[1])
        new = abs(far[0] - nv[0]) + abs(far[1] - nv[1])
        dot = (pv[0] - far[0]) * (nv[0] - far[0]) + (pv[1] - far[1]) * (nv[1] - far[1])
        if dot <= 0 or new < (min_end if kk in (0, n - 1) else 2.0):
            return False
        if new > old and a_hit(pv, nv, hard):
            return False
    if apply:
        pts[k], pts[k + 1] = nv0, nv1
    return True


def r4_separate(routes, hard_for, fixed_arr, gap, min_end, iters=5):
    """Splits overlapping or closely spaced PARALLEL segments lying on the same line into strips.
    Port segments and segments passing through doors are fixed; they do not enter box/wall/door blockers.
    Returns: (number of moved segments, {'fixed':.., 'no_space':..})."""
    stats = {"fixed": 0, "no_space": 0}
    total = 0
    for _it in range(iters):
        fixedflag = [[a_hit(p[k], p[k + 1], fixed_arr) for k in range(len(p) - 1)] for p in routes]
        segs = []
        for ri, pts in enumerate(routes):
            n = len(pts) - 1
            ff = fixedflag[ri]
            for k in range(n):
                p, q = pts[k], pts[k + 1]
                o = r4_orient(p, q)
                if o is None:
                    continue
                c = p[1] if o == "H" else p[0]
                a_, b_ = sorted((p[0], q[0])) if o == "H" else sorted((p[1], q[1]))
                mv = 0 < k < n - 1 and not ff[k] and not ff[k - 1] and not ff[k + 1]
                segs.append({"r": ri, "k": k, "o": o, "c": c, "a": a_, "b": b_, "mv": mv})
        parent = list(range(len(segs)))

        def find(i, parent=parent):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        conflict = False
        for o in "HV":
            idx = sorted([i for i, s in enumerate(segs) if s["o"] == o], key=lambda i: segs[i]["c"])
            for ii, i in enumerate(idx):
                si = segs[i]
                for j in idx[ii + 1:]:
                    sj = segs[j]
                    if sj["c"] - si["c"] >= gap * 0.999:
                        break
                    if si["r"] != sj["r"] and min(si["b"], sj["b"]) - max(si["a"], sj["a"]) > 1.0 \
                            and (si["mv"] or sj["mv"]):
                        parent[find(i)] = find(j)
                        conflict = True
        if not conflict:
            break
        comps = defaultdict(list)
        for i in range(len(segs)):
            comps[find(i)].append(i)
        moved = 0
        for members in comps.values():
            mv = [i for i in members if segs[i]["mv"]]
            stats["fixed"] += len(members) - len(mv)
            if len(members) < 2 or not mv:
                continue
            o = segs[members[0]]["o"]
            placed = [(segs[i]["c"], segs[i]["a"], segs[i]["b"], segs[i]["r"]) for i in members if not segs[i]["mv"]]

            def sign(i, o=o):
                s = segs[i]  # noqa: B023
                pts = routes[s["r"]]
                k = s["k"]

                def sg(nb, cur):
                    d = (nb[1] - cur[1]) if o == "H" else (nb[0] - cur[0])
                    return -1 if d < -0.6 else (1 if d > 0.6 else 0)

                return sg(pts[k - 1], pts[k]) + sg(pts[k + 2], pts[k + 1])

            mv.sort(key=lambda i: (sign(i), segs[i]["c"], segs[i]["r"]))
            cm = float(np.mean([segs[i]["c"] for i in members]))
            for rank, i in enumerate(mv):
                s = segs[i]
                pts = routes[s["r"]]
                k = s["k"]
                p, q = pts[k], pts[k + 1]
                a_, b_ = sorted((p[0], q[0])) if o == "H" else sorted((p[1], q[1]))
                desired = cm + (rank - (len(mv) - 1) / 2.0) * gap
                hard = hard_for(s["r"])
                chosen = None
                for c in sorted((desired + t * gap * 0.5 for t in range(-40, 41)), key=lambda c: abs(c - desired)):
                    if any(pr != s["r"] and abs(c - pc) < gap - 1e-6 and min(b_, pb) - max(a_, pa) > 1.0
                           for pc, pa, pb, pr in placed):
                        continue
                    if r4_try_move(pts, k, o, c, hard, min_end):
                        chosen = c
                        break
                if chosen is None:
                    stats["no_space"] += 1
                    placed.append((s["c"], a_, b_, s["r"]))
                else:
                    placed.append((chosen, a_, b_, s["r"]))
                    moved += 1
        for ri in range(len(routes)):
            routes[ri] = _simplify_points(routes[ri])
        total += moved
        if moved == 0:
            break
    return total, stats
