from __future__ import annotations

import numpy as np
import pandas as pd
from pipeline import _non_flow_df, format_duration, norm_code, safe_sort
from svg_engine import _classify_node_id


def build_level_transition_matrix(ops: pd.DataFrame, cfg, code_col: str, name_col: str | None = None):
    """Sequential events with the same level values are counted once; only
    actual movements between different locations are counted (SLP 'From-To' table).
    Returns: (matrix_df, pairs_df[from,to,count(,from_name,to_name)], name_lookup)."""
    ts_col = cfg.op_ts_col
    cols = [cfg.case_id_col, code_col, ts_col] + ([name_col] if name_col else [])
    frame = ops[cols].dropna(subset=[code_col, ts_col]).copy()
    frame[code_col] = frame[code_col].astype(str)
    frame = safe_sort(frame, [cfg.case_id_col, ts_col])
    name_lookup = {}
    if name_col:
        name_lookup = frame.drop_duplicates(code_col).set_index(code_col)[name_col].astype(str).to_dict()
    frame["prev_code"] = frame.groupby(cfg.case_id_col, sort=False)[code_col].shift(1)
    moves = frame[frame["prev_code"].notna() & (frame["prev_code"] != frame[code_col])]
    if moves.empty:
        empty = pd.DataFrame(columns=["from", "to", "count"])
        return pd.DataFrame(), empty, name_lookup
    pairs = (moves.groupby(["prev_code", code_col]).size().rename("count").reset_index()
                  .rename(columns={"prev_code": "from", code_col: "to"}))
    pairs = pairs.iloc[np.argsort(-pairs["count"].to_numpy(), kind="stable")].reset_index(drop=True)
    codes = sorted(set(pairs["from"]) | set(pairs["to"]))
    matrix = pd.DataFrame(0, index=codes, columns=codes, dtype=int)
    for r in pairs.itertuples(index=False):
        matrix.loc[r[0], r[1]] = r[2]
    if name_lookup:
        pairs["from_name"] = pairs["from"].map(name_lookup)
        pairs["to_name"] = pairs["to"].map(name_lookup)
    return matrix, pairs, name_lookup


def _duration_sec(ops: pd.DataFrame, cfg) -> pd.Series | None:
    if not cfg.show_start_finish_box:
        return None
    return (ops[cfg.end_timestamp_col] - ops[cfg.start_timestamp_col]).dt.total_seconds()


def describe_node(nid: str, pipeline: dict) -> str:
    kind, key = _classify_node_id(nid)
    if kind == "flow":
        label = pipeline["node_label_lookup"].get(nid, nid)
        attach = pipeline["flow_attach_info"].get(nid)
        return f"{label} @ {attach[1]}" if attach else str(label)
    parts = pipeline["base_paths"].get(key)
    if parts is None:
        return nid
    nm = pipeline["names"].get(parts, "")
    text = "/".join(parts) + (f" {nm}" if nm else "")
    return {"start": "▶ ", "finish": "■ ", "point": ""}[kind] + text


def level_ranking(ops: pd.DataFrame, cfg, level_idx: int, pairs: pd.DataFrame, names: dict) -> pd.DataFrame:
    lvl = cfg.hierarchy_levels[level_idx]
    code = lvl.code_col
    d = ops.copy()
    d[code] = d[code].astype(str)
    dur = _duration_sec(d, cfg)
    if dur is not None:
        d["_dur"] = dur
    g = d.groupby(code, observed=True)
    out = pd.DataFrame({
        "Operation events": g.size(),
        "Distinct orders": g[cfg.case_id_col].nunique(),
        "Distinct operations": g[cfg.activity_code_col].nunique(),
    })
    if dur is not None:
        out["Avg op. duration (sec)"] = g["_dur"].mean()
        out["Avg op. duration"] = out["Avg op. duration (sec)"].map(
            lambda v: format_duration(v) if pd.notna(v) else "-")
    partners = {}
    for a, b in zip(pairs["from"], pairs["to"]):
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    out[f"Connected to # other {lvl.label}s"] = [len(partners.get(c, ())) for c in out.index]
    out.insert(0, "Name", [names.get(c, "") for c in out.index])
    if level_idx > 0:
        parent = cfg.hierarchy_levels[level_idx - 1].code_col
        d[parent] = d[parent].astype(str)
        out.insert(1, f"Parent {cfg.hierarchy_levels[level_idx - 1].label}",
                   d.groupby(code, observed=True)[parent].agg(lambda s: s.mode().iat[0]))
    out.index.name = lvl.label
    out = out.iloc[np.argsort(-out["Operation events"].to_numpy(), kind="stable")]
    return out.reset_index()


def transfer_time_summary(ops: pd.DataFrame, cfg) -> pd.DataFrame:
    """The gap between consecutive operations (the end of the previous -> the start of the next),
    grouped by the level at which they diverge:
    'Same <inner level>' ... 'Different <outer level>'. The outermost divergence
    (e.g., different work center) is a strong layout signal."""
    levels = cfg.hierarchy_levels
    if not cfg.show_start_finish_box or not levels:
        return pd.DataFrame()
    code_cols = list(cfg.hier_code_cols)
    cols = [cfg.case_id_col, cfg.start_timestamp_col, cfg.end_timestamp_col] + code_cols
    seq = ops[cols].dropna(subset=[cfg.start_timestamp_col, cfg.end_timestamp_col]).copy()
    for c in code_cols:
        seq[c] = seq[c].astype(str)
    seq = safe_sort(seq, [cfg.case_id_col, cfg.start_timestamp_col])
    grp = seq.groupby(cfg.case_id_col, sort=False)
    nxt_start = grp[cfg.start_timestamp_col].shift(-1)
    valid = nxt_start.notna().to_numpy().copy()
    gap = (nxt_start - seq[cfg.end_timestamp_col]).dt.total_seconds().to_numpy()
    valid = valid & (gap >= 0)
    n = len(levels)
    first_diff = np.full(len(seq), n)
    for k in reversed(range(n)):
        nxt_code = grp[code_cols[k]].shift(-1).to_numpy()
        first_diff = np.where(seq[code_cols[k]].to_numpy() != nxt_code, k, first_diff)
    first_diff, gap = first_diff[valid], gap[valid]

    def bucket(k):
        if k == n:
            return f"Same {levels[-1].label}"
        if k == 0:
            return f"Different {levels[0].label}"
        return f"Same {levels[k - 1].label}, different {levels[k].label}"

    tmp = pd.DataFrame({"k": first_diff, "gap": gap})
    agg = tmp.groupby("k")["gap"].agg(["count", "mean", "median"])
    agg.columns = ["Moves", "Mean", "Median"]
    res = agg.copy()
    res.insert(0, "Transfer type", [bucket(k) for k in res.index])
    res["Avg wait"] = res["Mean"].map(format_duration)
    res["Median wait (sec)"] = res["Median"]
    return res.drop(columns=["Mean", "Median"]).reset_index(drop=True)


def level_year_share(ops: pd.DataFrame, cfg, level_idx: int) -> pd.DataFrame:
    """Row: code, Column: year, Value: total activity share in that year."""
    code = cfg.hierarchy_levels[level_idx].code_col
    d = ops.dropna(subset=[cfg.op_ts_col]).copy()
    d["year"] = d[cfg.op_ts_col].dt.year
    ct = pd.crosstab(d[code].astype(str), d["year"])
    return ct.div(ct.sum(axis=0), axis=1)


def share_change(year_share: pd.DataFrame, year_from: int, year_to: int) -> pd.DataFrame:
    a = year_share[year_from] if year_from in year_share.columns else 0.0
    b = year_share[year_to] if year_to in year_share.columns else 0.0
    ya, yb = str(year_from), str(year_to)
    out = pd.DataFrame({ya: a, yb: b}).fillna(0.0)
    out["Change (pp)"] = (out[yb] - out[ya]) * 100
    return out.iloc[np.argsort(-out["Change (pp)"].to_numpy(), kind="stable")].round(3)


# ----------------------------------------------------------------------------
# Helpers for readable node names
# ----------------------------------------------------------------------------
def _base_text(key: str, pipeline: dict) -> str:
    parts = pipeline["base_paths"].get(key)
    return "/".join(parts) if parts else key


def short_node(nid: str, pipeline: dict) -> str:
    kind, key = _classify_node_id(nid)
    if kind == "flow":
        return str(pipeline["node_label_lookup"].get(nid, nid))
    return _base_text(key, pipeline) + {"start": " ▶", "finish": " ■", "point": ""}[kind]


def _compress_steps(nids, pipeline: dict) -> list:
    """A Start immediately followed by the Finish of the same operation is shown as a single step."""
    out, i = [], 0
    while i < len(nids):
        kind, key = _classify_node_id(nids[i])
        if kind == "start" and i + 1 < len(nids):
            k2, key2 = _classify_node_id(nids[i + 1])
            if k2 == "finish" and key2 == key:
                out.append(_base_text(key, pipeline))
                i += 2
                continue
        out.append(short_node(nids[i], pipeline))
        i += 1
    return out


def _desc_order(values) -> np.ndarray:
    return np.argsort(-np.asarray(values, dtype=float), kind="stable")


# ----------------------------------------------------------------------------
# Case table (one row per order, flow events included)
# ----------------------------------------------------------------------------
def build_case_table(event_log: pd.DataFrame) -> pd.DataFrame:
    g = event_log.groupby("case_id", sort=False)
    cases = pd.DataFrame({
        "start": g["timestamp"].min(),
        "end": g["timestamp"].max(),
        "n_events": g.size(),
        "seq": g["node_id"].agg("|".join),
    })
    cases["duration_sec"] = (cases["end"] - cases["start"]).dt.total_seconds()
    return cases


def log_summary(pipeline: dict, cfg, cases: pd.DataFrame) -> pd.DataFrame:
    el = pipeline["event_log"]
    is_flow = el["node_id"].str.startswith("flow_L")
    rows = [
        ("Events (all)", f"{len(el):,}"),
        ("  of which flow events", f"{int(is_flow.sum()):,}"),
        ("  of which operation events" + (" (start + finish)" if cfg.show_start_finish_box else ""),
         f"{int((~is_flow).sum()):,}"),
        ("Orders (cases)", f"{len(cases):,}"),
        ("Distinct nodes", f"{el['node_id'].nunique():,}"),
        ("DFG edges (unfiltered)", f"{len(pipeline['dfg_full']):,}"),
        ("Trace variants", f"{cases['seq'].nunique():,}"),
        ("First event", str(el["timestamp"].min())),
        ("Last event", str(el["timestamp"].max())),
        ("Avg events per order", f"{cases['n_events'].mean():.1f}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ----------------------------------------------------------------------------
# DFG edges: frequency, duration, total time (unfiltered, flow events included)
# ----------------------------------------------------------------------------
def flow_table(pipeline: dict) -> pd.DataFrame:
    rows = []
    for (s, t), cnt in pipeline["dfg_full"].items():
        ks, bs = _classify_node_id(s)
        kt, bt = _classify_node_id(t)
        if ks == "flow" or kt == "flow":
            typ = "Flow event"
        elif ks == "start" and kt == "finish" and bs == bt:
            typ = "Processing"
        else:
            typ = "Transfer / wait"
        p = pipeline["perf_full"].get((s, t), {})
        mean = p.get("mean")
        rows.append({
            "From": describe_node(s, pipeline), "To": describe_node(t, pipeline), "Type": typ,
            "Frequency": cnt,
            "Avg duration": format_duration(mean) if mean is not None else "-",
            "Avg duration (sec)": mean if mean is not None else np.nan,
            "Min duration (sec)": p.get("min", np.nan),
            "Max duration (sec)": p.get("max", np.nan),
            "Total time (hours)": mean * cnt / 3600 if mean is not None else np.nan,
        })
    cols = ["From", "To", "Type", "Frequency", "Avg duration", "Avg duration (sec)",
            "Min duration (sec)", "Max duration (sec)", "Total time (hours)"]
    out = pd.DataFrame(rows, columns=cols)
    return out.iloc[_desc_order(out["Frequency"])].reset_index(drop=True)


def busiest_flows(pipeline: dict) -> pd.DataFrame:
    df = flow_table(pipeline)
    return df.iloc[_desc_order(df["Avg duration (sec)"])].reset_index(drop=True)


def bottleneck_flows(flows: pd.DataFrame, min_count: int = 30) -> pd.DataFrame:
    """Non-processing edges (waiting/transfer and flow-event edges) ranked by total time spent."""
    d = flows[(flows["Type"] != "Processing") & (flows["Frequency"] >= min_count)]
    d = d.dropna(subset=["Total time (hours)"])
    return d.iloc[_desc_order(d["Total time (hours)"])].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Nodes, start/end nodes
# ----------------------------------------------------------------------------
def node_stats(pipeline: dict) -> pd.DataFrame:
    el = pipeline["event_log"]
    g = el.groupby("node_id")
    out = pd.DataFrame({"Events": g.size(), "Orders": g["case_id"].nunique()})
    out["Avg position in order (0=start, 1=end)"] = [pipeline["scores"].get(n, np.nan) for n in out.index]
    out.insert(0, "Type", [{"start": "Start", "finish": "Finish", "point": "Event", "flow": "Flow event"}
                           [_classify_node_id(n)[0]] for n in out.index])
    out.insert(0, "Node", [describe_node(n, pipeline) for n in out.index])
    out = out.iloc[_desc_order(out["Events"])]
    return out.reset_index(drop=True)


def start_end_nodes(pipeline: dict) -> pd.DataFrame:
    g = pipeline["event_log"].groupby("case_id", sort=False)["node_id"]
    s, e = g.first().value_counts(), g.last().value_counts()
    nodes = s.index.union(e.index)
    out = pd.DataFrame({
        "Node": [describe_node(n, pipeline) for n in nodes],
        "Case starts": s.reindex(nodes).fillna(0).astype(int).to_numpy(),
        "Case ends": e.reindex(nodes).fillna(0).astype(int).to_numpy(),
    })
    out = out.iloc[_desc_order(out["Case starts"] + out["Case ends"])]
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Order lead time (throughput time) and variants
# ----------------------------------------------------------------------------
def case_duration_summary(cases: pd.DataFrame) -> pd.DataFrame:
    d = cases.assign(year=cases["start"].dt.year)
    groups = [("All", d)] + [(int(y), x) for y, x in d.groupby("year")]
    rows = []
    for label, x in groups:
        v = x["duration_sec"].dropna()
        if v.empty:
            continue
        rows.append({
            "Order start year": label, "Orders": len(x),
            "Mean": format_duration(v.mean()), "Median": format_duration(v.median()),
            "P90": format_duration(v.quantile(0.90)), "P95": format_duration(v.quantile(0.95)),
            "Max": format_duration(v.max()), "Median (sec)": v.median(),
        })
    return pd.DataFrame(rows)


def variant_table(cases: pd.DataFrame, pipeline: dict, top_n: int = 20, max_steps: int = 14) -> pd.DataFrame:
    v = cases.groupby("seq", sort=False)["duration_sec"].agg(["size", "mean", "median"]).reset_index()
    v = v.iloc[_desc_order(v["size"])].head(top_n).reset_index(drop=True)
    total = len(cases)
    cum = np.cumsum(v["size"].to_numpy()) / total * 100
    texts, nsteps = [], []
    for key in v["seq"]:
        steps = _compress_steps(key.split("|"), pipeline)
        nsteps.append(len(steps))
        texts.append(" → ".join(steps[:max_steps]) + (f" → … (+{len(steps) - max_steps})"
                                                       if len(steps) > max_steps else ""))
    return pd.DataFrame({
        "Rank": np.arange(1, len(v) + 1),
        "Orders": v["size"].to_numpy(),
        "Share (%)": (v["size"].to_numpy() / total * 100).round(2),
        "Cumulative (%)": cum.round(2),
        "Avg lead time": v["mean"].map(lambda x: format_duration(x) if pd.notna(x) else "-"),
        "Median lead time": v["median"].map(lambda x: format_duration(x) if pd.notna(x) else "-"),
        "Steps": nsteps,
        "Variant": texts,
    })


# ----------------------------------------------------------------------------
# Flow events (Set Order, Update Order, Request Order Complete ...)
# ----------------------------------------------------------------------------
def flow_rows(df: pd.DataFrame, cfg) -> pd.DataFrame:
    if not (cfg.is_flow_event_col and cfg.is_flow_event_col in df.columns):
        return df.iloc[0:0]
    return df[df[cfg.is_flow_event_col].astype(str).str.lower().isin(["true", "1"])]


def flow_event_summary(df: pd.DataFrame, cfg) -> pd.DataFrame:
    fr = flow_rows(df, cfg)
    if fr.empty:
        return pd.DataFrame()
    ts = cfg.flow_event_ts_col
    g = fr.groupby(cfg.activity_code_col, observed=True)
    out = pd.DataFrame({
        "Events": g.size(),
        "Distinct orders": g[cfg.case_id_col].nunique(),
        "First": g[ts].min(),
        "Last": g[ts].max(),
    })
    out.index.name = "Flow event"
    return out.iloc[_desc_order(out["Events"])].reset_index()


def flow_event_counts(df: pd.DataFrame, cfg, level_idx: int) -> dict:
    """Number of flow events attached to each code of the given hierarchy level."""
    fr = flow_rows(df, cfg)
    if fr.empty or not (cfg.flow_attach_level_col and cfg.flow_attach_code_col):
        return {}
    lv = pd.to_numeric(fr[cfg.flow_attach_level_col], errors="coerce")
    sel = fr[lv == level_idx]
    return sel[cfg.flow_attach_code_col].map(norm_code).value_counts().to_dict()


# ----------------------------------------------------------------------------
# Yearly overview (whole event log)
# ----------------------------------------------------------------------------
def yearly_overview(pipeline: dict, cfg, cases: pd.DataFrame) -> pd.DataFrame:
    el = pipeline["event_log"].copy()
    el["year"] = el["timestamp"].dt.year
    el["_flow"] = el["node_id"].str.startswith("flow_L")
    out = el.groupby("year").agg(**{
        "Events (all)": ("case_id", "size"),
        "Flow events": ("_flow", "sum"),
        "Distinct orders": ("case_id", "nunique"),
    })
    ops = _non_flow_df(pipeline["df"], cfg)
    d = ops.dropna(subset=[cfg.op_ts_col]).copy()
    d["year"] = d[cfg.op_ts_col].dt.year
    out["Operations"] = d.groupby("year").size()
    for lvl in cfg.hierarchy_levels:
        out[f"Distinct {lvl.label}"] = d.groupby("year")[lvl.code_col].nunique()
    dur = _duration_sec(d, cfg)
    if dur is not None:
        d["_dur"] = dur
        out["Avg. op duration"] = d.groupby("year")["_dur"].mean().map(format_duration)
    lead = cases.assign(year=cases["start"].dt.year).groupby("year")["duration_sec"].mean()
    out["Avg order lead time"] = lead.map(format_duration)
    return out


# ----------------------------------------------------------------------------
# ONLY ONE ENTRY POINT
# ----------------------------------------------------------------------------
def run_analysis(pipeline: dict, cfg) -> dict:
    df = pipeline["df"]
    ops = _non_flow_df(df, cfg)
    cases = build_case_table(pipeline["event_log"])
    flows = flow_table(pipeline)
    levels = []
    for i, lvl in enumerate(cfg.hierarchy_levels):
        _, pairs, names = build_level_transition_matrix(ops, cfg, lvl.code_col, lvl.name_col)
        ranking = level_ranking(ops, cfg, i, pairs, names)
        fe = flow_event_counts(df, cfg, i)
        if fe:
            ranking["Flow events attached"] = ranking[lvl.label].map(lambda c: fe.get(norm_code(c), 0))
        levels.append({"label": lvl.label, "level_idx": i, "ranking": ranking,
                       "year_share": level_year_share(ops, cfg, i)})
    return {
        "summary": log_summary(pipeline, cfg, cases),
        "cases": cases,
        "case_durations": case_duration_summary(cases),
        "variants": variant_table(cases, pipeline),
        "flows": flows,
        "busiest_flows": busiest_flows(pipeline),
        "flow_events": flow_event_summary(df, cfg),
        "bottlenecks": bottleneck_flows(flows),
        "nodes": node_stats(pipeline),
        "start_end": start_end_nodes(pipeline),
        "transfer": transfer_time_summary(ops, cfg),
        "yearly": yearly_overview(pipeline, cfg, cases),
        "levels": levels,
    }
