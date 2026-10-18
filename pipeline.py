from __future__ import annotations

import io
import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from config import EventLogConfig
else:
    EventLogConfig = Any


# ============================================================================
# 1) FILE LOADING — not depended on global config, just IO
# ============================================================================
def load_event_log(file_bytes: bytes, file_name: str, sheet_name=0) -> pd.DataFrame:
    """CSV or XLSX raw text (dtype=str) is read as-is. Column meanings
    are not yet known — this is resolved by prepare_dataframe() with the cfg."""
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), dtype=str)
    if file_name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, dtype=str, engine="openpyxl")
    raise ValueError(f"Unsupported file type: {file_name!r} (only .csv / .xlsx)")


# ============================================================================
# 2) VALIDATION + TYPE CONVERSION — the cfg-enabled 
# ============================================================================
def prepare_dataframe(raw: pd.DataFrame, cfg: EventLogConfig) -> pd.DataFrame:
    """Validates required columns and sets their types (str/category/datetime).
    Raises ValueError if any required column is missing — display in the calling
    context with st.error(), similar to the notebook."""
    df = raw.copy()

    required = [cfg.case_id_col, cfg.activity_code_col] + list(cfg.hier_code_cols)
    if cfg.show_start_finish_box:
        required += [cfg.start_timestamp_col, cfg.end_timestamp_col]
    else:
        required += [cfg.timestamp_col]
    if getattr(cfg, "flow_timestamp_col", None):
        required.append(cfg.flow_timestamp_col)
    missing = [c for c in required if c and c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}\nAvailable columns: {list(df.columns)}"
        )

    str_cols = list(cfg.hier_code_cols) + [cfg.activity_code_col]
    for lvl in cfg.hierarchy_levels:
        if lvl.name_col:
            str_cols.append(lvl.name_col)
    if cfg.activity_name_col:
        str_cols.append(cfg.activity_name_col)
    str_cols += list(cfg.resource_cols) + list(cfg.other_cols)
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    ts_cols = [c for c in [cfg.timestamp_col, cfg.start_timestamp_col, cfg.end_timestamp_col,
                           getattr(cfg, 'flow_timestamp_col', None)] if c]
    for col in ts_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in list(cfg.hier_code_cols) + [cfg.activity_code_col]:
        df[col] = df[col].astype("category")
    return df


# ============================================================================
# 3) HELPER FUNCTIONS — do not require cfg
# ============================================================================
def safe_sort(frame: pd.DataFrame, cols) -> pd.DataFrame:
    if isinstance(cols, str):
        cols = [cols]
    if len(frame) == 0:
        return frame.reset_index(drop=True)
    keys = [frame[c].to_numpy() for c in reversed(cols)]
    order = np.lexsort(keys)
    return frame.iloc[order].reset_index(drop=True)


def format_duration(seconds):
    seconds = abs(seconds)
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.0f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"


# ============================================================================
# 4) NODE-KEY — AGG_LEVEL/HIERARCHY_LEVELS is no longer global; it is passed as a parameter:
# cfg + agg_level. agg_level: int (0..len(hierarchy_levels)-1) or
# the string "activity" (the most detailed level).
# ============================================================================
def agg_level_index(cfg: EventLogConfig, agg_level) -> int:
    return len(cfg.hierarchy_levels) if agg_level == "activity" else int(agg_level)


def _base_from_parts(parts, idx: int) -> str:
    """The core of `node_key` — `compute_node_stats`/`build_base_paths` in Section 4 
    — was extracted into a separate function so that it could also generate the *same* base string."""
    return f"lvl{idx}_" + "_".join(re.sub(r"[^a-zA-Z0-9]", "", str(p)) or "NA" for p in parts)


def node_key(level_values, cfg: EventLogConfig, agg_level) -> str:
    idx = agg_level_index(cfg, agg_level)
    parts = [str(v) for v in level_values[: idx + 1]]
    return _base_from_parts(parts, idx)


def norm_code(x) -> str:
    """For code comparison: remove spaces, '9.0' -> '9', ignore leading zeros ('009' == '9')."""
    t = str(x).strip()
    t = re.sub(r"^(-?\d+)\.0+$", r"\1", t)
    return t.lstrip("0") or "0" if t.isdigit() else t.upper()


def flow_node_key(attach_level: int, attach_code, activity_label: str) -> str:
    clean_label = re.sub(r"[^a-zA-Z0-9]", "_", str(activity_label))
    return f"flow_L{attach_level}_{re.sub(r'[^a-zA-Z0-9]', '', str(attach_code))}__{clean_label}"


# ============================================================================
# 5) EVENT LISTS SETUP — fully cfg-based
# ============================================================================
def _non_flow_df(df: pd.DataFrame, cfg: EventLogConfig) -> pd.DataFrame:
    if cfg.is_flow_event_col and cfg.is_flow_event_col in df.columns:
        mask = df[cfg.is_flow_event_col].astype(str).str.lower().isin(["true", "1"])
        return df[~mask]
    return df


def _row_level_values(row, cfg: EventLogConfig):
    return tuple(row[c] for c in cfg.hier_code_cols) + (row[cfg.activity_code_col],)


def make_start_events(df: pd.DataFrame, cfg: EventLogConfig, agg_level) -> pd.DataFrame:
    frame = _non_flow_df(df, cfg).copy()
    frame = frame.dropna(subset=[cfg.start_timestamp_col])
    frame["timestamp"] = frame[cfg.start_timestamp_col].astype("datetime64[ns]").astype("int64")
    frame["node_id"] = "start_" + frame.apply(
        lambda r: node_key(_row_level_values(r, cfg), cfg, agg_level), axis=1
    )
    act_label = (frame[cfg.activity_name_col].astype(str)
                 if cfg.activity_name_col else frame[cfg.activity_code_col].astype(str))
    frame["activity"] = "Start: " + act_label
    frame["node_label"] = "Start"
    frame["case_id"] = frame[cfg.case_id_col].astype(str)
    return frame[["case_id", "timestamp", "activity", "node_id", "node_label"]]


def make_finish_events(df: pd.DataFrame, cfg: EventLogConfig, agg_level) -> pd.DataFrame:
    frame = _non_flow_df(df, cfg).copy()
    frame = frame.dropna(subset=[cfg.end_timestamp_col])
    frame["timestamp"] = frame[cfg.end_timestamp_col].astype("datetime64[ns]").astype("int64")
    frame["node_id"] = "finish_" + frame.apply(
        lambda r: node_key(_row_level_values(r, cfg), cfg, agg_level), axis=1
    )
    act_label = (frame[cfg.activity_name_col].astype(str)
                 if cfg.activity_name_col else frame[cfg.activity_code_col].astype(str))
    frame["activity"] = "Finish: " + act_label
    frame["node_label"] = "Finish"
    frame["case_id"] = frame[cfg.case_id_col].astype(str)
    return frame[["case_id", "timestamp", "activity", "node_id", "node_label"]]


def make_point_events(df: pd.DataFrame, cfg: EventLogConfig, agg_level) -> pd.DataFrame:
    frame = _non_flow_df(df, cfg).copy()
    frame = frame.dropna(subset=[cfg.timestamp_col])
    frame["timestamp"] = frame[cfg.timestamp_col].astype("datetime64[ns]").astype("int64")
    frame["node_id"] = frame.apply(
        lambda r: node_key(_row_level_values(r, cfg), cfg, agg_level), axis=1
    )
    act_label = (frame[cfg.activity_name_col].astype(str)
                 if cfg.activity_name_col else frame[cfg.activity_code_col].astype(str))
    frame["activity"] = act_label
    frame["node_label"] = None
    frame["case_id"] = frame[cfg.case_id_col].astype(str)
    return frame[["case_id", "timestamp", "activity", "node_id", "node_label"]]


def make_flow_events(df: pd.DataFrame, cfg: EventLogConfig):
    """Events like Place Order / Request Order Complete. If not present in the dataset
    (cfg.is_flow_event_col is not set), an empty result is returned."""
    empty = pd.DataFrame(columns=["case_id", "timestamp", "activity", "node_id", "node_label"]), {}
    if not (cfg.is_flow_event_col and cfg.is_flow_event_col in df.columns):
        return empty
    mask = df[cfg.is_flow_event_col].astype(str).str.lower().isin(["true", "1"])
    if not mask.any():
        return empty
    ts_col = cfg.flow_event_ts_col
    if not (cfg.flow_attach_level_col and cfg.flow_attach_code_col and ts_col):
        st.warning("There are IsFlowEvent marked rows but FlowAttachLevel/Code or "
                   "timestamp column is missing; flow events are being skipped.")
        return empty
    frame = df[mask].copy()
    frame = frame.dropna(subset=[ts_col, cfg.flow_attach_level_col, cfg.flow_attach_code_col])
    if frame.empty:
        return empty
    frame["timestamp"] = frame[ts_col].astype("datetime64[ns]").astype("int64")
    frame["node_id"] = frame.apply(
        lambda r: flow_node_key(int(float(r[cfg.flow_attach_level_col])),
                                r[cfg.flow_attach_code_col], r[cfg.activity_code_col]),
        axis=1,
    )
    frame["activity"] = frame[cfg.activity_code_col].astype(str)
    frame["node_label"] = frame["activity"]
    frame["case_id"] = frame[cfg.case_id_col].astype(str)
    attach_info = {}
    for _, r in frame.drop_duplicates("node_id").iterrows():
        attach_info[r["node_id"]] = (int(float(r[cfg.flow_attach_level_col])), r[cfg.flow_attach_code_col])
    return frame[["case_id", "timestamp", "activity", "node_id", "node_label"]], attach_info


def build_event_log(df: pd.DataFrame, cfg: EventLogConfig, agg_level):
    """start/finish/point/flow events is combined to produce a single event_log sorted by time."""
    empty = pd.DataFrame(columns=["case_id", "timestamp", "activity", "node_id", "node_label"])
    if cfg.show_start_finish_box:
        start_events = make_start_events(df, cfg, agg_level)
        finish_events = make_finish_events(df, cfg, agg_level)
        point_events = empty.copy()
    else:
        start_events = empty.copy()
        finish_events = empty.copy()
        point_events = make_point_events(df, cfg, agg_level)
    flow_events, flow_attach_info = make_flow_events(df, cfg)

    event_log = pd.concat(
        [e for e in [start_events, finish_events, point_events, flow_events] if not e.empty],
        ignore_index=True,
    )
    event_log = event_log.dropna(subset=["timestamp"])
    event_log["timestamp"] = pd.to_datetime(event_log["timestamp"])
    event_log = safe_sort(event_log, ["case_id", "timestamp"])
    return event_log, flow_attach_info


# ============================================================================
# 6) DFG DISCOVERY
# ============================================================================
def discover_dfg_full_vectorized(df, case_col="case_id", act_col="node_id", time_col="timestamp"):
    d = safe_sort(df[[case_col, act_col, time_col]], [case_col, time_col])
    nxt_act = d.groupby(case_col, sort=False)[act_col].shift(-1)
    pairs = pd.DataFrame({"src": d[act_col].to_numpy(), "dst": nxt_act.to_numpy()})
    pairs = pairs.dropna(subset=["dst"])
    dfg = {}
    for key, c in pairs.value_counts(["src", "dst"]).items():
        if not isinstance(key, tuple):
            continue
        s, t = key
        dfg[(s, t)] = int(c)
    g = d.groupby(case_col, sort=False)[act_col]
    start_activities = g.first().value_counts().to_dict()
    end_activities = g.last().value_counts().to_dict()
    return dfg, start_activities, end_activities


def discover_performance_dfg_vectorized(df, case_col="case_id", act_col="node_id", time_col="timestamp"):
    d = safe_sort(df[[case_col, act_col, time_col]], [case_col, time_col])
    d["next_act"] = d.groupby(case_col, sort=False)[act_col].shift(-1)
    d["next_time"] = d.groupby(case_col, sort=False)[time_col].shift(-1)
    d = d.dropna(subset=["next_act"])
    d["duration"] = (d["next_time"] - d[time_col]).dt.total_seconds()
    stats = d.groupby([act_col, "next_act"])["duration"].agg(["mean", "min", "max"])
    result = {}
    for idx, row in stats.iterrows():
        if not isinstance(idx, tuple) or len(idx) != 2:
            continue
        s, t = idx
        result[(s, t)] = {"mean": row["mean"], "min": row["min"], "max": row["max"]}
    return result


def bfs_reachable(start, adj):
    from collections import deque
    visited = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in visited:
                visited.add(v); q.append(v)
    return visited


def filter_dfg_on_paths_percentage_faithful(dfg0, start_activities0, end_activities0, activities_count0,
                                            percentage, keep_all_activities=False, keep_nodes=()):
    dfg = dict(dfg0)
    start_activities = dict(start_activities0)
    end_activities = dict(end_activities0)
    activities_count = dict(activities_count0)
    if len(activities_count) <= 1 or len(dfg) <= 1:
        return dfg, start_activities, end_activities, activities_count
    START, END = "_S_START_", "_S_END_"
    nodes = set(activities_count) | {a for e in dfg for a in e} | set(start_activities) | set(end_activities)
    adj = {n: set() for n in nodes}
    rev = {n: set() for n in nodes}
    adj[START] = set(start_activities)
    rev[START] = set()
    adj[END] = set()
    rev[END] = set(end_activities)
    for (a, b) in dfg:
        adj[a].add(b); rev[b].add(a)
    for a in start_activities:
        rev.setdefault(a, set()).add(START)
    for a in end_activities:
        adj.setdefault(a, set()).add(END)
    all_edges = (list(dfg.items())
                 + [((START, a), f) for a, f in start_activities.items()]
                 + [((a, END), f) for a, f in end_activities.items()])
    all_edges.sort(key=lambda x: (x[1], x[0]), reverse=True)
    cut_idx = math.ceil((len(all_edges) - 1) * percentage) + 1
    non_discardable = [e for e, _ in all_edges[:cut_idx]]
    discardable = [e for e, _ in all_edges[cut_idx:]]
    discardable.reverse()
    if keep_all_activities:
        must_keep = set(activities_count) | set(start_activities) | set(end_activities)
    else:
        must_keep = {u for (u, v) in non_discardable if u != START} | \
                    {v for (u, v) in non_discardable if v != END}
    # Flow events (Place Order, etc.) may be infrequent; they are always preserved
    # (added to the reachability condition) so that the path filter does not silently discard them.
    must_keep |= (set(keep_nodes) & nodes)
    for (u, v) in discardable:
        if v in adj.get(u, ()):
            adj[u].discard(v); rev[v].discard(u)
            if must_keep.issubset(bfs_reachable(START, adj)) and \
               must_keep.issubset(bfs_reachable(END, rev)):
                if (u, v) in dfg:
                    del dfg[(u, v)]
                elif u == START and v in start_activities:
                    del start_activities[v]
                elif v == END and u in end_activities:
                    del end_activities[u]
            else:
                adj[u].add(v); rev[v].add(u)
    reach_fwd = bfs_reachable(START, adj)
    reach_bwd = bfs_reachable(END, rev)
    keep = (reach_fwd & reach_bwd) - {START, END}
    activities_count = {a: c for a, c in activities_count.items() if a in keep}
    start_activities = {a: c for a, c in start_activities.items() if a in keep}
    end_activities = {a: c for a, c in end_activities.items() if a in keep}
    dfg = {(a, b): c for (a, b), c in dfg.items() if a in keep and b in keep}
    return dfg, start_activities, end_activities, activities_count


# ============================================================================
# 7) Derived data for the SVG engine (Section 4)
# Section 4 no longer calculates *any* data—it simply looks up values ​​in these dictionaries
# and performs the layout/drawing. This leaves a *single* source of truth (this file)
# instead of two parallel event/DFG implementations.
# ============================================================================
def build_base_paths(df: pd.DataFrame, cfg: EventLogConfig, agg_level) -> dict:
    """Maps base-key strings (e.g., 'lvl1_00900901') to the tuple of codes (code0, code1, ..., [activity code])
    required by Section 4 to build the box tree. The node_id string is not split — it is read from this dictionary."""
    idx = agg_level_index(cfg, agg_level)
    key_cols = (list(cfg.hier_code_cols) + [cfg.activity_code_col])[: idx + 1]
    ops = _non_flow_df(df, cfg)
    sub = ops[key_cols].drop_duplicates()
    out = {}
    for _, row in sub.iterrows():
        parts = tuple(str(row[c]) for c in key_cols)
        out[_base_from_parts(parts, idx)] = parts
    return out


def compute_node_stats(df: pd.DataFrame, cfg: EventLogConfig, agg_level) -> dict:
    """Computes event counts and (if applicable) duration statistics for each base-key. Section 4
    uses these to display 'n=...' / 'avg=...' in the box titles."""
    idx = agg_level_index(cfg, agg_level)
    key_cols = (list(cfg.hier_code_cols) + [cfg.activity_code_col])[: idx + 1]
    ops = _non_flow_df(df, cfg).copy()
    if cfg.show_start_finish_box:
        ops["_dur"] = (ops[cfg.end_timestamp_col] - ops[cfg.start_timestamp_col]).dt.total_seconds()
    g = ops.groupby(key_cols, observed=True)
    sizes = g.size()
    means = g["_dur"].mean() if cfg.show_start_finish_box else None
    mins = g["_dur"].min() if cfg.show_start_finish_box else None
    maxs = g["_dur"].max() if cfg.show_start_finish_box else None
    def _lookup_group_stat(stat: pd.Series | None, key_vals):
        if stat is None:
            return None
        vals = stat.loc[[key_vals]]
        if isinstance(vals, pd.Series):
            return float(vals.iloc[0])
        return float(vals)

    out = {}
    for key_vals, n in sizes.items():
        parts = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        base = _base_from_parts([str(p) for p in parts], idx)
        dur_mean = None
        dur_min = None
        dur_max = None
        if cfg.show_start_finish_box:
            dur_mean = _lookup_group_stat(means, key_vals)
            dur_min = _lookup_group_stat(mins, key_vals)
            dur_max = _lookup_group_stat(maxs, key_vals)
        out[base] = {
            "n": int(n),
            "dur_mean": dur_mean,
            "dur_min": dur_min,
            "dur_max": dur_max,
        }
    return out


def build_names(df: pd.DataFrame, cfg: EventLogConfig) -> dict:
    """Maps (code0, ..., codeN) tuples to readable names (Work Center name, Operation name, ...)."""
    ops = _non_flow_df(df, cfg)
    names = {}
    for i, lvl in enumerate(cfg.hierarchy_levels):
        if lvl.name_col and lvl.name_col in ops.columns:
            cc = list(cfg.hier_code_cols[: i + 1])
            for r in ops[cc + [lvl.name_col]].astype(str).drop_duplicates(cc).itertuples(index=False):
                names[tuple(r[:-1])] = r[-1]
    if cfg.activity_name_col and cfg.activity_name_col in ops.columns:
        cc = list(cfg.hier_code_cols) + [cfg.activity_code_col]
        for r in ops[cc + [cfg.activity_name_col]].astype(str).drop_duplicates(cc).itertuples(index=False):
            names[tuple(r[:-1])] = r[-1]
    return names


def compute_node_scores(event_log: pd.DataFrame) -> dict:
    """The event_log must be sorted by case_id (build_event_log performs this sorting).
    For each node_id, it returns the average relative position within a case (0 = start of case, 1 = end of case). 
    This value is used for positioning Section 4's flow boxes before/after the process and for the SVG_ORDER_BY='flow' sorting order."""
    case_arr = event_log["case_id"].to_numpy()
    n = len(case_arr)
    if n == 0:
        return {}
    starts = np.r_[0, np.flatnonzero(case_arr[1:] != case_arr[:-1]) + 1]
    sizes = np.diff(np.r_[starts, n])
    idx = np.arange(n) - np.repeat(starts, sizes)
    pos = idx / np.maximum(np.repeat(sizes, sizes) - 1, 1)
    return pd.Series(pos).groupby(event_log["node_id"].to_numpy()).mean().to_dict()


def compute_child_rows(df: pd.DataFrame, cfg: EventLogConfig, agg_level) -> dict:
    """It calculates the distribution of the level immediately below AGG_LEVEL for each base: at the hierarchy level, 
    this is the sub-hierarchy code (🏭); at the 'activity' level, 
    it includes ALL resource columns (👤, now multiple) + other dimensions (📎). 
    We do not concatenate the label text here—'n=...' will be displayed in 'freq' mode and 'avg=...' in 'perf' mode; 
    since Section 4 decides which one to show at render time, we return the raw fields (icon/label_base/n/dur_*). 
    Sorting is already done in descending order by count; the "top N rows" truncation is a VIEW setting, 
    so it is handled in Section 4, not here."""
    idx = agg_level_index(cfg, agg_level)
    ops = _non_flow_df(df, cfg).copy()
    if cfg.show_start_finish_box:
        ops["_dur"] = (ops[cfg.end_timestamp_col] - ops[cfg.start_timestamp_col]).dt.total_seconds()
    parent_cols = (list(cfg.hier_code_cols) + [cfg.activity_code_col])[: idx + 1]

    dim_specs = []   # (column, icon, name_column, label_prefix)
    if idx < len(cfg.hierarchy_levels):
        if idx + 1 < len(cfg.hierarchy_levels):
            nxt = cfg.hierarchy_levels[idx + 1]
            dim_specs.append((nxt.code_col, "🏭", nxt.name_col, ""))
        else:
            dim_specs.append((cfg.activity_code_col, "🏭", cfg.activity_name_col, ""))
    else:
        dim_specs += [(c, "👤", None, "") for c in cfg.resource_cols]
        dim_specs += [(c, "📎", None, f"{c}: ") for c in cfg.other_cols]

    out: dict = {}
    for col, icon, name_col, prefix in dim_specs:
        if not col or col not in ops.columns:
            continue
        group_cols = parent_cols + [col]
        g = ops.groupby(group_cols, observed=True)
        sizes = g.size().rename("cnt").reset_index()
        sizes["neg_cnt"] = -sizes["cnt"]
        sizes = safe_sort(sizes, parent_cols + ["neg_cnt"])
        dur = g["_dur"].agg(["mean", "min", "max"]) if cfg.show_start_finish_box else None
        name_lookup = {}
        if name_col and name_col in ops.columns:
            name_lookup = ops.drop_duplicates(col).set_index(col)[name_col].astype(str).to_dict()

        for _, r in sizes.iterrows():
            parent_key = tuple(str(r[c]) for c in parent_cols)
            base = _base_from_parts(parent_key, idx)
            full_key = tuple(r[c] for c in group_cols)
            d = dur.loc[full_key] if dur is not None else None
            label_base = f"{prefix}{r[col]}"
            if r[col] in name_lookup:
                label_base += f" {name_lookup[r[col]]}"
            out.setdefault(base, []).append({
                "icon": icon, "label_base": label_base, "n": int(r["cnt"]),
                "dur_mean": float(d["mean"]) if d is not None else None,
                "dur_min": float(d["min"]) if d is not None else None,
                "dur_max": float(d["max"]) if d is not None else None,
            })
    return out


def compute_node_degrees(dfg_filtered: dict) -> dict:
    """Compute the total connection weight in the filtered DFG — 
    used to separate isolated (unconnected) boxes to the bottom when SVG_ORDER_BY='degree'."""
    deg = {}
    for (s, t), c in dfg_filtered.items():
        deg[s] = deg.get(s, 0) + c
        deg[t] = deg.get(t, 0) + c
    return deg


# ============================================================================
# 8) MULTI-USER ORCHESTRATION — using `st.cache_data`. 
# It automatically recalculates if `cfg`, `agg_level`, or `path_filter_percentage` change; 
# returns the result from the cache if the same user returns with the same file and settings; 
# and creates a separate cache entry if a different user arrives with a different file or configuration.
# ============================================================================
@st.cache_data(show_spinner="Processing event log and extracting DFG...")
def build_pipeline(raw_df: pd.DataFrame, cfg: EventLogConfig,
                   agg_level, path_filter_percentage: float):
    df = prepare_dataframe(raw_df, cfg)
    event_log, flow_attach_info = build_event_log(df, cfg, agg_level)

    dfg_c, sa_c, ea_c = discover_dfg_full_vectorized(event_log)
    perf_c = discover_performance_dfg_vectorized(event_log)
    ac_c = event_log["node_id"].value_counts().to_dict()
    dfg_f, _, _, ac_f = filter_dfg_on_paths_percentage_faithful(
        dfg_c, sa_c, ea_c, ac_c, percentage=path_filter_percentage,
        keep_nodes=set(flow_attach_info)
    )
    node_label_lookup = event_log.drop_duplicates("node_id").set_index("node_id")["node_label"]

    return {
        "df": df, 
        "event_log": event_log,
        "dfg_full": dfg_c, 
        "perf_full": perf_c,
        "dfg_filtered": dfg_f, 
        "activities_filtered": ac_f,
        "flow_attach_info": flow_attach_info,
        "node_label_lookup": node_label_lookup,
        # ---- Section 4 (SVG engine) JUST reads these, doesn't compute them ----
        "base_paths": build_base_paths(df, cfg, agg_level),
        "node_stats": compute_node_stats(df, cfg, agg_level),
        "names": build_names(df, cfg),
        "scores": compute_node_scores(event_log),
        "deg": compute_node_degrees(dfg_f),
        "child_rows": compute_child_rows(df, cfg, agg_level),
    }


# ============================================================================
# 9) DIAGNOSIS — where are the flow events being lost?
# ============================================================================
def diagnose_flow(df: pd.DataFrame, cfg: EventLogConfig, pipeline: dict) -> dict:
    """Returns: {"steps": [(step, count)], "notes": [str], "flag_values": dict}."""
    steps, notes = [], []
    col = cfg.is_flow_event_col
    if not col or col not in df.columns:
        return {"steps": [("IsFlowEvent column is not selected", 0)], 
                "notes": ["Please select the '5. Flow events' checkbox in the sidebar and choose the columns."], 
                "flag_values": {}
                }
    flag_values = df[col].astype(str).value_counts().head(8).to_dict()
    mask = df[col].astype(str).str.lower().isin(["true", "1"])
    fl = df[mask]
    steps.append(("IsFlowEvent = true/1 rows", len(fl)))
    if len(fl) == 0:
        notes.append(f"Flag column values: {flag_values}. Only 'true' / '1' are accepted.")
        return {"steps": steps, "notes": notes, "flag_values": flag_values}
    ts = cfg.flow_event_ts_col
    n1 = int(fl[ts].notna().sum()) if ts in fl.columns else 0
    steps.append((f"... timestamp column '{ts}' is filled", n1))
    if n1 == 0:
        notes.append(f"'{ts}' flow rows are empty. Please select a column with filled values (e.g., OrderDate/timestamp)")
    n2 = int(fl[cfg.flow_attach_level_col].notna().sum())
    n3 = int(fl[cfg.flow_attach_code_col].notna().sum())
    steps.append((f"... '{cfg.flow_attach_level_col}' is filled", n2))
    steps.append((f"... '{cfg.flow_attach_code_col}' is filled", n3))
    attach = pipeline["flow_attach_info"]
    steps.append(("Generated flow nodes", len(attach)))
    steps.append(("In DFG (unfiltered)", len({n for e in pipeline["dfg_full"] for n in e if n in attach})))
    steps.append(("After filtering", len([n for n in pipeline["activities_filtered"] if n in attach])))
    avail = {}
    for parts in pipeline["base_paths"].values():
        for k in range(1, len(parts) + 1):
            avail.setdefault(k, set()).add(norm_code(parts[k - 1]))
    matched, bad = 0, []
    for (lvl, code) in attach.values():
        if norm_code(code) in avail.get(lvl + 1, set()):
            matched += 1
        else:
            bad.append((lvl, code))
    steps.append(("Target boxes found (drawable)", matched))
    if bad:
        samples = {k: sorted(v)[:5] for k, v in avail.items()}
        notes.append(f"Target boxes not found: {bad[:8]}. Available codes (level index+1 -> example): "
                     f"{samples}. FlowAttachLevel 0 = outer hierarchy level.")
    return {"steps": steps, "notes": notes, "flag_values": flag_values}
