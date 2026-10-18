from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

# ============================================================================
# SINGLE SOURCE OF TRUTH: all domain-config flows through here; no function
# reads bare module-level globals anymore. Every function accepts
# `cfg: EventLogConfig`. frozen=True -> protects against accidental
# mutation (a shared object in a multi-user environment cannot be
# modified inadvertently).
# ============================================================================

@dataclass(frozen=True)
class HierLevel:
    code_col: str
    name_col: str | None
    label: str


@dataclass(frozen=True)
class EventLogConfig:
    case_id_col: str
    activity_code_col: str
    activity_name_col: str | None
    timestamp_col: str | None          # point-event log if not None
    start_timestamp_col: str | None    # interval-event log if not None
    end_timestamp_col: str | None
    resource_cols: tuple[str, ...]     # no longer singular — multiple resource dimensions (User, Vehicle, Operator, ...)
    other_cols: tuple[str, ...]
    hierarchy_levels: tuple[HierLevel, ...]
    is_flow_event_col: str | None = None
    flow_attach_level_col: str | None = None
    flow_attach_code_col: str | None = None
    flow_timestamp_col: str | None = None   # The dedicated timestamp column for flow events (Place Order, etc.)

    @property
    def flow_event_ts_col(self) -> str | None:
        """Flow events timestamp column: the explicitly selected one, or the default flow_ts_col."""
        return self.flow_timestamp_col or self.flow_ts_col

    @property
    def show_start_finish_box(self) -> bool:
        return self.start_timestamp_col is not None and self.end_timestamp_col is not None

    @property
    def flow_ts_col(self) -> str | None:
        """Flow events timestamp column: the dedicated column for flow events (Place Order, etc.), or the default."""
        return self.timestamp_col if self.timestamp_col else self.start_timestamp_col

    @property
    def op_ts_col(self) -> str | None:
        """Timestamp of ordinary operation rows (start time if available, else the point timestamp)."""
        return self.start_timestamp_col if self.start_timestamp_col else self.timestamp_col

    @property
    def hier_code_cols(self) -> tuple[str, ...]:
        return tuple(lvl.code_col for lvl in self.hierarchy_levels)



# ============================================================================
# Sidebar: column mapping UI (Streamlit counterpart of the notebook's
# "Column mapping" settings cell).
# ============================================================================
_DEFAULT_LABELS = ("Workcenter", "Workstation")
_CODE_DEFAULTS = ("WorkCenterCode", "WorkStationCode")
_NAME_DEFAULTS = ("WorkCenterName", "WorkStationName")


def _pick(cols, *names, optional=False):
    """Index of the first candidate name found in cols (pre-selects sensible defaults)."""
    options = (["(none)"] if optional else []) + list(cols)
    for n in names:
        if n in options:
            return options.index(n)
    return 0


def build_config_ui(df: pd.DataFrame) -> EventLogConfig:
    cols = list(df.columns)

    st.sidebar.subheader("2. Columns")
    case_id_col = st.sidebar.selectbox("Order / Case ID", cols, index=_pick(cols, "OrderId"))
    activity_code_col = st.sidebar.selectbox("Operation code", cols, index=_pick(cols, "OperationCode"))
    activity_name_col = st.sidebar.selectbox("Operation name (optional)", ["(none)"] + cols, index=_pick(cols, "OperationName", optional=True))

    st.sidebar.subheader("3. Time model")
    st.sidebar.caption("Interval logs use Start/Finish; for flow events a single timestamp may also be given.")
    use_interval = st.sidebar.checkbox("Interval event log (Start/Finish)", value="StartTime" in cols)
    if use_interval:
        start_col = st.sidebar.selectbox("Start time", cols, index=_pick(cols, "StartTime"))
        end_col = st.sidebar.selectbox("Finish time", cols, index=_pick(cols, "FinishTime"))
        timestamp_col = None
    else:
        timestamp_col = st.sidebar.selectbox("Timestamp", cols, index=_pick(cols, "timestamp"))
        start_col = end_col = None

    st.sidebar.subheader("4. Resource / other dimensions")
    resource_cols = st.sidebar.multiselect("Resource columns (shown inside operation boxes)", cols, default=[c for c in ("User",) if c in cols])
    other_cols = st.sidebar.multiselect("Other dimensions", cols)

    st.sidebar.subheader("5. Hierarchy (outer to inner)")
    n_levels = int(st.sidebar.number_input("Number of levels", min_value=1, max_value=5, value=2))
    levels: list[HierLevel] = []
    for i in range(n_levels):
        st.sidebar.markdown(f"**Level {i}**")
        code_col = st.sidebar.selectbox(f"Code column #{i}", cols, key=f"lvl_code_{i}", index=_pick(cols, *(_CODE_DEFAULTS[i:i + 1])))
        name_col = st.sidebar.selectbox(f"Name column #{i} (optional)", ["(none)"] + cols, key=f"lvl_name_{i}", index=_pick(cols, *(_NAME_DEFAULTS[i:i + 1]), optional=True))
        label = st.sidebar.text_input(f"Label #{i}", value=_DEFAULT_LABELS[i] if i < 2 else f"Level {i}", key=f"lvl_label_{i}")
        levels.append(HierLevel(code_col, None if name_col == "(none)" else name_col, label))

    st.sidebar.subheader("6. Flow events (optional)")
    use_flow = st.sidebar.checkbox("Dataset contains flow events", value="IsFlowEvent" in cols)
    is_flow_col = flow_lvl_col = flow_code_col = flow_ts_col = None
    if use_flow:
        is_flow_col = st.sidebar.selectbox("IsFlowEvent column", cols, index=_pick(cols, "IsFlowEvent"))
        flow_lvl_col = st.sidebar.selectbox("FlowAttachLevel column (0 = outermost)", cols, 
                                            index=_pick(cols, "FlowAttachLevel"))
        flow_code_col = st.sidebar.selectbox("FlowAttachCode column", cols, 
                                             index=_pick(cols, "FlowAttachCode"))
        pick = st.sidebar.selectbox("Flow event time column", ["(Same as Start / Timestamp)"] + cols,
                                    index=_pick(cols, "timestamp", optional=True) if "timestamp" in cols else 0)
        flow_ts_col = None if pick.startswith("(") else pick

    return EventLogConfig(
        case_id_col=case_id_col,
        activity_code_col=activity_code_col,
        activity_name_col=None if activity_name_col == "(none)" else activity_name_col,
        timestamp_col=timestamp_col,
        start_timestamp_col=start_col,
        end_timestamp_col=end_col,
        resource_cols=tuple(resource_cols),
        other_cols=tuple(other_cols),
        hierarchy_levels=tuple(levels),
        is_flow_event_col=is_flow_col,
        flow_attach_level_col=flow_lvl_col,
        flow_attach_code_col=flow_code_col,
        flow_timestamp_col=flow_ts_col,
    )
