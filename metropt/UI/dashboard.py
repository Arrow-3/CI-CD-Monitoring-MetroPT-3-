from __future__ import annotations
import json
import time
import threading
import uuid
from collections import deque

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from metropt import kafka_utils
from metropt.config.settings import get_consumer_group_id

# ------------------------------------------------------------ config
BUFFER = 2000
REFRESH_MS = 1000
TOPICS = ["predictions", "coordinates_nd", "shift_events",
          "model_update_complete"]


# ------------------------------------------------------------ shared state
@st.cache_resource
def _get_state():
    """Single source of truth for buffers + lock, shared across all reruns
    and module reloads. This is the Streamlit-idiomatic way to share mutable
    state between background threads and the render loop."""
    return {
        "lock": threading.Lock(),
        "buffers": {t: deque(maxlen=BUFFER) for t in TOPICS},
        "rates": {t: deque(maxlen=30) for t in TOPICS},
    }


def _consume_forever():
    """Background thread: reads all UI topics into the shared state."""
    try:
        state = _get_state()
        buffers, rates, lock = state["buffers"], state["rates"], state["lock"]
        group = f"ui_dashboard_{uuid.uuid4().hex[:8]}"
        consumer = kafka_utils.get_consumer_multi(TOPICS, group)
        counts = {t: 0 for t in TOPICS}
        tick = time.time()
        for msg in consumer:
            try:
                payload = json.loads(msg.value)
                payload["_topic"] = msg.topic
                payload["_recv_ts"] = time.time()
                with lock:
                    buffers[msg.topic].append(payload)
                counts[msg.topic] += 1
            except Exception:
                continue
            now = time.time()
            if now - tick >= 1.0:
                with lock:
                    for t in TOPICS:
                        rates[t].append(counts[t])
                        counts[t] = 0
                tick = now
    except Exception as e:
        print(f"[UI] consumer crashed: {e}", flush=True)
        raise


@st.cache_resource
def _start_consumer_thread():
    """Cached: starts exactly once per process, no matter how many reruns."""
    t = threading.Thread(target=_consume_forever, daemon=True)
    t.start()
    return t


# ------------------------------------------------------------ helpers
def _df(topic: str) -> pd.DataFrame:
    state = _get_state()
    with state["lock"]:
        return pd.DataFrame(list(state["buffers"][topic]))


def _rate(topic: str) -> float:
    state = _get_state()
    with state["lock"]:
        r = list(state["rates"][topic])
    return round(sum(r) / max(len(r), 1), 1)


def _latest_severity() -> str:
    events = list(_get_state()["buffers"]["shift_events"])
    if not events:
        return "healthy"
    recent = events[-1]
    age = time.time() - recent["_recv_ts"]
    if age > 30:
        return "healthy"
    return recent["severity"]


def _latest_model_version() -> str:
    preds = list(_get_state()["buffers"]["predictions"])
    return preds[-1]["model_version"] if preds else "—"


# ------------------------------------------------------------ layout
st.set_page_config(page_title="MetroPT · PHM", layout="wide")
_start_consumer_thread()

st.title("MetroPT-3 · Adaptive Condition Monitoring")

# Header row: live metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Predictions/s", _rate("predictions"))
c2.metric("Coords/s", _rate("coordinates_nd"))
c3.metric("Drift events (buf)",
          len(_get_state()["buffers"]["shift_events"]))
sev = _latest_severity()
sev_color = {"healthy": "🟢", "warn": "🟡", "alert": "🟠", "critical": "🔴"}
c4.metric("Status", f"{sev_color.get(sev, '⚪')} {sev.upper()}")
mv = _latest_model_version()
c5.metric("Model", mv[-8:] if mv != "—" else "—")

st.divider()

# Left: scatter (2D feature space). Right: probability timeline.
left, right = st.columns([1, 1])

with left:
    st.subheader("Feature-space scatter (PCA)")
    coords = _df("coordinates_nd")
    preds = _df("predictions")
    if not coords.empty and not preds.empty:
        coords["ts"] = pd.to_datetime(coords["ts"])
        preds["ts"] = pd.to_datetime(preds["ts"])
        merged = pd.merge_asof(
            coords.sort_values("ts"),
            preds.sort_values("ts")[["ts", "prob"]],
            on="ts", direction="nearest", tolerance=pd.Timedelta("2s"),
        ).dropna(subset=["prob"])
        if not merged.empty:
            fig = go.Figure(go.Scattergl(
                x=merged["x"], y=merged["y"], mode="markers",
                marker=dict(size=6, color=merged["prob"],
                            colorscale="RdYlGn_r", cmin=0, cmax=1,
                            colorbar=dict(title="P(failure)")),
                text=merged["ts"].astype(str), hoverinfo="text",
            ))
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aligning coordinates + predictions...")
    else:
        st.info("Waiting for scatter data...")

with right:
    st.subheader("Failure probability over time")
    if not preds.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(preds["ts"]), y=preds["prob"],
            mode="lines", line=dict(width=1.5),
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="threshold")
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for predictions...")

st.divider()

# Drift score timeline
st.subheader("Distribution drift score")
shifts = _df("shift_events")
if not shifts.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(shifts["ts"]), y=shifts["score"],
        mode="lines+markers", marker=dict(size=6),
    ))
    for level, color, val in [("warn", "gold", 0.15),
                              ("alert", "orange", 0.25),
                              ("critical", "red", 0.40)]:
        fig.add_hline(y=val, line_dash="dot", line_color=color,
                      annotation_text=level)
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No drift events yet — the monitor is still calibrating.")

# Two tables side by side
tcol1, tcol2 = st.columns(2)

with tcol1:
    st.subheader("Recent drift events")
    if not shifts.empty:
        display = shifts.tail(10)[["ts", "severity", "score"]].copy()
        display["top_features"] = shifts.tail(10)["window"].apply(
            lambda w: ", ".join(f["name"] for f in w.get("top_features", [])[:3])
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.write("—")

with tcol2:
    st.subheader("Model update history")
    updates = _df("model_update_complete")
    if not updates.empty:
        display = updates.tail(10)[["ts", "promotion_decision"]].copy()
        display["auprc"] = updates.tail(10)["eval_metrics"].apply(
            lambda m: round(m.get("auprc", 0), 3))
        display["vs_incumbent"] = updates.tail(10)["eval_metrics"].apply(
            lambda m: round(m.get("incumbent_auprc", 0), 3))
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.write("—")

# Auto-refresh
time.sleep(REFRESH_MS / 1000)
st.rerun()