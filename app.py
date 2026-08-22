"""
Traffic Intelligence dashboard.

Reads only what the pipeline wrote to outputs/ - it runs no models and does no
detection, so it starts instantly and can be re-opened without touching the GPU.

    streamlit run app.py
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT_DIR = "outputs"

st.set_page_config(page_title="Drone Traffic Intelligence", page_icon="::", layout="wide")

# Dark operations-console styling; the video and charts carry the colour.
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {
        background: #171a1f; border: 1px solid #262b33;
        border-radius: 10px; padding: 12px 14px;
      }
      div[data-testid="stMetricLabel"] p {font-size: .72rem; letter-spacing: .06em;
        text-transform: uppercase; color: #8b95a5;}
      .caveat {font-size: .78rem; color: #8b95a5; border-left: 2px solid #3d4450;
        padding-left: .7rem; margin: .3rem 0 1rem 0;}
      .pill {display:inline-block; padding:2px 9px; border-radius:999px;
        font-size:.7rem; font-weight:600; letter-spacing:.04em;}
      h1, h2, h3 {letter-spacing: -0.01em;}
    </style>
    """,
    unsafe_allow_html=True,
)

CLASS_COLOR = {
    "car": "#50c878", "lgv": "#c85adc", "hgv": "#963cbe", "truck": "#c85adc",
    "bus": "#f0783c", "motorcycle": "#ffbe3c", "bicycle": "#64dcf0",
    "pedestrian": "#5a82ff",
}
SEV_COLOR = {"high": "#ff4d5a", "medium": "#ffa03c", "low": "#7d8794"}


# --------------------------------------------------------------------------- io
def available_tags() -> list[str]:
    """Discover pipeline runs by looking for summary JSONs."""
    tags = []
    for p in glob.glob(os.path.join(OUT_DIR, "summary*.json")):
        base = os.path.basename(p)[len("summary"):-len(".json")]
        tags.append(base.lstrip("_"))
    return sorted(tags, key=lambda t: (t != "", t))


def suffixed(name: str, ext: str, tag: str) -> str:
    return os.path.join(OUT_DIR, f"{name}{'_' + tag if tag else ''}{ext}")


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ load a run
tags = available_tags()
if not tags:
    st.title("Drone Traffic Intelligence")
    st.error("No pipeline output found. Run the pipeline first:")
    st.code("python run_pipeline.py", language="bash")
    st.stop()

with st.sidebar:
    st.markdown("### Run")
    tag = st.selectbox("pipeline output", tags, format_func=lambda t: t or "(default)")

summary = load_json(suffixed("summary", ".json", tag))
obs = load_csv(suffixed("trajectories", ".csv", tag))
events = load_csv(suffixed("events", ".csv", tag))
tracks = load_csv(suffixed("track_summary", ".csv", tag))
queues = load_csv(suffixed("queues", ".csv", tag))
congestion = load_csv(suffixed("congestion", ".csv", tag))
active = load_csv(suffixed("active_tracks", ".csv", tag))
turns = load_csv(suffixed("turning_movements", ".csv", tag))
flow = load_csv(suffixed("directional_flow", ".csv", tag))
inter = load_csv(suffixed("interactions", ".csv", tag))
headways = load_csv(suffixed("headways", ".csv", tag))

calib = summary.get("calibration", {})
calibrated = bool(calib.get("usable"))
counts = summary.get("counts", {})
speed = summary.get("speed", {})

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Calibration")
    if calibrated:
        st.success("Metric - DJI telemetry")
        st.metric("Altitude", f"{calib.get('altitude_m')} m")
        st.metric("Ground scale", f"{calib.get('metres_per_px_at_centre')} m/px")
        st.caption(
            f"gimbal pitch {calib.get('gimbal_pitch_deg')}deg, "
            f"yaw {calib.get('gimbal_yaw_deg')}deg\n\n"
            f"scene width ~{calib.get('bottom_edge_ground_width_m')} m "
            f"from {calib.get('records')} telemetry records"
        )
        hs = calib.get("hover_stability", {})
        if hs:
            st.caption(
                "**Hover stability** (justifies the single-pose model)\n\n"
                f"altitude spread {hs.get('rel_alt_p5_p95_spread', 0):.3f} m, "
                f"pitch {hs.get('pitch_deg_p5_p95_spread', 0):.2f}deg, "
                f"yaw {hs.get('yaw_deg_p5_p95_spread', 0):.2f}deg"
            )
        if calib.get("gps"):
            st.caption(f"GPS {calib['gps'][0]}, {calib['gps'][1]}")
    else:
        st.warning("No metric calibration")
        st.caption(
            f"{calib.get('reason', 'telemetry unavailable')}\n\n"
            "Analytics fall back to pixel space. **km/h is deliberately not reported** "
            "rather than fabricated."
        )

    st.markdown("### Model")
    det = summary.get("detector", {})
    st.caption(
        f"**{os.path.basename(str(det.get('weights', '?')))}** + ByteTrack\n\n"
        f"imgsz {det.get('imgsz')} - conf {det.get('conf')} - device {det.get('device')}"
    )
    v = summary.get("video", {})
    st.caption(
        f"{v.get('width')}x{v.get('height')} @ {round(float(v.get('fps', 0)), 1)} fps\n\n"
        f"{v.get('frames_processed')} frames at stride {v.get('stride')} "
        f"({v.get('processing_fps')} fps processing)"
    )

# --------------------------------------------------------------------- header
st.title("Drone Traffic Intelligence")
st.caption(
    "Trajectory-first traffic analytics - every insight below is derived from tracked "
    "road-user trajectories, not from isolated frame detections."
)

k = st.columns(6)
k[0].metric("Total road users", counts.get("total", 0))
k[1].metric("Motor vehicles", counts.get("motor_vehicles", 0))
k[2].metric("Vulnerable users", counts.get("vulnerable", 0),
            help="Pedestrians, cyclists and motorcyclists")
peak = int(active["active_tracks"].max()) if not active.empty else 0
k[3].metric("Peak simultaneous", peak)
if calibrated and speed.get("mean_moving_speed") is not None:
    k[4].metric("Mean moving speed", f"{speed['mean_moving_speed']} km/h",
                help="Estimate. " + str(speed.get("basis", "")))
else:
    k[4].metric("Mean speed", f"{speed.get('mean_moving_speed', '-')} px/s",
                help="No metric calibration, so km/h is not reported.")
n_conf = int(summary.get("interactions", {}).get("potential_conflicts", 0))
k[5].metric("Potential conflicts", n_conf)

cong_levels = summary.get("congestion", {}).get("levels", {})
if cong_levels:
    dominant = max(cong_levels, key=cong_levels.get)
    col = {"LOW": "#50c878", "MEDIUM": "#ffa03c", "HIGH": "#ff4d5a"}.get(dominant, "#7d8794")
    st.markdown(
        f"<span class='pill' style='background:{col}22;color:{col};border:1px solid {col}55'>"
        f"CONGESTION {dominant}</span>"
        f"<span class='caveat' style='display:inline;border:none;margin-left:.8rem'>"
        f"peak score {summary.get('congestion', {}).get('peak_score')} - analytical estimate for "
        f"operator triage, not a certified level of service</span>",
        unsafe_allow_html=True,
    )

st.divider()

# ------------------------------------------------------- video + traffic summary
left, right = st.columns([1.65, 1], gap="large")

with left:
    st.subheader("Annotated aerial view")
    vid = suffixed("annotated", ".mp4", tag)
    web = suffixed("annotated_web", ".mp4", tag)
    playable = web if os.path.exists(web) else vid
    if os.path.exists(playable):
        st.video(playable)
        st.markdown(
            "<div class='caveat'>Bounding box, class, track ID, live speed and trajectory "
            "trail per road user. Flagged objects switch to red/orange; potential conflicts "
            "are drawn as a line between the two interacting users.</div>",
            unsafe_allow_html=True,
        )
        if not os.path.exists(web):
            st.caption(
                "If the player shows nothing, the file is mp4v-encoded. Re-encode to H.264:"
            )
            st.code(f"ffmpeg -i {vid} -vcodec libx264 -crf 28 -y {web}", language="bash")
    else:
        st.info("No annotated video for this run.")

with right:
    st.subheader("Traffic composition")
    by_class = counts.get("by_class", {})
    if by_class:
        cdf = pd.DataFrame({"class": list(by_class), "road users": list(by_class.values())})
        fig = px.bar(
            cdf.sort_values("road users"), x="road users", y="class", orientation="h",
            color="class", color_discrete_map=CLASS_COLOR, text="road users",
        )
        fig.update_layout(
            showlegend=False, height=260, margin=dict(l=0, r=10, t=6, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title=None, yaxis_title=None,
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

        if calibrated and not tracks.empty and "class_source" in tracks:
            n_heur = int((tracks["class_source"] == "size_heuristic_from_calibration").sum())
            if n_heur:
                st.markdown(
                    f"<div class='caveat'><b>LGV / HGV caveat:</b> the detector has no LGV or HGV "
                    f"class. {n_heur} <code>truck</code> detections were split by measured ground "
                    f"footprint (threshold 7 m) using the telemetry calibration. This is a size "
                    f"heuristic, recorded in the CSV as "
                    f"<code>class_source=size_heuristic_from_calibration</code> - not a model output."
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.subheader("Identity stability")
    stab = summary.get("id_stability", {})
    if stab:
        c = st.columns(2)
        c[0].metric("Mean track life", f"{stab.get('mean_track_duration_s', 0)} s")
        c[1].metric("Longest track", f"{stab.get('max_track_duration_s', 0)} s")
        c = st.columns(2)
        c[0].metric("Occlusion recoveries", stab.get("gap_recoveries", 0),
                    help="Times a track was re-associated after missed detections instead "
                         "of being reborn as a new ID.")
        c[1].metric("Longest bridged gap", f"{stab.get('longest_bridged_gap_s', 0)} s")
        st.markdown(
            "<div class='caveat'>Occlusion recoveries are the measurable evidence for stable "
            "identity: ByteTrack's 90-frame buffer re-linked tracks across missed detections "
            "rather than issuing a new ID. Without hand-labelled ground truth this is the "
            "honest metric - we do not quote a MOTA/IDF1 score we cannot compute.</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ------------------------------------------------------------------ trajectories
st.subheader("Trajectory map")
if not obs.empty:
    metric_mode = calibrated and "sx" in obs and obs["sx"].notna().any()
    xcol, ycol = ("sx", "sy") if metric_mode else ("x", "y")
    d = obs.dropna(subset=[xcol, ycol])

    cc = st.columns([1, 1, 2])
    max_tracks = cc[0].slider("tracks to draw", 10, 400,
                              min(120, int(d["track_id"].nunique())), step=10)
    sel_classes = cc[1].multiselect("classes", sorted(d["class"].dropna().unique()),
                                    default=sorted(d["class"].dropna().unique()))
    d = d[d["class"].isin(sel_classes)]
    keep = d["track_id"].drop_duplicates().head(max_tracks)
    d = d[d["track_id"].isin(keep)]

    fig = go.Figure()
    for tid, g in d.groupby("track_id"):
        cls = g["class"].iloc[0]
        fig.add_trace(go.Scattergl(
            x=g[xcol], y=g[ycol], mode="lines",
            line=dict(width=1.4, color=CLASS_COLOR.get(cls, "#9aa4b2")),
            name=str(cls), legendgroup=str(cls), showlegend=False,
            hovertemplate=f"{cls} #{int(tid)}<br>%{{x:.1f}}, %{{y:.1f}}<extra></extra>",
        ))
    unit = "metres east of aircraft" if metric_mode else "pixels (image x)"
    unit_y = "metres north of aircraft" if metric_mode else "pixels (image y)"
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12151a",
        xaxis=dict(title=unit, gridcolor="#1f242c", zeroline=False),
        yaxis=dict(title=unit_y, gridcolor="#1f242c", zeroline=False,
                   scaleanchor="x", scaleratio=1,
                   autorange=True if metric_mode else "reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    frame_note = (
        "common metric ground frame - the single coordinate system a drone provides "
        "and fixed cameras cannot"
        if metric_mode else "image-space frame (no calibration available)"
    )
    st.markdown(
        f"<div class='caveat'>{len(keep)} trajectories in a {frame_note}. "
        f"Axes are equal-scaled, so shapes are geometrically faithful.</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("No trajectories for this run.")

st.divider()

# --------------------------------------------------------- congestion + demand
c1, c2 = st.columns(2, gap="large")

with c1:
    st.subheader("Congestion over time")
    if not congestion.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=congestion["t"], y=congestion["score"],
            marker_color=[{"LOW": "#50c878", "MEDIUM": "#ffa03c", "HIGH": "#ff4d5a"}
                          .get(l, "#7d8794") for l in congestion["level"]],
            hovertemplate="t=%{x}s<br>score %{y:.2f}<extra></extra>", name="score",
        ))
        if not active.empty:
            fig.add_trace(go.Scatter(
                x=active["t"], y=active["active_tracks"], mode="lines",
                line=dict(color="#64dcf0", width=1.5), name="active tracks", yaxis="y2",
            ))
        fig.update_layout(
            height=320, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12151a",
            xaxis=dict(title="time (s)", gridcolor="#1f242c"),
            yaxis=dict(title="congestion score", gridcolor="#1f242c", range=[0, 1]),
            yaxis2=dict(title="active tracks", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"<div class='caveat'>{summary.get('congestion', {}).get('method', '')}</div>",
            unsafe_allow_html=True,
        )

with c2:
    st.subheader("Turning movement demand")
    if not turns.empty:
        fig = px.bar(turns, x="turn_type", y="road_users", color="class",
                     color_discrete_map=CLASS_COLOR, barmode="stack")
        fig.update_layout(
            height=320, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12151a",
            xaxis=dict(title=None, gridcolor="#1f242c"),
            yaxis=dict(title="road users", gridcolor="#1f242c"),
            legend=dict(orientation="h", y=1.14, x=0, title=None),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "<div class='caveat'>Derived from entry vs exit heading of each trajectory. "
            "Tracks too short to judge reliably are excluded rather than forced into a bucket."
            "</div>",
            unsafe_allow_html=True,
        )
    elif not flow.empty:
        st.dataframe(flow, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough sustained trajectories to classify turning movements.")

st.divider()

# ---------------------------------------------------------------- events tabs
st.subheader("Detected events")
t1, t2, t3, t4, t5 = st.tabs(
    ["Event log", "Interactions & conflicts", "Queues", "Directional flow", "All tracks"]
)

with t1:
    if not events.empty:
        sev = st.multiselect("severity", ["high", "medium", "low"],
                             default=["high", "medium", "low"], key="sev")
        types = st.multiselect("event type", sorted(events["event_type"].unique()),
                               default=sorted(events["event_type"].unique()), key="et")
        e = events[events["severity"].isin(sev) & events["event_type"].isin(types)]
        st.caption(f"{len(e)} of {len(events)} events")
        for _, r in e.head(60).iterrows():
            col = SEV_COLOR.get(str(r["severity"]), "#7d8794")
            tid = f"#{int(r['track_id'])}" if pd.notna(r["track_id"]) else ""
            ts = f"{float(r['timestamp']):.1f}s" if pd.notna(r["timestamp"]) else "-"
            st.markdown(
                f"<div style='border-left:3px solid {col};background:#15181d;padding:.55rem .8rem;"
                f"margin-bottom:.4rem;border-radius:0 7px 7px 0'>"
                f"<span class='pill' style='background:{col}22;color:{col}'>"
                f"{str(r['severity']).upper()}</span>&nbsp;"
                f"<b>{str(r['event_type']).replace('_', ' ')}</b> {tid} "
                f"<span style='color:#8b95a5'>at {ts}</span><br>"
                f"<span style='font-size:.85rem;color:#c3cad4'>{r['description']}</span></div>",
                unsafe_allow_html=True,
            )
        with st.expander("raw events table"):
            st.dataframe(e, use_container_width=True, hide_index=True)
    else:
        st.info("No events detected.")

with t2:
    isum = summary.get("interactions", {})
    if isum.get("pairs"):
        c = st.columns(4)
        c[0].metric("Interacting pairs", isum.get("pairs", 0))
        c[1].metric("Potential conflicts", isum.get("potential_conflicts", 0))
        c[2].metric("Involving vulnerable users", isum.get("vulnerable_involved", 0))
        c[3].metric("Lowest TTC", f"{isum.get('min_ttc_s', '-')} s")
        st.markdown(
            "<div class='caveat'><b>Method:</b> for every nearby pair we solve for the closest "
            "point of approach under constant velocity. A pair is flagged only if it is genuinely "
            "closing (>=6 km/h along the line of centres), the projected miss distance falls inside "
            "the two users' combined physical envelope, and the trigger <b>persists over multiple "
            "frames</b> - the filter that separates a real conflict from tracker jitter. "
            "A constant-velocity model cannot know a driver was already braking, so these are "
            "conflict <b>indicators</b> for review, not collisions.</div>",
            unsafe_allow_html=True,
        )
        if not inter.empty:
            cols = [c for c in ["track_a", "class_a", "track_b", "class_b", "t", "ttc_s",
                                "min_distance", "d_cpa", "closing_speed", "frames_triggered",
                                "interaction_type", "severity", "unit"] if c in inter.columns]
            st.dataframe(inter[cols], use_container_width=True, hide_index=True)
        if not headways.empty:
            st.markdown("**Car-following headways** (gap / follower speed)")
            st.dataframe(headways, use_container_width=True, hide_index=True)
    else:
        st.info("No qualifying interactions detected.")

with t3:
    if not queues.empty:
        unit = queues["length_unit"].iloc[0]
        c = st.columns(3)
        c[0].metric("Queue detections", len(queues))
        c[1].metric("Largest queue", f"{int(queues['vehicles'].max())} vehicles")
        c[2].metric("Longest queue", f"{queues['length'].max():.0f} {unit}")
        st.dataframe(queues, use_container_width=True, hide_index=True)
        st.markdown(
            f"<div class='caveat'>Single-linkage spatial clustering of slow/stopped vehicles per "
            f"time bin. Length is the cluster's extent along its own principal axis, reported in "
            f"<b>{unit}</b>{' from the telemetry calibration' if unit == 'm' else ' - pixels, because no calibration was available'}. "
            f"Queue membership is also used to downgrade stationary vehicles that are merely "
            f"waiting at a signal.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No queues met the clustering threshold.")

with t4:
    if not flow.empty:
        fig = px.bar_polar(flow, r="road_users", theta="sector",
                           color="road_users", color_continuous_scale="Teal")
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Frame of reference: {flow['frame_of_reference'].iloc[0]}")
        if calibrated:
            st.markdown(
                "<div class='caveat'>These are <b>true compass bearings</b>, recovered from the "
                "gimbal yaw in the telemetry. Without that we would only be able to report "
                "image-space directions - we never invent geographic directions.</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No directional flow available.")

with t5:
    if not tracks.empty:
        show = [c for c in ["track_id", "class", "class_source", "first_t", "last_t", "visible_s",
                            "stationary_s", "longest_stop_s", "mean_speed_kph", "max_speed_kph",
                            "path_len_m", "turn_type", "footprint_p90_m", "n_obs"]
                if c in tracks.columns]
        st.dataframe(tracks[show], use_container_width=True, hide_index=True, height=460)
        st.download_button("Download track summary CSV", tracks.to_csv(index=False),
                           file_name="track_summary.csv", mime="text/csv")

# ------------------------------------------------------------------ limitations
st.divider()
with st.expander("Limitations and what we deliberately do not claim", expanded=False):
    st.markdown(
        """
| Area | What we claim | What we do **not** claim |
|---|---|---|
| **Classes** | Detector classes: pedestrian, bicycle, car, motorcycle, bus, truck | The model does **not** distinguish LGV from HGV. That split is a footprint-size heuristic from calibration, tagged `class_source` in the CSV |
| **Speed** | Estimated km/h from a telemetry-derived ground plane | Not survey-grade. Assumes flat ground; uses the box bottom as the road-contact point |
| **Stationary** | "Stationary vehicle **candidate**", cross-referenced against detected queues | Not an incident. One clip cannot separate a breakdown from a red light |
| **Wrong way** | Movement against the **observed dominant flow** of that part of the road, learned from the data | Not a legal violation - we have no HD map of legal carriageway directions |
| **Conflicts** | "**Potential** conflict" from a constant-velocity CPA projection, filtered by closing rate and persistence | Not a collision, and not a certified traffic-conflict technique score |
| **Congestion** | A transparent weighted score for operator triage | Not a certified level of service - that needs lane geometry and capacity |
| **Coordinates** | Metres relative to the aircraft, in a common ground frame | Not absolute survey coordinates, though GPS is recorded so results could be georeferenced |
"""
    )
