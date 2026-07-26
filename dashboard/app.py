"""
dashboard/app.py - Streamlit operator terminal dashboard for QCS Copilot.

Entry point:  streamlit run dashboard/app.py   (run from project root)

Acts as the single visual operator-facing terminal for replaying transitions,
monitoring live predicted risk, displaying plain-language explanation rationale
cards, showing corrective setpoint recommendation options, recording operator
accept/reject choices, and rendering the Trust Ledger calibration & Correlation panel.

CONSTITUTION RULES OBSERVED:
  Rule 4  - Dashboard contains only UI code; logic resides in src/ modules.
  Rule 6  - No hidden dependencies or file reads. Calls public interfaces.
  Rule 7  - Excludes `aggressive_ramp` from model feature inputs.
  Rule 14 - Operates on stored replay dataset, never live-simulates on the fly.
"""

from __future__ import annotations

import os
import sys

# Streamlit Cloud runs this script with only its own directory (dashboard/) on
# the import path, not the project root — so `from src.xxx import yyy` fails
# there even though it works locally (where the project root happens to
# already be on the path). This explicitly adds the project root (one level
# up from this file) so `src` and `data` resolve identically in both
# environments. Must run before any `from src...`/`from data...` import below.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import random
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

from src.data_loader import load_events, load_timeseries
from src.features import generate_features
from src.recommend import get_recommendations, assess_transition_novelty
from src.explain import explain_prediction
from src.feedback_store import log_feedback, get_calibration_stats, get_calibration_curve_data
from src.correlation_discovery import run_full_discovery
from src.cost_model import estimate_broke_cost, DEFAULT_PRICE_PER_TON_USD, DEFAULT_WEB_WIDTH_M

logger = logging.getLogger(__name__)

# Premium glassmorphism custom CSS
GLASS_CSS = """
<style>
    .reportview-container {
        background: #0f111a;
    }
    div.stButton > button:first-child {
        background-color: #1e3a8a;
        color: #ffffff;
        border-radius: 6px;
        border: 1px solid #3b82f6;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #3b82f6;
        border-color: #60a5fa;
        transform: scale(1.02);
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    .recommendation-card {
        background: rgba(15, 23, 42, 0.7);
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #3b82f6;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 12px;
    }
    .recommendation-card-rule {
        border-left: 4px solid #f59e0b;
    }
    .recommendation-card-model {
        border-left: 4px solid #10b981;
    }
</style>
"""

# Page Configuration
st.set_page_config(
    page_title="QCS Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(GLASS_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State Initialization
# ---------------------------------------------------------------------------
if "time_step" not in st.session_state:
    st.session_state["time_step"] = 300  # Default prediction cutoff
if "is_playing" not in st.session_state:
    st.session_state["is_playing"] = False
if "accepted_recs" not in st.session_state:
    st.session_state["accepted_recs"] = set()
if "rejected_recs" not in st.session_state:
    st.session_state["rejected_recs"] = set()


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------
@st.cache_data
def _load_cached_data():
    events = load_events()
    ts = load_timeseries()
    features = generate_features()
    return events, ts, features


@st.cache_resource
def _load_cached_models():
    clf = joblib.load("models/offspec_classifier.pkl")
    reg = joblib.load("models/stabilization_regressor.pkl")
    return clf, reg


@st.cache_data
def _load_cached_correlations():
    return run_full_discovery()


@st.cache_data
def _get_cached_recommendations(event_id: int):
    return get_recommendations(event_id)


try:
    events_df, ts_df, features_df = _load_cached_data()
    clf, reg = _load_cached_models()
except Exception as e:
    st.error(f"Error loading system files/models: {e}. Please ensure you ran the generator and models scripts.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar Event Replay Selector
# ---------------------------------------------------------------------------
st.sidebar.title("🎮 QCS Copilot Replay Control")

event_ids = events_df["event_id"].tolist()
_event_display = events_df.set_index("event_id")  # safe index for label lookup
selected_event_id = st.sidebar.selectbox(
    "Select Transition Event:",
    event_ids,
    format_func=lambda x: f"Event #{x} ({_event_display.loc[x, 'from_grade']} -> {_event_display.loc[x, 'to_grade']})"
)

# Load selected event details
event_row = events_df[events_df["event_id"] == selected_event_id].iloc[0]
event_ts = ts_df[ts_df["event_id"] == selected_event_id].copy().sort_values("t_seconds")

# Display event metadata
st.sidebar.markdown(f"""
<div class="glass-card">
    <h4 style="margin-top:0;">📋 Transition Details</h4>
    <p><b>From Grade:</b> {event_row['from_grade']}</p>
    <p><b>To Grade:</b> {event_row['to_grade']}</p>
    <p><b>Ambient Temp:</b> {event_row['ambient_temp_c']:.1f} °C</p>
    <p><b>Valve Wear Index:</b> {event_row['valve_wear_index']:.2f}</p>
    <p><b>Planned Ramp:</b> {event_row['ramp_duration_seconds']} seconds</p>
</div>
""", unsafe_allow_html=True)

# Cost model assumptions — illustrative defaults, adjustable for a real mill.
# See src/cost_model.py docstring for why these are exposed rather than baked in.
with st.sidebar.expander("💰 Cost Assumptions (edit for your mill)"):
    st.caption("Illustrative defaults — not real mill data. Adjust to your own figures.")
    price_per_ton_usd = st.number_input(
        "Paper price ($/metric ton)", min_value=100.0, max_value=5000.0,
        value=DEFAULT_PRICE_PER_TON_USD, step=50.0,
    )
    web_width_m = st.number_input(
        "Web width (meters)", min_value=1.0, max_value=12.0,
        value=DEFAULT_WEB_WIDTH_M, step=0.5,
    )

# Replay animation buttons
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("▶ Play"):
        st.session_state["is_playing"] = True
with col2:
    if st.button("⏸ Pause"):
        st.session_state["is_playing"] = False
with col3:
    if st.button("🔄 Reset"):
        st.session_state["time_step"] = 300
        st.session_state["is_playing"] = False
        st.session_state["accepted_recs"] = set()
        st.session_state["rejected_recs"] = set()

# Timeline Slider
st.session_state["time_step"] = st.sidebar.slider(
    "Timeline (seconds):",
    min_value=0,
    max_value=2400,
    value=st.session_state["time_step"],
    step=5
)

# Handle Play Animation Loop
if st.session_state["is_playing"]:
    if st.session_state["time_step"] < 2400:
        st.session_state["time_step"] += 30  # accelerate playback for demo smoothness
        time.sleep(0.08)
        st.rerun()
    else:
        st.session_state["is_playing"] = False


# ---------------------------------------------------------------------------
# Main Layout Tabs
# ---------------------------------------------------------------------------
tab_operator, tab_correlations, tab_calibration = st.tabs([
    "🖥️ Operator Control Console",
    "🔍 Discovered Correlations",
    "📈 Calibration & Trust Ledger"
])


# ===========================================================================
# Tab 1: Operator Control Console
# ===========================================================================
with tab_operator:
    st.title("QCS Copilot Console")
    
    current_time = st.session_state["time_step"]
    
    # 1. Gauge and Predictions Metrics Row
    col_metrics, col_rationales = st.columns([1, 2])
    
    # Restrict actual/measured timeline data to current_time (Anti-leakage)
    visible_ts = event_ts[event_ts["t_seconds"] <= current_time]
    
    with col_metrics:
        # Build plot representation of risk gauge
        if current_time >= 300:
            # Predict risk using t<=300 features (static windowed prediction)
            feat_row = features_df[features_df["event_id"] == selected_event_id]
            drop_cols = ["event_id", "offspec_label", "stabilization_time_seconds", "early_breach"]
            feat_cols = [c for c in feat_row.columns if c not in drop_cols]
            
            p_offspec = float(clf.predict_proba(feat_row[feat_cols])[0][1])
            pred_stab_s = float(reg.predict(feat_row[feat_cols])[0])
            
            # Adjust using the SPECIFIC accepted recommendation's own computed
            # impact (was previously a hardcoded flat -0.45 / -120s mock that
            # didn't match the numbers shown on the recommendation card itself).
            current_event_recs = _get_cached_recommendations(int(selected_event_id))
            accepted_this_event = [
                r for r in current_event_recs
                if f"{selected_event_id}_{r.setpoint_variable}_{r.suggested_value}"
                in st.session_state["accepted_recs"]
            ]
            modified = bool(accepted_this_event)

            if modified:
                # If more than one was accepted, use the strongest single
                # accepted recommendation's stated impact rather than summing
                # (the models were never re-predicted on a stacked/chained
                # adjustment, so summing would overstate the effect).
                best_rec = max(accepted_this_event, key=lambda r: r.expected_risk_reduction)
                p_offspec = max(0.02, p_offspec - best_rec.expected_risk_reduction)
                pred_stab_s = max(0.0, pred_stab_s - best_rec.expected_stab_reduction_s)
                
            risk_pct = p_offspec * 100
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                title={"text": "Predicted Off-Spec Risk (%)", "font": {"size": 16, "color": "white"}},
                number={"font": {"color": "white", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": "#ef4444" if risk_pct > 60 else ("#f59e0b" if risk_pct > 30 else "#10b981")},
                    "bgcolor": "rgba(255, 255, 255, 0.05)",
                    "borderwidth": 1,
                    "steps": [
                        {"range": [0, 30], "color": "rgba(16, 185, 129, 0.15)"},
                        {"range": [30, 60], "color": "rgba(245, 158, 11, 0.15)"},
                        {"range": [60, 100], "color": "rgba(239, 68, 68, 0.15)"}
                    ]
                }
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=220,
                margin=dict(t=30, b=10, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Predict remaining stabilization metrics
            st.metric(
                label="Predicted Stabilization Delay",
                value=f"{pred_stab_s:.0f} seconds" if pred_stab_s > 0 else "Stabilized"
            )

            # Illustrative broke-cost translation — makes the risk number
            # concrete for a non-technical reviewer. Uses the destination
            # grade's basis weight/machine speed at the current point in the
            # ramp, and the predicted stabilization delay as the assumed
            # off-spec duration.
            if not visible_ts.empty:
                current_bw_setpoint = float(visible_ts["basis_weight_setpoint"].iloc[-1])
                current_speed_setpoint = float(visible_ts["machine_speed_setpoint"].iloc[-1])
                est_cost = estimate_broke_cost(
                    basis_weight_gsm=current_bw_setpoint,
                    machine_speed_m_per_min=current_speed_setpoint,
                    duration_seconds=pred_stab_s,
                    price_per_ton_usd=price_per_ton_usd,
                    web_width_m=web_width_m,
                )
                st.metric(
                    label="Estimated Broke Cost If Off-Spec Persists",
                    value=f"${est_cost:,.0f}",
                )
                st.caption("Illustrative estimate based on the cost assumptions in the sidebar — not a real financial figure.")

            # Novelty / out-of-distribution check — is this transition unlike
            # anything in the training history? Confidence should be treated
            # with more caution when it is.
            novelty = assess_transition_novelty(int(selected_event_id))
            if novelty["is_novel"]:
                st.warning(
                    f"🧭 This transition sits outside {novelty['percentile']:.0f}% of historical "
                    f"experience (closest precedent: Event #{novelty['nearest_event_id']}). "
                    f"Treat automated predictions here with extra caution."
                )
            else:
                st.caption(
                    f"🧭 Within normal historical experience "
                    f"(closest precedent: Event #{novelty['nearest_event_id']})."
                )
        else:
            st.info("⌛ Live Risk Gauge active at t >= 300 seconds.")
            
    with col_rationales:
        st.subheader("💡 Co-Pilot Diagnostics")
        if current_time >= 300:
            feat_row = features_df[features_df["event_id"] == selected_event_id]
            pred_explanation = explain_prediction(feat_row, clf)
            
            st.markdown(f"""
            <div class="glass-card">
                <h5>🔮 Risk Driver Explanation</h5>
                <p style="font-size: 16px; line-height: 1.5; color: #e2e8f0;">{pred_explanation}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("Collecting process metrics... diagnostics will surface at t = 300s.")

    st.markdown("---")

    # 2. Main Live Trend Charts
    st.subheader("📈 Live Transition Trends")
    
    # --- Future-State Trajectory Projection ---
    # Extrapolates ONLY from data already visible up to current_time (no
    # leakage of real future values, consistent with the rest of the
    # pipeline). The future SETPOINT is legitimately known in advance (it's
    # the planned ramp trajectory, already stored for the whole event in
    # historian_timeseries.csv) — only the ACTUAL trajectory is projected.
    PROJECTION_HORIZON_S = 300
    PROJECTION_LOOKBACK_SAMPLES = 12  # last 60s, matches the settled-window definition

    projection_trace = None
    breach_message = None
    recent = visible_ts.tail(PROJECTION_LOOKBACK_SAMPLES)

    if len(recent) >= 2 and current_time < 2400:
        slope, intercept = np.polyfit(recent["t_seconds"], recent["basis_weight"], 1)
        last_t = float(visible_ts["t_seconds"].iloc[-1])
        last_bw = float(visible_ts["basis_weight"].iloc[-1])

        horizon_end = min(current_time + PROJECTION_HORIZON_S, 2395)
        future = event_ts[(event_ts["t_seconds"] > current_time) & (event_ts["t_seconds"] <= horizon_end)].copy()

        if not future.empty:
            future["projected_bw"] = slope * future["t_seconds"] + intercept
            future["projected_pct_dev"] = (
                (future["projected_bw"] - future["basis_weight_setpoint"])
                / future["basis_weight_setpoint"] * 100.0
            )

            projection_trace = go.Scatter(
                x=[last_t] + future["t_seconds"].tolist(),
                y=[last_bw] + future["projected_bw"].tolist(),
                mode="lines",
                name="Projected trend (if uncorrected)",
                line=dict(color="#a78bfa", dash="dash", width=2),
            )

            breach_rows = future[future["projected_pct_dev"].abs() > 2.5]
            if not breach_rows.empty:
                breach_in_s = float(breach_rows["t_seconds"].iloc[0]) - current_time
                breach_message = (
                    "warning",
                    f"⚠️ Projected trend crosses the ±2.5% spec limit in ~{breach_in_s:.0f}s "
                    f"if the current rate of change continues uncorrected."
                )
            else:
                breach_message = (
                    "success",
                    f"✅ Projected trend stays within spec limits for the next "
                    f"{PROJECTION_HORIZON_S}s at the current rate of change."
                )

    # Basis Weight Chart
    fig_bw = go.Figure()
    
    # Plot complete reference setpoint line for user context
    fig_bw.add_trace(go.Scatter(
        x=event_ts["t_seconds"],
        y=event_ts["basis_weight_setpoint"],
        mode="lines",
        name="Recipe setpoint",
        line=dict(color="#3b82f6", dash="dash", width=2)
    ))
    
    # Plot visible actuals
    fig_bw.add_trace(go.Scatter(
        x=visible_ts["t_seconds"],
        y=visible_ts["basis_weight"],
        mode="lines",
        name="Actual Basis Weight",
        line=dict(color="#ef4444" if not visible_ts.empty and abs(visible_ts["basis_weight_pct_dev"].iloc[-1]) > 2.5 else "#10b981", width=3)
    ))

    if projection_trace is not None:
        fig_bw.add_trace(projection_trace)
    
    # High-quality upper/lower tolerance bands (2.5%)
    fig_bw.add_trace(go.Scatter(
        x=event_ts["t_seconds"],
        y=event_ts["basis_weight_setpoint"] * 1.025,
        mode="lines",
        name="Spec Limit (+2.5%)",
        line=dict(color="rgba(239, 68, 68, 0.4)", width=1, dash="dot"),
        showlegend=False
    ))
    
    fig_bw.add_trace(go.Scatter(
        x=event_ts["t_seconds"],
        y=event_ts["basis_weight_setpoint"] * 0.975,
        mode="lines",
        name="Spec Limit (-2.5%)",
        line=dict(color="rgba(239, 68, 68, 0.4)", width=1, dash="dot"),
        showlegend=False
    ))

    fig_bw.update_layout(
        title="Basis Weight Quality Control vs Specification Limits",
        xaxis_title="Elapsed Time (seconds)",
        yaxis_title="Basis Weight (gsm)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickcolor="white", tickfont=dict(color="white")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickcolor="white", tickfont=dict(color="white")),
        height=320,
        margin=dict(t=40, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_bw, use_container_width=True)

    if breach_message is not None:
        level, text = breach_message
        (st.warning if level == "warning" else st.success)(text)

    # 3. Dynamic Corrective Action Recommendation Panel
    st.markdown("---")
    st.subheader("🛠️ Active Recommendations & Operator Control Decisions")
    
    if current_time >= 300:
        recs = _get_cached_recommendations(int(selected_event_id))
        
        if not recs:
            st.success("✅ Risk remains below alert thresholds. No corrective action required.")
        else:
            for rec in recs:
                rec_id = f"{selected_event_id}_{rec.setpoint_variable}_{rec.suggested_value}"
                
                # Assign styling based on source tag
                card_style = ""
                if "Recipe" in rec.source_tag:
                    card_style = "recommendation-card-rule"
                elif "Model" in rec.source_tag:
                    card_style = "recommendation-card-model"

                # Translate the recommendation's own stated stabilization-time
                # reduction into an illustrative $ savings figure, using the
                # same cost assumptions as the gauge above.
                savings_html = ""
                if rec.expected_stab_reduction_s > 0 and not visible_ts.empty:
                    est_savings = estimate_broke_cost(
                        basis_weight_gsm=float(visible_ts["basis_weight_setpoint"].iloc[-1]),
                        machine_speed_m_per_min=float(visible_ts["machine_speed_setpoint"].iloc[-1]),
                        duration_seconds=rec.expected_stab_reduction_s,
                        price_per_ton_usd=price_per_ton_usd,
                        web_width_m=web_width_m,
                    )
                    savings_html = f' | 💰 Estimated savings: <span style="color:#10b981; font-weight:bold;">${est_savings:,.0f}</span>'

                st.markdown(f"""
                <div class="recommendation-card {card_style}">
                    <h4 style="margin-top:0; color:#ffffff;">💡 Suggested Action: Adjust {rec.setpoint_variable.replace('_', ' ')}</h4>
                    <p><b>Original Target:</b> {rec.current_value} | <b>Recommended Target:</b> <span style="color:#10b981; font-weight:bold;">{rec.suggested_value}</span></p>
                    <p><b>Source Tag:</b> <span style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px;">{rec.source_tag}</span> | <b>Confidence:</b> {rec.confidence*100:.0f}%</p>
                    <p><b>Expected Output Improvement:</b> ⬇️ Reduces deviation probability by {rec.expected_risk_reduction*100:.0f}% | ⚡ Stabilization time reduction of {rec.expected_stab_reduction_s:.0f}s{savings_html}</p>
                    <p><b>Rationale Explanation:</b> <i>{rec.rationale}</i></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Dynamic interaction buttons
                col_btn1, col_btn2, _ = st.columns([1, 1, 6])
                with col_btn1:
                    if rec_id in st.session_state["accepted_recs"]:
                        st.success("Accepted ✅")
                    elif rec_id in st.session_state["rejected_recs"]:
                        st.write("")
                    else:
                        if st.button("Accept", key=f"acc_{rec_id}"):
                            st.session_state["accepted_recs"].add(rec_id)
                            # Was hardcoded to "clean" every time, which made the
                            # Trust Ledger's "Clean Run Rate (Accepted)" metric
                            # tautologically 100% by construction. Since there's
                            # no live re-simulation (replay-only, per constitution),
                            # we can't get a REAL post-correction outcome — but we
                            # can at least be honest about the uncertainty implied
                            # by the recommendation's own stated impact, rather
                            # than asserting success outright.
                            residual_risk = max(0.0, min(1.0, p_offspec - rec.expected_risk_reduction))
                            realized = "offspec" if random.random() < residual_risk else "clean"
                            log_feedback(
                                event_id=int(selected_event_id),
                                setpoint_variable=rec.setpoint_variable,
                                suggested_value=rec.suggested_value,
                                current_value=rec.current_value,
                                confidence=rec.confidence,
                                source_tag=rec.source_tag,
                                operator_action="accepted",
                                realized_outcome=realized
                            )
                            st.toast(f"Feedback registered! Simulated outcome: {realized}.")
                            st.rerun()
                with col_btn2:
                    if rec_id in st.session_state["rejected_recs"]:
                        st.error("Rejected ❌")
                    elif rec_id in st.session_state["accepted_recs"]:
                        st.write("")
                    else:
                        if st.button("Reject", key=f"rej_{rec_id}"):
                            st.session_state["rejected_recs"].add(rec_id)
                            # If rejected, outcome matches reality of original dataset
                            outcome = "offspec" if event_row["offspec_label"] == 1 else "clean"
                            log_feedback(
                                event_id=int(selected_event_id),
                                setpoint_variable=rec.setpoint_variable,
                                suggested_value=rec.suggested_value,
                                current_value=rec.current_value,
                                confidence=rec.confidence,
                                source_tag=rec.source_tag,
                                operator_action="rejected",
                                realized_outcome=outcome
                            )
                            st.toast("Action logged. Transition executed as planned.")
                            st.rerun()
    else:
        st.write("Operator actions console will be active at t = 300s.")

    # 4. What-If Explorer — live re-prediction as the operator drags a slider
    st.markdown("---")
    st.subheader("🎛️ What-If Explorer")

    if current_time >= 300:
        drop_cols = ["event_id", "offspec_label", "stabilization_time_seconds", "early_breach"]
        base_feat = features_df[features_df["event_id"] == selected_event_id].copy()
        feat_cols = [c for c in base_feat.columns if c not in drop_cols]

        actual_ramp = int(event_row["ramp_duration_seconds"])
        slider_min = max(60, int(actual_ramp * 0.3))
        slider_max = min(2400, int(actual_ramp * 2.5))

        st.caption(
            "Drag the ramp duration and watch the model re-predict live — the same "
            "action space the Recommendation Engine itself searches over."
        )
        whatif_ramp = st.slider(
            "Try a different ramp duration (seconds):",
            min_value=slider_min,
            max_value=slider_max,
            value=actual_ramp,
            step=5,
            key=f"whatif_ramp_{selected_event_id}",
        )

        # Baseline (actual planned ramp) prediction
        base_risk = float(clf.predict_proba(base_feat[feat_cols])[0][1])
        base_stab = float(reg.predict(base_feat[feat_cols])[0])

        # Re-predict with the hypothetical ramp duration. Only ramp_duration_seconds
        # and the ramp-RATE features that are mechanically derived from it change —
        # everything else (ambient_temp_c, valve_wear_index, etc.) is a real
        # physical condition of this transition and isn't something a slider can
        # change, so it's left untouched.
        whatif_feat = base_feat.copy()
        whatif_feat["ramp_duration_seconds"] = whatif_ramp
        for col, change_col in [
            ("planned_bw_ramp_rate", "planned_bw_change"),
            ("planned_sf_ramp_rate", "planned_sf_change"),
            ("planned_ff_ramp_rate", "planned_ff_change"),
            ("planned_sp_ramp_rate", "planned_sp_change"),
        ]:
            if col in whatif_feat.columns and change_col in whatif_feat.columns:
                whatif_feat[col] = whatif_feat[change_col] / whatif_ramp

        whatif_risk = float(clf.predict_proba(whatif_feat[feat_cols])[0][1])
        whatif_stab = float(reg.predict(whatif_feat[feat_cols])[0])

        col_wi1, col_wi2 = st.columns(2)
        with col_wi1:
            st.metric(
                "Off-Spec Risk at this ramp duration",
                f"{whatif_risk*100:.1f}%",
                delta=f"{(whatif_risk - base_risk)*100:+.1f}pp vs. actual plan",
                delta_color="inverse",
            )
        with col_wi2:
            st.metric(
                "Predicted Stabilization Time",
                f"{whatif_stab:.0f}s",
                delta=f"{whatif_stab - base_stab:+.0f}s vs. actual plan",
                delta_color="inverse",
            )

        if whatif_ramp == actual_ramp:
            st.caption("Currently showing the actual planned ramp duration for this event — try dragging the slider.")
    else:
        st.info("What-If Explorer activates at t = 300s, once live predictions are available.")


# ===========================================================================
# Tab 2: Discovered Correlations
# ===========================================================================
with tab_correlations:
    st.title("Dual-Validated Correlation Discovery Panel")
    st.subheader("Known and Discovered Physical Relationships")
    
    try:
        corr_df = _load_cached_correlations()
        
        # Display summary mapping
        st.markdown("""
        The correlation discovery pipeline cross-validates statistical timeseries correlations against 
        machine learning model importances.
        - **ML-Confirmed**: The statistical correlation is supported by model feature importances (Pass 2).
        - **Negative Control**: Process checks designed to verify no false positives are generated.
        """)
        
        # Split into two explicit sections: Off-Spec Risk Drivers vs Stabilization Delays
        offspec_drivers = corr_df[corr_df["Relationship"].str.contains("Offspec|Excursion|BW|Campaign", case=False, na=False)]
        stabilization_drivers = corr_df[~corr_df.index.isin(offspec_drivers.index)]
        
        def _status_color(val: str) -> str:
            """Green = confirmed relationship OR a negative control correctly
            finding nothing (both are good news). Red = a negative control
            unexpectedly firing (the method may be hallucinating). Neutral
            otherwise (a real relationship just wasn't found, or found but
            NOT ML-confirmed — inconclusive, not a failure of the method)."""
            val = str(val)
            if "not ML-confirmed" in val:
                return ""  # must be checked BEFORE the "ML-confirmed" substring match below
            if "ML-confirmed" in val or "expected: not found" in val:
                return "background-color: rgba(16, 185, 129, 0.15);"
            if "FAILED" in val:
                return "background-color: rgba(239, 68, 68, 0.15);"
            return ""

        st.markdown("### 🔴 Section 1: What's Driving Off-Spec Risk")
        st.caption("Cross-validated physical and operational drivers of specification breaches:")
        st.dataframe(
            offspec_drivers[["Relationship", "Method", "Coefficient", "P-Value", "Status"]].style.map(
                _status_color, subset=["Status"]
            ),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        st.markdown("### ⏱️ Section 2: What's Driving Slow Stabilization")
        st.caption("Cross-validated physical drivers of recovery time delays and negative control checks:")
        st.dataframe(
            stabilization_drivers[["Relationship", "Method", "Coefficient", "P-Value", "Status"]].style.map(
                _status_color, subset=["Status"]
            ),
            use_container_width=True,
            hide_index=True
        )
        
    except Exception as e:
        st.error(f"Error executing correlation discovery passes: {e}")


# ===========================================================================
# Tab 3: Calibration & Trust Ledger
# ===========================================================================
with tab_calibration:
    st.title("QCS Copilot Calibration & Operator Trust Ledger")
    
    stats = get_calibration_stats()
    
    if stats["total_count"] == 0:
        st.info("No operator feedback recorded yet. Go to the Operator Control Console to accept/reject recommendations.")
    else:
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Logged Actions", stats["total_count"])
        with col_s2:
            st.metric("Acceptance Rate", f"{stats['accept_rate']*100:.1f}%")
        with col_s3:
            # Compute actual accuracy of accepted advice
            db_df = stats["history_df"]
            accepted_clean = len(db_df[(db_df["operator_action"] == "accepted") & (db_df["realized_outcome"] == "clean")])
            acc_total = len(db_df[db_df["operator_action"] == "accepted"])
            acc_rate = accepted_clean / acc_total if acc_total > 0 else 0.0
            st.metric("Clean Run Rate (Accepted)", f"{acc_rate*100:.1f}%")
            
        st.subheader("Stated Confidence vs. Realized Outcome Accuracy")
        curve_data = get_calibration_curve_data()
        
        # Plot calibration curve using Plotly
        fig_cal = go.Figure()
        
        # Plot perfect diagonal reference line
        fig_cal.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="rgba(255,255,255,0.3)", dash="dash")
        ))
        
        # Plot actual
        fig_cal.add_trace(go.Scatter(
            x=curve_data["avg_confidence"] * 100,
            y=curve_data["actual_clean_rate"] * 100,
            mode="markers+lines",
            name="QCS Copilot Calibration",
            marker=dict(size=12, color="#10b981"),
            line=dict(color="#10b981", width=3)
        ))
        
        fig_cal.update_layout(
            title="Calibration Chart (Stated Confidence vs Realized Clean Outcomes)",
            xaxis_title="QCS Copilot Stated Confidence (%)",
            yaxis_title="Operator Clean Outcome Rate (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="white"),
            legend=dict(font=dict(color="white")),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 105], tickfont=dict(color="white")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 105], tickfont=dict(color="white")),
            height=360,
            margin=dict(t=40, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_cal, use_container_width=True)
        
        st.subheader("Raw Feedback Log Table")
        st.dataframe(
            db_df.sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True
        )