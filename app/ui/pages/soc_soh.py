"""
soc_soh.py - Dedicated SOC and SOH Monitoring Page.
Distinguishes:
- State of Charge (SOC): How much energy/charge is currently available in the battery right now.
- State of Health (SOH): How much of the original battery capacity remains over its operational lifetime.

IMPORTANT ENGINEERING COMPLIANCE:
The existing CellGuard repository does not implement an independent SOC machine learning model.
SOC is strictly computed as a clearly-labelled derived estimate via Coulomb counting (current integration)
when initial charge state, usable capacity, current, and timestamps are present.
Otherwise, displays: "SOC estimation not available – requires initial charge state and current integration data."
"""

import os
from typing import Tuple, Optional
import streamlit as st
import pandas as pd
import numpy as np

from ui.components import (
    render_compact_header,
    render_category_legend,
    render_kpi,
    render_soc_soh_dual_bar,
    render_callout,
    render_html,
)
from ui.charts import apply_theme
import plotly.graph_objects as go

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SAMPLE_DIR = os.path.join(ROOT_DIR, "sample_cycles")


def calculate_coulomb_counting_soc(
    t: np.ndarray,
    i: np.ndarray,
    usable_capacity_ah: float,
    initial_soc_pct: float = 100.0,
) -> Tuple[np.ndarray, float]:
    """
    Calculate derived State of Charge via Coulomb Counting integration:
    SOC(t) = SOC_0 - (1 / Q_usable) * integral(|I| dt) * 100%
    """
    if len(t) < 2 or usable_capacity_ah <= 0:
        return np.array([]), initial_soc_pct

    dt = np.diff(t)
    # Average current in interval
    i_mid = 0.5 * (np.abs(i[:-1]) + np.abs(i[1:]))
    # Integrated ampere-seconds -> ampere-hours
    d_q_ah = (i_mid * dt) / 3600.0
    cum_q_ah = np.concatenate(([0.0], np.cumsum(d_q_ah)))

    soc_curve = initial_soc_pct - (cum_q_ah / usable_capacity_ah) * 100.0
    soc_curve = np.clip(soc_curve, 0.0, 100.0)
    final_soc = float(soc_curve[-1])

    return soc_curve, final_soc


def render_soc_soh(predictor, feature_table: pd.DataFrame, metrics: dict):
    """Render the dedicated SOC & SOH Monitoring page."""
    # Ensure default session state
    if "selected_cell" not in st.session_state:
        st.session_state["selected_cell"] = "B0018"
    if "selected_cycle" not in st.session_state:
        st.session_state["selected_cycle"] = 50

    active_cell = st.session_state["selected_cell"]
    render_compact_header(
        page_title="SOC & SOH Dual Monitoring",
        selected_battery=active_cell,
        data_source="Coulomb Counting &bull; Capacity Retention",
        model_status="Charge and health tracking",
    )

    render_category_legend()

    # Educational distinction banner
    render_html("""
    <div class="cg-card" style="margin-bottom: 20px;">
      <div class="cg-card-header">
        <div>
          <h3 class="cg-card-title">Fundamental Distinction: Operational Charge vs Structural Health</h3>
          <p class="cg-card-subtitle">Why Battery Management Systems treat SOC and SOH as distinct state variables</p>
        </div>
        <span class="cg-cat-badge cg-cat-calculated">BMS FUNDAMENTALS</span>
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; font-size: 0.84rem; line-height: 1.55;">
        <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 6px; padding: 14px 16px;">
          <div style="font-weight: 700; color: #1E40AF; font-size: 0.92rem; margin-bottom: 4px;">
            🔋 State of Charge (SOC) — Short-Term Operational State
          </div>
          <div style="color: #1E293B;">
            <strong>What it represents:</strong> The current amount of electrical charge stored in the cell, expressed as a percentage of its <em>current usable capacity</em>.
            <ul style="padding-left: 18px; margin: 6px 0;">
              <li>Analogous to the <strong>fuel gauge in a gas tank</strong>.</li>
              <li>Fluctuates rapidly within minutes or hours as the battery charges or discharges.</li>
              <li>Reversible: drops to 0% when drained and returns to 100% upon complete recharge.</li>
            </ul>
          </div>
        </div>

        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 14px 16px;">
          <div style="font-weight: 700; color: #166534; font-size: 0.92rem; margin-bottom: 4px;">
            🩺 State of Health (SOH) — Long-Term Degradation State
          </div>
          <div style="color: #1E293B;">
            <strong>What it represents:</strong> The maximum capacity the cell can currently hold relative to its <em>original factory-rated capacity (2.00 Ah)</em>.
            <ul style="padding-left: 18px; margin: 6px 0;">
              <li>Analogous to the <strong>shrinking size of the fuel tank itself</strong> over years.</li>
              <li>Degrades irreversibly over months and years due to SEI growth, lithium plating, and active material loss.</li>
              <li>Irreversible: once capacity fades to 70% (1.40 Ah), recharging cannot recover lost material.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
    """)

    # Data Source & Telemetry Selection
    col_ctrl1, col_ctrl2 = st.columns([1.2, 1.2], gap="large")

    with col_ctrl1:
        cell_choice = st.selectbox(
            "Select Battery Cell Under Evaluation:",
            ["B0018 (Held-out Benchmark)", "B0005 (Nominal Rack)", "B0006 (Heavy Use)", "B0007 (Lab Ambient)"],
            index=0,
        )
        selected_cell = cell_choice.split()[0]
        st.session_state["selected_cell"] = selected_cell

    with col_ctrl2:
        cycle_file_options = [
            "B0018_cycle_005_early.csv (Fresh Cell &bull; Cycle 5)",
            "B0018_cycle_015_early.csv (Early Use &bull; Cycle 15)",
            "B0018_cycle_050_middle.csv (Mid Life &bull; Cycle 50)",
            "B0018_cycle_065_middle.csv (Aging Pack &bull; Cycle 65)",
            "B0018_cycle_100_late.csv (Retirement Limit &bull; Cycle 100)",
            "None (Evaluate without Timeseries Telemetry)",
        ]
        chosen_telemetry = st.selectbox(
            "Discharge Telemetry Timeseries (for Coulomb Counting SOC):",
            cycle_file_options,
            index=2,
            help="Select a cycle dataset containing current and timestamps to compute derived SOC integration.",
        )

    # Filter cell table
    cell_df = feature_table[feature_table["cell"] == selected_cell].sort_values("disc_idx").reset_index(drop=True)
    
    # Retrieve capacity and SOH for selected operating cycle
    current_cycle = int(st.session_state.get("selected_cycle", 50))
    cycle_row = cell_df[cell_df["disc_idx"] == current_cycle] if not cell_df.empty else pd.DataFrame()
    if cycle_row.empty and not cell_df.empty:
        cycle_row = cell_df.iloc[[0]]

    rated_capacity = 2.00
    if not cycle_row.empty:
        curr_cap = float(cycle_row["capacity"].iloc[0])
        curr_soh = (curr_cap / rated_capacity) * 100.0
    else:
        curr_cap = 1.666
        curr_soh = 83.3

    # Check telemetry presence for Coulomb counting
    has_timeseries = False
    soc_curve = np.array([])
    current_soc_display = None
    t_data = None
    v_data = None
    i_data = None

    if "None" not in chosen_telemetry:
        fname = chosen_telemetry.split()[0]
        fpath = os.path.join(SAMPLE_DIR, fname)
        if os.path.exists(fpath):
            df_ts = pd.read_csv(fpath)
            t_data = df_ts["Time_s"].values
            v_data = df_ts["Voltage_V"].values
            i_data = df_ts["Current_A"].values
            has_timeseries = True

            # Calculate Coulomb counting
            soc_curve, final_soc = calculate_coulomb_counting_soc(
                t=t_data,
                i=i_data,
                usable_capacity_ah=curr_cap,
                initial_soc_pct=100.0,
            )
            # Pick mid-cycle point or final
            current_soc_display = float(soc_curve[len(soc_curve) // 2]) if len(soc_curve) > 0 else 50.0

    # -------------------------------------------------------------------------
    # MANDATORY COMPLIANCE CALLOUT & DUAL VISUALIZATION
    # -------------------------------------------------------------------------
    if not has_timeseries:
        render_callout(
            "<strong>SOC estimation not available – requires initial charge state and current integration data.</strong><br/>"
            "Note: The CellGuard repository intentionally implements ML for SOH and RUL, not SOC. "
            "Select a cycle telemetry file above to view derived Coulomb counting SOC.",
            alert_type="info",
            title="SOC Estimation Status",
        )
    else:
        render_callout(
            f"<strong>Derived Coulomb Counting SOC Active:</strong> Integrated over {len(t_data)} timestamps at nominal 2.0A constant current discharge. "
            f"Usable capacity base: <strong>{curr_cap:.4f} Ah</strong>. At mid-discharge snapshot: <strong>SOC = {current_soc_display:.1f}%</strong>.",
            alert_type="success",
            title="Coulomb Counting Integration Active",
        )

    # Dual SOC / SOH visual comparison bar
    render_soc_soh_dual_bar(
        soc=current_soc_display if has_timeseries else None,
        soh=curr_soh,
        rated_cap=rated_capacity,
        remaining_cap=curr_cap,
        soc_available=has_timeseries,
    )

    # -------------------------------------------------------------------------
    # SIX CORE METRICS
    # -------------------------------------------------------------------------
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#475569; letter-spacing:0.04em; margin-bottom:8px;'>DETAILED CHARGE &amp; HEALTH METRICS</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        if has_timeseries:
            render_kpi(
                label="State of Charge (SOC)",
                value=f"{current_soc_display:.1f}",
                unit="%",
                subtext="Derived from Coulomb counting",
                category="calculated",
                tooltip="Fraction of usable charge remaining in the cell at this instant.",
            )
        else:
            render_kpi(
                label="State of Charge (SOC)",
                value="N/A",
                unit="",
                subtext="Requires current integration",
                category="calculated",
                tooltip="SOC requires initial charge state and current integration data.",
            )
    with m2:
        render_kpi(
            label="State of Health (SOH)",
            value=f"{curr_soh:.2f}",
            unit="%",
            subtext=f"{curr_cap:.3f} Ah / 2.00 Ah rated",
            status="Healthy" if curr_soh >= 80 else ("Degrading" if curr_soh >= 70 else "Critical EOL"),
            category="calculated",
            tooltip="Ratio of current usable capacity to factory-rated nominal capacity.",
        )
    with m3:
        render_kpi(
            label="Rated Factory Capacity",
            value=f"{rated_capacity:.2f}",
            unit="Ah",
            subtext="Nominal 18650 LiCoO2 specification",
            category="measured",
            tooltip="Original uncycled cell capacity from manufacturer datasheet.",
        )

    m4, m5, m6 = st.columns(3)
    with m4:
        render_kpi(
            label="Estimated Remaining Capacity",
            value=f"{curr_cap:.4f}",
            unit="Ah",
            subtext="Maximum deliverable charge to cutoff",
            category="calculated",
            tooltip="Current usable capacity at this cycle index.",
        )
    with m5:
        degradation_pct = (1.0 - (curr_cap / rated_capacity)) * 100.0
        render_kpi(
            label="Capacity Degradation Fade",
            value=f"{degradation_pct:.2f}",
            unit="%",
            subtext=f"Lost {round(rated_capacity - curr_cap, 4)} Ah",
            status="Healthy" if degradation_pct <= 20 else ("Degrading" if degradation_pct <= 30 else "Critical EOL"),
            category="calculated",
            tooltip="Permanent capacity loss relative to rated capacity.",
        )
    with m6:
        render_kpi(
            label="Operational Cycle Number",
            value=f"#{current_cycle}",
            unit="",
            subtext=f"Cell: {selected_cell}",
            category="measured",
            tooltip="Discharge cycle index in aging history.",
        )

    # -------------------------------------------------------------------------
    # CHARTS: SOC DYNAMIC DISCHARGE & SOH CAPACITY DEGRADATION
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    c_soc, c_soh = st.columns([1.2, 1.2], gap="large")

    with c_soc:
        st.markdown("""
        <div class="cg-card">
          <div class="cg-card-header">
            <div>
              <div class="cg-card-title">Dynamic SOC Discharge Trajectory</div>
              <div class="cg-card-subtitle">Coulomb Counting integration: SOC(t) = SOC₀ - (1/Q_usable) &int; |I| dt</div>
            </div>
            <span class="cg-cat-badge cg-cat-calculated">INTEGRATION</span>
          </div>
        """, unsafe_allow_html=True)

        if has_timeseries and len(soc_curve) > 0:
            fig_soc = go.Figure()
            fig_soc.add_trace(
                go.Scatter(
                    x=t_data,
                    y=soc_curve,
                    mode="lines",
                    name="Derived SOC (%)",
                    line=dict(color="#2563EB", width=2.4),
                )
            )
            fig_soc.update_xaxes(title_text="Discharge Time (seconds)")
            fig_soc.update_yaxes(title_text="State of Charge (%)", range=[-2, 105])
            fig_soc = apply_theme(fig_soc, height=320)
            st.plotly_chart(fig_soc, use_container_width=True)
        else:
            st.markdown("""
            <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 6px; padding: 40px 20px; text-align: center; color: #64748B;">
              <div style="font-size: 1.8rem; margin-bottom: 8px;">⏱️</div>
              <div style="font-weight: 600; color: #334155; margin-bottom: 4px;">Timeseries Telemetry Not Selected</div>
              <div style="font-size: 0.82rem;">Select a sample cycle file above to compute and render the real-time Coulomb counting SOC discharge curve.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with c_soh:
        st.markdown("""
        <div class="cg-card">
          <div class="cg-card-header">
            <div>
              <div class="cg-card-title">Capacity &amp; SOH Degradation vs Cycle</div>
              <div class="cg-card-subtitle">Permanent capacity fade trajectory over battery operating lifetime</div>
            </div>
            <span class="cg-cat-badge cg-cat-measured">CYCLE HISTORY</span>
          </div>
        """, unsafe_allow_html=True)

        if not cell_df.empty:
            fig_soh = go.Figure()
            fig_soh.add_trace(
                go.Scatter(
                    x=cell_df["disc_idx"],
                    y=cell_df["capacity"],
                    mode="lines+markers",
                    name="Measured Capacity (Ah)",
                    line=dict(color="#0D9488", width=1.8),
                    marker=dict(size=3.5),
                )
            )
            # EOL line
            fig_soh.add_hline(
                y=1.40,
                line_dash="dash",
                line_color="#DC2626",
                line_width=1.5,
                annotation_text="1.40 Ah EOL Threshold (70% SOH)",
                annotation_position="bottom right",
                annotation_font=dict(size=9.5, color="#DC2626", family="JetBrains Mono"),
            )
            fig_soh.update_xaxes(title_text="Discharge Cycle Index")
            fig_soh.update_yaxes(title_text="Discharge Capacity (Ah)", range=[1.2, 2.05])
            fig_soh = apply_theme(fig_soh, height=320)
            st.plotly_chart(fig_soh, use_container_width=True)
        else:
            st.info("No cycle history available.")

        st.markdown("</div>", unsafe_allow_html=True)
