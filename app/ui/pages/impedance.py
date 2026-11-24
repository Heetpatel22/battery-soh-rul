"""
impedance.py - Resistance Degradation Trends & Physical Impedance Analysis.
Analyzes Ohmic resistance (Re) and Charge-transfer resistance (Rct) evolution
across aging cycles and multiple battery cells.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.components import (
    render_compact_header,
    render_category_legend,
    render_kpi,
    render_html,
)
from ui.charts import (
    plot_re_rct_dual_trend,
    plot_multi_cell_impedance,
    apply_theme,
)


def render_impedance(predictor, feature_table: pd.DataFrame, metrics: dict):
    # Retrieve active session cell
    active_cell = st.session_state.get("selected_cell", "B0005")

    render_compact_header(
        page_title="Internal Resistance Degradation Trends (Re & Rct)",
        selected_battery=active_cell,
        data_source="Electrochemical Impedance Spectroscopy (EIS) Frequency Sweeps",
        model_status="Actual Laboratory Ground Truth & ML Trajectories",
    )

    render_category_legend()

    if feature_table.empty:
        st.error("Feature table is unavailable. Check models/feature_table.csv.")
        return

    # Cell & Cycle Controls
    c_sel, cyc_sel = st.columns([1, 2])
    with c_sel:
        cells = [c for c in ["B0005", "B0006", "B0007", "B0018"] if c in feature_table["cell"].unique()]
        if not cells:
            cells = sorted(feature_table["cell"].dropna().unique().tolist())
        selected_cell = st.selectbox(
            "Target Cell",
            cells,
            index=cells.index(active_cell) if active_cell in cells else 0,
            key="imp_trend_cell_select",
        )
        st.session_state["selected_cell"] = selected_cell

    cell_data = feature_table[feature_table["cell"] == selected_cell].sort_values("disc_idx").copy()
    valid_imp = cell_data.dropna(subset=["Re", "Rct"]).sort_values("disc_idx")
    max_c = int(cell_data["disc_idx"].max()) + 1 if not cell_data.empty else 168

    with cyc_sel:
        selected_cycle = st.slider(
            "Inspection Cycle",
            1,
            max_c,
            min(st.session_state.get("selected_cycle", 40), max_c),
            key="imp_trend_cycle_slider",
        )
        st.session_state["selected_cycle"] = selected_cycle

    # Selected cycle data
    cycle_idx = selected_cycle - 1
    closest = (cell_data["disc_idx"] - cycle_idx).abs().argsort()[:1]
    row = cell_data.iloc[closest].iloc[0]

    re_val = float(row.get("Re", np.nan))
    rct_val = float(row.get("Rct", np.nan))
    rint_val = (re_val + rct_val) if (np.isfinite(re_val) and np.isfinite(rct_val)) else np.nan

    # Fresh cycle baseline (first available)
    if not valid_imp.empty:
        fresh_row = valid_imp.iloc[0]
        re_fresh = float(fresh_row["Re"])
        rct_fresh = float(fresh_row["Rct"])
        rint_fresh = re_fresh + rct_fresh
        re_mult = (re_val / re_fresh) if np.isfinite(re_val) else 1.0
        rct_mult = (rct_val / rct_fresh) if np.isfinite(rct_val) else 1.0
        rint_mult = (rint_val / rint_fresh) if np.isfinite(rint_val) else 1.0
    else:
        re_mult = rct_mult = rint_mult = 1.0

    # KPI METRICS ROW
    st.markdown('<div class="cg-section-label">IMPEDANCE STATUS AT INSPECTION CYCLE</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        re_str = f"{re_val * 1000.0:.2f}" if np.isfinite(re_val) else "—"
        render_kpi(
            label="Ohmic Re (Bulk)",
            value=re_str,
            unit="mΩ",
            subtext=f"Growth: {re_mult:.2f}× fresh baseline",
            status="Healthy" if (np.isfinite(re_val) and re_val < 0.070) else "Elevated",
            category="measured" if np.isfinite(re_val) else "predicted",
            tooltip="Bulk resistance of electrolyte, separator, active material bulk, and foil collectors.",
        )

    with k2:
        rct_str = f"{rct_val * 1000.0:.2f}" if np.isfinite(rct_val) else "—"
        render_kpi(
            label="Charge-Transfer Rct",
            value=rct_str,
            unit="mΩ",
            subtext=f"Growth: {rct_mult:.2f}× fresh baseline",
            status="Normal" if (np.isfinite(rct_val) and rct_val < 0.090) else "Degraded",
            category="measured" if np.isfinite(rct_val) else "predicted",
            tooltip="Activation polarization resistance for lithium ion transfer across electrode-electrolyte interface.",
        )

    with k3:
        rint_str = f"{rint_val * 1000.0:.2f}" if np.isfinite(rint_val) else "—"
        render_kpi(
            label="Total R_int = Re + Rct",
            value=rint_str,
            unit="mΩ",
            subtext=f"Growth: {rint_mult:.2f}× fresh baseline",
            status="Healthy" if (np.isfinite(rint_val) and rint_val < 0.160) else "Severe Degradation",
            category="calculated",
            tooltip="Overall equivalent internal series resistance governing terminal voltage drop.",
        )

    with k4:
        # Ratio Rct / Re
        ratio_str = f"{rct_val / re_val:.2f}" if (np.isfinite(re_val) and np.isfinite(rct_val) and re_val > 0) else "—"
        render_kpi(
            label="Polarization Ratio (Rct/Re)",
            value=ratio_str,
            unit="ratio",
            subtext="Interfacial kinetics vs bulk",
            status="Interfacial Dominant" if (np.isfinite(rct_val / re_val) and rct_val / re_val > 1.0) else "Ohmic Dominant",
            category="calculated",
            tooltip="Ratio of charge-transfer to ohmic resistance; increases as SEI layer thickens.",
        )

    # MAIN CHARTS: RESISTANCE TRAJECTORY & MULTI-CELL BENCHMARK
    st.markdown('<div class="cg-section-label">RESISTANCE TRAJECTORY & MULTI-CELL EVOLUTION</div>', unsafe_allow_html=True)

    hdr1, hdr2 = st.columns(2, gap="medium", vertical_alignment="bottom")
    with hdr1:
        st.markdown(f"##### Cell {selected_cell} Component Evolution (Re, Rct, Total R_int)")
    with hdr2:
        param_choice = st.radio(
            "Multi-Cell Target Parameter",
            ["Re (Ohmic)", "Rct (Charge-Transfer)"],
            horizontal=True,
            key="mc_target_radio",
        )
        target_param = "Rct" if "Rct" in param_choice else "Re"

    ch1, ch2 = st.columns(2, gap="medium")

    with ch1:
        fig_dual = plot_re_rct_dual_trend(
            cell_df=feature_table,
            cell_id=selected_cell,
            current_cycle=selected_cycle,
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    with ch2:
        fig_mc = plot_multi_cell_impedance(feature_table, target=target_param)
        st.plotly_chart(fig_mc, use_container_width=True)

    # TECHNICAL INSIGHT PANEL
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:18px; margin-top:14px; box-shadow:0 1px 3px rgba(15,23,42,0.04);">
      <h4 style="font-size:0.95rem; font-weight:800; color:#0F172A; text-transform:uppercase; letter-spacing:0.04em; margin-top:0;">
        Why Charge-Transfer Resistance (Rct) Grows Faster Than Ohmic Resistance (Re)
      </h4>
      <div style="font-size:0.82rem; color:#334155; line-height:1.7;">
        <strong>1. Solid Electrolyte Interphase (SEI) Layer Thickening:</strong> During each charge-discharge cycle, parasitic electrochemical side reactions occur between the intercalated lithium and the organic liquid electrolyte solvents (EC/DMC). This continuously precipitates a resistive, passivating layer (primarily <em>Li₂CO₃</em>, <em>LiF</em>, and lithium alkyl carbonates) on the anode surface. This film directly increases <strong>Rct</strong> by creating a tortuous barrier to lithium-ion desolvation and charge transfer.
        <br/><br/>
        <strong>2. Ohmic Resistance (Re) Stability:</strong> Because the bulk electrolyte volume and metallic copper/aluminum current collectors remain relatively stable during normal cycling, <strong>Re</strong> increases at a much slower rate (typically 20% to 40% growth over cell life), whereas <strong>Rct</strong> routinely grows by <strong>100% to 250%</strong>.
        <br/><br/>
        <strong>3. Voltage Drop and Thermal Loss Implications:</strong> Under an operating discharge load <em>I</em>:
        <ul>
          <li>Instantaneous terminal voltage drop: <code>ΔV_ohmic = I × Re</code></li>
          <li>Transient polarization voltage drop: <code>ΔV_polarization = I × Rct × (1 - e^(-t / τ))</code></li>
          <li>Continuous internal heat generation rate: <code>P_loss = I² × (Re + Rct)</code> [Watts]</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)
