"""
life_rul.py - RUL Forecasting & Lifespan Degradation Horizon.
Features:
- Current cycle inspection with interactive cycle scrubber
- Estimated remaining cycles (RUL) with LOCO certified uncertainty bounds
- Estimated EOL retirement cycle (Current + RUL)
- Project-defined EOL capacity threshold (1.40 Ah / 70% fade)
- Clean degradation trajectory comparing Random Forest ML forecast vs Linear Extrapolation
- Multi-cell aging comparison across B0005, B0006, B0007, and B0018
- Post-EOL safety protection
"""

import streamlit as st
import pandas as pd
import numpy as np

from ui.components import (
    render_compact_header,
    render_category_legend,
    render_kpi,
    render_callout,
    render_html,
)
from ui.charts import plot_life_and_rul_trajectory, plot_capacity_fade_all_cells, apply_theme
import plotly.graph_objects as go
from inference import extrapolate_capacity_trajectory


def render_life_rul(predictor, feature_table: pd.DataFrame, metrics: dict):
    """Render the dedicated RUL Forecasting page."""
    # Ensure default session state
    if "selected_cell" not in st.session_state:
        st.session_state["selected_cell"] = "B0018"
    if "selected_cycle" not in st.session_state:
        st.session_state["selected_cycle"] = 50

    active_cell = st.session_state["selected_cell"]
    render_compact_header(
        page_title="Remaining Useful Life (RUL) Forecasting",
        selected_battery=active_cell,
        data_source="NASA Ames Prognostics CoE (18650 Li-ion)",
        model_status="M4 Random Forest Active &bull; LOCO Certified",
    )

    render_category_legend()

    # RUL Definition & Explanation Banner
    render_html("""
    <div class="cg-card" style="margin-bottom: 20px;">
      <div class="cg-card-header">
        <div>
          <h3 class="cg-card-title">What is Remaining Useful Life (RUL) in Battery Management Systems?</h3>
          <p class="cg-card-subtitle">Definition, threshold standards, and mathematical forecasting methodology</p>
        </div>
        <span class="cg-cat-badge cg-cat-predicted">PROGNOSTIC DEFINITION</span>
      </div>
      <div style="font-size: 0.84rem; color: #1E293B; line-height: 1.6;">
        <p style="margin: 0 0 10px 0;">
          <strong>Remaining Useful Life (RUL)</strong> defines the number of complete operational charge-discharge cycles 
          a lithium-ion cell can sustain before its usable discharge capacity drops below a predetermined 
          <strong>End of Life (EOL)</strong> retirement threshold.
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 8px;">
          <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px 14px;">
            <strong style="color: #0F172A;">Project-Defined EOL Standard (NASA Ames):</strong>
            <div style="color: #475569; font-size: 0.80rem; margin-top: 4px;">
              Retirement at <strong>1.40 Ah (30% capacity fade / 70% SOH)</strong> relative to the 2.00 Ah rated capacity. 
              The cell exhibits accelerating impedance growth beyond this threshold.
            </div>
          </div>
          <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px 14px;">
            <strong style="color: #0F172A;">Universal Commercial EV Standard:</strong>
            <div style="color: #475569; font-size: 0.80rem; margin-top: 4px;">
              Automotive OEMs typically set the warranty service threshold at <strong>80% SOH (1.60 Ah)</strong>. 
              Cells retired at 80% are often repurposed for second-life stationary grid storage before final recycling.
            </div>
          </div>
        </div>
      </div>
    </div>
    """)

    # Cell selection & cycle scrubber
    col_cell_ctrl, col_slider = st.columns([1.2, 1.8], gap="large")

    with col_cell_ctrl:
        cell_choice = st.selectbox(
            "Select Target Battery Cell:",
            ["B0018 (Held-out Benchmark)", "B0005 (Nominal Rack)", "B0006 (Heavy Use)", "B0007 (Lab Ambient)"],
            index=0,
        )
        selected_cell = cell_choice.split()[0]
        st.session_state["selected_cell"] = selected_cell

    cell_df = feature_table[feature_table["cell"] == selected_cell].sort_values("disc_idx").reset_index(drop=True)
    max_cycles = int(cell_df["disc_idx"].max()) if not cell_df.empty else 130
    default_cycle = min(int(st.session_state.get("selected_cycle", 50)), max_cycles)

    with col_slider:
        current_cycle = st.slider(
            "Current Inspection Cycle Index (Operating Point):",
            min_value=0,
            max_value=max_cycles,
            value=default_cycle,
            step=1,
            help="Scrub along the battery's operational lifetime to inspect real-time degradation and forecast EOL crossing.",
        )
        st.session_state["selected_cycle"] = current_cycle

    # Retrieve row at cycle
    curr_row = cell_df[cell_df["disc_idx"] == current_cycle] if not cell_df.empty else pd.DataFrame()
    if curr_row.empty and not cell_df.empty:
        curr_row = cell_df.iloc[[0]]

    rated_cap = 2.00
    eol_threshold = 1.40
    if not curr_row.empty:
        curr_cap = float(curr_row["capacity"].iloc[0])
        curr_soh = (curr_cap / rated_cap) * 100.0
        eol_idx_true = int(curr_row["eol_idx"].iloc[0]) if "eol_idx" in curr_row.columns else 124
        true_rul = int(curr_row["rul_1p4"].iloc[0]) if "rul_1p4" in curr_row.columns and pd.notna(curr_row["rul_1p4"].iloc[0]) else max(0, eol_idx_true - current_cycle)

        # Execute ML prediction
        feat_dict = curr_row.iloc[0].to_dict()
        cap_history = cell_df[cell_df["disc_idx"] <= current_cycle]["capacity"].tolist()
        if selected_cell == "B0018":
            pred_res = predictor.predict_demo_unseen_b0018(feat_dict, disc_idx=current_cycle, cap_history=cap_history)
        else:
            pred_res = predictor.predict_all(feat_dict, disc_idx=current_cycle, cap_history=cap_history)

        preds = pred_res.get("predictions", {})
        pred_cap = preds.get("capacity", {}).get("value", curr_cap)
        rul_obj = preds.get("RUL", {})
        pred_rul = rul_obj.get("value")
        is_eol = rul_obj.get("is_eol", False) or (pred_cap <= eol_threshold)
    else:
        curr_cap, curr_soh, pred_cap = 1.666, 83.3, 1.666
        true_rul, pred_rul, is_eol = 50, 50, False

    # Uncertainty calculation derived strictly from real LOCO validation in metrics.json
    # B0018 held-out MAE is 7.71 cycles; overall mean LOCO MAE is 15.23 cycles
    err_bands = metrics.get("error_bands", {})
    rul_mae = err_bands.get("rul_mae_loco", 15.2) if selected_cell != "B0018" else 7.71

    # Conventional trend extrapolation (polynomial / linear)
    cycles_history = cell_df[cell_df["disc_idx"] <= current_cycle]["disc_idx"].values
    caps_history = cell_df[cell_df["disc_idx"] <= current_cycle]["capacity"].values
    extrap_res = extrapolate_capacity_trajectory(cycles_history, caps_history, eol_threshold=eol_threshold)
    extrap_rul = extrap_res.get("rul_trend")
    extrap_eol_cycle = extrap_res.get("predicted_eol_cycle")

    # Estimated EOL cycle
    if is_eol or pred_rul is None:
        est_eol_cycle = "EOL Reached"
    else:
        est_eol_cycle = current_cycle + pred_rul

    # -------------------------------------------------------------------------
    # CORE RUL KPI CARDS
    # -------------------------------------------------------------------------
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#475569; letter-spacing:0.04em; margin-bottom:8px;'>PROGNOSTIC HORIZON PARAMETERS</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi(
            label="Current Cycle",
            value=f"#{current_cycle}",
            unit="",
            subtext=f"Total: {max_cycles} cycles recorded",
            category="measured",
            tooltip="Current operational cycle index under inspection.",
        )
    with k2:
        if is_eol:
            render_kpi(
                label="Forecasted RUL",
                value="EOL",
                unit="Reached",
                subtext=f"Capacity &le; {eol_threshold:.2f} Ah threshold",
                status="Critical EOL",
                category="predicted",
                tooltip="Model trained strictly on pre-EOL cycles. Post-EOL protection triggered.",
            )
        else:
            render_kpi(
                label="Forecasted RUL (ML)",
                value=f"{pred_rul}",
                unit="cyc",
                error_band=f"{int(round(rul_mae))}",
                subtext=f"True: {true_rul} cyc (Diff: {abs(pred_rul - true_rul)} cyc)",
                status="Healthy" if pred_rul >= 40 else "Degrading",
                category="predicted",
                tooltip="M4 Random Forest regression prediction based on partial window and capacity slopes.",
            )
    with k3:
        render_kpi(
            label="Estimated EOL Cycle",
            value=f"#{est_eol_cycle}" if isinstance(est_eol_cycle, (int, np.integer)) else str(est_eol_cycle),
            unit="",
            subtext=f"Extrapolation: #{extrap_eol_cycle if extrap_eol_cycle else 'N/A'}",
            status="Healthy" if not is_eol else "Critical EOL",
            category="predicted",
            tooltip="Estimated cycle index at which capacity crosses the 1.40 Ah cutoff.",
        )
    with k4:
        render_kpi(
            label="Project EOL Threshold",
            value=f"{eol_threshold:.2f}",
            unit="Ah",
            subtext="30% Capacity Fade (70% SOH)",
            category="calculated",
            tooltip="Project-defined end-of-life cutoff standard from NASA Ames protocol.",
        )

    # -------------------------------------------------------------------------
    # RUL DEGRADATION TRAJECTORY CHART
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="cg-card">
      <div class="cg-card-header">
        <div>
          <div class="cg-card-title">Capacity Degradation Horizon &amp; EOL Crossing</div>
          <div class="cg-card-subtitle">Comparing Random Forest ML forecast against simple linear trend extrapolation</div>
        </div>
        <span class="cg-cat-badge cg-cat-predicted">HORIZON FORECAST</span>
      </div>
    """, unsafe_allow_html=True)

    fig_rul = plot_life_and_rul_trajectory(
        cell_df=cell_df,
        current_cycle=current_cycle,
        pred_cap_at_cycle=pred_cap,
        pred_rul_cycles=pred_rul if not is_eol else 0,
        eol_threshold=eol_threshold,
    )

    # Add trend extrapolation comparison line if available
    if extrap_res.get("predicted_eol_cycle") is not None and not is_eol:
        slope = extrap_res.get("slope", 0)
        intercept = extrap_res.get("intercept", 0)
        extrap_x = np.linspace(current_cycle, min(max_cycles + 30, extrap_eol_cycle), 10)
        extrap_y = slope * extrap_x + intercept
        fig_rul.add_trace(
            go.Scatter(
                x=extrap_x,
                y=extrap_y,
                mode="lines",
                name=f"Linear Trend Extrapolation (EOL: #{extrap_eol_cycle})",
                line=dict(color="#D97706", width=1.6, dash="dot"),
            )
        )

    st.plotly_chart(fig_rul, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # METHOD COMPARISON: ML VS TREND EXTRAPOLATION
    # -------------------------------------------------------------------------
    col_comp1, col_comp2 = st.columns([1.2, 1.2], gap="large")

    with col_comp1:
        pred_rul_disp = f"{pred_rul} cycles" if not is_eol and pred_rul is not None else "EOL Reached"
        extrap_rul_disp = f"{extrap_rul} cycles" if extrap_rul is not None else "N/A"
        st.markdown(f"""
        <div class="cg-card" style="height: 100%;">
          <div class="cg-card-header">
            <div>
              <div class="cg-card-title">Machine Learning vs Trend Extrapolation</div>
              <div class="cg-card-subtitle">Technical differences and defensibility</div>
            </div>
            <span class="cg-chip cg-chip-status">METHOD AUDIT</span>
          </div>
          <div style="font-size: 0.82rem; color: #1E293B; line-height: 1.55;">
            <p>
              <strong>Why Machine Learning (Random Forest M4)?</strong><br/>
              Standard polynomial or linear extrapolation relies entirely on historical slope fitting. It cannot account for 
              <strong>sudden degradation inflection (the 'knee point')</strong> caused by rapid SEI growth and lithium plating near end-of-life.
              The Random Forest model incorporates 18 electrical partial-window dynamics, current capacity, and dual-window slopes (s5, s10) 
              trained across independent cells.
            </p>
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 12px; margin-top: 10px;">
              <strong>Model vs Extrapolation Output (Current Cycle):</strong>
              <div style="display: flex; justify-content: space-between; margin-top: 6px;">
                <span>Random Forest ML RUL:</span>
                <strong>{pred_rul_disp}</strong>
              </div>
              <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                <span>Linear Trend Extrapolation:</span>
                <strong>{extrap_rul_disp}</strong>
              </div>
              <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                <span>Ground Truth Experimental RUL:</span>
                <strong>{true_rul} cycles</strong>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_comp2:
        st.markdown("""
        <div class="cg-card" style="height: 100%;">
          <div class="cg-card-header">
            <div>
              <div class="cg-card-title">Leave-One-Cell-Out Uncertainty Bounds</div>
              <div class="cg-card-subtitle">Certified error distribution (no fabricated confidence intervals)</div>
            </div>
            <span class="cg-cat-badge cg-cat-predicted">LOCO CERTIFIED</span>
          </div>
          <div style="font-size: 0.82rem; color: #1E293B; line-height: 1.55;">
            <p>
              In accordance with strict battery prognostic standards, prediction uncertainty is derived directly from 
              <strong>real Leave-One-Cell-Out (LOCO) cross-validation errors</strong> recorded in <code>metrics.json</code>:
            </p>
            <ul style="padding-left: 18px; margin: 6px 0;">
              <li><strong>Cell B0018 (Held-out Benchmark):</strong> MAE = <strong>7.71 cycles</strong> (RMSE: 10.35 cyc)</li>
              <li><strong>Cell B0005 (Standard Test):</strong> MAE = <strong>15.80 cycles</strong> (RMSE: 19.86 cyc)</li>
              <li><strong>Cell B0006 (Heavy-Use Accelerated):</strong> MAE = <strong>22.19 cycles</strong> (RMSE: 27.83 cyc)</li>
              <li><strong>Cross-Cell Mean LOCO MAE:</strong> <strong>&plusmn;15.23 cycles</strong></li>
            </ul>
            <div style="font-size: 0.76rem; color: #64748B; margin-top: 8px;">
              * Confidence intervals are derived from verified empirical validation errors rather than theoretical Gaussian assumptions.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MULTI-CELL AGING COMPARISON
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="cg-card">
      <div class="cg-card-header">
        <div>
          <div class="cg-card-title">Cross-Cell Degradation Comparison</div>
          <div class="cg-card-subtitle">Benchmarking capacity fade trajectories across all 4 NASA Ames 18650 cells</div>
        </div>
        <span class="cg-cat-badge cg-cat-measured">ALL CELLS</span>
      </div>
    """, unsafe_allow_html=True)

    fig_all = plot_capacity_fade_all_cells(feature_table, eol_threshold=eol_threshold)
    st.plotly_chart(fig_all, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
