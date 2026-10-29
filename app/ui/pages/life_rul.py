"""
life_rul.py - Battery Life and Remaining Useful Life (RUL) page.
Provides cell lifecycle scrubbing, capacity degradation tracking, RUL forecasting,
and conventional trend extrapolation comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np

from ui.components import (
    render_kpi,
    status_chip_html,
    alert_box,
    render_html,
)
from ui.charts import plot_life_and_rul_trajectory
from inference import extrapolate_capacity_trajectory


def render_life_rul(predictor, feature_table: pd.DataFrame, metrics: dict):
    """Render the Battery Life and RUL page."""
    st.markdown("### Battery Degradation Tracking & RUL Forecasting")
    st.markdown(
        "<p style='color: var(--text-secondary); margin-bottom: 20px; font-size: 0.92rem;'>"
        "Observe capacity fade across cycles, monitor approaching End of Life (1.40 Ah threshold), and forecast retirement horizons."
        "</p>",
        unsafe_allow_html=True,
    )

    tab_cell_demo, tab_trend_extrap = st.tabs(["Cell Lifecycle Explorer", "Upload Capacity History (Trend Extrapolation)"])

    with tab_cell_demo:
        col_ctrl, col_stats = st.columns([1.5, 1], gap="large")

        with col_ctrl:
            col_cell, col_mode = st.columns([1, 1.2])
            with col_cell:
                cell_choice = st.selectbox(
                    "Select Benchmark Cell:",
                    ["B0018 (Held-Out Benchmark)", "B0005 (In Training)", "B0006 (In Training)", "B0007 (In Training)"],
                    index=0,
                )
                cell_name = cell_choice.split()[0]

            with col_mode:
                if cell_name == "B0018":
                    st.caption("Benchmark status: **Held-out unseen cell** (using model trained on B0005/6/7 only).")
                else:
                    st.caption(f"Benchmark status: **In training set** (4-cell deployed Ridge model).")

            # Filter data for chosen cell
            cell_df = feature_table[feature_table["cell"] == cell_name].sort_values("disc_idx").reset_index(drop=True)
            max_cycles = int(cell_df["disc_idx"].max())
            eol_idx_true = int(cell_df["eol_idx"].iloc[0])
            is_pseudo_eol = bool(cell_df["eol_is_pseudo"].iloc[0]) if "eol_is_pseudo" in cell_df.columns else False

            # Scrub cycle slider
            default_cycle = min(50, max_cycles)
            current_cycle = st.slider(
                "Scrub Cycle Number:",
                min_value=0,
                max_value=max_cycles,
                value=default_cycle,
                step=1,
                help="Step forward or backward along the cell life to observe prognostic updates.",
            )

        # Retrieve row data at selected cycle
        row_at_cycle = cell_df[cell_df["disc_idx"] == current_cycle]
        if row_at_cycle.empty:
            row_at_cycle = cell_df.iloc[[0]]

        curr_true_cap = float(row_at_cycle["capacity"].iloc[0])
        curr_soh = (curr_true_cap / 2.0) * 100.0
        true_rul_val = int(row_at_cycle["rul_1p4"].iloc[0]) if "rul_1p4" in row_at_cycle.columns else (eol_idx_true - current_cycle)

        # Compute ML prediction at current cycle
        features_dict = row_at_cycle.iloc[0].to_dict()
        cap_history = cell_df[cell_df["disc_idx"] <= current_cycle]["capacity"].tolist()

        if cell_name == "B0018":
            pred_res = predictor.predict_demo_unseen_b0018(features_dict, disc_idx=current_cycle, cap_history=cap_history)
        else:
            pred_res = predictor.predict_all(features_dict, disc_idx=current_cycle, cap_history=cap_history)

        pred_cap = pred_res["predictions"]["capacity"]["value"] if pred_res.get("success") else curr_true_cap
        pred_rul_obj = pred_res["predictions"]["RUL"] if pred_res.get("success") else {}
        is_eol = pred_rul_obj.get("is_eol", False) or (pred_cap <= 1.40)
        pred_rul = pred_rul_obj.get("value")

        with col_stats:
            st.markdown(
                "<div style='font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-secondary); margin-bottom: 8px;'>Prognostic Diagnostics</div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                render_kpi(
                    label="Current Capacity",
                    value=f"{pred_cap:.3f}",
                    unit="Ah",
                    error_band="0.046",
                    subtext=f"True: {curr_true_cap:.3f} Ah",
                    status="Healthy" if pred_cap >= 1.60 else ("Degrading" if pred_cap > 1.40 else "End of Life"),
                )
            with c2:
                if is_eol:
                    render_kpi(
                        label="Predicted RUL",
                        value="EOL",
                        unit="",
                        error_band=None,
                        subtext="Threshold reached (≤ 1.40 Ah)",
                        status="End of Life",
                    )
                else:
                    rul_err = pred_rul_obj.get("error_band", 15)
                    render_kpi(
                        label="Predicted RUL",
                        value=f"{pred_rul}",
                        unit="cycles",
                        error_band=f"{rul_err}",
                        subtext=f"True RUL: {max(0, true_rul_val)} cycles",
                        status="Healthy" if pred_rul and pred_rul > 30 else "Degrading",
                    )

            # Life Progress Bar
            progress_pct = max(0.0, min(100.0, (current_cycle / float(eol_idx_true)) * 100.0))
            st.markdown(
                f"""
                <div style="margin-top: 10px;">
                  <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #93A0B8; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace;">
                    <span>Life Consumed: {progress_pct:.1f}%</span>
                    <span>True EOL: Cycle {eol_idx_true} {'(Pseudo EOL)' if is_pseudo_eol else ''}</span>
                  </div>
                  <div style="background: rgba(255,255,255,0.08); height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #22D3EE, #8B5CF6); height: 100%; width: {progress_pct}%;"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Capacity vs Cycle Plot
        fig_life = plot_life_and_rul_trajectory(
            cell_df,
            current_cycle=current_cycle,
            pred_cap_at_cycle=pred_cap,
            pred_rul_cycles=pred_rul if not is_eol else None,
        )
        st.plotly_chart(fig_life, use_container_width=True)

        # Physics & Engineering Note
        render_html(f"""
        <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 14px 18px; margin-top: 10px; font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5;">
          <strong>Engineering Safeguard:</strong> The headline capacity model is strictly linear (Ridge) to guarantee robust downward extrapolation past training extremes.
          Notice the upward capacity bumps along the curve: these are <em>capacity regeneration spikes</em> occurring after rest periods between cycling blocks.
          The RUL model is deliberately constrained to active cycles; once capacity drops below 1.40 Ah, the cell is flagged as <strong>End of life reached</strong>.
        </div>
        """)

    with tab_trend_extrap:
        st.markdown("#### Conventional Trend Extrapolation Baseline")
        st.markdown(
            "<p style='color: var(--text-secondary); font-size: 0.85rem;'>"
            "Paste or upload a simple two-column history (<code>cycle, capacity</code>) to fit a linear degradation slope over the last 10 cycles and extrapolate the 1.40 Ah crossing point."
            "</p>",
            unsafe_allow_html=True,
        )

        col_up, col_res = st.columns([1, 1.2], gap="large")

        with col_up:
            sample_csv_text = "cycle,capacity\n10,1.82\n20,1.78\n30,1.75\n40,1.71\n50,1.67\n60,1.63\n70,1.58\n80,1.52\n90,1.46\n"
            user_text = st.text_area(
                "Paste CSV data (cycle, capacity):",
                value=sample_csv_text,
                height=180,
            )
            up_hist = st.file_uploader("Or upload history CSV:", type=["csv"], key="hist_uploader")

        with col_res:
            hist_df = None
            if up_hist:
                try:
                    hist_df = pd.read_csv(up_hist)
                except Exception as e:
                    alert_box(f"CSV read failed: {str(e)}", alert_type="error")
            elif user_text.strip():
                try:
                    import io
                    hist_df = pd.read_csv(io.StringIO(user_text))
                except Exception as e:
                    alert_box(f"Text parse failed: {str(e)}", alert_type="error")

            if hist_df is not None and "cycle" in hist_df.columns and "capacity" in hist_df.columns:
                c_arr = hist_df["cycle"].values
                q_arr = hist_df["capacity"].values
                extrap = extrapolate_capacity_trajectory(c_arr, q_arr, eol_threshold=1.40)

                eol_c = extrap.get("predicted_eol_cycle")
                rul_t = extrap.get("rul_trend")
                slope = extrap.get("slope", 0.0)

                k1, k2 = st.columns(2)
                with k1:
                    render_kpi(
                        label="Projected EOL Cycle",
                        value=f"{eol_c}" if eol_c else "N/A",
                        unit="cycle",
                        subtext="Trend extrapolation baseline",
                        status="Healthy" if eol_c and eol_c > c_arr[-1] else "Degrading",
                    )
                with k2:
                    render_kpi(
                        label="Extrapolated RUL",
                        value=f"{rul_t}" if rul_t else "N/A",
                        unit="cycles",
                        subtext=f"Degradation: {slope*1000:.2f} mAh/cyc",
                        status="Healthy" if rul_t and rul_t > 20 else "Degrading",
                    )

                render_html(f"""
                <div class="cg-card" style="margin-top: 12px; font-size: 0.85rem; color: var(--text-secondary);">
                  <strong>Notice:</strong> This is a simple polynomial slope extrapolation (conventional EE baseline).
                  Unlike the ML model which extracts electrochemical features from terminal voltage dynamics, linear trend extrapolation cannot account for non-linear SEI growth or rollover knee degradation.
                </div>
                """)
            else:
                alert_box("Please supply columns 'cycle' and 'capacity' in the input.", alert_type="warn")
