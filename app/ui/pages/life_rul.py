"""
life_rul.py - CellGuard Battery Lifespan & Range Forecasting.
Tracks battery degradation across life cycles, projects remaining useful life,
and explains the 70% retirement threshold in plain English.
"""

import streamlit as st
import pandas as pd
import numpy as np

from ui.components import (
    render_kpi,
    status_chip_html,
    alert_box,
    render_html,
    get_health_grade,
    render_human_insight_card,
)
from ui.charts import plot_life_and_rul_trajectory
from inference import extrapolate_capacity_trajectory


def render_life_rul(predictor, feature_table: pd.DataFrame, metrics: dict):
    """Render the Battery Lifespan and RUL page."""
    st.markdown("""
    <div style="margin-bottom: 18px;">
      <h2 style="font-size: 1.85rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">
        ⏳ Battery Lifespan &amp; Range Forecasting
      </h2>
      <p style="font-size: 0.92rem; color: #94A3B8; margin: 0;">
        Track how a battery ages charge-by-charge, observe declining driving range, and project exactly when replacement will be needed.
      </p>
    </div>
    """, unsafe_allow_html=True)

    tab_cell_demo, tab_trend_extrap = st.tabs([
        "🔋 Battery Lifecycle Explorer",
        "📈 Simple Trend Extrapolation (Baseline Comparison)",
    ])

    with tab_cell_demo:
        col_ctrl, col_stats = st.columns([1.4, 1.1], gap="large")

        with col_ctrl:
            col_cell, col_mode = st.columns([1.1, 1.2])
            with col_cell:
                cell_choice = st.selectbox(
                    "Select Battery Cell to Examine:",
                    ["B0018 (Held-Out Benchmark)", "B0005 (Standard Test Pack)", "B0006 (Heavy-Use Cell)", "B0007 (Lab Cell)"],
                    index=0,
                )
                cell_name = cell_choice.split()[0]

            with col_mode:
                if cell_name == "B0018":
                    st.caption("Benchmark status: **Held-out unseen cell** (evaluates pipeline generalization on real battery).")
                else:
                    st.caption("Benchmark status: **Verified laboratory cell** (NASA test rack).")

            # Filter data for chosen cell
            cell_df = feature_table[feature_table["cell"] == cell_name].sort_values("disc_idx").reset_index(drop=True)
            max_cycles = int(cell_df["disc_idx"].max())
            eol_idx_true = int(cell_df["eol_idx"].iloc[0])
            is_pseudo_eol = bool(cell_df["eol_is_pseudo"].iloc[0]) if "eol_is_pseudo" in cell_df.columns else False

            # Scrub cycle slider
            default_cycle = min(50, max_cycles)
            current_cycle = st.slider(
                "Step Through Battery Age (Charges Completed):",
                min_value=0,
                max_value=max_cycles,
                value=default_cycle,
                step=1,
                help="Move slider along the battery's lifespan to observe how health and range degrade.",
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

        pred_soh = (pred_cap / 2.0) * 100.0
        est_range_km = int(round((pred_soh / 100.0) * 320.0))

        with col_stats:
            st.markdown(
                "<div style='font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #22D3EE; margin-bottom: 8px;'>DIAGNOSTIC STATUS AT CURRENT AGE</div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                render_kpi(
                    label="Health Score (SOH)",
                    value=f"{pred_soh:.1f}",
                    unit="%",
                    subtext=f"Usable: {pred_cap:.3f} Ah (~{est_range_km} km)",
                    status="Healthy" if pred_soh >= 80 else ("Degrading" if pred_soh >= 70 else "End of Life"),
                )
            with c2:
                if is_eol:
                    render_kpi(
                        label="Charges Left",
                        value="Service",
                        unit="Due",
                        subtext="Under 70% threshold (≤1.40 Ah)",
                        status="End of Life",
                    )
                else:
                    rul_err = pred_rul_obj.get("error_band", 15)
                    months_left = max(1, int(round(pred_rul / 3.5))) if pred_rul else 0
                    render_kpi(
                        label="Charges Left",
                        value=f"{pred_rul}",
                        unit="charges",
                        subtext=f"~{months_left} months (True: {max(0, true_rul_val)})",
                        status="Healthy" if pred_rul and pred_rul > 30 else "Degrading",
                    )

            # Battery Lifespan Consumption Progress Bar
            progress_pct = max(0.0, min(100.0, (current_cycle / float(eol_idx_true)) * 100.0))
            st.markdown(
                f"""
                <div style="margin-top: 10px; background: rgba(12, 16, 25, 0.85); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 14px;">
                  <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: #94A3B8; margin-bottom: 6px;">
                    <span>Lifespan Consumed: <strong style="color: #FFFFFF;">{progress_pct:.1f}%</strong></span>
                    <span>Retirement Due: <strong style="color: #22D3EE;">Cycle {eol_idx_true}</strong></span>
                  </div>
                  <div class="cg-progress-bar-bg">
                    <div class="cg-progress-bar-fill" style="width: {progress_pct}%; background: linear-gradient(90deg, #10B981, #F59E0B 80%, #EF4444 100%);"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Plain English Diagnosis Card
        render_human_insight_card(
            soh=pred_soh,
            capacity_ah=pred_cap,
            rul_val=pred_rul,
            is_eol=is_eol,
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Capacity vs Cycle Plot with Health Zones
        fig_life = plot_life_and_rul_trajectory(
            cell_df,
            current_cycle=current_cycle,
            pred_cap_at_cycle=pred_cap,
            pred_rul_cycles=pred_rul if not is_eol else None,
        )
        st.plotly_chart(fig_life, use_container_width=True)

        # Plain English Explanation of Battery Science
        render_html("""
        <div class="cg-card" style="margin-top: 10px; border-left: 3px solid #22D3EE;">
          <div style="font-weight: 700; font-size: 0.92rem; color: #FFFFFF; margin-bottom: 6px;">
            💡 Why do batteries get retired at 70% health (1.40 Ah)?
          </div>
          <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.55;">
            Automotive and aerospace standards establish the 30% fade mark (&le; 1.40 Ah for a 2.0 Ah cell) as <strong>End of Life (EOL)</strong>.
            Past this point, lithium ions begin plating on the anode, the curve enters an accelerated "rollover knee" where capacity plunges unpredictably, and fast-charging creates excessive heat.
            <br/><br/>
            <strong>Notice the small upward spikes along the curve?</strong> Those are natural <em>capacity regeneration recoveries</em>! When a battery is left to rest between usage cycles, lithium ions redistribute evenly across the electrodes, providing a temporary slight boost in usable charge.
          </div>
        </div>
        """)

    with tab_trend_extrap:
        st.markdown("#### Conventional Linear Trend Extrapolation Baseline")
        st.markdown(
            "<p style='color: var(--text-secondary); font-size: 0.86rem;'>"
            "See why smart AI beats simple straight-line extrapolation. Paste or upload historical capacity points (<code>cycle, capacity</code>) to fit a linear trend line."
            "</p>",
            unsafe_allow_html=True,
        )

        col_up, col_res = st.columns([1, 1.2], gap="large")

        with col_up:
            sample_csv_text = "cycle,capacity\n10,1.82\n20,1.78\n30,1.75\n40,1.71\n50,1.67\n60,1.63\n70,1.58\n80,1.52\n90,1.46\n"
            user_text = st.text_area(
                "Paste history data (cycle, capacity):",
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
                        subtext="Linear extrapolation baseline",
                        status="Healthy" if eol_c and eol_c > c_arr[-1] else "Degrading",
                    )
                with k2:
                    render_kpi(
                        label="Extrapolated RUL",
                        value=f"{rul_t}" if rul_t else "N/A",
                        unit="charges",
                        subtext=f"Degradation: {slope*1000:.2f} mAh/cycle",
                        status="Healthy" if rul_t and rul_t > 20 else "Degrading",
                    )

                render_html("""
                <div class="cg-card" style="margin-top: 12px; font-size: 0.84rem; color: #CBD5E1; line-height: 1.5;">
                  <strong>Why Smart AI Wins Over Linear Trend:</strong> A simple straight-line assumes the battery degrades at a constant rate forever.
                  In reality, batteries degrade in phases: slow early aging, a steady plateau, and then sudden accelerated decline.
                  CellGuard's AI reads early electrochemical voltage signals to capture real internal degradation rather than drawing a naive straight line.
                </div>
                """)
            else:
                alert_box("Please supply columns 'cycle' and 'capacity' in the input.", alert_type="warn")
