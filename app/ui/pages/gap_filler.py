"""
gap_filler.py - Missing-data gap-filling demonstration for impedance (M2: Re and Rct).
Simulates missing EIS cycles under random (10/20/30%) and contiguous block masking.
Compares ML model against within-cell linear interpolation, LOCF, and mean fill.
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
from ui.charts import plot_gap_filling_comparison, plot_model_baseline_bar


def render_gap_filler(predictor, feature_table: pd.DataFrame, metrics: dict):
    """Render the Missing-Data Gap Filler page."""
    st.markdown("""
    <div style="margin-bottom: 16px;">
      <h2 style="font-size: 1.85rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">
        🧩 Smart Sensor Gap Filler
      </h2>
      <p style="font-size: 0.92rem; color: #94A3B8; margin: 0;">
        How CellGuard recovers missing battery resistance telemetry when laboratory sensors are unavailable.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Friendly Plain-English Explanation
    render_html("""
    <div class="cg-launch-hero" style="padding: 16px 22px; margin-bottom: 18px;">
      <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <span class="cg-launch-badge">💡 REAL-WORLD SENSOR RECOVERY</span>
        <span style="font-size: 0.82rem; color: #E2E8F0; font-weight: 700;">No Expensive Lab Hardware Required</span>
      </div>
      <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.55;">
        In real electric vehicles, measuring internal chemical resistance (EIS) requires expensive laboratory equipment that cannot fit inside a car. 
        What happens when sensor readings are missing? 
        CellGuard's AI accurately estimates internal electrolyte flow resistance (<code>Re</code>) and reaction speed (<code>Rct</code>) 
        across missing cycles, outperforming naive averages especially during long data gaps!
      </div>
    </div>
    """)

    # Controls row
    col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
    with col_c1:
        cell_choice = st.selectbox(
            "Select Cell to Audit:",
            ["B0018", "B0005", "B0006", "B0007"],
            index=0,
            help="B0018 is the primary held-out benchmark.",
        )
    with col_c2:
        mask_mode = st.selectbox(
            "Masking Experiment:",
            ["Contiguous Block (15 Cycles Gap)", "Random Mask 10%", "Random Mask 20%", "Random Mask 30%"],
            index=0,
        )
    with col_c3:
        target_param = st.selectbox(
            "Impedance Parameter:",
            ["Rct (Charge-Transfer Resistance)", "Re (Ohmic Resistance)"],
            index=0,
        )
        target_col = "Rct" if "Rct" in target_param else "Re"

    # Filter cell rows with valid EIS measurements
    cdf = feature_table[(feature_table["cell"] == cell_choice) & (feature_table[target_col].notna())].copy()
    cdf = cdf.sort_values("disc_idx").reset_index(drop=True)

    n_samples = len(cdf)
    if n_samples < 10:
        alert_box(f"Insufficient EIS measurements for cell {cell_choice}.", alert_type="warn")
        return

    # Deterministic masking logic
    rng = np.random.RandomState(42)
    masked_mask = np.zeros(n_samples, dtype=bool)

    if "Block" in mask_mode:
        block_len = min(15, n_samples // 3)
        start_idx = n_samples // 3
        masked_mask[start_idx : start_idx + block_len] = True
    elif "10%" in mask_mode:
        masked_mask[rng.choice(n_samples, size=int(0.10 * n_samples), replace=False)] = True
    elif "20%" in mask_mode:
        masked_mask[rng.choice(n_samples, size=int(0.20 * n_samples), replace=False)] = True
    else:  # 30%
        masked_mask[rng.choice(n_samples, size=int(0.30 * n_samples), replace=False)] = True

    cycles = cdf["disc_idx"].values
    measured_vals = cdf[target_col].values

    # 1. Linear Interpolation baseline
    s_interp = pd.Series(measured_vals).copy()
    s_interp[masked_mask] = np.nan
    interp_vals = s_interp.interpolate(method="linear").bfill().ffill().values

    # 2. LOCF baseline
    s_locf = pd.Series(measured_vals).copy()
    s_locf[masked_mask] = np.nan
    locf_vals = s_locf.ffill().bfill().values

    # 3. Mean fill baseline
    known_mean = np.mean(measured_vals[~masked_mask])
    mean_vals = np.full_like(measured_vals, known_mean)

    # 4. ML Model Prediction
    model_wrapper = predictor.model_rct if target_col == "Rct" else predictor.model_re
    pipe = model_wrapper["pipeline"]
    feats_req = model_wrapper["features"]

    X_eval = cdf[feats_req].copy()
    ml_vals = pipe.predict(X_eval)

    # Compute errors on masked cycles
    true_masked = measured_vals[masked_mask]
    mae_interp = float(np.mean(np.abs(interp_vals[masked_mask] - true_masked)))
    mae_locf = float(np.mean(np.abs(locf_vals[masked_mask] - true_masked)))
    mae_mean = float(np.mean(np.abs(mean_vals[masked_mask] - true_masked)))
    mae_ml = float(np.mean(np.abs(ml_vals[masked_mask] - true_masked)))

    # Chart display
    fig_gap = plot_gap_filling_comparison(
        cycles=cycles,
        measured=measured_vals,
        masked_mask=masked_mask,
        ml_preds=ml_vals,
        interp_preds=interp_vals,
        target_name=target_col,
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    # Metrics Summary Row
    st.markdown(f"#### Benchmarking Results on {int(np.sum(masked_mask))} Masked Cycles ({target_col})")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi(
            label="Linear Interp MAE",
            value=f"{mae_interp * 1000:.2f}",
            unit="mΩ",
            subtext="Within-cell linear baseline",
            status="Healthy" if mae_interp <= mae_ml else "Degrading",
        )
    with k2:
        render_kpi(
            label="ML Model MAE",
            value=f"{mae_ml * 1000:.2f}",
            unit="mΩ",
            subtext="Cross-cell regression pipeline",
            status="Healthy" if mae_ml <= mae_interp else "Degrading",
        )
    with k3:
        render_kpi(
            label="LOCF Baseline MAE",
            value=f"{mae_locf * 1000:.2f}",
            unit="mΩ",
            subtext="Last observation carried forward",
            status="Degrading",
        )
    with k4:
        render_kpi(
            label="Mean Fill MAE",
            value=f"{mae_mean * 1000:.2f}",
            unit="mΩ",
            subtext="Training set average fill",
            status="End of Life",
        )

    # Honest Finding Callout Card (strictly per metrics.json m2_impedance findings)
    render_html(f"""
    <div class="cg-card" style="margin-top: 14px; border-left: 3px solid #22D3EE;">
      <div style="font-size: 0.92rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;">
        🔬 Academic &amp; Engineering Finding (from metrics.json verification)
      </div>
      <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.55;">
        <strong>Honest Finding:</strong> Within-cell linear interpolation beats cross-cell ML for Re and Rct in most small random masking cases (10% and 20%) 
        because impedance changes very smoothly between adjacent cycles within a single cell.
        <br/><br/>
        <strong>Where ML Wins:</strong> Machine learning outperforms interpolation specifically for 
        <strong>large contiguous block gaps</strong> (15+ cycles without EIS) and for <strong>Rct at 30% masking</strong> 
        (MAE 0.00121 &Omega; for ML vs 0.00123 &Omega; for interpolation; block gap MAE 0.00125 &Omega; for ML vs 0.00146 &Omega; for interpolation).
      </div>
    </div>
    """)
