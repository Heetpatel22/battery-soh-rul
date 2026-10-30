"""
predict.py - CellGuard Smart Battery Health Check.
Designed for both everyday users (car owners, fleet managers) and battery engineers.
Supports 1-click real vehicle presets, everyday driving habit simulation,
clean CSV telemetry upload, and an optional expandable direct-parameter laboratory mode.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np

from ui.components import (
    render_kpi,
    tachometer_soh_gauge_svg,
    prismatic_battery_svg,
    alert_box,
    render_html,
    get_health_grade,
    render_human_insight_card,
)
from ui.charts import plot_discharge_curve

SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sample_cycles"))

# Human-friendly real-world vehicle profiles
SAMPLE_CYCLES_HUMAN = {
    "B0018_cycle_005_early.csv": {
        "title": "Brand New Battery (Factory Fresh)",
        "stage": "Showroom Condition",
        "icon": "🌟",
        "cycle": 5,
        "est_range_km": 292,
        "true_cap": 1.8285,
        "true_soh": 91.43,
        "human_desc": "Completed just 5 charges. Excellent health, maximum range, and cool operation.",
    },
    "B0018_cycle_015_early.csv": {
        "title": "Gently Used (6 Months Daily Commute)",
        "stage": "Early Life",
        "icon": "🚙",
        "cycle": 15,
        "est_range_km": 283,
        "true_cap": 1.7712,
        "true_soh": 88.56,
        "human_desc": "6 months of typical gentle driving. Minor nominal break-in fade.",
    },
    "B0018_cycle_050_middle.csv": {
        "title": "Mid-Life Vehicle (1.5 - 2 Years Used)",
        "stage": "Mid Life (Nominal)",
        "icon": "🚗",
        "cycle": 50,
        "est_range_km": 266,
        "true_cap": 1.6664,
        "true_soh": 83.32,
        "human_desc": "Typical 1-2 year old daily commuter. Solid 83% health, smooth power delivery.",
    },
    "B0018_cycle_065_middle.csv": {
        "title": "High Mileage (Noticeable Aging)",
        "stage": "Moderate Aging",
        "icon": "🚕",
        "cycle": 65,
        "est_range_km": 245,
        "true_cap": 1.5316,
        "true_soh": 76.58,
        "human_desc": "Noticeable capacity degradation. Driver will notice ~15-20% shorter range.",
    },
    "B0018_cycle_100_late.csv": {
        "title": "Near Retirement (End-of-Life Warning)",
        "stage": "Retirement Due",
        "icon": "⚠️",
        "cycle": 100,
        "est_range_km": 221,
        "true_cap": 1.3849,
        "true_soh": 69.25,
        "human_desc": "Reached the 30% fade limit (1.40 Ah threshold). Service or replacement needed.",
    },
    "B0018_cycle_115_late.csv": {
        "title": "Aged Pack (Post-Retirement Storage)",
        "stage": "Post Retirement",
        "icon": "🚨",
        "cycle": 115,
        "est_range_km": 220,
        "true_cap": 1.3882,
        "true_soh": 69.41,
        "human_desc": "Heavily degraded cell suitable only for low-power stationary energy storage.",
    },
}


def load_reference_curve():
    """Load B0018 Cycle 5 as reference early-life curve."""
    p = os.path.join(SAMPLE_DIR, "B0018_cycle_005_early.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        return df["Time_s"].values, df["Voltage_V"].values
    return None, None


def render_predict(predictor, schema: dict, metrics: dict):
    """Render the human-friendly Battery Health Check diagnostic page."""
    st.markdown("""
    <div style="margin-bottom: 18px;">
      <h2 style="font-size: 1.85rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">
        🩺 Smart Battery Health Check
      </h2>
      <p style="font-size: 0.92rem; color: #94A3B8; margin: 0;">
        Check any battery's health, driving range, and remaining charges in seconds.
        Pick a real-world vehicle stage, simulate everyday driving habits, or upload your battery log.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_results = st.columns([1.1, 1.4], gap="large")

    df_raw = None
    manual_features = None
    inferred_disc_idx = 50
    data_source_label = ""

    with col_input:
        st.markdown("<div style='font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #22D3EE; margin-bottom: 8px;'>1. CHOOSE HOW YOU WANT TO TEST</div>", unsafe_allow_html=True)

        input_mode = st.radio(
            "Input Mode Selector",
            ["🚗 Real Vehicle Presets", "🎮 Everyday Usage Simulator", "📁 Upload Battery Log (CSV)"],
            label_visibility="collapsed",
            horizontal=False,
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # MODE 1: Real Vehicle Presets
        if "Real Vehicle Presets" in input_mode:
            sample_keys = list(SAMPLE_CYCLES_HUMAN.keys())
            default_ix = 2  # Cycle 50
            if "selected_sample" in st.session_state and st.session_state["selected_sample"] in sample_keys:
                default_ix = sample_keys.index(st.session_state["selected_sample"])

            selected_key = st.selectbox(
                "Select a Real Vehicle Battery Life Stage:",
                sample_keys,
                index=default_ix,
                format_func=lambda k: f"{SAMPLE_CYCLES_HUMAN[k]['icon']} {SAMPLE_CYCLES_HUMAN[k]['title']}",
            )
            meta = SAMPLE_CYCLES_HUMAN[selected_key]
            inferred_disc_idx = meta["cycle"]
            data_source_label = meta["title"]

            fpath = os.path.join(SAMPLE_DIR, selected_key)
            if os.path.exists(fpath):
                df_raw = pd.read_csv(fpath)

            render_html(f"""
            <div style="background: rgba(14, 20, 33, 0.75); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px 16px; margin-top: 8px;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.85rem; font-weight: 700; color: #FFFFFF;">{meta['icon']} {meta['title']}</span>
                <span style="font-size: 0.72rem; color: #22D3EE; font-family: 'JetBrains Mono', monospace;">Cycle {meta['cycle']}</span>
              </div>
              <div style="font-size: 0.80rem; color: #CBD5E1; line-height: 1.45; margin-bottom: 8px;">
                {meta['human_desc']}
              </div>
              <div style="display: flex; gap: 14px; font-size: 0.74rem; color: #94A3B8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 6px;">
                <span>Verified Benchmark: <strong style="color: #FFFFFF;">{meta['true_cap']:.3f} Ah</strong></span>
                <span>Actual Health: <strong style="color: #10B981;">{meta['true_soh']:.1f}%</strong></span>
              </div>
            </div>
            """)

        # MODE 2: Everyday Usage Simulator (Replaces raw confusing sliders!)
        elif "Everyday Usage Simulator" in input_mode:
            st.markdown("""
            <div style="background: rgba(14, 20, 33, 0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.82rem; color: #CBD5E1;">
              Adjust intuitive everyday factors like battery age and driving style. Our engine automatically translates these into physical electrochemical signals.
            </div>
            """, unsafe_allow_html=True)

            sim_charges = st.slider("Battery Age (Completed Full Charges):", 0, 150, 45, step=5, help="Approximate number of full charge-discharge cycles completed.")
            inferred_disc_idx = sim_charges

            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                driving_style = st.selectbox(
                    "Daily Driving Style:",
                    ["Gentle Eco Driving", "Normal Daily Commute", "Aggressive / Frequent Fast-Charge"],
                    index=1,
                    help="Aggressive driving and frequent fast-charging slightly increase internal battery heating and resistance.",
                )
            with col_sim2:
                climate = st.selectbox(
                    "Operating Climate:",
                    ["Mild (24°C / Ideal)", "Hot Summer (38°C)", "Cold Winter (5°C)"],
                    index=0,
                    help="Extreme heat or freezing cold impacts battery chemical kinetics.",
                )

            # Map everyday inputs into physically valid partial-window features
            age_frac = float(sim_charges) / 140.0
            style_penalty = 1.05 if "Aggressive" in driving_style else (0.96 if "Gentle" in driving_style else 1.0)
            temp_offset = 15.0 if "Hot" in climate else (5.0 if "Cold" in climate else 0.0)

            # Realistic feature synthesis adhering to training bounds
            manual_features = {}
            manual_features["T_40_38"] = float(np.clip(350.0 - (120.0 * age_frac * style_penalty), 100.0, 480.0))
            manual_features["T_38_36"] = float(np.clip(770.0 - (210.0 * age_frac * style_penalty), 350.0, 1000.0))
            manual_features["T_36_34"] = float(np.clip(1320.0 - (380.0 * age_frac * style_penalty), 600.0, 1650.0))
            manual_features["T_40_36"] = manual_features["T_40_38"] + manual_features["T_38_36"]
            manual_features["T_38_34"] = manual_features["T_38_36"] + manual_features["T_36_34"]
            manual_features["Q_40_36"] = float(np.clip(0.62 - (0.22 * age_frac * style_penalty), 0.22, 0.80))
            manual_features["V_at_60s"] = float(np.clip(3.93 - (0.08 * age_frac), 3.78, 3.98))
            manual_features["V_at_300s"] = float(np.clip(3.83 - (0.10 * age_frac), 3.65, 3.88))
            manual_features["V_at_600s"] = float(np.clip(3.73 - (0.12 * age_frac), 3.50, 3.78))
            manual_features["V_at_1200s"] = float(np.clip(3.58 - (0.14 * age_frac), 3.35, 3.68))

            manual_features["V_start"] = 3.98
            manual_features["V_mean_first600s"] = (manual_features["V_at_60s"] + manual_features["V_at_300s"] + manual_features["V_at_600s"]) / 3.0
            manual_features["V_slope_mid"] = -0.00030 * style_penalty
            manual_features["T_start"] = 24.0 + temp_offset
            manual_features["T_max"] = 39.0 + (5.0 * age_frac) + temp_offset
            manual_features["T_rise"] = manual_features["T_max"] - manual_features["T_start"]
            manual_features["T_mean"] = (manual_features["T_start"] + manual_features["T_max"]) / 2.0
            manual_features["R_dc_proxy"] = float(np.clip(0.095 + (0.045 * age_frac * style_penalty), 0.08, 0.22))
            data_source_label = f"Usage Simulator ({sim_charges} charges, {driving_style.split()[0]})"

        # MODE 3: Upload Battery Log (CSV)
        else:
            uploaded_file = st.file_uploader(
                "Upload Raw Battery Discharge CSV:",
                type=["csv"],
                help="Columns needed: Time_s, Voltage_V, Current_A, Temperature_C",
            )
            if uploaded_file is not None:
                try:
                    df_raw = pd.read_csv(uploaded_file)
                    req_cols = ["Time_s", "Voltage_V", "Current_A", "Temperature_C"]
                    missing = [c for c in req_cols if c not in df_raw.columns]
                    if missing:
                        alert_box(
                            f"CSV is missing required headers: {', '.join(missing)}.",
                            alert_type="error",
                            title="CSV Header Validation Failed",
                        )
                        df_raw = None
                    else:
                        st.success(f"✓ Valid telemetry loaded ({len(df_raw)} samples recorded).")
                        data_source_label = uploaded_file.name
                except Exception as e:
                    alert_box(f"CSV Parse error: {str(e)}", alert_type="error")
                    df_raw = None
            else:
                render_html("""
                <div style="border: 1px dashed rgba(34,211,238,0.3); border-radius: 10px; padding: 24px; text-align: center; color: #94A3B8; margin-top: 8px;">
                  <div style="font-size: 0.95rem; font-weight: 600; color: #FFFFFF; margin-bottom: 4px;">Drop Your Battery Log Here</div>
                  <div style="font-size: 0.76rem; color: #64748B;">Supported columns: Time_s, Voltage_V, Current_A, Temperature_C</div>
                </div>
                """)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Optional Laboratory Mode Drawer (Tucked away so normal people aren't confused)
        with st.expander("🔬 Direct Micro-Signal Parameter Lab (For Battery Engineers)", expanded=False):
            st.markdown("""
            <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 8px;">
              Directly override micro-second plateau traversal timings ($4.0\\text{ V} \\to 3.4\\text{ V}$) for custom bench testing.
            </div>
            """, unsafe_allow_html=True)

            lab_override = st.checkbox("Enable Direct Parameter Overrides", value=False)
            if lab_override:
                if manual_features is None:
                    manual_features = {}
                col_l1, col_l2 = st.columns(2)
                with col_l1:
                    manual_features["T_40_38"] = st.slider("T_40_38: Time from 4.0V to 3.8V (s)", 50.0, 500.0, 300.0, 5.0)
                    manual_features["T_38_36"] = st.slider("T_38_36: Time from 3.8V to 3.6V (s)", 250.0, 1050.0, 670.0, 10.0)
                    manual_features["T_36_34"] = st.slider("T_36_34: Time from 3.6V to 3.4V (s)", 500.0, 1700.0, 1140.0, 10.0)
                    manual_features["Q_40_36"] = st.slider("Q_40_36: Partial Charge (Ah)", 0.18, 0.85, 0.54, 0.01)
                with col_l2:
                    manual_features["T_40_36"] = manual_features["T_40_38"] + manual_features["T_38_36"]
                    manual_features["T_38_34"] = manual_features["T_38_36"] + manual_features["T_36_34"]
                    manual_features["V_at_60s"] = st.slider("V_at_60s: Voltage at 60s (V)", 3.75, 4.00, 3.90, 0.01)
                    manual_features["V_at_300s"] = st.slider("V_at_300s: Voltage at 300s (V)", 3.60, 3.90, 3.79, 0.01)
                    manual_features["V_at_600s"] = st.slider("V_at_600s: Voltage at 600s (V)", 3.45, 3.80, 3.68, 0.01)
                    manual_features["V_at_1200s"] = st.slider("V_at_1200s: Voltage at 1200s (V)", 3.30, 3.70, 3.53, 0.01)
                data_source_label = "Direct Parameter Lab"

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        inferred_disc_idx = st.number_input(
            "Service Cycle Age (Charges completed):",
            min_value=0,
            max_value=250,
            value=inferred_disc_idx,
            step=1,
            help="Number of full charge cycles the battery has completed since brand new.",
        )

        use_holdout = st.toggle(
            "Strict Unseen Battery Mode (Held-out NASA B0018)",
            value=True,
            help="Evaluates strictly on an AI model that never saw cell B0018 during training to prove true generalization.",
        )

    # Right Column: Instant Diagnostic Output Console
    with col_results:
        st.markdown("<div style='font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #22D3EE; margin-bottom: 8px;'>2. DIAGNOSTIC REPORT</div>", unsafe_allow_html=True)

        result = None
        if df_raw is not None:
            t = df_raw["Time_s"].values
            v = df_raw["Voltage_V"].values
            i = df_raw["Current_A"].values
            temp = df_raw["Temperature_C"].values

            from battery_features import extract_discharge_features
            extracted_feats = extract_discharge_features(t, v, i, temp)

            is_valid, issues = predictor.validate_features(extracted_feats)
            if not is_valid:
                for issue in issues:
                    alert_box(issue, alert_type="warn", title="Voltage Window Incomplete")
                st.info("💡 The diagnostic scan requires the battery curve to start at 4.0 V and reach down to 3.4 V (the first 5 minutes of discharge).")
            else:
                if use_holdout:
                    result = predictor.predict_demo_unseen_b0018(extracted_feats, disc_idx=inferred_disc_idx)
                else:
                    result = predictor.predict_all(extracted_feats, disc_idx=inferred_disc_idx)

        elif manual_features is not None:
            if use_holdout:
                result = predictor.predict_demo_unseen_b0018(manual_features, disc_idx=inferred_disc_idx)
            else:
                result = predictor.predict_all(manual_features, disc_idx=inferred_disc_idx)

        if result and result.get("success"):
            preds = result["predictions"]
            pred_cap = preds["capacity"]["value"]
            pred_soh = preds["soh"]["value"]
            re_val = preds["Re"]["value"]
            re_err = preds["Re"]["error_band"]
            rct_val = preds["Rct"]["value"]
            rct_err = preds["Rct"]["error_band"]
            rul_data = preds["RUL"]
            cap_err = preds["capacity"]["error_band"]

            grade, grade_badge, grade_desc, grade_color = get_health_grade(pred_soh)
            est_range_km = int(round((pred_soh / 100.0) * 320.0))

            # Visual instruments
            col_g, col_b = st.columns([1.2, 1])
            with col_g:
                tacho_svg = tachometer_soh_gauge_svg(pred_soh, capacity_ah=pred_cap, size=185)
                render_html(f'<div style="background: rgba(12, 16, 25, 0.9); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 10px; display: flex; align-items: center; justify-content: center;">{tacho_svg}</div>')
            with col_b:
                cell_svg = prismatic_battery_svg(pred_soh, width=105, height=185)
                render_html(f'<div style="background: rgba(12, 16, 25, 0.9); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 10px; display: flex; align-items: center; justify-content: center;">{cell_svg}</div>')

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            # 4 Human-friendly headline KPI tiles
            k1, k2 = st.columns(2)
            with k1:
                render_kpi(
                    label="Battery Health Score",
                    value=f"{pred_soh:.1f}",
                    unit="%",
                    subtext=f"Holds {pred_cap:.3f} Ah (Original 2.00 Ah)",
                    status="Healthy" if pred_soh >= 80 else ("Degrading" if pred_soh >= 70 else "End of Life"),
                    tooltip="State of Health: Total stored energy compared to when the battery was brand new.",
                )
            with k2:
                render_kpi(
                    label="Usable Driving Range",
                    value=f"{est_range_km}",
                    unit="km",
                    subtext="Baseline: 320 km on showroom battery",
                    status="Healthy" if est_range_km >= 260 else ("Degrading" if est_range_km >= 230 else "End of Life"),
                    tooltip="Estimated driving range on a full 100% charge.",
                )

            k3, k4 = st.columns(2)
            with k3:
                if rul_data.get("is_eol") or (pred_cap <= 1.40):
                    render_kpi("Lifespan Countdown", "Service Due", "", subtext="Reached 70% retirement threshold", status="End of Life")
                else:
                    rul_val = rul_data.get("value", 0)
                    render_kpi(
                        label="Lifespan Countdown",
                        value=f"{rul_val}",
                        unit="charges",
                        subtext=f"Retirement ~ cycle {inferred_disc_idx + rul_val}",
                        status="Healthy" if rul_val >= 35 else "Degrading",
                    )
            with k4:
                re_mohm = re_val * 1000.0
                re_status = "Optimal" if re_mohm < 170 else ("Moderate" if re_mohm < 200 else "High Heat")
                render_kpi(
                    label="Internal Flow & Heat",
                    value=re_status,
                    unit="",
                    subtext=f"Resistance: {re_mohm:.1f} mΩ (Cool)",
                    status="Healthy" if re_mohm < 170 else ("Degrading" if re_mohm < 200 else "End of Life"),
                )

            # Plain English Insight & Recommendation Card
            render_human_insight_card(
                soh=pred_soh,
                capacity_ah=pred_cap,
                rul_val=rul_data.get("value"),
                is_eol=rul_data.get("is_eol", False) or (pred_cap <= 1.40),
                re_val=re_val,
            )

            # Optional Technical Lab Drawer for Engineers
            with st.expander("🔬 View Certified Lab Error Bounds & ECM Impedances (For Engineers)", expanded=False):
                st.markdown(f"""
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #CBD5E1; line-height: 1.6; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px;">
                  • Discharged Capacity: <strong>{pred_cap:.4f} &plusmn; {cap_err:.4f} Ah</strong><br/>
                  • Electrolyte Resistance (Re / R0): <strong>{re_val*1000:.2f} &plusmn; {re_err*1000:.2f} mΩ</strong><br/>
                  • Charge-Transfer Resistance (Rct / R1): <strong>{rct_val*1000:.2f} &plusmn; {rct_err*1000:.2f} mΩ</strong><br/>
                  • Remaining Useful Life (RUL): <strong>{rul_data.get('value')} &plusmn; {rul_data.get('error_band', 15)} cycles</strong>
                </div>
                """, unsafe_allow_html=True)

                export_payload = {
                    "source": data_source_label,
                    "cycle": inferred_disc_idx,
                    "model_mode": "held_out_b0018" if use_holdout else "deployed_4_cells",
                    "predictions": preds,
                }
                st.download_button(
                    "📥 Export Full Diagnostic Report (JSON)",
                    data=json.dumps(export_payload, indent=2),
                    file_name=f"cellguard_telemetry_cycle_{inferred_disc_idx}.json",
                    mime="application/json",
                )

        else:
            render_html("""
            <div style="background: rgba(12, 16, 25, 0.85); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 48px 24px; text-align: center; color: #64748B;">
              <div style="font-size: 1.1rem; font-weight: 700; color: #E2E8F0; margin-bottom: 6px;">Ready to Inspect Battery</div>
              <div style="font-size: 0.84rem; color: #94A3B8;">Select a real vehicle preset on the left or simulate everyday driving habits.</div>
            </div>
            """)

    # Bottom: Voltage Profile Chart
    if df_raw is not None:
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        col_c_hdr, col_c_tog = st.columns([1.5, 1])
        with col_c_hdr:
            st.markdown("#### Battery Discharge Profile & 5-Minute Inspection Window")
            st.caption("Notice how CellGuard only needs the early shaded window (4.0 V &rarr; 3.4 V) to determine full health — no full discharge needed!")
        with col_c_tog:
            overlay_ref = st.checkbox("Compare with Brand New Battery (Cycle 5)", value=True)

        ref_t, ref_v = (None, None)
        if overlay_ref:
            ref_t, ref_v = load_reference_curve()

        t = df_raw["Time_s"].values
        v = df_raw["Voltage_V"].values
        temp = df_raw["Temperature_C"].values if "Temperature_C" in df_raw.columns else None

        fig_curve = plot_discharge_curve(t, v, temp=temp, ref_t=ref_t, ref_v=ref_v)
        st.plotly_chart(fig_curve, use_container_width=True)
