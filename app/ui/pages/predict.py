"""
predict.py - Precision cycle diagnosis page.
Supports bundled cycle selection, CSV telemetry upload, and manual feature entry.
Displays tachometer SOH gauge, prismatic cell visual, verified error bands, and interactive voltage profile.
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
)
from ui.charts import plot_discharge_curve

SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sample_cycles"))

SAMPLE_CYCLES_META = {
    "B0018_cycle_005_early.csv": {"stage": "Early Life", "cycle": 5, "true_cap": 1.8285, "true_soh": 91.43},
    "B0018_cycle_015_early.csv": {"stage": "Early Life", "cycle": 15, "true_cap": 1.7712, "true_soh": 88.56},
    "B0018_cycle_050_middle.csv": {"stage": "Mid Life", "cycle": 50, "true_cap": 1.6664, "true_soh": 83.32},
    "B0018_cycle_065_middle.csv": {"stage": "Mid Life", "cycle": 65, "true_cap": 1.5316, "true_soh": 76.58},
    "B0018_cycle_100_late.csv": {"stage": "Post EOL", "cycle": 100, "true_cap": 1.3849, "true_soh": 69.25},
    "B0018_cycle_115_late.csv": {"stage": "Post EOL", "cycle": 115, "true_cap": 1.3882, "true_soh": 69.41},
}


def load_reference_curve():
    """Load B0018 Cycle 5 as reference early-life curve."""
    p = os.path.join(SAMPLE_DIR, "B0018_cycle_005_early.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        return df["Time_s"].values, df["Voltage_V"].values
    return None, None


def render_predict(predictor, schema: dict, metrics: dict):
    """Render the Predict Cycle diagnostic page."""
    col_input, col_results = st.columns([1.1, 1.4], gap="large")

    df_raw = None
    manual_features = None
    inferred_disc_idx = 50
    data_source_label = ""

    with col_input:
        st.markdown("#### Input Signal Acquisition")

        # Segmented radio control
        input_mode = st.radio(
            "Input Mode Selector",
            ["Sample Cycle", "Upload CSV", "Manual Features"],
            label_visibility="collapsed",
            horizontal=True,
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        if input_mode == "Sample Cycle":
            sample_keys = list(SAMPLE_CYCLES_META.keys())
            default_ix = 2  # cycle 50
            if "selected_sample" in st.session_state and st.session_state["selected_sample"] in sample_keys:
                default_ix = sample_keys.index(st.session_state["selected_sample"])

            selected_key = st.selectbox(
                "Select Bundled NASA Cell B0018 Cycle:",
                sample_keys,
                index=default_ix,
                format_func=lambda k: f"{k} ({SAMPLE_CYCLES_META[k]['stage']} &bull; Cycle {SAMPLE_CYCLES_META[k]['cycle']})",
            )
            meta = SAMPLE_CYCLES_META[selected_key]
            inferred_disc_idx = meta["cycle"]
            data_source_label = selected_key

            fpath = os.path.join(SAMPLE_DIR, selected_key)
            if os.path.exists(fpath):
                df_raw = pd.read_csv(fpath)
                render_html(f"""
                <div style="background: #0B0F19; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 14px; margin-top: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: #94A3B8;">
                  <div><span style="color: #64748B;">STAGE:</span> <span style="color: #E2E8F0; font-weight: 600;">{meta['stage']}</span> &bull; <span style="color: #64748B;">CYCLE:</span> <span style="color: #22D3EE;">{meta['cycle']}</span></div>
                  <div><span style="color: #64748B;">BENCHMARK TRUTH:</span> <span style="color: #FFFFFF;">{meta['true_cap']:.4f} Ah</span> ({meta['true_soh']:.2f}% SOH)</div>
                </div>
                """)

        elif input_mode == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload Raw Discharge CSV:",
                type=["csv"],
                help="Requires columns: Time_s, Voltage_V, Current_A, Temperature_C",
            )
            if uploaded_file is not None:
                try:
                    df_raw = pd.read_csv(uploaded_file)
                    req_cols = ["Time_s", "Voltage_V", "Current_A", "Temperature_C"]
                    missing = [c for c in req_cols if c not in df_raw.columns]
                    if missing:
                        alert_box(
                            f"Missing required columns in CSV: {', '.join(missing)}.",
                            alert_type="error",
                            title="CSV Header Validation Failed",
                        )
                        df_raw = None
                    else:
                        st.success(f"✓ Valid discharge profile uploaded ({len(df_raw)} samples).")
                        data_source_label = uploaded_file.name
                except Exception as e:
                    alert_box(f"CSV Parse error: {str(e)}", alert_type="error")
                    df_raw = None
            else:
                render_html("""
                <div style="border: 1px dashed rgba(255,255,255,0.1); border-radius: 8px; padding: 24px; text-align: center; color: #64748B; margin-top: 8px;">
                  <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #94A3B8; margin-bottom: 4px;">Awaiting Discharge CSV Telemetry</div>
                  <div style="font-size: 0.74rem;">Columns: Time_s, Voltage_V, Current_A, Temperature_C</div>
                </div>
                """)

        else:
            # Manual Slider Mode
            manual_features = {}
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                manual_features["T_40_38"] = st.slider("T_40_38 (s)", 50.0, 500.0, 300.0, 5.0)
                manual_features["T_38_36"] = st.slider("T_38_36 (s)", 250.0, 1050.0, 670.0, 10.0)
                manual_features["T_36_34"] = st.slider("T_36_34 (s)", 500.0, 1700.0, 1140.0, 10.0)
                manual_features["Q_40_36"] = st.slider("Q_40_36 (Ah)", 0.18, 0.85, 0.54, 0.01)
            with col_s2:
                manual_features["T_40_36"] = manual_features["T_40_38"] + manual_features["T_38_36"]
                manual_features["T_38_34"] = manual_features["T_38_36"] + manual_features["T_36_34"]
                manual_features["V_at_60s"] = st.slider("V_at_60s (V)", 3.75, 4.00, 3.90, 0.01)
                manual_features["V_at_300s"] = st.slider("V_at_300s (V)", 3.60, 3.90, 3.79, 0.01)
                manual_features["V_at_600s"] = st.slider("V_at_600s (V)", 3.45, 3.80, 3.68, 0.01)
                manual_features["V_at_1200s"] = st.slider("V_at_1200s (V)", 3.30, 3.70, 3.53, 0.01)

            manual_features["V_start"] = 3.98
            manual_features["V_mean_first600s"] = 3.80
            manual_features["V_slope_mid"] = -0.00032
            manual_features["T_start"] = 24.0
            manual_features["T_max"] = 39.5
            manual_features["T_rise"] = 15.5
            manual_features["T_mean"] = 32.3
            manual_features["R_dc_proxy"] = 0.105
            data_source_label = "Manual Sliders"

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        inferred_disc_idx = st.number_input(
            "Service Cycle Index (disc_idx):",
            min_value=0,
            max_value=250,
            value=inferred_disc_idx,
            step=1,
            help="Cycle count since start of battery operation.",
        )

        use_holdout_model = st.toggle(
            "Use Demo Holdout Model (Strictly Unseen B0018)",
            value=True,
            help="Pipeline trained on B0005, B0006, B0007 only. Ensures B0018 is completely unseen.",
        )

    # Right Column: Diagnostic Output Console
    with col_results:
        st.markdown("#### Diagnostic Telemetry Console")

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
                st.info("💡 The M1 capacity model requires the curve to traverse from 4.0 V down to 3.4 V.")
            else:
                if use_holdout_model:
                    result = predictor.predict_demo_unseen_b0018(extracted_feats, disc_idx=inferred_disc_idx)
                else:
                    result = predictor.predict_all(extracted_feats, disc_idx=inferred_disc_idx)

        elif manual_features is not None:
            if use_holdout_model:
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

            # Visual instrument row
            col_g, col_b = st.columns([1.2, 1])
            with col_g:
                tacho_svg = tachometer_soh_gauge_svg(pred_soh, capacity_ah=pred_cap, size=190)
                render_html(f'<div style="background: #0B0F19; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 6px; display: flex; align-items: center; justify-content: center;">{tacho_svg}</div>')
            with col_b:
                cell_svg = prismatic_battery_svg(pred_soh, width=110, height=185)
                render_html(f'<div style="background: #0B0F19; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 6px; display: flex; align-items: center; justify-content: center;">{cell_svg}</div>')

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

            # Precision KPI Readouts
            k1, k2 = st.columns(2)
            with k1:
                render_kpi("Discharged Capacity", f"{pred_cap:.3f}", "Ah", error_band=f"{cap_err:.3f}", subtext=f"Rated: 2.00 Ah ({pred_soh:.1f}% SOH)")
            with k2:
                if rul_data.get("is_eol") or (pred_cap <= 1.40):
                    render_kpi("Remaining Life", "EOL", "", subtext="Capacity &le; 1.40 Ah threshold", status="End of Life")
                else:
                    rul_val = rul_data.get("value", 0)
                    rul_err = rul_data.get("error_band", 15)
                    render_kpi("Remaining Life", f"{rul_val}", "cyc", error_band=f"{rul_err}", subtext=f"Retirement ~ cycle {inferred_disc_idx + rul_val}", status="Healthy" if rul_val > 30 else "Degrading")

            k3, k4 = st.columns(2)
            with k3:
                render_kpi("Ohmic Impedance (Re)", f"{re_val*1000:.1f}", "mΩ", error_band=f"{re_err*1000:.1f}", subtext="Electrolyte series resistance")
            with k4:
                render_kpi("Charge-Transfer (Rct)", f"{rct_val*1000:.1f}", "mΩ", error_band=f"{rct_err*1000:.1f}", subtext="SEI polarization impedance")

            # Diagnostic Status Banner
            if pred_cap <= 1.40:
                render_html("""
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 10px 14px; margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #FCA5A5;">
                  &bull; <strong>CRITICAL EOL STATE</strong>: 30% capacity fade threshold reached (&le; 1.40 Ah). Decommission pack immediately.
                </div>
                """)
            else:
                render_html(f"""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 8px 14px; margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: #6EE7B7;">
                  &bull; <strong>DIAGNOSTIC NOMINAL</strong>: Cell SOH is {pred_soh:.1f}% ({pred_cap:.3f} Ah). Estimated {rul_data.get('value')} operational cycles remaining.
                </div>
                """)

            # JSON Download
            export_payload = {
                "source": data_source_label,
                "cycle": inferred_disc_idx,
                "model_mode": "held_out_b0018" if use_holdout_model else "deployed_4_cells",
                "predictions": preds,
            }
            st.download_button(
                "📥 Export Diagnostic Telemetry (JSON)",
                data=json.dumps(export_payload, indent=2),
                file_name=f"cellguard_telemetry_cycle_{inferred_disc_idx}.json",
                mime="application/json",
            )

        elif df_raw is None and manual_features is None:
            render_html("""
            <div style="background: #0B0F19; border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 40px; text-align: center; color: #64748B;">
              <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94A3B8; margin-bottom: 4px;">Awaiting Telemetry Stream</div>
              <div style="font-size: 0.76rem;">Select a bundled NASA cycle on the left or upload a discharge profile.</div>
            </div>
            """)

    # Voltage Profile Bottom Plot
    if df_raw is not None:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        col_c_hdr, col_c_tog = st.columns([1.5, 1])
        with col_c_hdr:
            st.markdown("#### Terminal Voltage Profile & Partial Window Bounds")
        with col_c_tog:
            overlay_ref = st.checkbox("Overlay Cycle 5 Early Reference", value=True)

        ref_t, ref_v = (None, None)
        if overlay_ref:
            ref_t, ref_v = load_reference_curve()

        t = df_raw["Time_s"].values
        v = df_raw["Voltage_V"].values
        temp = df_raw["Temperature_C"].values if "Temperature_C" in df_raw.columns else None

        fig_curve = plot_discharge_curve(t, v, temp=temp, ref_t=ref_t, ref_v=ref_v)
        st.plotly_chart(fig_curve, use_container_width=True)
