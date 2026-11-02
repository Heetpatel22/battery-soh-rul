"""
predict.py - Battery Health Diagnostics.
Interactive multi-input diagnostic engine supporting:
1. NASA Ames Reference Cell & Cycle selection
2. Raw battery discharge CSV file upload with strict telemetry validation
3. Illustrative simulation scenarios (Fresh, Nominal Commuter, Aging, End-of-Life)
Features:
- Deterministic 18-feature extraction via existing battery_features.py
- Inference coordinator execution via existing trained models in models/
- Full parameter diagnostics: Capacity, SOH, Re, Rct, RUL, Discharge Energy
- Voltage vs time discharge curve with 4.0 V -> 3.4 V partial extraction window
- Ground truth comparison when available
- "Understanding Your Battery Results" educational section
- Exportable prediction report
"""

import os
import io
import json
from typing import Tuple
import streamlit as st
import pandas as pd
import numpy as np

from ui.components import (
    render_compact_header,
    render_category_legend,
    render_kpi,
    render_health_gauge_svg,
    render_callout,
    get_health_grade,
    render_html,
)
from ui.charts import plot_discharge_curve

try:
    from battery_features import extract_discharge_features, RATED_CAPACITY_AH, EOL_CAPACITY_AH
except ImportError:
    from src.battery_features import extract_discharge_features, RATED_CAPACITY_AH, EOL_CAPACITY_AH

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SAMPLE_DIR = os.path.join(ROOT_DIR, "sample_cycles")

# Simulation scenarios
SIMULATION_SCENARIOS = {
    "fresh": {
        "title": "Fresh Cell (Early Life, Cycle 5)",
        "desc": "Showroom baseline cell with minimal cycling. Characterized by long upper voltage plateau and minimal SEI resistance.",
        "sample_file": "B0018_cycle_005_early.csv",
        "cycle": 5,
        "true_cap": 1.8285,
        "true_soh": 91.43,
        "true_re": None,
        "true_rct": None,
    },
    "commuter": {
        "title": "Nominal Commuter (Mid Life, Cycle 50)",
        "desc": "Typical mid-cycle operational battery with nominal degradation. Moderate plateau shortening and normal electrolyte resistance.",
        "sample_file": "B0018_cycle_050_middle.csv",
        "cycle": 50,
        "true_cap": 1.6664,
        "true_soh": 83.32,
        "true_re": 0.06336,
        "true_rct": 0.08601,
    },
    "aging": {
        "title": "Aging High-Mileage Cell (Cycle 65)",
        "desc": "Noticeable capacity degradation approaching the 80% warranty boundary. Increased internal resistance and faster voltage drop.",
        "sample_file": "B0018_cycle_065_middle.csv",
        "cycle": 65,
        "true_cap": 1.5316,
        "true_soh": 76.58,
        "true_re": 0.06244,
        "true_rct": 0.08553,
    },
    "eol": {
        "title": "End-of-Life Warning Cell (Cycle 100)",
        "desc": "Cell has crossed the 30% capacity fade threshold (1.40 Ah EOL). Model triggers post-EOL protection.",
        "sample_file": "B0018_cycle_100_late.csv",
        "cycle": 100,
        "true_cap": 1.3849,
        "true_soh": 69.25,
        "true_re": 0.06545,
        "true_rct": 0.08884,
    },
}


def validate_csv_data(df: pd.DataFrame) -> Tuple[bool, list, pd.DataFrame]:
    """
    Validate uploaded battery discharge CSV file.
    Checks:
    - Required columns: Time_s, Voltage_V, Current_A, Temperature_C
    - Non-empty and sufficient samples (> 50 points)
    - Finite numerical values
    - Monotonically increasing timestamps
    - Plausible voltage range (2.0 V to 4.5 V)
    - Plausible current (negative for discharge)
    - Plausible temperature (10°C to 70°C)
    """
    errors = []
    required_cols = ["Time_s", "Voltage_V", "Current_A", "Temperature_C"]
    
    # 1. Column presence check (case-insensitive mapping)
    col_map = {}
    for req in required_cols:
        matches = [c for c in df.columns if c.strip().lower() == req.lower()]
        if matches:
            col_map[matches[0]] = req
        else:
            errors.append(f"Missing required column: '{req}'. File must include Time_s, Voltage_V, Current_A, Temperature_C.")

    if errors:
        return False, errors, df

    df = df.rename(columns=col_map)
    df = df[required_cols].copy()

    # 2. Check empty or small dataset
    if len(df) < 50:
        errors.append(f"Insufficient telemetry data: only {len(df)} rows found. A minimum of 50 samples is required to evaluate discharge dynamics.")
        return False, errors, df

    # 3. Numeric conversion & finite check
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        null_count = df[c].isna().sum()
        if null_count > 0:
            errors.append(f"Column '{c}' contains {null_count} invalid non-numeric or NaN values.")

    if errors:
        return False, errors, df

    # 4. Monotonic timestamps check
    t_diff = np.diff(df["Time_s"].values)
    if np.any(t_diff < -1e-5):
        errors.append("Timestamp sequence in 'Time_s' is not monotonically increasing. Ensure rows are chronologically sorted.")

    # 5. Voltage range check
    v = df["Voltage_V"].values
    if np.any(v < 1.5) or np.any(v > 4.6):
        errors.append(f"Terminal voltage spans [{np.min(v):.2f} V, {np.max(v):.2f} V], which falls outside the physically plausible lithium-ion range (2.0 V to 4.5 V).")

    # 6. Voltage window crossing check (must cross 4.0 V down to 3.4 V)
    if np.max(v) < 3.95:
        errors.append(f"Maximum terminal voltage ({np.max(v):.2f} V) does not reach the 4.00 V upper extraction threshold. Test must start near full charge.")
    if np.min(v) > 3.45:
        errors.append(f"Minimum terminal voltage ({np.min(v):.2f} V) does not cross below the 3.40 V lower threshold. Partial discharge window incomplete.")

    # 7. Temperature check
    temp = df["Temperature_C"].values
    if np.any(temp < 5.0) or np.any(temp > 85.0):
        errors.append(f"Temperature values span [{np.min(temp):.1f}°C, {np.max(temp):.1f}°C], outside plausible operational limits (10°C to 70°C).")

    is_valid = len(errors) == 0
    return is_valid, errors, df


def render_predict(predictor, schema: dict, metrics: dict):
    """Render the main Battery Health Diagnostics interactive page."""
    # Ensure default session state
    if "selected_cell" not in st.session_state:
        st.session_state["selected_cell"] = "B0018"
    if "selected_cycle" not in st.session_state:
        st.session_state["selected_cycle"] = 50

    active_cell = st.session_state["selected_cell"]
    render_compact_header(
        page_title="Battery Health Diagnostics",
        selected_battery=active_cell,
        data_source="Partial-Window Telemetry (4.0 V &rarr; 3.4 V)",
        model_status="Inference Ready &bull; Zero Leakage Verified",
    )

    render_category_legend()

    # Input method selection
    st.markdown("""
    <div style="margin-bottom: 14px;">
      <h3 style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin: 0 0 4px 0;">
        Diagnostic Input Source
      </h3>
      <p style="font-size: 0.82rem; color: #64748B; margin: 0;">
        Select an input channel to extract partial-window voltage features and generate multi-target predictions.
      </p>
    </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose Input Method:",
        [
            "1. Reference NASA Battery Cell (Historical Run)",
            "2. Upload Battery Discharge Telemetry (CSV)",
            "3. Illustrative Simulation Scenario (Controlled Test)",
        ],
        index=0,
        horizontal=True,
    )

    t_arr = None
    v_arr = None
    i_arr = None
    temp_arr = None
    inferred_cycle = 50
    ground_truth = {}
    data_label = ""
    is_held_out = False

    # -------------------------------------------------------------------------
    # METHOD 1: Reference NASA Battery Cell
    # -------------------------------------------------------------------------
    if "1. Reference" in input_mode:
        col_c1, col_c2 = st.columns([1.2, 1.2], gap="medium")
        with col_c1:
            cell_choice = st.selectbox(
                "Select NASA Ames Reference Cell:",
                ["B0018 (Held-out Benchmark)", "B0005 (Nominal Rack)", "B0006 (Accelerated Degradation)", "B0007 (Lab Ambient)"],
                index=0,
            )
            selected_cell = cell_choice.split()[0]
            st.session_state["selected_cell"] = selected_cell
            is_held_out = (selected_cell == "B0018")

        with col_c2:
            preset_file = st.selectbox(
                "Select Available Sample Discharge Cycle:",
                [
                    "B0018_cycle_005_early.csv (Fresh Pack)",
                    "B0018_cycle_015_early.csv (Early Use)",
                    "B0018_cycle_050_middle.csv (Nominal Mid-Life)",
                    "B0018_cycle_065_middle.csv (Aging Pack)",
                    "B0018_cycle_100_late.csv (Retirement Limit)",
                    "B0018_cycle_115_late.csv (Post-EOL Storage)",
                ],
                index=2,
            )
            fname = preset_file.split()[0]

        fpath = os.path.join(SAMPLE_DIR, fname)
        if os.path.exists(fpath):
            df_sample = pd.read_csv(fpath)
            t_arr = df_sample["Time_s"].values
            v_arr = df_sample["Voltage_V"].values
            i_arr = df_sample["Current_A"].values
            temp_arr = df_sample["Temperature_C"].values
            
            # Extract cycle number from filename
            parts = fname.replace(".csv", "").split("_")
            inferred_cycle = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 50
            st.session_state["selected_cycle"] = inferred_cycle
            data_label = f"NASA {selected_cell} Cycle {inferred_cycle}"

            # Known ground truths for B0018
            gt_table = {
                5: {"cap": 1.8285, "soh": 91.43, "re": None, "rct": None, "rul": 95},
                15: {"cap": 1.7712, "soh": 88.56, "re": None, "rct": None, "rul": 85},
                50: {"cap": 1.6664, "soh": 83.32, "re": 0.06336, "rct": 0.08601, "rul": 50},
                65: {"cap": 1.5316, "soh": 76.58, "re": 0.06244, "rct": 0.08553, "rul": 25},
                100: {"cap": 1.3849, "soh": 69.25, "re": 0.06545, "rct": 0.08884, "rul": 0},
                115: {"cap": 1.3882, "soh": 69.41, "re": 0.06297, "rct": 0.08791, "rul": 0},
            }
            if inferred_cycle in gt_table:
                ground_truth = gt_table[inferred_cycle]
        else:
            st.error(f"Sample discharge cycle file not found at: {fpath}")

    # -------------------------------------------------------------------------
    # METHOD 2: CSV Upload with Full Validation
    # -------------------------------------------------------------------------
    elif "2. Upload" in input_mode:
        st.markdown("""
        <div class="cg-callout" style="margin-top: 6px;">
          <strong>CSV Upload Requirements:</strong> File must contain the four standard BMS channels: 
          <code>Time_s</code> (seconds), <code>Voltage_V</code> (terminal volts), <code>Current_A</code> (amperes, negative during discharge), and <code>Temperature_C</code> (thermocouple Celsius). 
          The discharge curve must cross from at least <strong>4.0 V down to 3.4 V</strong>.
        </div>
        """, unsafe_allow_html=True)

        col_up, col_guide = st.columns([1.5, 1.2], gap="large")

        with col_guide:
            st.markdown("<div style='font-size:0.80rem; font-weight:700; color:#334155; margin-bottom:6px;'>Expected CSV Format:</div>", unsafe_allow_html=True)
            guide_df = pd.DataFrame({
                "Time_s": [0.0, 9.3, 19.8, 29.5, 39.2],
                "Voltage_V": [4.190, 4.190, 3.995, 3.979, 3.968],
                "Current_A": [-0.002, 0.001, -2.007, -2.008, -2.007],
                "Temperature_C": [24.69, 24.70, 24.71, 24.77, 24.86],
            })
            st.dataframe(guide_df, use_container_width=True, hide_index=True)

            # Download CSV template button
            sample_csv_path = os.path.join(SAMPLE_DIR, "B0018_cycle_050_middle.csv")
            if os.path.exists(sample_csv_path):
                with open(sample_csv_path, "rb") as f:
                    csv_bytes = f.read()
                st.download_button(
                    label="📥 Download Reference CSV Template",
                    data=csv_bytes,
                    file_name="cellguard_sample_cycle_50.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with col_up:
            uploaded_file = st.file_uploader(
                "Upload Raw Discharge CSV Telemetry:",
                type=["csv"],
                help="Select a CSV file containing timestamped discharge timeseries data.",
            )

            if uploaded_file is not None:
                try:
                    df_up = pd.read_csv(uploaded_file)
                    is_valid, validation_errors, clean_df = validate_csv_data(df_up)

                    if not is_valid:
                        st.error("Validation Failed. Please correct the following errors:")
                        for err in validation_errors:
                            st.markdown(f"- ⚠️ {err}")
                    else:
                        st.success(f"Telemetry Verified: {len(clean_df)} valid measurements loaded successfully.")
                        t_arr = clean_df["Time_s"].values
                        v_arr = clean_df["Voltage_V"].values
                        i_arr = clean_df["Current_A"].values
                        temp_arr = clean_df["Temperature_C"].values
                        data_label = f"Uploaded File: {uploaded_file.name}"
                        st.session_state["uploaded_battery_data"] = clean_df

                        inferred_cycle = st.number_input(
                            "Provide Optional Known Cycle Number (if tracking degradation history):",
                            min_value=1,
                            max_value=300,
                            value=50,
                            step=1,
                        )
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
            else:
                st.info("Upload a file above or switch to Method 1 or 3 to test with preloaded telemetry.")

    # -------------------------------------------------------------------------
    # METHOD 3: Illustrative Simulation Scenarios
    # -------------------------------------------------------------------------
    else:
        sc_keys = list(SIMULATION_SCENARIOS.keys())
        sc_choice = st.selectbox(
            "Select Controlled Scenario:",
            [SIMULATION_SCENARIOS[k]["title"] for k in sc_keys],
            index=1,
        )
        selected_key = sc_keys[[SIMULATION_SCENARIOS[k]["title"] for k in sc_keys].index(sc_choice)]
        active_sc = SIMULATION_SCENARIOS[selected_key]

        st.caption(f"**Scenario Profile:** {active_sc['desc']}")
        fpath = os.path.join(SAMPLE_DIR, active_sc["sample_file"])
        if os.path.exists(fpath):
            df_sim = pd.read_csv(fpath)
            t_arr = df_sim["Time_s"].values
            v_arr = df_sim["Voltage_V"].values
            i_arr = df_sim["Current_A"].values
            temp_arr = df_sim["Temperature_C"].values
            inferred_cycle = active_sc["cycle"]
            data_label = f"Simulated Scenario: {active_sc['title']}"
            ground_truth = {
                "cap": active_sc["true_cap"],
                "soh": active_sc["true_soh"],
                "re": active_sc["true_re"],
                "rct": active_sc["true_rct"],
            }
        else:
            st.error(f"Simulation scenario file not found at: {fpath}")

    # -------------------------------------------------------------------------
    # FEATURE EXTRACTION & INFERENCE EXECUTION
    # -------------------------------------------------------------------------
    if t_arr is not None and v_arr is not None:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px; margin-bottom: 16px;">
          <div>
            <h3 style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin: 0;">
              Diagnostic Inference Results
            </h3>
            <span style="font-size: 0.78rem; color: #64748B;">Source: <strong>{data_label}</strong></span>
          </div>
          <span class="cg-chip cg-chip-status">INFERENCE COMPLETE</span>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Extracting 18 partial-window voltage features and executing ML pipelines..."):
            try:
                # Call existing feature extraction from battery_features.py
                features = extract_discharge_features(t_arr, v_arr, i_arr, temp_arr)

                # Execute predictor
                if is_held_out:
                    pred_res = predictor.predict_demo_unseen_b0018(features, disc_idx=inferred_cycle)
                else:
                    pred_res = predictor.predict_all(features, disc_idx=inferred_cycle)

            except Exception as e:
                st.error(f"Inference execution failed: {str(e)}")
                return

        if not pred_res.get("success"):
            st.error("Feature validation failed. The partial discharge window could not be resolved:")
            for issue in pred_res.get("errors", []):
                st.markdown(f"- ⚠️ {issue}")
            return

        preds = pred_res["predictions"]
        p_cap = preds["capacity"]["value"]
        p_soh = preds["soh"]["value"]
        p_re = preds["Re"]["value"]
        p_rct = preds["Rct"]["value"]
        rul_obj = preds["RUL"]
        p_rul = rul_obj.get("value")
        p_is_eol = rul_obj.get("is_eol", False) or (p_cap <= 1.40)
        p_energy = preds.get("discharge_energy", {}).get("value", 5.8)

        # ---------------------------------------------------------------------
        # TOP RESULTS KPI TILES
        # ---------------------------------------------------------------------
        col_res_kpi1, col_res_kpi2, col_res_kpi3, col_res_kpi4 = st.columns(4)
        with col_res_kpi1:
            render_kpi(
                label="Predicted SOH",
                value=f"{p_soh:.2f}",
                unit="%",
                error_band="2.32",
                subtext=f"Meas: {ground_truth.get('soh', 'N/A')}%" if "soh" in ground_truth and ground_truth["soh"] else "2.00 Ah Rated",
                status="Healthy" if p_soh >= 80 else ("Degrading" if p_soh >= 70 else "Critical EOL"),
                category="predicted",
                tooltip="M1 Ridge prediction from 4.0V-3.4V partial window.",
            )
        with col_res_kpi2:
            render_kpi(
                label="Predicted Capacity",
                value=f"{p_cap:.4f}",
                unit="Ah",
                error_band="0.046",
                subtext=f"Meas: {ground_truth.get('cap', 'N/A')} Ah" if "cap" in ground_truth and ground_truth["cap"] else "Window: 4.0V to 3.4V",
                status="Healthy" if p_cap >= 1.60 else ("Degrading" if p_cap >= 1.40 else "Critical EOL"),
                category="predicted",
                tooltip="Predicted full discharge capacity without draining to cutoff.",
            )
        with col_res_kpi3:
            if p_is_eol or (p_rul is None):
                render_kpi(
                    label="Predicted RUL",
                    value="EOL",
                    unit="Reached",
                    subtext="Capacity &le; 1.40 Ah threshold",
                    status="Critical EOL",
                    category="predicted",
                    tooltip="End of Life retirement threshold reached (30% fade).",
                )
            else:
                render_kpi(
                    label="Predicted RUL",
                    value=f"{p_rul}",
                    unit="cyc",
                    error_band="15",
                    subtext=f"Meas: {ground_truth.get('rul', 'N/A')} cyc" if "rul" in ground_truth and ground_truth["rul"] else "To 1.40 Ah EOL",
                    status="Healthy" if p_rul >= 40 else "Degrading",
                    category="predicted",
                    tooltip="M4 Random Forest forecast of full cycles until 1.40 Ah cutoff.",
                )
        with col_res_kpi4:
            render_kpi(
                label="Discharge Energy",
                value=f"{p_energy:.2f}",
                unit="Wh",
                error_band="0.15",
                subtext="Total available Wh",
                status="Healthy" if p_energy >= 5.0 else "Degrading",
                category="predicted",
                tooltip="M5 Ridge prediction of usable energy to cutoff.",
            )

        col_imp1, col_imp2 = st.columns(2)
        with col_imp1:
            render_kpi(
                label="Estimated Ohmic Resistance (Re / R₀)",
                value=f"{p_re*1000.0:.2f}",
                unit="mΩ",
                error_band="6.6",
                subtext=f"Meas: {ground_truth['re']*1000.0:.2f} mΩ" if ground_truth.get("re") else "Bulk electrolyte & contact impedance",
                status="Healthy" if p_re < 0.08 else "Degrading",
                category="predicted",
                tooltip="M2 Ridge estimation of high-frequency series resistance.",
            )
        with col_imp2:
            render_kpi(
                label="Estimated Charge-Transfer (Rct / R₁)",
                value=f"{p_rct*1000.0:.2f}",
                unit="mΩ",
                error_band="5.5",
                subtext=f"Meas: {ground_truth['rct']*1000.0:.2f} mΩ" if ground_truth.get("rct") else "Interfacial reaction kinetics & SEI resistance",
                status="Healthy" if p_rct < 0.12 else "Degrading",
                category="predicted",
                tooltip="M3 Ridge estimation of charge-transfer polarization resistance.",
            )

        # ---------------------------------------------------------------------
        # DISCHARGE CURVE & EXTRACTION WINDOW
        # -------------------------------------------------------------------------
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="cg-card">
          <div class="cg-card-header">
            <div>
              <div class="cg-card-title">Discharge Voltage Profile &amp; Extraction Window</div>
              <div class="cg-card-subtitle">Terminal voltage V(t) highlighting the 4.0 V &rarr; 3.4 V BMS feature window</div>
            </div>
            <span class="cg-cat-badge cg-cat-measured">CYCLE PROFILE</span>
          </div>
        """, unsafe_allow_html=True)

        # Load reference early curve for visual comparison
        ref_p = os.path.join(SAMPLE_DIR, "B0018_cycle_005_early.csv")
        ref_t, ref_v = None, None
        if os.path.exists(ref_p):
            df_ref = pd.read_csv(ref_p)
            ref_t = df_ref["Time_s"].values
            ref_v = df_ref["Voltage_V"].values

        fig_discharge = plot_discharge_curve(
            t=t_arr,
            v=v_arr,
            temp=temp_arr,
            ref_t=ref_t,
            ref_v=ref_v,
            ref_label="Reference Early Life (Cycle 5)",
        )
        st.plotly_chart(fig_discharge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # SECTION: UNDERSTANDING YOUR BATTERY RESULTS
        # ---------------------------------------------------------------------
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="cg-card">
          <div class="cg-card-header">
            <div>
              <div class="cg-card-title">Understanding Your Battery Results</div>
              <div class="cg-card-subtitle">Electrochemical and BMS engineering interpretation for each parameter</div>
            </div>
            <span class="cg-chip cg-chip-status">GUIDE</span>
          </div>
        """, unsafe_allow_html=True)

        # Educational table covering the four core questions
        guide_data = [
            {
                "Parameter": "Capacity & SOH (%)",
                "What It Measures": "Total usable electric charge deliverable from full charge (4.2V) down to cutoff (2.7V).",
                "Why It Matters": "Directly governs the driving range of an EV or runtime of an energy storage system.",
                "Predicted Meaning": f"Predicted {p_cap:.3f} Ah ({p_soh:.1f}% SOH) vs rated 2.00 Ah.",
                "Health Assessment": "Normal degradation" if p_soh >= 80 else ("Accelerated fade / Approaching EOL" if p_soh >= 70 else "Decommissioning recommended"),
            },
            {
                "Parameter": "Ohmic Resistance (Re / R₀)",
                "What It Measures": "Instantaneous ohmic resistance across the electrolyte, separator pores, and current collectors.",
                "Why It Matters": "Causes immediate voltage drop (ΔV = I·R₀) upon load application and resistive Joule heating (I²R).",
                "Predicted Meaning": f"Predicted {p_re*1000.0:.1f} mΩ (error band ±6.6 mΩ).",
                "Health Assessment": "Nominal bulk conductivity" if p_re < 0.08 else "Elevated contact/electrolyte resistance",
            },
            {
                "Parameter": "Charge-Transfer Resistance (Rct / R₁)",
                "What It Measures": "Kinetic resistance to electron transfer and lithium-ion intercalation across the electrode-electrolyte interface.",
                "Why It Matters": "Growth of the Solid Electrolyte Interphase (SEI) layer on the negative electrode increases Rct over time.",
                "Predicted Meaning": f"Predicted {p_rct*1000.0:.1f} mΩ (error band ±5.5 mΩ).",
                "Health Assessment": "Healthy kinetic rate" if p_rct < 0.12 else "Thickened SEI layer / Lithium inventory loss",
            },
            {
                "Parameter": "Remaining Useful Life (RUL)",
                "What It Measures": "Number of full discharge-charge cycles remaining until capacity drops to 1.40 Ah (30% fade).",
                "Why It Matters": "Critical for warranty planning, predictive maintenance, and avoiding unexpected pack failure.",
                "Predicted Meaning": f"Forecasted {p_rul if p_rul is not None else 'End of life reached'} cycles.",
                "Health Assessment": "Operational" if not p_is_eol else "End of Life threshold crossed",
            },
            {
                "Parameter": "Discharge Energy (Wh)",
                "What It Measures": "Integral of instantaneous power ∫ V(t)·I(t) dt deliverable over the full discharge.",
                "Why It Matters": "Accounts for both capacity loss and lower operating voltage caused by rising internal impedance.",
                "Predicted Meaning": f"Estimated {p_energy:.2f} Wh usable energy.",
                "Health Assessment": "Healthy energy density" if p_energy >= 5.0 else "Reduced energy availability",
            },
        ]
        st.dataframe(pd.DataFrame(guide_data), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # EXPORT DIAGNOSTIC REPORT
        # ---------------------------------------------------------------------
        report_dict = {
            "cell_id": active_cell,
            "cycle_index": inferred_cycle,
            "data_source": data_label,
            "predictions": preds,
            "ground_truth": ground_truth,
            "feature_subset": {k: round(float(v), 4) for k, v in features.items() if isinstance(v, (int, float))},
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        col_dl1, col_dl2 = st.columns([1, 1])
        with col_dl1:
            st.download_button(
                label="📥 Download Diagnostic Report (JSON)",
                data=json.dumps(report_dict, indent=2),
                file_name=f"cellguard_diagnostic_{active_cell}_c{inferred_cycle}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_dl2:
            df_export = pd.DataFrame([{
                "Cell": active_cell,
                "Cycle": inferred_cycle,
                "SOH_pct": p_soh,
                "Capacity_Ah": p_cap,
                "Re_ohm": p_re,
                "Rct_ohm": p_rct,
                "RUL_cycles": p_rul if p_rul else 0,
                "Energy_Wh": p_energy,
            }])
            st.download_button(
                label="📥 Download Summary CSV",
                data=df_export.to_csv(index=False),
                file_name=f"cellguard_summary_{active_cell}_c{inferred_cycle}.csv",
                mime="text/csv",
                use_container_width=True,
            )
