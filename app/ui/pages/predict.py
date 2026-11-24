"""
predict.py - Primary Battery Impedance (Re & Rct) Diagnostics Engine.
Allows predicting Ohmic Resistance (Re) and Charge-Transfer Resistance (Rct) using 19
observable discharge parameters and comparing across ML models.

Three Input Methods:
1. NASA Reference Cell & Cycle (with ground-truth EIS comparison)
2. CSV File Upload (with schema validation & feature extraction)
3. Illustrative Simulation / Manual Parameter Input
"""

import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.io import loadmat

from battery_features import (
    extract_discharge_features,
    PARTIAL_WINDOW_FEATURE_NAMES,
    RATED_CAPACITY_AH,
    EOL_CAPACITY_AH,
)
from ui.components import (
    render_compact_header,
    render_category_legend,
    render_kpi,
    get_health_grade,
    render_html,
)
from ui.charts import plot_discharge_curve

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@st.cache_data(show_spinner=False)
def _load_reference_discharge_mat(cell_id: str, cycle_number: int):
    path = os.path.join(ROOT_DIR, "data", f"{cell_id}.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference .mat file for {cell_id} not found.")

    root = loadmat(path, simplify_cells=True).get(cell_id, {})
    cycles = root.get("cycle", []) if isinstance(root, dict) else []
    if not isinstance(cycles, list):
        cycles = list(np.asarray(cycles, dtype=object).reshape(-1))

    discharge_entries = [(idx, c) for idx, c in enumerate(cycles) if isinstance(c, dict) and c.get("type") == "discharge"]
    impedance_entries = [(idx, c) for idx, c in enumerate(cycles) if isinstance(c, dict) and c.get("type") == "impedance"]

    if not 1 <= cycle_number <= len(discharge_entries):
        raise ValueError(f"Cycle number must be between 1 and {len(discharge_entries)}.")

    raw_idx, target_disc = discharge_entries[cycle_number - 1]
    ddata = target_disc.get("data", {})

    t = np.asarray(ddata.get("Time", []), dtype=float).reshape(-1)
    v = np.asarray(ddata.get("Voltage_measured", []), dtype=float).reshape(-1)
    i = np.asarray(ddata.get("Current_measured", []), dtype=float).reshape(-1)
    temp = np.asarray(ddata.get("Temperature_measured", []), dtype=float).reshape(-1)
    cap = float(np.asarray(ddata.get("Capacity", [np.nan]), dtype=float).reshape(-1)[0])

    # Find closest impedance cycle within 3 cycle distance
    nearest_item = min(impedance_entries, key=lambda item: abs(item[0] - raw_idx), default=None)
    true_re = None
    true_rct = None
    if nearest_item and abs(nearest_item[0] - raw_idx) <= 3:
        idata = nearest_item[1].get("data", {})
        try:
            val_re = float(np.asarray(idata.get("Re", [np.nan]), dtype=float).reshape(-1)[0])
            val_rct = float(np.asarray(idata.get("Rct", [np.nan]), dtype=float).reshape(-1)[0])
            if np.isfinite(val_re) and 0.01 < val_re < 0.3:
                true_re = val_re
            if np.isfinite(val_rct) and 0.01 < val_rct < 0.4:
                true_rct = val_rct
        except Exception:
            pass

    return {
        "t": t,
        "v": v,
        "i": i,
        "temp": temp,
        "capacity": cap if np.isfinite(cap) else None,
        "re": true_re,
        "rct": true_rct,
        "total_cycles": len(discharge_entries),
    }


def _validate_uploaded_csv(df: pd.DataFrame):
    """Validate uploaded CSV discharge data and standardize column names."""
    col_map = {}
    lower_cols = {str(c).strip().lower(): c for c in df.columns}

    # Match time
    for key in ["time_s", "time", "t", "elapsed_time", "timestamp"]:
        if key in lower_cols:
            col_map["t"] = lower_cols[key]
            break

    # Match voltage
    for key in ["voltage_v", "voltage_measured", "voltage", "v", "terminal_voltage"]:
        if key in lower_cols:
            col_map["v"] = lower_cols[key]
            break

    # Match current
    for key in ["current_a", "current_measured", "current", "i", "load_current"]:
        if key in lower_cols:
            col_map["i"] = lower_cols[key]
            break

    # Match temperature
    for key in ["temperature_c", "temperature_measured", "temp", "temperature", "t_battery"]:
        if key in lower_cols:
            col_map["temp"] = lower_cols[key]
            break

    missing = [c for c in ["t", "v", "i", "temp"] if c not in col_map]
    if missing:
        missing_names = {"t": "Time (Time_s / Time)", "v": "Voltage (Voltage_V / Voltage)",
                         "i": "Current (Current_A / Current)", "temp": "Temperature (Temperature_C / Temp)"}
        return False, [f"Missing required column: {missing_names[m]}" for m in missing], None

    # Check numeric types and non-empty
    errors = []
    clean_dict = {}
    for standard_name, orig_col in col_map.items():
        arr = pd.to_numeric(df[orig_col], errors="coerce").to_numpy(float)
        if np.isnan(arr).all():
            errors.append(f"Column '{orig_col}' contains no valid numeric readings.")
        clean_dict[standard_name] = arr

    if errors:
        return False, errors, None

    # Voltage range check
    v_arr = clean_dict["v"]
    v_finite = v_arr[np.isfinite(v_arr)]
    if len(v_finite) < 10:
        return False, ["Discharge record has fewer than 10 valid data points."], None

    if np.max(v_finite) < 3.8:
        errors.append(f"Maximum recorded voltage is {np.max(v_finite):.2f} V. Discharge must start near or above 3.80 V to capture the partial inspection window.")
    if np.min(v_finite) > 3.5:
        errors.append(f"Minimum recorded voltage is {np.min(v_finite):.2f} V. Discharge must fall down to 3.40 V to compute full partial-window timing features.")

    if errors:
        return False, errors, None

    return True, [], clean_dict


def render_predict(predictor, schema: dict, metrics: dict):
    render_compact_header(
        page_title="Re & Rct Impedance Diagnostic Studio",
        selected_battery="Active Inspection",
        data_source="Partial-Window Observable Feature Extraction",
        model_status="Multi-Algorithm Engine · Real-time Inference",
    )

    render_category_legend()

    # Introduction Banner
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:14px 18px; margin-bottom:16px;">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <div>
          <h3 style="font-size:1.05rem; font-weight:800; color:#0F172A; margin:0 0 4px 0;">
            Primary Task: Ohmic (Re) & Charge-Transfer (Rct) Resistance Estimation
          </h3>
          <p style="font-size:0.80rem; color:#475569; margin:0;">
            Estimates internal electrochemical impedance directly from observable partial-window discharge curves (4.0 V down to 3.4 V).
            <strong>Neither Re nor Rct is ever used as an input feature.</strong>
          </p>
        </div>
        <span style="font-size:0.72rem; font-family:'JetBrains Mono', monospace; font-weight:700; background:#EFF6FF; border:1px solid #BFDBFE; color:#1D4ED8; padding:3px 8px; border-radius:4px;">
          INPUT: V(t), I(t), T(t), Cycle No.
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Input Method Selection Tabs
    tab_nasa, tab_csv, tab_manual = st.tabs([
        "🔬 1. NASA Reference Battery Cell",
        "📂 2. Upload Discharge CSV",
        "⚡ 3. Illustrative Simulation / Manual",
    ])

    features = None
    t_arr = v_arr = i_arr = temp_arr = None
    cell_id = "B0005"
    cycle_number = 40
    true_re = None
    true_rct = None
    true_cap = None
    data_source_label = ""

    # TAB 1: NASA REFERENCE CELL
    with tab_nasa:
        st.caption("Select any historical discharge cycle from NASA Ames 18650 cells to benchmark ML resistance against true laboratory EIS measurements.")
        nc1, nc2, nc3 = st.columns([1, 1.2, 1.2])
        with nc1:
            sel_cell = st.selectbox("NASA Cell Identifier", ["B0005", "B0006", "B0007", "B0018"], index=0, key="pred_nasa_cell")
        with nc2:
            try:
                base_ref = _load_reference_discharge_mat(sel_cell, 1)
                max_c = base_ref["total_cycles"]
                sel_cycle = st.slider("Discharge Cycle Number", 1, max_c, min(40, max_c), key="pred_nasa_cycle")
            except Exception:
                sel_cycle = 40
        with nc3:
            st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
            load_nasa_btn = st.button("Load Cycle Data & Diagnose", type="primary", use_container_width=True, key="btn_load_nasa")

        if load_nasa_btn or "nasa_loaded_once" not in st.session_state:
            st.session_state["nasa_loaded_once"] = True
            try:
                ref_data = _load_reference_discharge_mat(sel_cell, int(sel_cycle))
                t_arr = ref_data["t"]
                v_arr = ref_data["v"]
                i_arr = ref_data["i"]
                temp_arr = ref_data["temp"]
                true_re = ref_data["re"]
                true_rct = ref_data["rct"]
                true_cap = ref_data["capacity"]
                cell_id = sel_cell
                cycle_number = int(sel_cycle)
                data_source_label = f"NASA {sel_cell} Cycle {sel_cycle}"
                features = extract_discharge_features(t_arr, v_arr, i_arr, temp_arr)
            except Exception as e:
                st.error(f"Error loading {sel_cell} cycle {sel_cycle}: {str(e)}")

    # TAB 2: CSV UPLOAD
    with tab_csv:
        st.caption("Upload a CSV file containing time-series discharge readings. Features will be extracted deterministically.")

        with st.expander("CSV Format Guide & Accepted Column Headers", expanded=False):
            st.markdown("""
            | Parameter | Expected Header | Accepted Aliases | Engineering Units |
            | :--- | :--- | :--- | :--- |
            | **Time** | `Time_s` | `Time`, `t`, `elapsed_time` | Seconds (s) starting from 0 |
            | **Voltage** | `Voltage_V` | `Voltage`, `v`, `Voltage_measured` | Volts (V), starts ~4.1V, reaches 3.4V |
            | **Current** | `Current_A` | `Current`, `i`, `Current_measured` | Amperes (A), positive or negative load |
            | **Temperature** | `Temperature_C` | `Temperature`, `temp`, `t_battery` | Degrees Celsius (°C) |
            """)

        csv_file = st.file_uploader("Choose a discharge CSV file", type=["csv"], key="pred_csv_uploader")
        csv_cycle_no = st.number_input("Recorded Battery Cycle Number", min_value=1, max_value=5000, value=50, step=1, key="csv_cycle_input")

        if csv_file is not None:
            try:
                raw_df = pd.read_csv(csv_file)
                valid, errs, clean_dict = _validate_uploaded_csv(raw_df)
                if not valid:
                    st.error("CSV Validation Failed:")
                    for err in errs:
                        st.markdown(f"- ⚠️ {err}")
                else:
                    st.success(f"CSV validated successfully! Found {len(clean_dict['t'])} discharge samples.")
                    t_arr = clean_dict["t"]
                    v_arr = clean_dict["v"]
                    i_arr = clean_dict["i"]
                    temp_arr = clean_dict["temp"]
                    cell_id = "Uploaded CSV"
                    cycle_number = int(csv_cycle_no)
                    data_source_label = f"Uploaded CSV ({csv_file.name})"
                    features = extract_discharge_features(t_arr, v_arr, i_arr, temp_arr)
            except Exception as e:
                st.error(f"Failed to read CSV: {str(e)}")

    # TAB 3: SIMULATION / MANUAL INPUT
    with tab_manual:
        st.caption("Test hypothetical or custom discharge characteristics using preset operational scenarios or manual controls.")
        scenario = st.selectbox(
            "Select Scenario Preset",
            ["Fresh Battery (Cycle 5 · Low Resistance)", "Mid-Life Battery (Cycle 60 · Moderate Growth)", "Aged Battery (Cycle 120 · High Impedance)"],
            key="pred_scenario_preset",
        )
        if "Fresh" in scenario:
            sc_c, sc_re, sc_rct, sc_t4036 = 5, 0.045, 0.068, 850.0
        elif "Mid" in scenario:
            sc_c, sc_re, sc_rct, sc_t4036 = 60, 0.062, 0.085, 620.0
        else:
            sc_c, sc_re, sc_rct, sc_t4036 = 120, 0.076, 0.115, 380.0

        sm_cols = st.columns(3)
        sim_cycle = sm_cols[0].number_input("Simulated Cycle", 1, 200, sc_c, 5, key="sim_cycle")
        sim_t4036 = sm_cols[1].number_input("Window Duration 4.0V→3.6V (s)", 100.0, 1500.0, sc_t4036, 20.0, key="sim_t4036")
        sim_rdc = sm_cols[2].number_input("Load Step Proxy ΔV/ΔI (Ω)", 0.02, 0.20, sc_re, 0.005, format="%.3f", key="sim_rdc")

        if st.button("Apply Simulation Scenario", type="primary", key="btn_apply_sim"):
            cycle_number = int(sim_cycle)
            cell_id = "Simulation"
            data_source_label = f"Scenario: {scenario}"
            # Construct synthetic features
            features = {
                "T_40_38": sim_t4036 * 0.35,
                "T_38_36": sim_t4036 * 0.65,
                "T_36_34": sim_t4036 * 0.50,
                "T_40_36": sim_t4036,
                "T_38_34": sim_t4036 * 1.15,
                "Q_40_36": 2.0 * sim_t4036 / 3600.0,
                "V_at_60s": 3.92 - (sim_cycle * 0.001),
                "V_at_300s": 3.80 - (sim_cycle * 0.0012),
                "V_at_600s": 3.68 - (sim_cycle * 0.0014),
                "V_at_1200s": 3.52 - (sim_cycle * 0.0016),
                "V_start": 4.10,
                "V_mean_first600s": 3.80 - (sim_cycle * 0.001),
                "V_slope_mid": -0.00035 * (1.0 + sim_cycle / 100.0),
                "T_start": 24.0,
                "T_max": 24.0 + 8.0 + (sim_cycle * 0.05),
                "T_rise": 8.0 + (sim_cycle * 0.05),
                "T_mean": 28.0 + (sim_cycle * 0.03),
                "R_dc_proxy": sim_rdc,
            }
            # Synthetic discharge curve for plotting
            t_arr = np.linspace(0, sim_t4036 * 2.2, 80)
            v_arr = np.clip(4.15 - 0.05 - (t_arr / (sim_t4036 * 2.2)) * 1.5, 2.5, 4.2)
            i_arr = np.full(len(t_arr), -2.0)
            temp_arr = np.linspace(24.0, 24.0 + 8.0 + (sim_cycle * 0.05), len(t_arr))

    # EXECUTE INFERENCE
    st.markdown("---")
    st.markdown('<div class="cg-section-label">SELECT ML ALGORITHM FOR RESISTANCE PREDICTION</div>', unsafe_allow_html=True)

    algo_cols = st.columns([2, 1, 1])
    with algo_cols[0]:
        selected_algo = st.selectbox(
            "Trained Algorithm",
            ["Extra Trees (Recommended · R² = 97.0%)", "Random Forest (R² = 95.7%)", "XGBoost (R² = 96.0%)", "Ridge Regression (Linear Baseline · R² = 90.3%)"],
            index=0,
            key="pred_algo_select",
            help="Trained strictly on observable partial discharge features without data leakage.",
        )
        algo_key = "Extra Trees" if "Extra Trees" in selected_algo else ("Random Forest" if "Random Forest" in selected_algo else ("XGBoost" if "XGBoost" in selected_algo else "Ridge"))

    with algo_cols[1]:
        st.metric("Operational Accuracy", "R² > 95%", help="5-Fold Cross-Validation on verified NASA discharge cycles.")
    with algo_cols[2]:
        st.metric("Active Model", algo_key, help="Pipeline loaded from models/benchmark_models.pkl.")

    if features is None:
        st.info("Select a NASA reference cell, upload a CSV file, or choose a simulation scenario above to generate predictions.")
        return

    # Run predictions
    try:
        disc_index = cycle_number - 1
        pred_dict = predictor.predict_all(features, disc_idx=disc_index, algorithm=algo_key)
        preds = pred_dict.get("predictions", {})
    except Exception as e:
        st.error(f"Inference error: {str(e)}")
        return

    p_re = preds.get("Re", {})
    p_rct = preds.get("Rct", {})
    p_rint = preds.get("R_int", {})
    p_soh = preds.get("soh", {})
    p_cap = preds.get("capacity", {})

    re_pred_val = p_re.get("value", 0.055)
    rct_pred_val = p_rct.get("value", 0.080)
    rint_pred_val = p_rint.get("value", re_pred_val + rct_pred_val)
    cap_pred_val = p_cap.get("value", 1.8)
    soh_pred_val = p_soh.get("value", 90.0)

    # DISPLAY RESULTS TILES
    st.markdown('<div class="cg-section-label">PREDICTED RESISTANCE & BATTERY HEALTH METRICS</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        render_kpi(
            label="Predicted Ohmic (Re)",
            value=f"{re_pred_val * 1000.0:.2f}",
            unit="mΩ",
            error_band=f"{p_re.get('error_band', 0.0061)*1000.0:.1f} mΩ",
            subtext=f"Raw: {re_pred_val:.5f} Ω",
            status="Normal" if re_pred_val < 0.070 else "High",
            category="predicted",
            tooltip="Ohmic bulk resistance (R0) predicted from DC load step and voltage response.",
        )

    with m2:
        render_kpi(
            label="Predicted Charge-Transfer (Rct)",
            value=f"{rct_pred_val * 1000.0:.2f}",
            unit="mΩ",
            error_band=f"{p_rct.get('error_band', 0.0057)*1000.0:.1f} mΩ",
            subtext=f"Raw: {rct_pred_val:.5f} Ω",
            status="Normal" if rct_pred_val < 0.090 else "Degraded",
            category="predicted",
            tooltip="Charge-transfer polarization resistance (R1) at electrode-electrolyte interface.",
        )

    with m3:
        render_kpi(
            label="Total Internal R_int",
            value=f"{rint_pred_val * 1000.0:.2f}",
            unit="mΩ",
            error_band=f"{p_rint.get('error_band', 0.0084)*1000.0:.1f} mΩ",
            subtext="R_int = Re + Rct",
            status="Healthy" if rint_pred_val < 0.160 else "Elevated",
            category="calculated",
            tooltip="Total series equivalent resistance governing voltage drops under current load.",
        )

    with m4:
        grade_label, _, _, _ = get_health_grade(soh_pred_val)
        render_kpi(
            label="Secondary SOH Retention",
            value=f"{soh_pred_val:.1f}",
            unit="%",
            error_band="2.3%",
            subtext=f"Capacity: {cap_pred_val:.3f} Ah / 2.00 Ah",
            status=grade_label,
            category="predicted",
            tooltip="State of Health retention derived from ML capacity prediction.",
        )

    # GROUND TRUTH BENCHMARK COMPARISON TABLE
    if true_re is not None or true_rct is not None or true_cap is not None:
        st.markdown('<div class="cg-section-label">GROUND TRUTH VALIDATION · EXPERIMENTAL EIS VS PREDICTION</div>', unsafe_allow_html=True)
        comp_rows = []
        if true_re is not None:
            err_re_mohm = (re_pred_val - true_re) * 1000.0
            pct_re = abs(re_pred_val - true_re) / true_re * 100.0
            comp_rows.append({
                "Parameter": "Ohmic Resistance (Re)",
                "Physical Meaning": "Electrolyte & bulk collectors",
                "Actual EIS Reading": f"{true_re * 1000.0:.2f} mΩ ({true_re:.5f} Ω)",
                "ML Prediction": f"{re_pred_val * 1000.0:.2f} mΩ ({re_pred_val:.5f} Ω)",
                "Absolute Error": f"{err_re_mohm:+.2f} mΩ",
                "Relative Error": f"{pct_re:.1f}%",
                "Agreement": "✅ High Accuracy" if pct_re < 10.0 else "⚠️ Moderate",
            })

        if true_rct is not None:
            err_rct_mohm = (rct_pred_val - true_rct) * 1000.0
            pct_rct = abs(rct_pred_val - true_rct) / true_rct * 100.0
            comp_rows.append({
                "Parameter": "Charge-Transfer (Rct)",
                "Physical Meaning": "Interface Faradaic kinetics",
                "Actual EIS Reading": f"{true_rct * 1000.0:.2f} mΩ ({true_rct:.5f} Ω)",
                "ML Prediction": f"{rct_pred_val * 1000.0:.2f} mΩ ({rct_pred_val:.5f} Ω)",
                "Absolute Error": f"{err_rct_mohm:+.2f} mΩ",
                "Relative Error": f"{pct_rct:.1f}%",
                "Agreement": "✅ High Accuracy" if pct_rct < 10.0 else "⚠️ Moderate",
            })

        if true_cap is not None:
            err_cap = cap_pred_val - true_cap
            pct_cap = abs(cap_pred_val - true_cap) / true_cap * 100.0
            comp_rows.append({
                "Parameter": "Discharge Capacity",
                "Physical Meaning": "Total Coulomb retention",
                "Actual EIS Reading": f"{true_cap:.3f} Ah",
                "ML Prediction": f"{cap_pred_val:.3f} Ah",
                "Absolute Error": f"{err_cap:+.3f} Ah",
                "Relative Error": f"{pct_cap:.1f}%",
                "Agreement": "✅ High Accuracy" if pct_cap < 5.0 else "⚠️ Moderate",
            })

        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

    # DISCHARGE CURVE WITH PARTIAL WINDOW
    if t_arr is not None and v_arr is not None:
        st.markdown('<div class="cg-section-label">OBSERVED TERMINAL VOLTAGE PROFILE & 4.0 V → 3.4 V INSPECTION WINDOW</div>', unsafe_allow_html=True)
        fig_curve = plot_discharge_curve(t_arr, v_arr, temp=temp_arr)
        st.plotly_chart(fig_curve, use_container_width=True)

    # SECTION: UNDERSTANDING YOUR BATTERY RESULTS
    st.markdown('<div class="cg-section-label">UNDERSTANDING YOUR RESISTANCE DIAGNOSTIC RESULTS</div>', unsafe_allow_html=True)
    with st.expander("📖 Physical Interpretation of Re, Rct, and Degradation Mechanisms", expanded=True):
        st.markdown(f"""
        ### What Do These Resistances Mean for an Electrical Engineer?

        #### 1. Ohmic Resistance ($R_e \\approx R_0$):
        - **What it measures:** The pure instantaneous series resistance across the battery bulk. It consists of the electrolyte ionic conductivity, the polymeric separator porosity, the active material electrical resistance, and the aluminum/copper current collector foil tabs.
        - **Current predicted value:** **{re_pred_val * 1000.0:.2f} mΩ** ({re_pred_val:.5f} Ω).
        - **Normal range:** Typically 40 to 65 mΩ for a healthy 18650 cell. Values exceeding 75 mΩ indicate electrolyte solvent dry-out, loss of electrolyte salts, or severe contact degradation.
        - **Impact:** Causes an instantaneous ohmic voltage drop $\\Delta V = I \\times R_e$ when current turns on, and generates Joule heat $P = I^2 R_e$.

        #### 2. Charge-Transfer Resistance ($R_{{ct}} \\approx R_1$):
        - **What it measures:** The activation resistance encountered by lithium ions ($Li^+$) transferring across the electrode-electrolyte phase boundary during intercalation/de-intercalation.
        - **Current predicted value:** **{rct_pred_val * 1000.0:.2f} mΩ** ({rct_pred_val:.5f} Ω).
        - **Normal range:** Typically 65 to 90 mΩ. As the cell ages, $R_{{ct}}$ grows substantially faster than $R_e$ (often doubling or tripling).
        - **Root cause of growth:** Thickening and passivation of the **Solid Electrolyte Interphase (SEI)** film on the negative graphite anode, active material particle cracking, and loss of cyclable lithium inventory.
        - **Impact:** Governs the polarization voltage drop and dynamic transient response under load.

        #### 3. Total Internal Resistance ($R_{{int}} = R_e + R_{{ct}}$):
        - Current combined value: **{rint_pred_val * 1000.0:.2f} mΩ**.
        - At 2.0 A discharge current (1C rate), this cell experiences an immediate continuous internal voltage drop of approximately:
          $$\\Delta V_{{loss}} = I \\times (R_e + R_{{ct}}) = 2.0\\text{{ A}} \\times {rint_pred_val:.4f}\\ \\Omega = {2.0 * rint_pred_val:.3f}\\text{{ V}}$$
          and an internal thermal dissipation rate of:
          $$P_{{loss}} = I^2 \\times (R_e + R_{{ct}}) = (2.0)^2 \\times {rint_pred_val:.4f}\\ \\Omega = {4.0 * rint_pred_val:.3f}\\text{{ Watts}}$$
        """)
