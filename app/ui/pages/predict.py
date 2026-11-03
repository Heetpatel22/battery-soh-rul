"""
predict.py - Battery Health Diagnostics.
Guided input for a NASA reference cycle or user-uploaded BMS telemetry,
with clear battery-health estimates and a downloadable prediction report.
"""

import os
import json
from typing import Tuple
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from battery_features import extract_discharge_features
except ImportError:
    from src.battery_features import extract_discharge_features

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SAMPLE_DIR = os.path.join(ROOT_DIR, "sample_cycles")

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


@st.cache_data(show_spinner=False)
def _load_reference_discharge(cell_id: str, cycle_number: int):
    """Read one discharge cycle from the bundled NASA reference .mat files."""
    from scipy.io import loadmat
    source = os.path.join(ROOT_DIR, "data", f"{cell_id}.mat")
    if not os.path.exists(source):
        raise FileNotFoundError(f"Reference dataset not found: {source}")
    root = loadmat(source, simplify_cells=True).get(cell_id, {})
    cycles = root.get("cycle", []) if isinstance(root, dict) else []
    if not isinstance(cycles, list):
        cycles = list(np.asarray(cycles, dtype=object).reshape(-1))
    discharge = [c for c in cycles if isinstance(c, dict) and c.get("type") == "discharge"]
    if cycle_number < 1 or cycle_number > len(discharge):
        raise ValueError(f"Choose a cycle between 1 and {len(discharge)}.")
    selected = discharge[cycle_number - 1]
    data = selected.get("data", {})
    return {
        "t": np.asarray(data.get("Time", []), dtype=float).reshape(-1),
        "v": np.asarray(data.get("Voltage_measured", []), dtype=float).reshape(-1),
        "i": np.asarray(data.get("Current_measured", []), dtype=float).reshape(-1),
        "temp": np.asarray(data.get("Temperature_measured", []), dtype=float).reshape(-1),
        "capacity": float(np.asarray(data.get("Capacity", [np.nan]), dtype=float).reshape(-1)[0]),
        "count": len(discharge),
    }


@st.cache_data(show_spinner=False)
def _load_cycle_history():
    path = os.path.join(ROOT_DIR, "models", "feature_table.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def render_predict(predictor, schema: dict, metrics: dict):
    """Simple, guided workflow for historical examples or user telemetry."""
    st.markdown("""
    <div class="cg-hero">
      <div class="cg-eyebrow">CELLGUARD · BATTERY HEALTH</div>
      <h1>Run a battery check</h1>
      <p>Choose a NASA reference cycle or upload your own discharge measurements. CellGuard extracts the voltage-window features and estimates battery health.</p>
      <div class="cg-step-row"><span>01&nbsp; Choose data</span><b>→</b><span>02&nbsp; Run estimate</span><b>→</b><span>03&nbsp; Review results</span></div>
    </div>
    """, unsafe_allow_html=True)

    input_method = st.radio("Choose your data source", ["Use NASA reference data", "Upload my measurements"],
                            horizontal=True, key="diagnostic_input_method")
    t_arr = v_arr = i_arr = temp_arr = None
    cell_id, cycle_number, data_label = "B0018", 50, ""
    truth_capacity = None

    if input_method == "Use NASA reference data":
        left, right = st.columns([1, 1.3], gap="large")
        with left:
            cell_id = st.selectbox("Battery", ["B0005", "B0006", "B0007", "B0018"], index=3,
                                   help="These four cells are the curated reference set bundled with the project.")
            try:
                from scipy.io import loadmat
                mat = loadmat(os.path.join(ROOT_DIR, "data", f"{cell_id}.mat"), simplify_cells=True)
                raw_cycles = mat[cell_id].get("cycle", [])
                if not isinstance(raw_cycles, list):
                    raw_cycles = list(np.asarray(raw_cycles, dtype=object).reshape(-1))
                total = sum(isinstance(c, dict) and c.get("type") == "discharge" for c in raw_cycles)
            except Exception:
                total = 1
            cycle_number = st.number_input("Discharge cycle", min_value=1, max_value=max(1, total),
                                           value=min(50, max(1, total)), step=1)
            st.caption("A discharge curve must cross 4.0 V down to 3.4 V for the capacity estimate.")
        with right:
            st.markdown("#### What you need")
            st.markdown("Select a cell and cycle to view bundled NASA data. To check your own battery, upload timestamped voltage, current, and temperature measurements.")
            st.info("Health estimates use the wider NASA battery collection. This selector offers four bundled reference cells.", icon="ℹ️")
        try:
            ref = _load_reference_discharge(cell_id, int(cycle_number))
            t_arr, v_arr, i_arr, temp_arr = ref["t"], ref["v"], ref["i"], ref["temp"]
            truth_capacity = ref["capacity"]
            data_label = f"NASA {cell_id} · discharge {cycle_number}"
        except Exception as exc:
            st.error(str(exc))

    if input_method == "Upload my measurements":
        st.markdown("#### Upload a discharge CSV")
        st.write("Include one row per measurement and these columns. Current should be negative while discharging.")
        template = pd.DataFrame({"Time_s": [0, 10, 20], "Voltage_V": [4.19, 3.99, 3.97],
                                 "Current_A": [-0.01, -2.0, -2.0], "Temperature_C": [24.5, 24.6, 24.7]})
        st.dataframe(template, hide_index=True, use_container_width=True)
        sample_csv = os.path.join(SAMPLE_DIR, "B0018_cycle_050_middle.csv")
        if os.path.exists(sample_csv):
            with open(sample_csv, "rb") as example_file:
                st.download_button("Download a valid example CSV", example_file.read(),
                                   file_name="cellguard_example_discharge.csv", mime="text/csv")
        uploaded = st.file_uploader("Choose your CSV file", type=["csv"], key="clean_diagnostics_upload")
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)
                ok, errors, cleaned = validate_csv_data(frame)
                if not ok:
                    for error in errors:
                        st.error(error)
                else:
                    t_arr = cleaned["Time_s"].to_numpy()
                    v_arr = cleaned["Voltage_V"].to_numpy()
                    i_arr = cleaned["Current_A"].to_numpy()
                    temp_arr = cleaned["Temperature_C"].to_numpy()
                    data_label = uploaded.name
                    col1, col2 = st.columns(2)
                    with col1:
                        cell_id = st.text_input("Optional battery label", "My battery")
                    with col2:
                        cycle_number = st.number_input("Known cycle number (optional)", 1, 5000, 50)
            except Exception as exc:
                st.error(f"Could not read this CSV: {exc}")

    if t_arr is None:
        st.info("Choose reference data or upload a valid CSV to see predictions.")
        return

    st.markdown("### Your estimate")
    if st.button("Estimate battery health", type="primary", use_container_width=True):
        features = extract_discharge_features(t_arr, v_arr, i_arr, temp_arr)
        result = predictor.predict_all(features, disc_idx=int(cycle_number) - 1)
        if not result.get("success"):
            st.error("This curve is missing part of the required discharge window.")
            for error in result.get("errors", []):
                st.write(f"• {error}")
            return
        st.session_state["last_cellguard_result"] = {
            "result": result, "features": features, "label": data_label,
            "cell": str(cell_id), "cycle": int(cycle_number), "truth_capacity": truth_capacity,
            "t": t_arr, "v": v_arr, "temp": temp_arr,
        }

    saved = st.session_state.get("last_cellguard_result")
    if not saved or saved.get("label") != data_label or saved.get("cell") != str(cell_id) or saved.get("cycle") != int(cycle_number):
        st.caption("Predictions appear here after you select **Estimate battery health**.")
        return
    result = saved["result"]
    preds = result["predictions"]
    capacity = preds["capacity"]
    soh = preds["soh"]
    rul = preds["RUL"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Estimated capacity", f"{capacity['value']:.3f} Ah", f"± {capacity['error_band']:.3f} Ah")
    k2.metric("State of health", f"{soh['value']:.1f}%", soh["status"])
    k3.metric("Remaining useful life", "End of life" if rul.get("is_eol") else f"{rul.get('value', '—')} cycles",
              "1.40 Ah threshold" if rul.get("is_eol") else f"± {rul.get('error_band', '—')} cycles")
    k4, k5, k6 = st.columns(3)
    k4.metric("Ohmic resistance · Re", f"{preds['Re']['value']:.4f} Ω", f"± {preds['Re']['error_band']:.4f} Ω")
    k5.metric("Charge-transfer · Rct", f"{preds['Rct']['value']:.4f} Ω", f"± {preds['Rct']['error_band']:.4f} Ω")
    energy = preds.get("discharge_energy", {})
    k6.metric("Discharge energy", f"{energy.get('value', float('nan')):.2f} Wh",
              f"± {energy.get('error_band', float('nan')):.2f} Wh")
    st.success("Battery check complete. Explore the discharge curve and cycle history below.")
    if saved.get("truth_capacity") is not None and np.isfinite(saved["truth_capacity"]):
        st.caption(f"Reference capacity recorded in the dataset: {saved['truth_capacity']:.3f} Ah. Model estimate and recorded value are shown separately.")
    st.markdown("#### Sensor trace")
    sensor_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                               subplot_titles=("Voltage", "Discharge current", "Temperature"))
    for row, values, color, name in [
        (1, saved["v"], "#2563EB", "Voltage"),
        (2, i_arr, "#0D9488", "Current"),
        (3, saved["temp"], "#F59E0B", "Temperature"),
    ]:
        sensor_fig.add_trace(go.Scatter(x=saved["t"], y=values, mode="lines", name=name,
                                        line={"color": color, "width": 2}), row=row, col=1)
    sensor_fig.update_yaxes(title_text="V", row=1, col=1)
    sensor_fig.update_yaxes(title_text="A", row=2, col=1)
    sensor_fig.update_yaxes(title_text="°C", row=3, col=1)
    sensor_fig.update_xaxes(title_text="Elapsed time (seconds)", row=3, col=1)
    sensor_fig.update_layout(height=650, showlegend=False, margin={"l": 10, "r": 10, "t": 55, "b": 10},
                             paper_bgcolor="white", plot_bgcolor="white", font={"color": "#172554"})
    st.plotly_chart(sensor_fig, use_container_width=True)

    if saved["label"].startswith("NASA "):
        st.markdown("#### Battery history")
        history = _load_cycle_history()
        history = history[(history["cell"] == saved["cell"])].sort_values("disc_idx") if not history.empty else history
        if not history.empty:
            history_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                                        subplot_titles=("Capacity over life", "Ohmic resistance history", "Charge-transfer resistance history"))
            history_fig.add_trace(go.Scatter(x=history.disc_idx, y=history.capacity, mode="lines",
                                             name="Capacity", line={"color": "#2563EB", "width": 2.5}), row=1, col=1)
            cycle_row = history[history.disc_idx == saved["cycle"] - 1]
            if not cycle_row.empty:
                history_fig.add_trace(go.Scatter(x=cycle_row.disc_idx, y=cycle_row.capacity, mode="markers",
                                                 name="Selected cycle", marker={"color": "#F97316", "size": 10}), row=1, col=1)
            history_fig.add_trace(go.Scatter(x=history.disc_idx, y=history.Re, mode="lines",
                                             name="Re", line={"color": "#0D9488", "width": 2}), row=2, col=1)
            history_fig.add_trace(go.Scatter(x=history.disc_idx, y=history.Rct, mode="lines",
                                             name="Rct", line={"color": "#8B5CF6", "width": 2}), row=3, col=1)
            history_fig.update_yaxes(title_text="Ah", row=1, col=1)
            history_fig.update_yaxes(title_text="Ω", row=2, col=1)
            history_fig.update_yaxes(title_text="Ω", row=3, col=1)
            history_fig.update_xaxes(title_text="Discharge cycle", row=3, col=1)
            history_fig.update_layout(height=720, showlegend=False, margin={"l": 10, "r": 10, "t": 55, "b": 10},
                                      paper_bgcolor="white", plot_bgcolor="white", font={"color": "#172554"})
            st.plotly_chart(history_fig, use_container_width=True)
    st.caption(f"Input data: {saved['label']} · Expected variation is based on the battery histories used for this project.")
    export = {"input": saved["label"], "cell": saved["cell"], "cycle": saved["cycle"],
              "predictions": preds, "features": saved["features"]}
    st.download_button("Download prediction report (JSON)", json.dumps(export, indent=2),
                       file_name="cellguard_prediction.json", mime="application/json")
