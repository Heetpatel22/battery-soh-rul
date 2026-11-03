"""Guided battery health estimate from a NASA cycle or entered measurements."""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from battery_features import extract_discharge_features

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@st.cache_data(show_spinner=False)
def _load_reference_discharge(cell_id: str, cycle_number: int):
    from scipy.io import loadmat
    path = os.path.join(ROOT_DIR, "data", f"{cell_id}.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference data not found for {cell_id}.")
    root = loadmat(path, simplify_cells=True).get(cell_id, {})
    cycles = root.get("cycle", []) if isinstance(root, dict) else []
    if not isinstance(cycles, list):
        cycles = list(np.asarray(cycles, dtype=object).reshape(-1))
    discharge = [item for item in cycles if isinstance(item, dict) and item.get("type") == "discharge"]
    if not 1 <= cycle_number <= len(discharge):
        raise ValueError(f"Select a discharge cycle from 1 to {len(discharge)}.")
    record = discharge[cycle_number - 1].get("data", {})
    return {
        "t": np.asarray(record.get("Time", []), dtype=float).reshape(-1),
        "v": np.asarray(record.get("Voltage_measured", []), dtype=float).reshape(-1),
        "i": np.asarray(record.get("Current_measured", []), dtype=float).reshape(-1),
        "temp": np.asarray(record.get("Temperature_measured", []), dtype=float).reshape(-1),
        "capacity": float(np.asarray(record.get("Capacity", [np.nan]), dtype=float).reshape(-1)[0]),
        "count": len(discharge),
    }


@st.cache_data(show_spinner=False)
def _load_cycle_history():
    path = os.path.join(ROOT_DIR, "models", "feature_table.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def _manual_feature_form():
    st.markdown("#### Enter discharge measurements")
    st.caption("Enter measurements from one discharge cycle. Times are seconds from the start of discharge; current is entered as a positive magnitude.")
    with st.form("manual_battery_measurements"):
        cycle_number = st.number_input("Battery cycle number", min_value=1, max_value=5000, value=50, step=1)
        st.markdown("**Voltage crossing times**")
        tcols = st.columns(3)
        t40 = tcols[0].number_input("At 4.0 V (s)", 0.0, 20000.0, 0.0, 10.0, key="manual_t40")
        t38 = tcols[1].number_input("At 3.8 V (s)", 1.0, 20000.0, 280.0, 10.0, key="manual_t38")
        t375 = tcols[2].number_input("At 3.75 V (s)", 1.0, 20000.0, 350.0, 10.0, key="manual_t375")
        tcols = st.columns(3)
        t365 = tcols[0].number_input("At 3.65 V (s)", 1.0, 20000.0, 650.0, 10.0, key="manual_t365")
        t36 = tcols[1].number_input("At 3.6 V (s)", 1.0, 20000.0, 760.0, 10.0, key="manual_t36")
        t34 = tcols[2].number_input("At 3.4 V (s)", 1.0, 20000.0, 1250.0, 10.0, key="manual_t34")

        st.markdown("**Voltage samples**")
        vcols = st.columns(3)
        vstart = vcols[0].number_input("Start voltage (V)", 3.4, 4.3, 4.02, 0.01, key="manual_vstart")
        v60 = vcols[1].number_input("At 60 s (V)", 2.5, 4.3, 3.92, 0.01, key="manual_v60")
        v300 = vcols[2].number_input("At 300 s (V)", 2.5, 4.3, 3.79, 0.01, key="manual_v300")
        vcols = st.columns(3)
        v600 = vcols[0].number_input("At 600 s (V)", 2.5, 4.3, 3.68, 0.01, key="manual_v600")
        v1200 = vcols[1].number_input("At 1200 s (V)", 2.5, 4.3, 3.53, 0.01, key="manual_v1200")
        vmean = vcols[2].number_input("Mean first 600 s (V)", 2.5, 4.3, 3.84, 0.01, key="manual_vmean")

        st.markdown("**Operating conditions**")
        tcols = st.columns(3)
        current = tcols[0].number_input("Current magnitude (A)", 0.1, 20.0, 2.0, 0.1, key="manual_current")
        temp_start = tcols[1].number_input("Start temperature (°C)", -20.0, 100.0, 24.0, 0.5, key="manual_temp_start")
        temp_max = tcols[2].number_input("Maximum temperature (°C)", -20.0, 120.0, 40.0, 0.5, key="manual_temp_max")
        tcols = st.columns(3)
        temp_mean = tcols[0].number_input("Mean temperature (°C)", -20.0, 100.0, 32.0, 0.5, key="manual_temp_mean")
        rdc = tcols[1].number_input("Load-step resistance (Ω)", 0.0, 1.0, 0.105, 0.001, format="%.3f", key="manual_rdc")

        label = st.text_input("Battery label", "My battery", key="manual_battery_label")
        submitted = st.form_submit_button("Estimate battery health", type="primary", use_container_width=True)

    features = {
        "T_40_38": t38 - t40,
        "T_38_36": t36 - t38,
        "T_36_34": t34 - t36,
        "T_40_36": t36 - t40,
        "T_38_34": t34 - t38,
        "Q_40_36": current * max(0.0, t36 - t40) / 3600.0,
        "V_at_60s": v60,
        "V_at_300s": v300,
        "V_at_600s": v600,
        "V_at_1200s": v1200,
        "V_start": vstart,
        "V_mean_first600s": vmean,
        "V_slope_mid": (3.65 - 3.75) / (t365 - t375) if t365 > t375 else np.nan,
        "T_start": temp_start,
        "T_max": temp_max,
        "T_rise": temp_max - temp_start,
        "T_mean": temp_mean,
        "R_dc_proxy": rdc,
    }
    times = np.asarray([t40, t38, t375, t365, t36, t34, 60, 300, 600, 1200], dtype=float)
    volts = np.asarray([4.0, 3.8, 3.75, 3.65, 3.6, 3.4, v60, v300, v600, v1200], dtype=float)
    sort_order = np.argsort(times)
    times, volts = times[sort_order], volts[sort_order]
    times, unique_idx = np.unique(times, return_index=True)
    volts = volts[unique_idx]
    temps = np.interp(times, [0, 600, 1200], [temp_start, temp_mean, temp_max])
    currents = np.full(len(times), -abs(current), dtype=float)
    return submitted, int(cycle_number), features, times, volts, currents, temps, label.strip() or "My battery"


def _signature(features: dict) -> str:
    return json.dumps({key: (float(value) if np.isfinite(value) else None) for key, value in features.items()}, sort_keys=True)


def render_predict(predictor, schema: dict, metrics: dict):
    st.markdown("""
    <section class="cg-page-hero cg-diagnostics-hero">
      <div class="cg-eyebrow">CELLGUARD <span>PERSONAL BATTERY CHECK</span></div>
      <h1>Enter your readings.<br/>Get a clear battery picture.</h1>
      <p>Choose a NASA reference cycle or fill in the discharge measurements from your own battery.</p>
      <div class="cg-hero-foot"><span><i></i> CAPACITY · HEALTH · RESISTANCE · ENERGY</span><span>INPUT WINDOW 4.0 V → 3.4 V</span></div>
    </section>
    """, unsafe_allow_html=True)

    input_method = st.radio("Choose an input method", ["NASA reference cycle", "Enter my measurements"],
                            horizontal=True, key="diagnostic_input_method")
    features = None
    t_arr = v_arr = i_arr = temp_arr = None
    cycle_number, cell_id, data_label, truth_capacity = 50, "B0018", "", None
    estimate_now = False

    if input_method == "NASA reference cycle":
        col1, col2 = st.columns([1, 1.5], gap="large")
        with col1:
            cell_id = st.selectbox("Reference battery", ["B0005", "B0006", "B0007", "B0018"], index=3)
            try:
                ref = _load_reference_discharge(cell_id, 1)
                cycle_number = st.number_input("Discharge cycle", min_value=1, max_value=ref["count"], value=min(50, ref["count"]), step=1)
                ref = _load_reference_discharge(cell_id, int(cycle_number))
                t_arr, v_arr, i_arr, temp_arr = ref["t"], ref["v"], ref["i"], ref["temp"]
                truth_capacity = ref["capacity"]
                features = extract_discharge_features(t_arr, v_arr, i_arr, temp_arr)
                data_label = f"NASA {cell_id} · cycle {cycle_number}"
            except Exception as exc:
                st.error(str(exc))
        with col2:
            st.markdown("#### Try a recorded battery cycle")
            st.write("Explore a real NASA battery discharge trace, or switch to manual entry to check your own readings.")
            st.info("The selected discharge needs to pass through the 4.0 V to 3.4 V window.", icon="ℹ️")
        estimate_now = st.button("Estimate this cycle", type="primary", use_container_width=True)
    else:
        submitted, cycle_number, features, t_arr, v_arr, i_arr, temp_arr, cell_id = _manual_feature_form()
        data_label = f"Manual entry · {cell_id}"
        estimate_now = submitted

    signature = _signature(features) if features else ""
    if estimate_now and features:
        result = predictor.predict_all(features, disc_idx=int(cycle_number) - 1)
        if not result.get("success"):
            st.error("Check the crossing times and measurements, then try again.")
            for error in result.get("errors", []):
                st.write(f"• {error}")
        else:
            st.session_state["last_cellguard_result"] = {
                "result": result, "features": features, "signature": signature,
                "label": data_label, "cell": str(cell_id), "cycle": int(cycle_number),
                "truth_capacity": truth_capacity, "t": t_arr, "v": v_arr,
                "i": i_arr, "temp": temp_arr,
            }

    saved = st.session_state.get("last_cellguard_result")
    if not saved or saved.get("signature") != signature or saved.get("label") != data_label or saved.get("cycle") != int(cycle_number):
        if input_method == "NASA reference cycle":
            st.caption("Select a cycle and choose **Estimate this cycle** to see the results.")
        return

    preds = saved["result"]["predictions"]
    st.markdown('<div class="cg-section-label">BATTERY HEALTH SNAPSHOT</div>', unsafe_allow_html=True)
    cap, soh, rul = preds["capacity"], preds["soh"], preds["RUL"]
    cols = st.columns(3)
    cols[0].metric("Estimated capacity", f"{cap['value']:.3f} Ah", f"± {cap['error_band']:.3f} Ah")
    cols[1].metric("State of health", f"{soh['value']:.1f}%", soh["status"])
    cols[2].metric("Remaining useful life", "End of life" if rul.get("is_eol") else f"{rul.get('value', '—')} cycles",
                   "1.40 Ah reference threshold" if rul.get("is_eol") else f"± {rul.get('error_band', '—')} cycles")
    cols = st.columns(3)
    cols[0].metric("Ohmic resistance · Re", f"{preds['Re']['value']:.4f} Ω", f"± {preds['Re']['error_band']:.4f} Ω")
    cols[1].metric("Charge-transfer · Rct", f"{preds['Rct']['value']:.4f} Ω", f"± {preds['Rct']['error_band']:.4f} Ω")
    energy = preds.get("discharge_energy", {})
    cols[2].metric("Discharge energy", f"{energy.get('value', np.nan):.2f} Wh", f"± {energy.get('error_band', np.nan):.2f} Wh")
    if saved.get("truth_capacity") is not None and np.isfinite(saved["truth_capacity"]):
        st.caption(f"Recorded reference capacity: {saved['truth_capacity']:.3f} Ah · estimate shown separately.")

    st.markdown('<div class="cg-section-label">MEASURED DISCHARGE PROFILE</div>', unsafe_allow_html=True)
    sensor_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.08,
                               subplot_titles=("Terminal voltage", "Discharge current", "Cell temperature"))
    for row, values, color, name in [(1, saved["v"], "#476BFF", "Voltage"),
                                     (2, saved["i"], "#16BFA8", "Current"),
                                     (3, saved["temp"], "#F3AA58", "Temperature")]:
        sensor_fig.add_trace(go.Scatter(x=saved["t"], y=values, name=name, mode="lines+markers" if input_method == "Enter my measurements" else "lines",
                                        marker={"size": 6}, line={"color": color, "width": 2.5}), row=row, col=1)
    sensor_fig.update_yaxes(title_text="V", row=1, col=1)
    sensor_fig.update_yaxes(title_text="A", row=2, col=1)
    sensor_fig.update_yaxes(title_text="°C", row=3, col=1)
    sensor_fig.update_xaxes(title_text="Elapsed time (seconds)", row=3, col=1)
    sensor_fig.update_layout(height=630, showlegend=False, margin={"l": 10, "r": 10, "t": 55, "b": 10},
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#D5E1F2"})
    st.plotly_chart(sensor_fig, use_container_width=True)

    if saved["label"].startswith("NASA "):
        st.markdown('<div class="cg-section-label">BATTERY HISTORY</div>', unsafe_allow_html=True)
        history = _load_cycle_history()
        history = history[history.cell == saved["cell"]].sort_values("disc_idx") if not history.empty else history
        if not history.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history.disc_idx + 1, y=history.capacity, name="Capacity", mode="lines",
                                     line={"color": "#42D6C3", "width": 2.5}))
            row = history[history.disc_idx == saved["cycle"] - 1]
            if not row.empty:
                fig.add_trace(go.Scatter(x=row.disc_idx + 1, y=row.capacity, name="Selected cycle", mode="markers",
                                         marker={"color": "#FFFFFF", "size": 10, "line": {"color": "#42D6C3", "width": 3}}))
            fig.add_hline(y=1.4, line_dash="dash", line_color="#F2738D", annotation_text="1.40 Ah threshold")
            fig.update_layout(title=f"Capacity history · {saved['cell']}", xaxis_title="Discharge cycle", yaxis_title="Capacity (Ah)",
                              height=380, margin={"l": 10, "r": 10, "t": 48, "b": 10}, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font={"color": "#D5E1F2"})
            st.plotly_chart(fig, use_container_width=True)

    export = {"input": saved["label"], "cycle": saved["cycle"], "predictions": preds, "measurements": saved["features"]}
    st.download_button("Download this battery report", json.dumps(export, indent=2),
                       file_name="cellguard_battery_report.json", mime="application/json")
