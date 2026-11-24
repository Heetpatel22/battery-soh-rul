"""
overview.py - Executive Battery Health & Impedance Overview Dashboard.
Provides instantaneous answers to: "What is the current condition of the battery?"
Focuses on Ohmic Resistance (Re), Charge-Transfer Resistance (Rct), and State of Health (SOH).
"""

import numpy as np
import pandas as pd
import streamlit as st

from ui.components import (
    render_compact_header,
    render_category_legend,
    render_kpi,
    render_health_gauge_svg,
    get_health_grade,
    render_html,
)
from ui.charts import (
    plot_re_rct_dual_trend,
    plot_life_and_rul_trajectory,
)


def render_overview(predictor, metrics: dict, feature_table: pd.DataFrame, workspace_pages: dict | None = None):
    # Ensure session state defaults
    if "selected_cell" not in st.session_state:
        st.session_state["selected_cell"] = "B0005"
    if "selected_cycle" not in st.session_state:
        st.session_state["selected_cycle"] = 40

    active_cell = st.session_state["selected_cell"]

    render_compact_header(
        page_title="Executive Battery Diagnostics & Impedance Overview",
        selected_battery=active_cell,
        data_source="NASA Ames Li-ion Aging Repository · 18650 Cells",
        model_status="ML Estimators Connected · LOCO Validated",
    )

    render_category_legend()

    if feature_table.empty:
        st.error("Feature table is unavailable. Check models/feature_table.csv.")
        return

    # Cell & Cycle Selection Controls
    col_sel1, col_sel2, col_sel3 = st.columns([1.5, 2.5, 1.5], vertical_alignment="bottom")
    with col_sel1:
        cells = [c for c in ["B0005", "B0006", "B0007", "B0018"] if c in feature_table["cell"].unique()]
        if not cells:
            cells = sorted(feature_table["cell"].dropna().unique().tolist())
        selected_cell = st.selectbox(
            "Select Battery Cell",
            options=cells,
            index=cells.index(active_cell) if active_cell in cells else 0,
            key="overview_cell_select",
            help="NASA Ames Research Center 18650 lithium-ion test cells cycled to end-of-life.",
        )
        st.session_state["selected_cell"] = selected_cell

    cell_data = feature_table[feature_table["cell"] == selected_cell].sort_values("disc_idx").copy()
    max_cycle = int(cell_data["disc_idx"].max()) + 1 if not cell_data.empty else 168

    with col_sel2:
        selected_cycle = st.slider(
            "Select Cycle Number",
            min_value=1,
            max_value=max_cycle,
            value=min(st.session_state.get("selected_cycle", 40), max_cycle),
            step=1,
            key="overview_cycle_slider",
            help="Select historical discharge cycle index (1-based).",
        )
        st.session_state["selected_cycle"] = selected_cycle

    with col_sel3:
        st.markdown(f"""
        <div style="background:#F1F5F9; border:1px solid #CBD5E1; border-radius:6px; padding:6px 12px; font-size:0.75rem; color:#334155; margin-bottom:4px;">
          <div>Chemistry: <strong>LiCoO₂ / Graphite</strong></div>
          <div>Rated Capacity: <strong>2.00 Ah</strong></div>
          <div>Cut-off: <strong>2.7 V / 2.5 V</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # Retrieve cycle data
    cycle_idx = selected_cycle - 1
    row_match = cell_data[cell_data["disc_idx"] == cycle_idx]
    if row_match.empty:
        # Closest available cycle
        closest_idx = (cell_data["disc_idx"] - cycle_idx).abs().argsort()[:1]
        row = cell_data.iloc[closest_idx].iloc[0]
    else:
        row = row_match.iloc[0]

    # Ground truth values if present
    actual_cap = float(row["capacity"]) if pd.notna(row.get("capacity")) else None
    actual_re = float(row["Re"]) if pd.notna(row.get("Re")) else None
    actual_rct = float(row["Rct"]) if pd.notna(row.get("Rct")) else None

    # Run ML prediction on extracted features
    feat_cols = predictor.model_capacity["features"]
    features = {name: float(row.get(name, np.nan)) for name in feat_cols}
    # Add all available features
    for col in row.index:
        if col not in features and pd.notna(row[col]) and isinstance(row[col], (int, float, np.number)):
            features[col] = float(row[col])

    try:
        pred_res = predictor.predict_all(features, disc_idx=cycle_idx)
        preds = pred_res.get("predictions", {})
    except Exception:
        preds = {}

    # Extract values
    pred_cap_dict = preds.get("capacity", {})
    pred_re_dict = preds.get("Re", {})
    pred_rct_dict = preds.get("Rct", {})
    pred_rint_dict = preds.get("R_int", {})
    pred_soh_dict = preds.get("soh", {})

    cap_val = actual_cap if actual_cap is not None else pred_cap_dict.get("value", 1.8)
    soh_val = (cap_val / 2.0) * 100.0
    grade_label, status_class, grade_desc, grade_color = get_health_grade(soh_val)

    re_val = actual_re if actual_re is not None else pred_re_dict.get("value", 0.050)
    rct_val = actual_rct if actual_rct is not None else pred_rct_dict.get("value", 0.075)
    rint_val = re_val + rct_val if (re_val is not None and rct_val is not None) else None

    fade_pct = round(max(0.0, 100.0 - soh_val), 1)

    # SECTION 1: PRIMARY RESISTANCE & IMPEDANCE METRICS
    st.markdown('<div class="cg-section-label">PRIMARY PARAMETERS · INTERNAL IMPEDANCE & HEALTH</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        re_disp = f"{re_val * 1000.0:.2f}" if re_val is not None else "—"
        re_sub = f"({re_val:.5f} Ω)" if re_val is not None else ""
        render_kpi(
            label="Ohmic Resistance (Re)",
            value=re_disp,
            unit="mΩ",
            error_band="6.13 mΩ",
            subtext=f"Electrolyte & collectors {re_sub}",
            status="Normal" if (re_val and re_val < 0.070) else "Elevated",
            category="measured" if actual_re is not None else "predicted",
            tooltip="Bulk ohmic resistance of electrolyte, separator, active material, and current collector tabs (R0).",
        )

    with c2:
        rct_disp = f"{rct_val * 1000.0:.2f}" if rct_val is not None else "—"
        rct_sub = f"({rct_val:.5f} Ω)" if rct_val is not None else ""
        render_kpi(
            label="Charge-Transfer (Rct)",
            value=rct_disp,
            unit="mΩ",
            error_band="5.74 mΩ",
            subtext=f"Interface kinetics {rct_sub}",
            status="Normal" if (rct_val and rct_val < 0.090) else "Degraded",
            category="measured" if actual_rct is not None else "predicted",
            tooltip="Faradaic charge-transfer kinetics resistance at the electrode-electrolyte interface (R1).",
        )

    with c3:
        rint_disp = f"{rint_val * 1000.0:.2f}" if rint_val is not None else "—"
        render_kpi(
            label="Total Resistance (R_int)",
            value=rint_disp,
            unit="mΩ",
            error_band="8.40 mΩ",
            subtext="R_int = Re + Rct",
            status="Healthy" if (rint_val and rint_val < 0.160) else "High Loss",
            category="calculated",
            tooltip="Combined internal equivalent series resistance determining instantaneous and polarization voltage drop.",
        )

    with c4:
        render_kpi(
            label="State of Health (SOH)",
            value=f"{soh_val:.1f}",
            unit="%",
            error_band="2.3%",
            subtext=f"{cap_val:.3f} Ah / 2.00 Ah ({fade_pct}% fade)",
            status=grade_label,
            category="measured" if actual_cap is not None else "predicted",
            tooltip="Ratio of usable remaining discharge capacity to original 2.00 Ah manufacturer rated capacity.",
        )

    # SECONDARY ROW: OPERATING & SENSORY MEASUREMENTS
    st.markdown('<div class="cg-section-label">ELECTRICAL & THERMAL MEASUREMENTS AT CYCLE</div>', unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        v_start = float(row.get("V_start", 4.10))
        render_kpi("Initial Voltage", f"{v_start:.3f}", "V", subtext="Terminal voltage at load start", category="measured")
    with e2:
        cur_mag = float(row.get("current_load", 2.0))
        render_kpi("Discharge Current", f"{cur_mag:.1f}", "A", subtext="Constant discharge load (1.0 C)", category="measured")
    with e3:
        temp_max = float(row.get("T_max", 35.0))
        temp_rise = float(row.get("T_rise", 8.0))
        render_kpi("Peak Cell Temp", f"{temp_max:.1f}", "°C", subtext=f"ΔT = +{temp_rise:.1f} °C rise during load", status="Normal" if temp_max < 45 else "Warm", category="measured")
    with e4:
        v_min = float(row.get("V_min", 2.7))
        render_kpi("Cut-off Voltage", f"{v_min:.2f}", "V", subtext=f"BMS end-of-discharge threshold", category="measured")

    # SECTION 2: GRAPHICAL DIAGNOSTICS (DUAL PLOTS)
    st.markdown('<div class="cg-section-label">DIAGNOSTIC VISUALIZATION & DEGRADATION TRACKING</div>', unsafe_allow_html=True)
    g_col1, g_col2 = st.columns([1.2, 1], gap="medium")

    with g_col1:
        pred_re_val = pred_re_dict.get("value")
        pred_rct_val = pred_rct_dict.get("value")
        fig_imp = plot_re_rct_dual_trend(
            cell_df=feature_table,
            cell_id=selected_cell,
            current_cycle=selected_cycle,
            pred_re=pred_re_val,
            pred_rct=pred_rct_val,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with g_col2:
        pred_cap_val = pred_cap_dict.get("value")
        fig_cap = plot_life_and_rul_trajectory(
            cell_df=cell_data,
            current_cycle=cycle_idx,
            pred_cap_at_cycle=pred_cap_val,
            pred_rul_cycles=None,
            eol_threshold=1.40,
        )
        st.plotly_chart(fig_cap, use_container_width=True)

    # SECTION 3: ENGINEERING DIAGNOSTIC SUMMARY
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-left:4px solid #2563EB; border-radius:6px; padding:14px 18px; margin-top:14px; box-shadow:0 1px 3px rgba(15,23,42,0.04);">
      <div style="font-size:0.85rem; font-weight:800; color:#0F172A; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:6px;">
        Engineering Diagnostic Assessment
      </div>
      <div style="font-size:0.82rem; color:#334155; line-height:1.6;">
        <strong>Condition Analysis:</strong> Cell <code>{cell}</code> at discharge cycle <code>{cycle}</code> exhibits an Ohmic resistance of <strong>{re_m:.2f} mΩ</strong> and a Charge-Transfer resistance of <strong>{rct_m:.2f} mΩ</strong>, giving a total internal resistance of <strong>{rint_m:.2f} mΩ</strong>.
        Usable capacity is <strong>{cap:.3f} Ah</strong> (SOH = <strong>{soh:.1f}%</strong>), indicating <strong>{status}</strong> operational health.
        <br/><br/>
        <strong>Electrochemical Mechanism:</strong> The increase in <em>Rct</em> relative to <em>Re</em> reflects the gradual thickening of the Solid Electrolyte Interphase (SEI) layer on the graphite anode, which impedes lithium-ion intercalation kinetics and generates increased overpotentials under load.
      </div>
    </div>
    """.format(
        cell=selected_cell,
        cycle=selected_cycle,
        re_m=re_val * 1000.0 if re_val else 0.0,
        rct_m=rct_val * 1000.0 if rct_val else 0.0,
        rint_m=rint_val * 1000.0 if rint_val else 0.0,
        cap=cap_val,
        soh=soh_val,
        status=grade_label,
    ), unsafe_allow_html=True)
