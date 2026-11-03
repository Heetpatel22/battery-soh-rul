"""Plain-language guide to CellGuard's battery health quantities."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_calculation_metrics(feature_table: pd.DataFrame):
    st.markdown("""
    <section class="cg-page-hero cg-metrics-hero">
      <div class="cg-eyebrow">REFERENCE GUIDE <span>UNITS · DEFINITIONS · DERIVATIONS</span></div>
      <h1>Calculation metrics</h1>
      <p>One clear place to see what each battery value means, how it is calculated, and how the values change through a cell's life.</p>
    </section>
    """, unsafe_allow_html=True)
    if feature_table.empty:
        st.warning("Cycle records are not available.")
        return

    cells = sorted(feature_table.cell.dropna().unique())
    col1, col2 = st.columns([1, 1.5])
    with col1:
        cell = st.selectbox("Battery", cells, key="metrics_cell")
    cell_df = feature_table[feature_table.cell == cell].sort_values("disc_idx").reset_index(drop=True)
    max_cycle = max(1, int(cell_df.disc_idx.max()) + 1)
    with col2:
        cycle = st.slider("Cycle", 1, max_cycle, min(50, max_cycle), key="metrics_cycle")
    row = cell_df.iloc[(cell_df.disc_idx - cycle + 1).abs().argsort()[:1]].iloc[0]

    values = [
        ("Capacity", row.get("capacity"), "Ah", "Measured discharge capacity"),
        ("State of health", float(row.capacity) / 2.0 * 100.0 if pd.notna(row.capacity) else np.nan, "%", "Capacity retained vs. 2.0 Ah rated capacity"),
        ("Ohmic resistance · Re", row.get("Re"), "Ω", "Electrolyte / ohmic component"),
        ("Charge-transfer · Rct", row.get("Rct"), "Ω", "Charge-transfer component"),
        ("Remaining life", row.get("rul_1p4", row.get("rul")), "cycles", "Cycles to the 1.40 Ah threshold"),
        ("Discharge energy", row.get("E_total_Wh"), "Wh", "Energy delivered over this discharge"),
    ]
    st.markdown(f'<div class="cg-section-label">{cell} · CYCLE {cycle:03d}</div>', unsafe_allow_html=True)
    columns = st.columns(3)
    for index, (label, value, unit, help_text) in enumerate(values):
        display = f"{float(value):.3f} {unit}" if pd.notna(value) else "Not recorded"
        columns[index % 3].metric(label, display)
    st.caption("Reference values are taken from the NASA cycle record. Some cells have incomplete EIS or end-of-life observations.")

    st.markdown('<div class="cg-section-label">HOW EACH VALUE IS DERIVED</div>', unsafe_allow_html=True)
    cards = [
        ("Capacity & SOH", "Capacity is recorded during a full reference discharge. SOH rescales it against the nominal 2.0 Ah rating.", "SOH (%) = capacity ÷ 2.0 Ah × 100"),
        ("Capacity fade", "Shows the fraction of rated capacity no longer available.", "Fade (%) = (2.0 − capacity) ÷ 2.0 × 100"),
        ("RUL", "Counts discharge cycles until the first valid capacity observation at or below 1.40 Ah.", "RUL = threshold cycle − current cycle"),
        ("Discharge energy", "Integrates terminal voltage and discharge current over elapsed time.", "Energy (Wh) = ∫ V(t) × |I(t)| dt ÷ 3600"),
        ("Re & Rct", "Impedance values are recorded during electrochemical impedance sweeps near the discharge cycle.", "Reported in ohms (Ω)"),
        ("SOC", "Instantaneous charge remaining. It requires a known starting charge state and current integration.", "SOC cannot be derived from capacity alone"),
    ]
    cols = st.columns(3)
    for i, (title, description, formula) in enumerate(cards):
        cols[i % 3].markdown(f'<div class="cg-formula-card"><div class="cg-formula-kicker">{i+1:02d}</div><h3>{title}</h3><p>{description}</p><code>{formula}</code></div>', unsafe_allow_html=True)

    st.markdown('<div class="cg-section-label">CYCLE HISTORY</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cell_df.disc_idx + 1, y=cell_df.capacity, name="Capacity (Ah)", mode="lines",
                             line={"color": "#42D6C3", "width": 3}, fill="tozeroy", fillcolor="rgba(66,214,195,.10)"))
    fig.add_hline(y=1.4, line_dash="dash", line_color="#F2738D", annotation_text="1.40 Ah threshold")
    fig.add_trace(go.Scatter(x=[cycle], y=[row.capacity], name="Selected cycle", mode="markers",
                             marker={"color": "#FFFFFF", "size": 11, "line": {"color": "#42D6C3", "width": 3}}))
    fig.update_layout(title="Capacity remaining through the battery's life", xaxis_title="Discharge cycle", yaxis_title="Capacity (Ah)",
                      height=400, margin={"l": 10, "r": 10, "t": 50, "b": 10}, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"color": "#D5E1F2"}, showlegend=False)
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)")
    st.plotly_chart(fig, use_container_width=True)
