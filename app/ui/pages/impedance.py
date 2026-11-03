"""Resistance history and relationship explorer."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def _style(fig, height=470):
    fig.update_layout(height=height, margin={"l": 8, "r": 12, "t": 48, "b": 10},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#D5E1F2"}, legend={"orientation": "h", "y": 1.12, "x": 0})
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)")
    return fig


def render_impedance(predictor, feature_table: pd.DataFrame, metrics: dict):
    st.markdown("""
    <section class="cg-page-hero">
      <div class="cg-eyebrow">ELECTRICAL BEHAVIOR <span>RESISTANCE OVER TIME</span></div>
      <h1>Impedance history</h1>
      <p>Track the cell's ohmic and charge-transfer resistance alongside capacity through repeated discharge cycles.</p>
    </section>
    """, unsafe_allow_html=True)
    cells = sorted(feature_table.cell.dropna().unique()) if not feature_table.empty else []
    cell = st.selectbox("Battery", cells, key="impedance_history_cell")
    data = feature_table[feature_table.cell == cell].sort_values("disc_idx").copy()
    if data.empty:
        st.info("No impedance observations are available for this battery.")
        return
    cycle = st.slider("Focus cycle", 1, int(data.disc_idx.max()) + 1, min(50, int(data.disc_idx.max()) + 1), key="impedance_history_cycle")
    row = data.iloc[(data.disc_idx - cycle + 1).abs().argsort()[:1]].iloc[0]
    re_value, rct_value = row.get("Re"), row.get("Rct")

    st.markdown('<div class="cg-section-label">FOCUSED READINGS</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ohmic resistance · Re", f"{re_value:.4f} Ω" if pd.notna(re_value) else "Not recorded")
    c2.metric("Charge-transfer · Rct", f"{rct_value:.4f} Ω" if pd.notna(rct_value) else "Not recorded")
    c3.metric("Capacity", f"{row.capacity:.3f} Ah" if pd.notna(row.capacity) else "—")
    c4.metric("Cycle", f"{cycle:03d}")

    st.markdown('<div class="cg-section-label">RESISTANCE TRAJECTORIES</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.13,
                        subplot_titles=("Ohmic resistance · Re", "Charge-transfer resistance · Rct"))
    fig.add_trace(go.Scatter(x=data.disc_idx + 1, y=data.Re, name="Re", mode="lines+markers", marker={"size": 4}, line={"color": "#42D6C3", "width": 2.8}), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.disc_idx + 1, y=data.Rct, name="Rct", mode="lines+markers", marker={"size": 4}, line={"color": "#CA8BFA", "width": 2.8}), row=2, col=1)
    fig.add_vline(x=cycle, line_dash="dot", line_color="#FFFFFF")
    fig.update_yaxes(title_text="Ω", row=1, col=1)
    fig.update_yaxes(title_text="Ω", row=2, col=1)
    fig.update_xaxes(title_text="Discharge cycle", row=2, col=1)
    st.plotly_chart(_style(fig, 570), use_container_width=True)

    st.markdown('<div class="cg-section-label">CAPACITY AND RESISTANCE</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    for col, target, color, title in [(left, "Re", "#42D6C3", "Capacity vs Re"), (right, "Rct", "#CA8BFA", "Capacity vs Rct")]:
        valid = data[["capacity", target, "disc_idx"]].dropna()
        fig = go.Figure(go.Scatter(x=valid.capacity, y=valid[target], mode="markers",
                                   marker={"color": valid.disc_idx, "colorscale": [[0, color], [1, "#F3AA58"]], "size": 8,
                                           "showscale": True, "colorbar": {"title": "Cycle"}},
                                   hovertemplate="Capacity %{x:.3f} Ah<br>Resistance %{y:.4f} Ω<extra></extra>"))
        fig.update_layout(title=title, xaxis_title="Capacity (Ah)", yaxis_title="Resistance (Ω)")
        col.plotly_chart(_style(fig, 330), use_container_width=True)
    st.caption("Resistance readings are intermittent because impedance sweeps were not recorded on every discharge cycle.")
