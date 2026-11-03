"""Immersive CellGuard overview and battery history dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PALETTE = ["#42D6C3", "#7798FF", "#F3AA58", "#CA8BFA", "#F2738D", "#65B9F5"]


def _style(fig, height=330):
    fig.update_layout(
        height=height, margin={"l": 8, "r": 14, "t": 42, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#D5E1F2", "family": "Inter, sans-serif"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        xaxis={"gridcolor": "rgba(148,163,184,.12)", "zerolinecolor": "rgba(148,163,184,.15)"},
        yaxis={"gridcolor": "rgba(148,163,184,.12)", "zerolinecolor": "rgba(148,163,184,.15)"},
    )
    return fig


def render_overview(predictor, metrics: dict, feature_table: pd.DataFrame):
    st.markdown("""
    <section class="cg-dashboard-hero">
      <div class="cg-hero-orb cg-orb-one"></div><div class="cg-hero-orb cg-orb-two"></div>
      <div class="cg-eyebrow">CELLGUARD <span>BATTERY INTELLIGENCE</span></div>
      <h1>See the health behind<br/><em>every cycle.</em></h1>
      <p>Explore battery condition, electrical behavior, and remaining life across the NASA cell archive.</p>
      <div class="cg-hero-foot"><span><i></i> REFERENCE ARCHIVE CONNECTED</span><span>NASA AMES · LI-ION 18650</span></div>
    </section>
    """, unsafe_allow_html=True)

    if feature_table.empty:
        st.error("Battery history is unavailable. Check the feature table in the models folder.")
        return

    cohort_cells = sorted(feature_table["cell"].dropna().unique().tolist())
    reference_cells = [c for c in ["B0005", "B0006", "B0007", "B0018"] if c in cohort_cells]
    cells = st.multiselect("Compare battery histories", cohort_cells,
                           default=reference_cells, max_selections=8,
                           help="Choose up to eight cells to compare capacity fade.")
    if not cells:
        st.info("Choose at least one battery to explore its history.")
        return

    cell = st.selectbox("Focused battery", cells, index=0, key="overview_focus_cell")
    cell_df = feature_table[feature_table.cell == cell].sort_values("disc_idx").reset_index(drop=True)
    if cell_df.empty:
        st.warning("No cycle records are available for this battery.")
        return
    max_cycle = max(1, int(cell_df.disc_idx.max()) + 1)
    cycle_number = st.slider("Cycle", 1, max_cycle, min(max_cycle, 50), key="overview_cycle")
    selected = cell_df.iloc[(cell_df.disc_idx - (cycle_number - 1)).abs().argsort()[:1]].iloc[0]
    capacity = float(selected.capacity) if pd.notna(selected.capacity) else np.nan
    soh = capacity / 2.0 * 100.0 if np.isfinite(capacity) else np.nan

    st.markdown('<div class="cg-section-label">LIVE BATTERY SNAPSHOT</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Capacity", f"{capacity:.3f} Ah" if np.isfinite(capacity) else "—")
    m2.metric("State of health", f"{soh:.1f}%" if np.isfinite(soh) else "—")
    m3.metric("Ohmic resistance", f"{selected.Re:.4f} Ω" if pd.notna(selected.Re) else "—")
    m4.metric("Charge-transfer", f"{selected.Rct:.4f} Ω" if pd.notna(selected.Rct) else "—")
    m5.metric("Discharge energy", f"{selected.E_total_Wh:.2f} Wh" if pd.notna(selected.E_total_Wh) else "—")

    if np.isfinite(soh):
        if soh <= 70:
            st.markdown(f'<div class="cg-health-banner cg-health-low"><span>●</span> End-of-life capacity range · {soh:.1f}% health at cycle {cycle_number}</div>', unsafe_allow_html=True)
        elif soh < 80:
            st.markdown(f'<div class="cg-health-banner cg-health-mid"><span>●</span> Degradation detected · {soh:.1f}% health at cycle {cycle_number}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="cg-health-banner cg-health-good"><span>●</span> Healthy capacity range · {soh:.1f}% health at cycle {cycle_number}</div>', unsafe_allow_html=True)

    st.markdown('<div class="cg-section-label">DEGRADATION & ELECTRICAL TRENDS</div>', unsafe_allow_html=True)
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        fig = go.Figure()
        for idx, cell_name in enumerate(cells):
            trace = feature_table[feature_table.cell == cell_name].sort_values("disc_idx")
            fig.add_trace(go.Scatter(x=trace.disc_idx + 1, y=trace.capacity, name=cell_name,
                                     mode="lines", line={"color": PALETTE[idx % len(PALETTE)], "width": 2.5},
                                     hovertemplate="Cycle %{x}<br>Capacity %{y:.3f} Ah<extra>%{fullData.name}</extra>"))
        fig.add_hline(y=1.4, line_dash="dash", line_color="#F2738D", annotation_text="1.40 Ah service threshold")
        fig.add_trace(go.Scatter(x=[cycle_number], y=[capacity], name="Focused cycle", mode="markers",
                                 marker={"color": "#FFFFFF", "size": 10, "line": {"color": "#42D6C3", "width": 3}}))
        fig.update_layout(title="Capacity fade across selected cells", xaxis_title="Discharge cycle", yaxis_title="Capacity (Ah)")
        st.plotly_chart(_style(fig, 390), use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cell_df.disc_idx + 1, y=cell_df.Re, name="Re · Ohmic", mode="lines+markers",
                                 marker={"size": 4}, line={"color": "#42D6C3", "width": 2.5}))
        fig.add_trace(go.Scatter(x=cell_df.disc_idx + 1, y=cell_df.Rct, name="Rct · Charge transfer", mode="lines+markers",
                                 marker={"size": 4}, line={"color": "#CA8BFA", "width": 2.5}))
        fig.add_vline(x=cycle_number, line_dash="dot", line_color="#FFFFFF")
        fig.update_layout(title=f"Resistance history · {cell}", xaxis_title="Discharge cycle", yaxis_title="Resistance (Ω)")
        st.plotly_chart(_style(fig, 390), use_container_width=True)

    st.markdown('<div class="cg-section-label">EXPLORE THE WORKSPACE</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="cg-nav-card"><span>01</span><h3>Battery check</h3><p>Upload a discharge trace and review a complete health snapshot.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="cg-nav-card"><span>02</span><h3>Calculation metrics</h3><p>Understand capacity, health, resistance, life, and energy values.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="cg-nav-card"><span>03</span><h3>Battery life</h3><p>Follow capacity fade and explore the end-of-life horizon.</p></div>', unsafe_allow_html=True)
