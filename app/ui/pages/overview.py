"""Immersive CellGuard overview and battery history dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PALETTE = ["#42D6C3", "#7798FF", "#F3AA58", "#CA8BFA", "#F2738D", "#65B9F5"]


def _style(fig, height=330):
    fig.update_layout(
        height=height, margin={"l": 8, "r": 14, "t": 92, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#D5E1F2", "family": "Inter, sans-serif"},
        legend={"orientation": "h", "y": 1.13, "x": 0, "xanchor": "left", "yanchor": "bottom"},
        xaxis={"gridcolor": "rgba(148,163,184,.12)", "zerolinecolor": "rgba(148,163,184,.15)"},
        yaxis={"gridcolor": "rgba(148,163,184,.12)", "zerolinecolor": "rgba(148,163,184,.15)"},
    )
    return fig


def render_overview(predictor, metrics: dict, feature_table: pd.DataFrame, workspace_pages: dict | None = None):
    st.markdown("""
    <section class="cg-dashboard-hero">
      <div class="cg-hero-orb cg-orb-one"></div><div class="cg-hero-orb cg-orb-two"></div>
      <div class="cg-hero-art" aria-hidden="true">
        <svg viewBox="0 0 420 270" role="presentation">
          <defs>
            <linearGradient id="batteryGlow" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#58E0CC"/><stop offset="1" stop-color="#6B91FF"/></linearGradient>
            <linearGradient id="batteryFill" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#42D6C3" stop-opacity=".12"/><stop offset="1" stop-color="#42D6C3" stop-opacity=".72"/></linearGradient>
          </defs>
          <path d="M27 219 C82 197 106 208 149 164 S226 180 272 112 S347 122 398 49" fill="none" stroke="url(#batteryGlow)" stroke-width="3" stroke-linecap="round" opacity=".82"/>
          <path d="M27 235 C84 213 111 225 154 181 S229 195 277 127 S350 138 402 65" fill="none" stroke="#86A6FF" stroke-width="1" stroke-dasharray="4 8" opacity=".4"/>
          <g transform="translate(184 38) rotate(8 78 90)">
            <rect x="53" y="12" width="54" height="13" rx="5" fill="url(#batteryGlow)" opacity=".9"/>
            <rect x="20" y="24" width="120" height="190" rx="25" fill="#0B1B36" fill-opacity=".82" stroke="url(#batteryGlow)" stroke-width="2.5"/>
            <rect x="31" y="91" width="98" height="111" rx="15" fill="url(#batteryFill)"/>
            <path d="M88 57 66 112h20l-9 43 30-59H86l12-39Z" fill="#A5FFF0" opacity=".95"/>
            <path d="M42 78h76" stroke="#A5BCE0" stroke-opacity=".4" stroke-width="1"/>
            <circle cx="126" cy="45" r="3" fill="#77F1D9"/>
          </g>
          <circle cx="73" cy="73" r="4" fill="#8FA9FF" opacity=".7"/><circle cx="349" cy="205" r="5" fill="#54DDC8" opacity=".65"/>
        </svg>
      </div>
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
        fig.update_layout(title={"text": "Capacity fade across selected cells", "x": 0.01, "xanchor": "left", "y": 0.99, "yanchor": "top"},
                          xaxis_title="Discharge cycle", yaxis_title="Capacity (Ah)")
        st.plotly_chart(_style(fig, 390), use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cell_df.disc_idx + 1, y=cell_df.Re, name="Re · Ohmic", mode="lines+markers",
                                 marker={"size": 4}, line={"color": "#42D6C3", "width": 2.5}))
        fig.add_trace(go.Scatter(x=cell_df.disc_idx + 1, y=cell_df.Rct, name="Rct · Charge transfer", mode="lines+markers",
                                 marker={"size": 4}, line={"color": "#CA8BFA", "width": 2.5}))
        fig.add_vline(x=cycle_number, line_dash="dot", line_color="#FFFFFF")
        fig.update_layout(title={"text": f"Resistance history · {cell}", "x": 0.01, "xanchor": "left", "y": 0.99, "yanchor": "top"},
                          xaxis_title="Discharge cycle", yaxis_title="Resistance (Ω)")
        st.plotly_chart(_style(fig, 390), use_container_width=True)

    st.markdown('<div class="cg-section-label">EXPLORE THE WORKSPACE</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="cg-nav-card"><span>01</span><h3>Battery check</h3><p>Enter discharge measurements and review a complete health snapshot.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="cg-nav-card"><span>02</span><h3>Calculation metrics</h3><p>Understand capacity, health, resistance, life, and energy values.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="cg-nav-card"><span>03</span><h3>Battery life</h3><p>Follow capacity fade and explore the end-of-life horizon.</p></div>', unsafe_allow_html=True)
    if workspace_pages:
        c1.page_link(workspace_pages["Battery Health Check"], label="Open battery check", icon="🩺", use_container_width=True)
        c2.page_link(workspace_pages["Calculation Metrics"], label="Open calculation metrics", icon="🧮", use_container_width=True)
        c3.page_link(workspace_pages["Battery Life"], label="Open battery life", icon="⏳", use_container_width=True)
