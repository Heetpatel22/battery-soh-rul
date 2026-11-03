"""Battery life planning and capacity trajectory page."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_life_rul(predictor, feature_table: pd.DataFrame, metrics: dict):
    st.markdown("""
    <section class="cg-page-hero">
      <div class="cg-eyebrow">LIFETIME PLANNING <span>CAPACITY · SERVICE HORIZON</span></div>
      <h1>Battery life outlook</h1>
      <p>Follow capacity fade and review the estimated distance to the 1.40 Ah service threshold.</p>
    </section>
    """, unsafe_allow_html=True)
    cells = sorted(feature_table.cell.dropna().unique()) if not feature_table.empty else []
    cell = st.selectbox("Battery", cells, key="life_cell")
    data = feature_table[feature_table.cell == cell].sort_values("disc_idx").copy()
    if data.empty:
        st.warning("No discharge history is available for this battery.")
        return
    cycle = st.slider("Review cycle", 1, int(data.disc_idx.max()) + 1, min(50, int(data.disc_idx.max()) + 1), key="life_cycle")
    row = data.iloc[(data.disc_idx - cycle + 1).abs().argsort()[:1]].iloc[0]
    history = data[data.disc_idx < cycle].capacity.dropna().tail(10).tolist()
    feature_names = predictor.model_capacity["features"]
    features = {name: row.get(name, np.nan) for name in feature_names}
    try:
        result = predictor.predict_all(features, disc_idx=cycle - 1, cap_history=history)
        forecast = result.get("predictions", {}).get("RUL", {}) if result.get("success") else {}
    except Exception:
        forecast = {}
    predicted_rul = forecast.get("value")
    actual_rul = row.get("rul_1p4", row.get("rul"))

    st.markdown('<div class="cg-section-label">SERVICE HORIZON</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated cycles remaining", "End of life" if forecast.get("is_eol") else (f"{predicted_rul} cycles" if predicted_rul is not None else "Unavailable"))
    c2.metric("Recorded capacity", f"{row.capacity:.3f} Ah" if pd.notna(row.capacity) else "—")
    c3.metric("Reference cycle distance", f"{int(actual_rul)} cycles" if pd.notna(actual_rul) else "No threshold label")

    history_for_plot = data[data.disc_idx < cycle]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_for_plot.disc_idx + 1, y=history_for_plot.capacity, name="Recorded capacity",
                             mode="lines+markers", marker={"size": 4}, line={"color": "#42D6C3", "width": 2.8}))
    fig.add_hline(y=1.4, line_dash="dash", line_color="#F2738D", annotation_text="1.40 Ah threshold")
    fig.add_trace(go.Scatter(x=[cycle], y=[row.capacity], name="Current cycle", mode="markers",
                             marker={"color": "#FFFFFF", "size": 11, "line": {"color": "#42D6C3", "width": 3}}))
    if predicted_rul is not None and not forecast.get("is_eol") and pd.notna(row.capacity):
        eol_cycle = cycle + int(predicted_rul)
        fig.add_trace(go.Scatter(x=[cycle, eol_cycle], y=[row.capacity, 1.4], name="Estimated horizon",
                                 mode="lines", line={"color": "#F3AA58", "width": 2, "dash": "dot"}))
    fig.update_layout(title=f"Capacity trajectory · {cell}", xaxis_title="Discharge cycle", yaxis_title="Capacity (Ah)",
                      height=450, margin={"l": 8, "r": 12, "t": 48, "b": 10}, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"color": "#D5E1F2"}, legend={"orientation": "h", "y": 1.12, "x": 0})
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="cg-formula-card"><div class="cg-formula-kicker">END OF LIFE GUIDE</div><h3>What does 1.40 Ah mean?</h3><p>The reference study defines end of life as a 30% fade from the cell’s nominal 2.0 Ah capacity. Cycle-to-cycle capacity can fluctuate, so use the horizon as a planning aid.</p></div>', unsafe_allow_html=True)
