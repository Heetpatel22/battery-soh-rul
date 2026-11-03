"""Interactive missing-reading recovery exercise for impedance history."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_gap_filler(predictor, feature_table: pd.DataFrame, metrics: dict):
    st.markdown("""
    <section class="cg-page-hero">
      <div class="cg-eyebrow">IMPEDANCE TOOL <span>REPAIR A MISSING STRETCH</span></div>
      <h1>Fill missing readings</h1>
      <p>Hide a section of a recorded resistance history, restore it, and compare the recovered values with the original measurements.</p>
    </section>
    """, unsafe_allow_html=True)
    controls = st.columns(4)
    cells = sorted(feature_table.cell.dropna().unique()) if not feature_table.empty else []
    with controls[0]:
        cell = st.selectbox("Battery", cells, key="gap_cell")
    with controls[1]:
        target = st.selectbox("Reading", ["Re", "Rct"], key="gap_target")
    with controls[2]:
        mode = st.selectbox("Missing pattern", ["Short random gaps", "One long gap"])
    with controls[3]:
        fill_method = st.selectbox("Recovery method", ["Linear interpolation", "Carry previous value", "Median fill"])

    data = feature_table[(feature_table.cell == cell) & feature_table[target].notna()].sort_values("disc_idx").copy()
    if len(data) < 8:
        st.warning("This cell has too few resistance records for a gap exercise.")
        return
    cycles = data.disc_idx.to_numpy(int)
    actual = data[target].to_numpy(float)
    gap_size = st.slider("Gap size", min_value=2, max_value=max(2, min(20, len(data) - 2)), value=min(10, max(2, len(data) - 2)))
    rng = np.random.default_rng(42)
    if mode == "One long gap":
        start = int(rng.integers(1, max(2, len(data) - gap_size)))
        missing = np.arange(start, min(start + gap_size, len(data) - 1))
    else:
        choices = np.arange(1, len(data) - 1)
        missing = np.sort(rng.choice(choices, size=min(gap_size, len(choices)), replace=False))

    observed = actual.copy()
    observed[missing] = np.nan
    series = pd.Series(observed)
    if fill_method == "Linear interpolation":
        filled = series.interpolate(limit_direction="both").to_numpy()
    elif fill_method == "Carry previous value":
        filled = series.ffill().bfill().to_numpy()
    else:
        filled = series.fillna(float(np.nanmedian(observed))).to_numpy()
    mae = float(np.mean(np.abs(actual[missing] - filled[missing])))
    rmse = float(np.sqrt(np.mean((actual[missing] - filled[missing]) ** 2)))

    st.markdown('<div class="cg-section-label">RECOVERY SUMMARY</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Readings hidden", len(missing))
    b.metric("Average recovery error", f"{mae * 1000:.2f} mΩ")
    c.metric("Largest gap error", f"{rmse * 1000:.2f} mΩ", help="Root-mean-square error across the hidden points")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cycles, y=actual, name="Original record", mode="lines", line={"color": "#536783", "width": 1.8, "dash": "dot"}))
    fig.add_trace(go.Scatter(x=cycles, y=filled, name="Recovered history", mode="lines+markers", line={"color": "#42D6C3", "width": 2.8}, marker={"size": 4}))
    fig.add_trace(go.Scatter(x=cycles[missing], y=actual[missing], name="Hidden values · reference", mode="markers", marker={"color": "#F3AA58", "size": 9, "symbol": "x"}))
    fig.update_layout(title=f"{cell} · {target} history", xaxis_title="Discharge cycle", yaxis_title="Resistance (Ω)", height=430,
                      margin={"l": 8, "r": 12, "t": 48, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#D5E1F2"}, legend={"orientation": "h", "y": 1.12, "x": 0})
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This exercise hides known reference values so the recovered section can be compared with the original record.")
    export = pd.DataFrame({"cycle": cycles, "original_ohm": actual, "recovered_ohm": filled, "was_hidden": np.isin(np.arange(len(data)), missing)})
    st.download_button("Download recovered history", export.to_csv(index=False), file_name=f"{cell}_{target}_recovered.csv", mime="text/csv")
