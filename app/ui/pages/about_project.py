"""Friendly project and battery fundamentals page."""

import streamlit as st

from ui.components import render_thevenin_circuit_svg, render_html


def render_about_project(metrics: dict):
    dataset = metrics.get("expanded_dataset_validation", {}).get("dataset", {})
    st.markdown("""
    <section class="cg-page-hero">
      <div class="cg-eyebrow">ABOUT CELLGUARD <span>BATTERY FUNDAMENTALS</span></div>
      <h1>Understand the cell.<br/>Read its signals.</h1>
      <p>CellGuard turns ordinary discharge measurements into a clear view of battery capacity, resistance, energy, and service life.</p>
    </section>
    """, unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("NASA cells in archive", dataset.get("unique_files", 34))
    b.metric("Discharge records", f"{dataset.get('usable_discharge_rows', 2538):,}")
    c.metric("Reference chemistry", "Li-ion 18650")

    st.markdown('<div class="cg-section-label">A SIMPLE ELECTRICAL VIEW</div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        render_html(render_thevenin_circuit_svg())
    with right:
        st.markdown("""
        <div class="cg-formula-card">
          <div class="cg-formula-kicker">CELL MODEL</div>
          <h3>Voltage under load</h3>
          <p>A battery's terminal voltage falls under load because of its internal resistance and slower electrochemical polarization.</p>
          <code>Vterminal = OCV − I·R₀ − Vpolarization</code>
          <p style="margin-top:14px">A quick voltage drop reflects ohmic resistance. Slower voltage recovery and curve changes reveal additional cell dynamics.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="cg-section-label">WHAT THE READINGS TELL YOU</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    items = [
        ("Capacity", "How much charge the cell can deliver during a discharge."),
        ("SOH", "Capacity retained compared with the rated capacity."),
        ("Re / Rct", "The ohmic and charge-transfer parts of internal resistance."),
        ("Energy", "Usable electrical energy delivered during discharge."),
    ]
    for col, (name, description) in zip(cols, items):
        col.markdown(f'<div class="cg-nav-card"><span>BATTERY SIGNAL</span><h3>{name}</h3><p>{description}</p></div>', unsafe_allow_html=True)
    st.caption("Reference data: NASA Ames Prognostics Center of Excellence Li-ion Battery Data Set.")
