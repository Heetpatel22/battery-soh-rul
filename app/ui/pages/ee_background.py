"""
ee_background.py - Electrical Engineering Analysis, Thevenin ECM & Power Loss Calculator.
Dedicated EEE section for 4th-year final year project and viva defense.
Presents:
- 1st-Order Thevenin Equivalent Circuit Model (ECM) schematic & governing equations
- Interactive Terminal Voltage Drop & Internal Power Loss (I²R) Calculator
- Physical degradation electrochemistry (SEI growth, lithium plating, impedance rise)
- Electrochemical Impedance Spectroscopy (EIS) Nyquist interpretation
- Examiner Viva Q&A Guide
"""

import math
import numpy as np
import pandas as pd
import streamlit as st

from ui.components import (
    render_compact_header,
    render_category_legend,
    render_kpi,
    render_thevenin_circuit_svg,
    render_html,
)


def render_ee_background():
    render_compact_header(
        page_title="Electrical Engineering Analysis & Power Loss Calculator",
        selected_battery="18650 Li-ion Cell",
        data_source="1st-Order Thevenin Equivalent Circuit & Heat Dissipation Physics",
        model_status="BMS Firmware Architecture · EEE Final-Year Focus",
    )

    render_category_legend()

    # Introduction Banner
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-left:4px solid #0D9488; border-radius:6px; padding:14px 18px; margin-bottom:18px; box-shadow:0 1px 3px rgba(15,23,42,0.04);">
      <div style="font-size:0.90rem; font-weight:800; color:#0F172A; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:4px;">
        Core EEE Concept: Why Impedance (Re & Rct) Dictates Battery Performance
      </div>
      <div style="font-size:0.82rem; color:#334155; line-height:1.6;">
        In electrical power engineering, internal battery resistance is not just an indicator of aging; it directly governs <strong>terminal voltage drop (voltage sag)</strong>, <strong>internal I²R Joule heating</strong>, and <strong>energy delivery efficiency</strong>.
        As a lithium-ion cell ages, interfacial charge-transfer resistance (<em>Rct</em>) and ohmic bulk resistance (<em>Re</em>) escalate, constraining acceleration and generating destructive heat.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # SECTION 1: THEVENIN EQUIVALENT CIRCUIT MODEL
    st.markdown('<div class="cg-section-label">1ST-ORDER THEVENIN EQUIVALENT CIRCUIT MODEL (ECM)</div>', unsafe_allow_html=True)

    render_html(render_thevenin_circuit_svg())

    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:16px 20px; margin:14px 0;">
      <h4 style="font-size:0.92rem; font-weight:800; color:#0F172A; text-transform:uppercase; margin-top:0;">
        Governing State-Space Equations
      </h4>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin:10px 0;">
        <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:6px; padding:12px; font-family:'JetBrains Mono', monospace; font-size:0.88rem; color:#0F172A;">
          <strong>Terminal Voltage Output Equation:</strong><br/>
          V_terminal(t) = OCV(SOC) - I(t) &times; R₀ - V₁(t)
        </div>
        <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:6px; padding:12px; font-family:'JetBrains Mono', monospace; font-size:0.88rem; color:#0F172A;">
          <strong>Polarization Dynamic State Equation:</strong><br/>
          dV₁(t) / dt = - V₁(t) / (R₁ &times; C₁) + I(t) / C₁
        </div>
      </div>
      <div style="font-size:0.82rem; color:#475569; line-height:1.6;">
        <ul>
          <li><strong>OCV(SOC):</strong> Open-Circuit Voltage as a non-linear function of State of Charge.</li>
          <li><strong>R₀ &approx; Re:</strong> Ohmic series resistance (bulk electrolyte, separator, active material bulk, current collector foils). Governs instantaneous step voltage drop.</li>
          <li><strong>R₁ &approx; Rct:</strong> Charge-transfer resistance (electrochemical activation overpotential at electrode-electrolyte interface).</li>
          <li><strong>C₁ &approx; C_dl:</strong> Double-layer capacitance across the electrode-electrolyte interface (typically 1,000 to 5,000 Farads). Creates exponential transient response with time constant &tau; = R₁ &times; C₁.</li>
          <li><strong>Steady-State Condition:</strong> Under sustained constant current, dV₁/dt &to; 0, so V₁ = I &times; R₁, and total voltage drop simplifies to: <code>ΔV_total = I &times; (Re + Rct)</code>.</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # SECTION 2: INTERACTIVE VOLTAGE DROP & POWER LOSS CALCULATOR
    st.markdown('<div class="cg-section-label">INTERACTIVE VOLTAGE DROP & INTERNAL POWER LOSS CALCULATOR</div>', unsafe_allow_html=True)

    with st.container():
        st.caption("Adjust operating load current and resistance values to calculate instantaneous voltage drop, continuous thermal dissipation (I²R), and electrical efficiency.")

        calc_c1, calc_c2, calc_c3, calc_c4 = st.columns(4)
        with calc_c1:
            load_current = st.slider("Load Current I (Amperes)", 0.5, 10.0, 2.0, 0.5, key="calc_curr_slider", help="Discharge current magnitude. 2.0A corresponds to 1C rate for 2.0Ah cell.")
        with calc_c2:
            input_re_mohm = st.number_input("Ohmic Re (mΩ)", 10.0, 200.0, 52.0, 1.0, key="calc_re_input", help="Bulk ohmic resistance (R0).")
        with calc_c3:
            input_rct_mohm = st.number_input("Charge-Transfer Rct (mΩ)", 20.0, 300.0, 78.0, 1.0, key="calc_rct_input", help="Interfacial charge-transfer resistance (R1).")
        with calc_c4:
            input_ocv = st.number_input("Open-Circuit Voltage OCV (V)", 3.0, 4.3, 3.90, 0.05, key="calc_ocv_input", help="Rest potential at current state of charge.")

        # Calculations
        re_ohm = input_re_mohm / 1000.0
        rct_ohm = input_rct_mohm / 1000.0
        rint_ohm = re_ohm + rct_ohm

        delta_v_ohmic = load_current * re_ohm
        delta_v_ct = load_current * rct_ohm
        delta_v_total = load_current * rint_ohm
        v_terminal = max(0.0, input_ocv - delta_v_total)

        p_ohmic = (load_current ** 2) * re_ohm
        p_ct = (load_current ** 2) * rct_ohm
        p_total = (load_current ** 2) * rint_ohm

        efficiency = (v_terminal / input_ocv) * 100.0 if input_ocv > 0 else 0.0
        # Energy lost in 1 hour discharge (Wh)
        e_lost_wh = p_total * 1.0

        # Display calculated KPIs
        k_c1, k_c2, k_c3, k_c4 = st.columns(4)
        with k_c1:
            render_kpi("Ohmic Drop (ΔV_ohmic)", f"{delta_v_ohmic:.3f}", "V", subtext=f"I × Re = {load_current:.1f}A × {input_re_mohm:.1f}mΩ", category="calculated")
        with k_c2:
            render_kpi("Polarization Drop (ΔV_ct)", f"{delta_v_ct:.3f}", "V", subtext=f"I × Rct = {load_current:.1f}A × {input_rct_mohm:.1f}mΩ", category="calculated")
        with k_c3:
            render_kpi("Total Terminal Sag (ΔV)", f"{delta_v_total:.3f}", "V", subtext=f"Terminal V: {v_terminal:.3f} V", status="Acceptable" if delta_v_total < 0.40 else "High Sag", category="calculated")
        with k_c4:
            render_kpi("Heat Dissipation Rate (P_loss)", f"{p_total:.3f}", "Watts", subtext=f"I² × R_int = ({load_current:.1f}A)² × {rint_ohm*1000:.1f}mΩ", status="Safe" if p_total < 2.0 else "High Heat", category="calculated")

        # Summary Breakdown Box
        st.markdown(f"""
        <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:6px; padding:12px 16px; margin-top:8px; font-size:0.80rem; color:#1E293B; line-height:1.6;">
          <strong>Electrical Engineering Results Summary:</strong>
          At a discharge current of <strong>{load_current:.1f} A</strong>, internal resistance causes an immediate ohmic voltage drop of <strong>{delta_v_ohmic*1000.0:.1f} mV</strong> and an overpotential sag of <strong>{delta_v_ct*1000.0:.1f} mV</strong>, pulling terminal voltage down from <strong>{input_ocv:.2f} V</strong> to <strong>{v_terminal:.3f} V</strong>.
          Electrical delivery efficiency is <strong>{efficiency:.1f}%</strong>.
          The cell generates internal Joule heat at a continuous rate of <strong>{p_total:.3f} Watts</strong> ({p_ohmic:.3f} W ohmic + {p_ct:.3f} W charge-transfer), dissipating <strong>{e_lost_wh:.3f} Wh</strong> ({e_lost_wh * 3600.0:.0f} Joules) of thermal energy per hour of load.
        </div>
        """, unsafe_allow_html=True)

    # SECTION 3: ELECTROCHEMICAL DEGRADATION MECHANISMS
    st.markdown('<div class="cg-section-label">ELECTROCHEMICAL AGING & DEGRADATION MECHANISMS</div>', unsafe_allow_html=True)

    deg_c1, deg_c2 = st.columns(2, gap="medium")
    with deg_c1:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:16px; height:100%;">
          <h4 style="font-size:0.90rem; font-weight:800; color:#0F172A; text-transform:uppercase; margin-top:0;">
            1. Solid Electrolyte Interphase (SEI) Layer Growth
          </h4>
          <div style="font-size:0.82rem; color:#475569; line-height:1.6;">
            <strong>Primary Cause of Rct Increase:</strong>
            The graphite negative electrode operates at an electrochemical potential (~0.1 V vs Li/Li⁺) outside the thermodynamic stability window of non-aqueous carbonate electrolytes.
            Solvent molecules (ethylene carbonate) undergo irreversible reduction, consuming active lithium ions and forming a passivating film (SEI).
            <br/><br/>
            Over cycling, continuous volume expansion and contraction cracks this film, exposing fresh graphite and causing continuous SEI regeneration. This thickens the barrier, increasing <strong>Rct</strong> and causing capacity fade.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with deg_c2:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:16px; height:100%;">
          <h4 style="font-size:0.90rem; font-weight:800; color:#0F172A; text-transform:uppercase; margin-top:0;">
            2. Lithium Plating & Thermal Runaway Hazard
          </h4>
          <div style="font-size:0.82rem; color:#475569; line-height:1.6;">
            <strong>Why High Impedance is Hazardous:</strong>
            As internal resistance climbs, internal power dissipation ($P = I^2 R$) generates substantial heat. If heat generation exceeds the pack cooling capacity, temperature escalates.
            <br/><br/>
            Furthermore, under high charge currents or low temperatures, high overpotentials push anode potential below 0 V vs Li/Li⁺, causing metallic lithium to plate onto the anode surface rather than intercalating. Metallic lithium forms dendrites that can pierce the polymeric separator, causing micro-short circuits and catastrophic <strong>thermal runaway</strong>.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # SECTION 4: ELECTROCHEMICAL IMPEDANCE SPECTROSCOPY (EIS) NYQUIST PLOT EXPLANATION
    st.markdown('<div class="cg-section-label">ELECTROCHEMICAL IMPEDANCE SPECTROSCOPY (EIS) NYQUIST INTERPRETATION</div>', unsafe_allow_html=True)
    st.markdown(r"""
    <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:16px 20px;">
      <h4 style="font-size:0.92rem; font-weight:800; color:#0F172A; text-transform:uppercase; margin-top:0;">
        How NASA Measures Re and Rct via Frequency Sweeps (0.1 Hz → 5 kHz)
      </h4>
      <div style="font-size:0.82rem; color:#334155; line-height:1.7;">
        In the NASA laboratory experiments, an AC sinusoidal excitation current $I(t) = I_0 \sin(\omega t)$ was injected into the battery across frequencies ranging from 0.1 Hz to 5 kHz. The resulting voltage response $V(t) = V_0 \sin(\omega t + \theta)$ yields the complex impedance:
        $$Z(\omega) = \frac{V(\omega)}{I(\omega)} = Z_{\text{real}} + j Z_{\text{imag}}$$
        Plotting $-Z_{\text{imag}}$ versus $Z_{\text{real}}$ produces the classic <strong>Nyquist Arc</strong>:
        <ul>
          <li><strong>High-Frequency Intercept on the Real Axis ($Z_{\text{imag}} = 0$):</strong> At frequencies above 1 kHz, capacitive reactance is shorted ($X_C \to 0$), leaving only the bulk series resistance: <code>Z_real = Re</code>.</li>
          <li><strong>Semicircle Arc Diameter:</strong> In the mid-frequency range (10 Hz to 1 kHz), the parallel RC network produces a semi-circular arc. The diameter of this arc corresponds directly to the Charge-Transfer Resistance: <code>Diameter = Rct</code>.</li>
          <li><strong>Low-Frequency Tail (45° Slope):</strong> At low frequencies (< 1 Hz), semi-infinite solid-state diffusion of lithium ions through the active material particles appears as the <strong>Warburg Impedance</strong> ($Z_W$).</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # SECTION 5: VIVA PREPARATION GUIDE
    st.markdown('<div class="cg-section-label">4TH-YEAR EEE FINAL PROJECT VIVA PREPARATION GUIDE</div>', unsafe_allow_html=True)
    with st.expander("🎓 Examiner Viva Q&A Cheat Sheet", expanded=True):
        st.markdown(r"""
        #### Q1: What is the main objective of your project?
        **Answer:** To develop a machine learning-based diagnostic system that estimates the internal electrochemical impedance—specifically Ohmic Resistance ($R_e$) and Charge-Transfer Resistance ($R_{ct}$)—directly from observable, partial-window discharge parameters (voltage, current, temperature, and cycle number) without requiring expensive offline Electrochemical Impedance Spectroscopy (EIS) laboratory equipment.

        #### Q2: What is the physical difference between Re and Rct?
        **Answer:** 
        - **$R_e$ (Ohmic Resistance)** is the bulk electrical and ionic resistance of the electrolyte solvent, separator, active material bulk, and metallic foil current collectors. It acts as a pure resistor causing an instantaneous voltage drop $\Delta V = I R_e$.
        - **$R_{ct}$ (Charge-Transfer Resistance)** is the kinetic activation resistance that lithium ions encounter when transferring across the interface between the electrolyte and the active electrode material. It is in parallel with the double-layer capacitance ($C_{dl}$) and governs dynamic polarization.

        #### Q3: Why does Rct grow much faster than Re during battery cycling?
        **Answer:** Because the primary degradation mechanism in lithium-ion cells is the progressive thickening and cracking of the Solid Electrolyte Interphase (SEI) layer on the graphite anode, along with loss of cyclable lithium inventory. This interfacial film directly impedes lithium-ion intercalation, causing $R_{ct}$ to double or triple over the battery's lifespan, whereas bulk electrolyte volume and current collectors change relatively little.

        #### Q4: How did you ensure your ML models do not suffer from data leakage?
        **Answer:** By implementing strict **Leave-One-Cell-Out (LOCO) cross-validation**. In battery cycling, adjacent cycles from the same battery cell are temporally correlated. Standard random train/test splits leak cycles from the test cell into the training set, producing falsely optimistic scores. Under LOCO, the models are trained on 3 cells and evaluated on the 4th held-out cell across rotating folds. Furthermore, neither $R_e$ nor $R_{ct}$ was included in the feature set.

        #### Q5: How does internal resistance affect thermal management in an EV Battery Management System?
        **Answer:** Internal resistance causes continuous Joule heating at a rate of $P_{\text{loss}} = I^2 (R_e + R_{ct})$. For example, at a 2.0 A load and 130 mΩ total resistance, the cell dissipates approximately 0.52 Watts of thermal energy. As resistance doubles in an aged pack, heat dissipation doubles for the exact same operating current, requiring increased cooling power and raising the risk of thermal runaway.
        """)
