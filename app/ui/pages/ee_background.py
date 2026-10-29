"""
ee_background.py - Electrical Engineering fundamentals and BMS architecture page.
Features an interactive SVG schematic of the 1st-order Thevenin Equivalent Circuit Model (ECM),
parameter mappings (Re -> R0, Rct -> R1), SOH/RUL definitions, and embedded BMS feasibility.
"""

import os
import streamlit as st
from ui.components import render_html

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "figures"))


def render_thevenin_circuit_svg() -> str:
    """Generate SVG schematic of 1st-order Thevenin Equivalent Circuit Model."""
    svg = """
    <div style="background: rgba(14, 20, 33, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 24px; text-align: center;">
      <svg width="680" height="240" viewBox="0 0 680 240" fill="none" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <!-- Definitions -->
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#22D3EE" />
          </marker>
        </defs>

        <!-- Main Wire Loop -->
        <path d="M 90 60 L 220 60" stroke="#93A0B8" stroke-width="2.5" />
        <path d="M 280 60 L 360 60" stroke="#93A0B8" stroke-width="2.5" />
        
        <!-- Parallel Branch Top (R1) -->
        <path d="M 360 60 L 360 25 L 400 25" stroke="#93A0B8" stroke-width="2.5" />
        <path d="M 460 25 L 500 25 L 500 60" stroke="#93A0B8" stroke-width="2.5" />
        
        <!-- Parallel Branch Bottom (C1) -->
        <path d="M 360 60 L 360 95 L 420 95" stroke="#93A0B8" stroke-width="2.5" />
        <path d="M 440 95 L 500 95 L 500 60" stroke="#93A0B8" stroke-width="2.5" />
        
        <!-- Output Wire Top -->
        <path d="M 500 60 L 610 60" stroke="#93A0B8" stroke-width="2.5" />
        
        <!-- Bottom Return Wire -->
        <path d="M 90 190 L 610 190" stroke="#93A0B8" stroke-width="2.5" />

        <!-- Current Arrow I(t) -->
        <path d="M 130 45 L 180 45" stroke="#22D3EE" stroke-width="2" marker-end="url(#arrow)" />
        <text x="155" y="38" fill="#22D3EE" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" text-anchor="middle">I(t)</text>

        <!-- Voltage Source (OCV) -->
        <line x1="90" y1="60" x2="90" y2="105" stroke="#93A0B8" stroke-width="2.5" />
        <line x1="90" y1="145" x2="90" y2="190" stroke="#93A0B8" stroke-width="2.5" />
        <circle cx="90" cy="125" r="20" stroke="#FFFFFF" stroke-width="2.5" fill="#141C2E" />
        <text x="90" y="121" fill="#FFFFFF" font-family="Inter, sans-serif" font-size="14" font-weight="700" text-anchor="middle">+</text>
        <text x="90" y="137" fill="#FFFFFF" font-family="Inter, sans-serif" font-size="14" font-weight="700" text-anchor="middle">-</text>
        <text x="90" y="165" fill="#38BDF8" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="700" text-anchor="middle">OCV(SOC)</text>

        <!-- Series Resistor R0 (Ohmic Resistance Re) -->
        <rect x="220" y="48" width="60" height="24" rx="4" fill="#1E293B" stroke="#22D3EE" stroke-width="2" />
        <text x="250" y="64" fill="#22D3EE" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" text-anchor="middle">R₀</text>
        <text x="250" y="90" fill="#93A0B8" font-family="'JetBrains Mono', monospace" font-size="11" text-anchor="middle">≈ Re (Ohmic)</text>

        <!-- Polarization Resistor R1 (Charge-Transfer Resistance Rct) -->
        <rect x="400" y="13" width="60" height="24" rx="4" fill="#1E293B" stroke="#A78BFA" stroke-width="2" />
        <text x="430" y="29" fill="#A78BFA" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" text-anchor="middle">R₁</text>
        <text x="430" y="4" fill="#93A0B8" font-family="'JetBrains Mono', monospace" font-size="10" text-anchor="middle">≈ Rct (Charge-Transfer)</text>

        <!-- Polarization Capacitor C1 -->
        <line x1="422" y1="83" x2="422" y2="107" stroke="#34D399" stroke-width="3" />
        <line x1="432" y1="83" x2="432" y2="107" stroke="#34D399" stroke-width="3" />
        <text x="427" y="125" fill="#34D399" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" text-anchor="middle">C₁</text>

        <!-- Terminals -->
        <circle cx="610" cy="60" r="5" fill="#22D3EE" stroke="#FFFFFF" stroke-width="1.5" />
        <circle cx="610" cy="190" r="5" fill="#93A0B8" stroke="#FFFFFF" stroke-width="1.5" />
        <text x="635" y="65" fill="#22D3EE" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="700">+</text>
        <text x="635" y="195" fill="#93A0B8" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="700">-</text>
        <text x="645" y="130" fill="#FFFFFF" font-family="'JetBrains Mono', monospace" font-size="14" font-weight="800">V_term(t)</text>
      </svg>
    </div>
    """
    return svg


def render_ee_background():
    """Render the EE Background page."""
    st.markdown("### Electrical Engineering Fundamentals & BMS Integration")
    st.markdown(
        "<p style='color: var(--text-secondary); margin-bottom: 20px; font-size: 0.92rem;'>"
        "Bridging machine learning predictions back to Equivalent Circuit Models (ECM), electrochemical kinetics, and embedded BMS hardware constraints."
        "</p>",
        unsafe_allow_html=True,
    )

    # Circuit Diagram Header & SVG
    st.markdown("#### 1st-Order Thevenin Equivalent Circuit Model")
    render_html(render_thevenin_circuit_svg())

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Core Equations Callout
    render_html("""
    <div class="cg-card">
      <div class="cg-card-title">Mathematical Governing Equations</div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; color: #FFFFFF; background: rgba(0,0,0,0.3); padding: 12px 18px; border-radius: 8px; margin-bottom: 12px;">
        V_term(t) = OCV(SOC) - I(t) &middot; R₀ - V₁(t)
      </div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.92rem; color: #93A0B8; background: rgba(0,0,0,0.3); padding: 12px 18px; border-radius: 8px;">
        dV₁(t) / dt = - V₁(t) / (R₁ &middot; C₁) + I(t) / C₁
      </div>
      <p style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 10px; line-height: 1.5;">
        <strong>Parameter Correspondence:</strong> Ohmic resistance <code>R₀</code> corresponds to high-frequency electrolyte/collector resistance (<code>Re</code>), 
        while polarization resistance <code>R₁</code> corresponds to intermediate-frequency charge-transfer resistance (<code>Rct</code>).
      </p>
    </div>
    """)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Definitions Grid
    col_d1, col_d2 = st.columns(2, gap="medium")
    with col_d1:
        render_html("""
        <div class="cg-card" style="height: 100%;">
          <div class="cg-card-title">State of Health (SOH) Definitions</div>
          <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6;">
            <strong>1. Capacity-Based SOH (Primary):</strong>
            <br/>
            <code>SOH_C = (C_present / C_rated) &times; 100%</code>
            <br/>
            Where rated capacity is <strong>2.00 Ah</strong>. When capacity reaches <strong>1.40 Ah</strong> (30% fade), the battery has reached its End of Life (70% SOH).
            <br/><br/>
            <strong>2. Resistance-Based SOH:</strong>
            <br/>
            <code>SOH_R = (R_initial / R_present) &times; 100%</code>
            <br/>
            Internal resistance roughly doubles over cell lifespan. Tracking <code>Re</code> and <code>Rct</code> independently identifies whether degradation is bulk electrolyte drying or SEI layer impedance growth.
          </div>
        </div>
        """)
    with col_d2:
        render_html("""
        <div class="cg-card" style="height: 100%;">
          <div class="cg-card-title">Charging Profile Dynamics (CC-CV)</div>
          <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6;">
            <strong>Standard Protocol:</strong> Constant Current (CC) 1.5 A to 4.2 V, followed by Constant Voltage (CV) until current decays to 20 mA.
            <br/><br/>
            <strong>Degradation Signature:</strong>
            As a cell ages and internal resistance (<code>R₀ + R₁</code>) rises, the terminal voltage reaches the 4.2 V ceiling prematurely under the 1.5 A load step.
            <br/><br/>
            Consequently, <strong>CC duration shrinks</strong> and <strong>CV duration expands</strong>. While informative, CellGuard’s headline models strictly use discharge features so diagnostics can be executed during vehicle operation.
          </div>
        </div>
        """)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # BMS Embedded Feasibility Section
    render_html("""
    <div class="cg-card" style="border-left: 3px solid #22D3EE;">
      <div style="font-size: 1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
        ⚡ Embedded Feasibility: Could this Run in a Real BMS Microcontroller?
      </div>
      <div style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.6;">
        <p>
          <strong>Computational Footprint:</strong> Yes, with near-zero overhead. Because CellGuard’s deployed capacity and impedance models are regularized 
          <strong>Ridge regression pipelines</strong>, inference on an embedded target requires evaluating a single 18-element dot product: 
          <code>y = &sum; (w_i &times; (x_i - &mu;_i) / &sigma;_i) + b</code>. On an ARM Cortex-M4 (e.g. STM32G4 / ESP32) running at 80 MHz, 
          this computation executes in <strong>under 5 microseconds</strong> with single-precision FPU hardware and requires 
          <strong>less than 2 KB of Flash memory</strong> for weights and scaling statistics.
        </p>
        <p style="margin-bottom: 0;">
          <strong>Sensor &amp; Sampling Requirements:</strong> The model does not require high-frequency electrochemical sweeps. 
          Sampling voltage, current, and temperature at <strong>0.1 Hz to 1 Hz</strong> with a standard 12-bit ADC is entirely sufficient to capture 
          threshold crossing timestamps between 4.0 V and 3.4 V. The entire feature extraction and inference pipeline can be triggered as a low-priority background RTOS task without interrupting safety-critical over-voltage protection loops.
        </p>
      </div>
    </div>
    """)
