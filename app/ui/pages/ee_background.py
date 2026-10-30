"""
ee_background.py - How Batteries Work & BMS Engineering Architecture.
Presents the everyday "Water Pipe & Tank" analogy for normal people,
alongside the 1st-order Thevenin Equivalent Circuit Model (ECM), governing equations,
and embedded microcontroller feasibility for BMS engineers.
"""

import os
import streamlit as st
from ui.components import render_html


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
        <text x="250" y="90" fill="#93A0B8" font-family="'JetBrains Mono', monospace" font-size="11" text-anchor="middle">≈ Re (Ohmic Flow)</text>

        <!-- Polarization Resistor R1 (Charge-Transfer Resistance Rct) -->
        <rect x="400" y="13" width="60" height="24" rx="4" fill="#1E293B" stroke="#A78BFA" stroke-width="2" />
        <text x="430" y="29" fill="#A78BFA" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" text-anchor="middle">R₁</text>
        <text x="430" y="4" fill="#93A0B8" font-family="'JetBrains Mono', monospace" font-size="10" text-anchor="middle">≈ Rct (Chemical Transfer)</text>

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
    """Render the How Batteries Work & EE Background page."""
    st.markdown("""
    <div style="margin-bottom: 18px;">
      <h2 style="font-size: 1.85rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">
        💡 How Batteries Work &amp; Why They Age
      </h2>
      <p style="font-size: 0.92rem; color: #94A3B8; margin: 0;">
        Understand battery health through everyday human analogies, electrical circuit models (ECM), and BMS hardware integration.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Everyday Analogy: The Water Pipe & Tank
    render_html("""
    <div class="cg-launch-hero" style="padding: 20px 24px; margin-bottom: 20px;">
      <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <span class="cg-launch-badge">🚿 THE EVERYDAY ANALOGY</span>
        <span style="font-size: 0.82rem; color: #E2E8F0; font-weight: 600;">Think of a Battery Like a Water Tank &amp; Pipes</span>
      </div>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; font-size: 0.84rem; color: #CBD5E1; line-height: 1.55; margin-top: 10px;">
        <div style="background: rgba(0,0,0,0.3); border-radius: 8px; padding: 12px;">
          <div style="font-weight: 700; color: #22D3EE; margin-bottom: 4px;">🛢️ Usable Capacity (Ah) = Tank Size</div>
          <div>How much water (charge) the tank can hold. As a battery ages, sediment forms inside the tank, so it holds less water over time.</div>
        </div>
        <div style="background: rgba(0,0,0,0.3); border-radius: 8px; padding: 12px;">
          <div style="font-weight: 700; color: #10B981; margin-bottom: 4px;">⚡ Internal Resistance (Re) = Pipe Cleanliness</div>
          <div>How freely water flows out. When the pipe is clean, water flows easily with no friction. When clogged, the pipe resists flow and gets hot!</div>
        </div>
        <div style="background: rgba(0,0,0,0.3); border-radius: 8px; padding: 12px;">
          <div style="font-weight: 700; color: #A78BFA; margin-bottom: 4px;">🔄 Chemical Reaction (Rct) = Valve Speed</div>
          <div>How quickly the tap opens and closes. In an aged battery, the valve gets stiff, slowing down fast acceleration and charging.</div>
        </div>
        <div style="background: rgba(0,0,0,0.3); border-radius: 8px; padding: 12px;">
          <div style="font-weight: 700; color: #F59E0B; margin-bottom: 4px;">🔋 Voltage (V) = Water Pressure</div>
          <div>The pressure pushing water through the pipe. A fully charged battery has high pressure (4.2V), dropping to 3.0V when empty.</div>
        </div>
      </div>
    </div>
    """)

    # 2. Formal Engineering Model: Thevenin Equivalent Circuit
    st.markdown("#### 1st-Order Thevenin Equivalent Circuit Model (ECM)")
    st.caption("How electrical engineers model the battery inside vehicle software:")
    render_html(render_thevenin_circuit_svg())

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 3. Governing Equations Callout
    render_html("""
    <div class="cg-card">
      <div class="cg-card-title">Mathematical Governing Equations</div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.02rem; color: #FFFFFF; background: rgba(0,0,0,0.3); padding: 12px 18px; border-radius: 8px; margin-bottom: 12px;">
        V_term(t) = OCV(SOC) - I(t) &middot; R₀ - V₁(t)
      </div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; color: #94A3B8; background: rgba(0,0,0,0.3); padding: 12px 18px; border-radius: 8px;">
        dV₁(t) / dt = - V₁(t) / (R₁ &middot; C₁) + I(t) / C₁
      </div>
      <p style="font-size: 0.84rem; color: var(--text-secondary); margin-top: 10px; line-height: 1.5;">
        <strong>Parameter Mapping:</strong> Ohmic resistance <code>R₀</code> corresponds to electrolyte series resistance (<code>Re</code>), 
        while polarization resistance <code>R₁</code> corresponds to charge-transfer kinetics (<code>Rct</code>).
      </p>
    </div>
    """)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 4. Definitions Grid
    col_d1, col_d2 = st.columns(2, gap="medium")
    with col_d1:
        render_html("""
        <div class="cg-card" style="height: 100%;">
          <div class="cg-card-title">State of Health (SOH) Definition</div>
          <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6;">
            <strong>Capacity-Based SOH (Automotive Standard):</strong>
            <br/>
            <code>SOH = (Current Capacity / Rated Capacity) &times; 100%</code>
            <br/>
            Where rated capacity is <strong>2.00 Ah</strong>. When capacity reaches <strong>1.40 Ah</strong> (30% drop), the battery reaches its End of Life (70% SOH).
            <br/><br/>
            <strong>Why Not Full Discharges?</strong>
            A normal driver rarely drives from 100% to 0%. CellGuard only inspects the first 5 minutes of discharge ($4.0\\text{ V} \\to 3.4\\text{ V}$) to estimate full capacity accurately.
          </div>
        </div>
        """)
    with col_d2:
        render_html("""
        <div class="cg-card" style="height: 100%;">
          <div class="cg-card-title">Charging Profile Dynamics (CC-CV)</div>
          <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6;">
            <strong>Standard Charging:</strong> Constant Current (CC) until 4.2 V, followed by Constant Voltage (CV) until current trickles down.
            <br/><br/>
            <strong>How Aging Affects Charging:</strong>
            As a battery ages and internal resistance (<code>R₀ + R₁</code>) climbs, voltage hits the 4.2 V ceiling earlier.
            <br/><br/>
            Consequently, <strong>fast-charging duration shrinks</strong> and the slow trickle phase expands. This is why older phones and EVs seem to take longer to reach 100%!
          </div>
        </div>
        """)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 5. BMS Embedded Feasibility
    render_html("""
    <div class="cg-card" style="border-left: 3px solid #22D3EE;">
      <div style="font-size: 1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
        ⚡ Embedded Feasibility: Can this run directly inside an automotive BMS microcontroller?
      </div>
      <div style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.6;">
        <p>
          <strong>Yes, with near-zero power and memory footprint:</strong> CellGuard's deployed capacity and impedance models are regularized 
          <strong>Ridge regression pipelines</strong>. Inference requires evaluating a single 18-element dot product: 
          <code>y = &sum; (w_i &times; (x_i - &mu;_i) / &sigma;_i) + b</code>.
        </p>
        <p>
          On a standard automotive microcontroller (such as an ARM Cortex-M4 / STM32 / ESP32 running at 80 MHz), 
          this calculation runs in <strong>under 5 microseconds</strong> and requires <strong>less than 2 KB of memory</strong>.
        </p>
        <p style="margin-bottom: 0;">
          <strong>No Special Hardware Needed:</strong> Telemetry is sampled at standard <strong>0.1 Hz to 1 Hz</strong> frequencies with existing onboard voltage and temperature sensors.
        </p>
      </div>
    </div>
    """)
