"""
components.py - Precision UI components for CellGuard Battery Management System.
Features:
- Compact technical page header with cell & source metadata
- Data category legend (Measured, Calculated, ML-Predicted, Simulated)
- Precision industrial SOH health gauge and dual SOC/SOH comparison meters
- Executive engineering KPI tiles with units, error bands, and category badges
- 1st-Order Thevenin Equivalent Circuit Model SVG schematic
- Sidebar brand header, telemetry status, and dataset footer
"""

import os
import math
import streamlit as st
from typing import Optional, Tuple

UI_DIR = os.path.dirname(os.path.abspath(__file__))
THEME_CSS_PATH = os.path.join(UI_DIR, "theme.css")


def render_html(html_content: str):
    """Safely render raw HTML across Streamlit versions."""
    if hasattr(st, "html"):
        st.html(html_content)
    else:
        st.markdown(html_content, unsafe_allow_html=True)


def inject_theme():
    """Inject theme.css into the Streamlit app."""
    if os.path.exists(THEME_CSS_PATH):
        with open(THEME_CSS_PATH, "r", encoding="utf-8") as f:
            css = f.read()
        render_html(f"<style>{css}</style>")


def render_compact_header(
    page_title: str,
    selected_battery: str = "B0018",
    data_source: str = "NASA Ames Li-ion Aging Repository",
    model_status: str = "LOCO Certified &bull; 5 Pipelines Active",
):
    """Render modern compact page header with live battery and model metadata."""
    html = f"""
    <div style="display: flex; align-items: center; justify-content: space-between; background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 10px 16px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(15,23,42,0.05); flex-wrap: wrap; gap: 10px;">
      <div style="display: flex; align-items: center; gap: 10px;">
        <div style="width: 32px; height: 32px; background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 6px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="2" y="7" width="16" height="12" rx="2"></rect>
            <path d="M20 11v4"></path>
            <path d="M7 13l2-2 2 4 2-2"></path>
          </svg>
        </div>
        <div style="font-size: 1.10rem; font-weight: 800; color: #0F172A !important; letter-spacing: -0.01em;">
          {page_title}
        </div>
      </div>
      <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
        <span style="display: inline-flex; align-items: center; gap: 5px; font-size: 0.74rem; font-family: 'JetBrains Mono', monospace; padding: 3px 8px; border-radius: 4px; font-weight: 700; background: #EFF6FF; border: 1px solid #93C5FD; color: #1D4ED8 !important;">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="2" y="7" width="16" height="12" rx="2"></rect><path d="M20 11v4"></path></svg>
          CELL: {selected_battery}
        </span>
        <span style="display: inline-flex; align-items: center; gap: 5px; font-size: 0.74rem; font-family: 'JetBrains Mono', monospace; padding: 3px 8px; border-radius: 4px; font-weight: 600; background: #F8FAFC; border: 1px solid #CBD5E1; color: #334155 !important;">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
          {data_source}
        </span>
        <span style="display: inline-flex; align-items: center; gap: 5px; font-size: 0.74rem; font-family: 'JetBrains Mono', monospace; padding: 3px 8px; border-radius: 4px; font-weight: 700; background: #F0FDF4; border: 1px solid #86EFAC; color: #15803D !important;">
          <span style="width: 6px; height: 6px; border-radius: 50%; background: #16A34A;"></span>
          {model_status}
        </span>
      </div>
    </div>
    """
    render_html(html)


def render_category_legend():
    """Render scientific legend distinguishing Measured, Calculated, ML-Predicted, and Simulated data."""
    html = """
    <div class="cg-category-legend">
      <span class="cg-legend-title">Data Provenance:</span>
      <span class="cg-cat-badge cg-cat-measured" title="Direct experimental telemetry from battery cycler sensors">
        ● Measured (Hardware Sensor)
      </span>
      <span class="cg-cat-badge cg-cat-calculated" title="Deterministic physical calculation (e.g. Coulomb counting, SOH ratio)">
        ● Calculated (Deterministic)
      </span>
      <span class="cg-cat-badge cg-cat-predicted" title="Estimated via machine learning model with LOCO error bounds">
        ● ML-Predicted (Statistical Inference)
      </span>
      <span class="cg-cat-badge cg-cat-simulated" title="Hypothetical or illustrative aging scenario">
        ● Simulated (Scenario Model)
      </span>
    </div>
    """
    render_html(html)


def get_health_grade(
    soh: float,
    eol_threshold: float = 70.0,
    warranty_threshold: float = 80.0,
) -> Tuple[str, str, str, str]:
    """
    Return scientifically meaningful health classification.
    Distinguishes project-defined EOL threshold (70% NASA dataset standard)
    from universal EV warranty threshold (80%).
    Returns: (grade_label, status_class, description, hex_color)
    """
    s = float(soh)
    if s >= 90.0:
        return ("Optimal", "cg-status-healthy", "Nominal capacity &bull; Minimal SEI growth", "#16A34A")
    elif s >= warranty_threshold:
        return ("Good", "cg-status-healthy", "Normal degradation &bull; Above warranty limit", "#16A34A")
    elif s >= eol_threshold:
        return ("Degrading", "cg-status-warning", f"Below {warranty_threshold:.0f}% warranty standard &bull; Approaching EOL", "#D97706")
    else:
        return ("End of Life (EOL)", "cg-status-critical", f"Below {eol_threshold:.0f}% EOL limit &bull; Decommissioning advised", "#DC2626")


def render_kpi(
    label: str,
    value: str,
    unit: str = "",
    error_band: str = None,
    subtext: str = "",
    status: str = None,
    category: str = None,  # "measured", "calculated", "predicted", "simulated"
    tooltip: str = "",
):
    """Render clean precision engineering KPI tile without wrapping cut-offs."""
    # Compact status pill at top right
    status_markup = ""
    if status:
        st_lower = str(status).lower()
        if "good" in st_lower or "optim" in st_lower or "health" in st_lower:
            status_markup = f'<span style="display:inline-flex; align-items:center; gap:3px; font-size:0.65rem; font-weight:700; padding:2px 6px; border-radius:4px; background:#F0FDF4; border:1px solid #BBF7D0; color:#15803D; white-space:nowrap; flex-shrink:0;">&bull; {status}</span>'
        elif "degrad" in st_lower or "warn" in st_lower or "mod" in st_lower:
            status_markup = f'<span style="display:inline-flex; align-items:center; gap:3px; font-size:0.65rem; font-weight:700; padding:2px 6px; border-radius:4px; background:#FFFBEB; border:1px solid #FDE68A; color:#B45309; white-space:nowrap; flex-shrink:0;">&bull; {status}</span>'
        else:
            status_markup = f'<span style="display:inline-flex; align-items:center; gap:3px; font-size:0.65rem; font-weight:700; padding:2px 6px; border-radius:4px; background:#FEF2F2; border:1px solid #FECACA; color:#B91C1C; white-space:nowrap; flex-shrink:0;">&bull; {status}</span>'

    # Category badge at footer right
    cat_tag = ""
    if category:
        cat_lower = str(category).lower()
        if "meas" in cat_lower:
            cat_tag = '<span style="font-size:0.58rem; font-family:\'JetBrains Mono\', monospace; font-weight:700; padding:1px 5px; border-radius:3px; background:#ECFDF5; border:1px solid #A7F3D0; color:#065F46; white-space:nowrap; flex-shrink:0;">MEASURED</span>'
        elif "calc" in cat_lower:
            cat_tag = '<span style="font-size:0.58rem; font-family:\'JetBrains Mono\', monospace; font-weight:700; padding:1px 5px; border-radius:3px; background:#EFF6FF; border:1px solid #BFDBFE; color:#1E40AF; white-space:nowrap; flex-shrink:0;">CALCULATED</span>'
        elif "pred" in cat_lower:
            cat_tag = '<span style="font-size:0.58rem; font-family:\'JetBrains Mono\', monospace; font-weight:700; padding:1px 5px; border-radius:3px; background:#F5F3FF; border:1px solid #DDD6FE; color:#5B21B6; white-space:nowrap; flex-shrink:0;">ML PREDICTED</span>'
        elif "sim" in cat_lower:
            cat_tag = '<span style="font-size:0.58rem; font-family:\'JetBrains Mono\', monospace; font-weight:700; padding:1px 5px; border-radius:3px; background:#FFFBEB; border:1px solid #FDE68A; color:#92400E; white-space:nowrap; flex-shrink:0;">SIMULATED</span>'

    err_markup = ""
    if error_band:
        err_markup = f'<span style="font-family:\'JetBrains Mono\', monospace; font-size:0.68rem; font-weight:600; color:#1D4ED8; background:#EFF6FF; border:1px solid #BFDBFE; padding:1px 4px; border-radius:3px; white-space:nowrap;">&plusmn;{error_band}</span>'

    title_attr = f'title="{tooltip}"' if tooltip else ""

    html = f"""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:10px 12px; box-shadow:0 1px 2px rgba(15,23,42,0.04); height:100%; min-height:102px; display:flex; flex-direction:column; justify-content:space-between;" {title_attr}>
      <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:4px; margin-bottom:4px;">
        <div style="font-size:0.72rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.02em; line-height:1.2; word-break:normal; overflow-wrap:normal; hyphens:none;">
          {label}
        </div>
        {status_markup}
      </div>
      <div style="display:flex; align-items:baseline; gap:4px; margin:2px 0 4px 0; flex-wrap:wrap;">
        <span style="font-family:'JetBrains Mono', monospace; font-size:1.40rem; font-weight:800; color:#0F172A; line-height:1;">
          {value}
        </span>
        <span style="font-family:'JetBrains Mono', monospace; font-size:0.80rem; font-weight:600; color:#475569;">
          {unit}
        </span>
        {err_markup}
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid #F1F5F9; padding-top:4px; margin-top:auto; font-size:0.69rem; color:#64748B; gap:6px;">
        <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:70%;" title="{subtext}">{subtext}</span>
        {cat_tag}
      </div>
    </div>
    """
    render_html(html)


def render_health_gauge_svg(
    soh: float,
    capacity_ah: float = None,
    eol_threshold: float = 70.0,
    warranty_threshold: float = 80.0,
    size: int = 240,
) -> str:
    """Industrial precision semi-circular gauge rendered with explicit SVG vector geometry."""
    clamped_soh = max(0.0, min(100.0, float(soh)))
    cx = 120.0
    cy = 110.0
    r = 82.0

    # Calculate exact active arc end point via cos and sin
    # Angle goes from 180° (left) to 360° (right) through 270° (top)
    rad_end = math.radians(180.0 + (clamped_soh / 100.0) * 180.0)
    x_start = cx - r
    y_start = cy
    x_end = cx + r * math.cos(rad_end)
    y_end = cy + r * math.sin(rad_end)

    # Color mapping
    if clamped_soh >= warranty_threshold:
        stroke_color = "#16A34A"
        status_label = "OPTIMAL / GOOD"
        status_color = "#16A34A"
    elif clamped_soh >= eol_threshold:
        stroke_color = "#D97706"
        status_label = "DEGRADING (PRE-EOL)"
        status_color = "#D97706"
    else:
        stroke_color = "#DC2626"
        status_label = "END OF LIFE REACHED"
        status_color = "#DC2626"

    # Tick marks at key thresholds: 0, 50, 70 (EOL), 80 (Warranty), 100
    ticks_svg = []
    for pct in [0, 25, 50, 70, 80, 100]:
        deg = 180.0 + (pct / 100.0) * 180.0
        trad = math.radians(deg)
        is_key = pct in [70, 80]
        t_len = 8 if is_key else 5
        t_color = "#DC2626" if pct == 70 else ("#D97706" if pct == 80 else "#CBD5E1")
        x1 = cx + (r - 2) * math.cos(trad)
        y1 = cy + (r - 2) * math.sin(trad)
        x2 = cx + (r - 2 - t_len) * math.cos(trad)
        y2 = cy + (r - 2 - t_len) * math.sin(trad)
        ticks_svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{t_color}" stroke-width="{2 if is_key else 1}" />')

    ticks_markup = "\n".join(ticks_svg)
    cap_text = f"{capacity_ah:.3f} Ah" if capacity_ah else ""

    # Arc path definition: use dual-quarter arcs for robust SVG rendering across all browsers
    track_svg = f'<path d="M {x_start:.1f} {y_start:.1f} A {r:.1f} {r:.1f} 0 0 1 {cx:.1f} {cy-r:.1f} A {r:.1f} {r:.1f} 0 0 1 {cx + r:.1f} {cy:.1f}" fill="none" stroke="#E2E8F0" stroke-width="12" stroke-linecap="round" />'
    if clamped_soh > 0.5:
        if clamped_soh <= 50.0:
            active_arc_svg = f'<path d="M {x_start:.1f} {y_start:.1f} A {r:.1f} {r:.1f} 0 0 1 {x_end:.1f} {y_end:.1f}" fill="none" stroke="{stroke_color}" stroke-width="12" stroke-linecap="round" />'
        else:
            active_arc_svg = f'<path d="M {x_start:.1f} {y_start:.1f} A {r:.1f} {r:.1f} 0 0 1 {cx:.1f} {cy-r:.1f} A {r:.1f} {r:.1f} 0 0 1 {x_end:.1f} {y_end:.1f}" fill="none" stroke="{stroke_color}" stroke-width="12" stroke-linecap="round" />'
    else:
        active_arc_svg = ""

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%;">
      <svg width="{size}" height="150" viewBox="0 0 {size} 150">
        <!-- Background Track -->
        {track_svg}
        
        <!-- Active SOH Arc -->
        {active_arc_svg}

        <!-- Calibration Ticks -->
        {ticks_markup}

        <!-- Center Numerical Vector Typography -->
        <text x="{cx}" y="62" fill="#64748B" font-family="'Inter', sans-serif" font-size="9.5" font-weight="700" text-anchor="middle" letter-spacing="0.08em">STATE OF HEALTH</text>
        <text x="{cx}" y="92" fill="#0F172A" font-family="'JetBrains Mono', monospace" font-size="26" font-weight="800" text-anchor="middle">{clamped_soh:.1f}%</text>
        <text x="{cx}" y="110" fill="#475569" font-family="'JetBrains Mono', monospace" font-size="11.5" font-weight="600" text-anchor="middle">{cap_text}</text>
        <text x="{cx}" y="132" fill="{status_color}" font-family="'Inter', sans-serif" font-size="10.5" font-weight="700" text-anchor="middle" letter-spacing="0.04em">&bull; {status_label}</text>
      </svg>
    </div>
    """


def render_soc_soh_dual_bar(
    soc: Optional[float] = None,
    soh: float = 90.0,
    rated_cap: float = 2.0,
    remaining_cap: Optional[float] = None,
    soc_available: bool = True,
):
    """
    Render clean dual comparison between SOC (operational state) and SOH (health/degradation state).
    Addresses prompt requirement: SOC = charge available now vs SOH = capacity remaining over lifetime.
    """
    clamped_soh = max(0.0, min(100.0, float(soh)))
    soh_color = "#16A34A" if clamped_soh >= 80 else ("#D97706" if clamped_soh >= 70 else "#DC2626")
    
    if soc is not None and soc_available:
        clamped_soc = max(0.0, min(100.0, float(soc)))
        soc_color = "#2563EB"
        soc_val_str = f"{clamped_soc:.1f}%"
        soc_bar_w = f"{clamped_soc:.1f}%"
        soc_sub = "Calculated via Coulomb Counting integration"
    else:
        soc_color = "#94A3B8"
        soc_val_str = "N/A"
        soc_bar_w = "0%"
        soc_sub = "Requires initial charge state and current integration data"

    cap_display = f"{remaining_cap:.3f} Ah" if remaining_cap is not None else f"{(clamped_soh/100.0)*rated_cap:.3f} Ah"
    fade_pct = round(max(0.0, 100.0 - clamped_soh), 1)

    html = f"""
    <div class="cg-card" style="margin-bottom: 20px;">
      <div class="cg-card-header">
        <div>
          <h4 class="cg-card-title">Dual State Analysis: Operational Charge (SOC) vs Battery Longevity (SOH)</h4>
          <p class="cg-card-subtitle">Comparing short-term usable energy against irreversible electrochemical degradation</p>
        </div>
        <span class="cg-cat-badge cg-cat-calculated">COMPARATIVE METRICS</span>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
        <!-- SOC Column -->
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
          <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
            <div>
              <span style="font-size: 0.76rem; font-weight: 700; color: #475569; text-transform: uppercase;">STATE OF CHARGE (SOC)</span>
              <div style="font-size: 0.72rem; color: #64748B;">Charge currently available in the cell right now</div>
            </div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; font-weight: 800; color: {soc_color};">
              {soc_val_str}
            </span>
          </div>

          <!-- Progress Bar -->
          <div style="width: 100%; height: 12px; background: #E2E8F0; border-radius: 6px; overflow: hidden; margin: 10px 0;">
            <div style="width: {soc_bar_w}; height: 100%; background: {soc_color}; border-radius: 6px; transition: width 0.3s ease;"></div>
          </div>
          <div style="font-size: 0.74rem; color: #64748B; line-height: 1.4;">
            {soc_sub}
          </div>
        </div>

        <!-- SOH Column -->
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
          <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
            <div>
              <span style="font-size: 0.76rem; font-weight: 700; color: #475569; text-transform: uppercase;">STATE OF HEALTH (SOH)</span>
              <div style="font-size: 0.72rem; color: #64748B;">Total retention of original 2.00 Ah rated capacity</div>
            </div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; font-weight: 800; color: {soh_color};">
              {clamped_soh:.1f}%
            </span>
          </div>

          <!-- Progress Bar with EOL Marker -->
          <div style="position: relative; width: 100%; height: 12px; background: #E2E8F0; border-radius: 6px; overflow: hidden; margin: 10px 0;">
            <div style="width: {clamped_soh:.1f}%; height: 100%; background: {soh_color}; border-radius: 6px;"></div>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: #64748B;">
            <span>Current: <strong>{cap_display}</strong></span>
            <span>Capacity Fade: <strong>-{fade_pct}%</strong></span>
            <span>EOL Threshold: <strong>1.40 Ah (70%)</strong></span>
          </div>
        </div>
      </div>
    </div>
    """
    render_html(html)


def render_thevenin_circuit_svg() -> str:
    """Generate high-contrast, professional SVG schematic of 1st-order Thevenin Equivalent Circuit Model."""
    svg = """
    <div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 22px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
      <svg width="680" height="230" viewBox="0 0 680 230" fill="none" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <!-- Definitions -->
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB" />
          </marker>
        </defs>

        <!-- Main Wire Loop -->
        <path d="M 90 60 L 210 60" stroke="#475569" stroke-width="2.5" />
        <path d="M 280 60 L 360 60" stroke="#475569" stroke-width="2.5" />
        
        <!-- Parallel Branch Top (R1) -->
        <path d="M 360 60 L 360 25 L 400 25" stroke="#475569" stroke-width="2.5" />
        <path d="M 470 25 L 510 25 L 510 60" stroke="#475569" stroke-width="2.5" />
        
        <!-- Parallel Branch Bottom (C1) -->
        <path d="M 360 60 L 360 95 L 420 95" stroke="#475569" stroke-width="2.5" />
        <path d="M 450 95 L 510 95 L 510 60" stroke="#475569" stroke-width="2.5" />
        
        <!-- Output Wire Top -->
        <path d="M 510 60 L 610 60" stroke="#475569" stroke-width="2.5" />
        
        <!-- Bottom Return Wire -->
        <path d="M 90 180 L 610 180" stroke="#475569" stroke-width="2.5" />

        <!-- Current Arrow I(t) -->
        <path d="M 125 45 L 175 45" stroke="#2563EB" stroke-width="2.5" marker-end="url(#arrow)" />
        <text x="150" y="38" fill="#2563EB" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" text-anchor="middle">Discharge Current I(t)</text>

        <!-- Voltage Source (OCV) -->
        <line x1="90" y1="60" x2="90" y2="100" stroke="#475569" stroke-width="2.5" />
        <line x1="90" y1="140" x2="90" y2="180" stroke="#475569" stroke-width="2.5" />
        <circle cx="90" cy="120" r="20" stroke="#0F172A" stroke-width="2.5" fill="#EFF6FF" />
        <text x="90" y="116" fill="#0F172A" font-family="'Inter', sans-serif" font-size="14" font-weight="700" text-anchor="middle">+</text>
        <text x="90" y="132" fill="#0F172A" font-family="'Inter', sans-serif" font-size="14" font-weight="700" text-anchor="middle">-</text>
        <text x="90" y="158" fill="#2563EB" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" text-anchor="middle">OCV(SOC)</text>

        <!-- Series Resistor R0 (Ohmic Resistance Re) -->
        <rect x="210" y="48" width="70" height="24" rx="4" fill="#FFFFFF" stroke="#0D9488" stroke-width="2" />
        <text x="245" y="64" fill="#0D9488" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" text-anchor="middle">R₀</text>
        <text x="245" y="88" fill="#64748B" font-family="'JetBrains Mono', monospace" font-size="10" text-anchor="middle">≈ Re (Ohmic)</text>

        <!-- Polarization Resistor R1 (Charge-Transfer Resistance Rct) -->
        <rect x="400" y="13" width="70" height="24" rx="4" fill="#FFFFFF" stroke="#7C3AED" stroke-width="2" />
        <text x="435" y="29" fill="#7C3AED" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" text-anchor="middle">R₁</text>
        <text x="435" y="6" fill="#64748B" font-family="'JetBrains Mono', monospace" font-size="10" text-anchor="middle">≈ Rct (Polarization)</text>

        <!-- Polarization Capacitor C1 -->
        <line x1="430" y1="83" x2="430" y2="107" stroke="#0D9488" stroke-width="3" />
        <line x1="440" y1="83" x2="440" y2="107" stroke="#0D9488" stroke-width="3" />
        <text x="435" y="125" fill="#0D9488" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" text-anchor="middle">C₁ (Double Layer)</text>

        <!-- Terminals -->
        <circle cx="610" cy="60" r="5" fill="#2563EB" stroke="#0F172A" stroke-width="1.5" />
        <circle cx="610" cy="180" r="5" fill="#64748B" stroke="#0F172A" stroke-width="1.5" />
        <text x="630" y="65" fill="#2563EB" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="700">+</text>
        <text x="630" y="185" fill="#64748B" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="700">-</text>
        <text x="645" y="125" fill="#0F172A" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="800">V_terminal(t)</text>
      </svg>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #1E293B; margin-top: 8px; font-weight: 600;">
        V_terminal(t) = OCV(SOC) - I(t) &times; R₀ - V₁(t)
      </div>
    </div>
    """
    return svg


def render_sidebar_header():
    """Render modern engineering sidebar identity."""
    with st.sidebar:
        html = """
        <div style="display: flex; align-items: center; gap: 10px; padding: 4px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 14px;">
          <div style="width: 34px; height: 34px; background: rgba(37, 99, 235, 0.2); border: 1px solid #3B82F6; border-radius: 8px; display: flex; align-items: center; justify-content: center;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="2" y="7" width="16" height="12" rx="2"></rect>
              <path d="M20 11v4"></path>
              <path d="M7 13l2-2 2 4 2-2"></path>
            </svg>
          </div>
          <div>
            <div style="font-weight: 800; font-size: 1.05rem; color: #FFFFFF; letter-spacing: 0.02em;">CellGuard</div>
            <div style="font-size: 0.68rem; color: #94A3B8; font-family: 'Inter', sans-serif; font-weight: 500;">
              Intelligent Battery Diagnostics &amp; Prognostics
            </div>
          </div>
        </div>

        <div style="background: rgba(19, 31, 63, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 10px 12px; margin-bottom: 16px; font-size: 0.74rem;">
          <div style="color: #94A3B8; font-weight: 600; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">OPERATIONAL MODE</div>
          <div style="color: #10B981; font-weight: 700; display: flex; align-items: center; gap: 6px;">
            <span style="width: 7px; height: 7px; border-radius: 50%; background: #10B981;"></span>
            BMS Online &bull; M1-M5 Ready
          </div>
          <div style="color: #64748B; font-size: 0.70rem; margin-top: 3px;">
            Partial-Window 4.0V &rarr; 3.4V Non-Destructive
          </div>
        </div>
        """
        render_html(html)


def render_sidebar_footer():
    """Render technical metadata at the bottom of the sidebar."""
    with st.sidebar:
        html = """
        <div class="cg-sidebar-footer">
          <div style="font-weight: 700; color: #E2E8F0; margin-bottom: 4px;">PLATFORM METADATA</div>
          <div><strong>Dataset:</strong> NASA Ames Li-ion Aging (B0005, 06, 07, 18)</div>
          <div><strong>Model Suite:</strong> Ridge &amp; Random Forest (LOCO Certified)</div>
          <div><strong>Version:</strong> CellGuard v3.2 Enterprise</div>
        </div>
        """
        render_html(html)


def render_footer(metrics: dict = None):
    """Render clean application footer."""
    versions = "scikit-learn 1.6.1 | numpy 2.1.3 | pandas 2.2.3"
    if metrics and "environment_versions" in metrics:
        v = metrics["environment_versions"]
        versions = f"scikit-learn {v.get('scikit_learn', '1.6.1')} | numpy {v.get('numpy', '2.1.3')} | pandas {v.get('pandas', '2.2.3')}"

    html = f"""
    <div class="cg-footer">
      <div>
        <strong>CellGuard</strong> &bull; Machine Learning-Based Battery State of Health, Impedance and RUL Estimation
      </div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">
        {versions}
      </div>
      <div>
        NASA Ames PCoE Dataset &bull; Non-Destructive Early Scan (No Full Discharge Needed)
      </div>
    </div>
    """
    render_html(html)


def render_callout(message: str, alert_type: str = "info", title: str = None):
    """Render technical callout banner."""
    type_cls = "cg-callout"
    if alert_type == "warn":
        type_cls += " cg-callout-warning"
    elif alert_type == "critical" or alert_type == "error":
        type_cls += " cg-callout-critical"
    elif alert_type == "success":
        type_cls += " cg-callout-success"

    title_html = f'<div style="font-weight: 700; margin-bottom: 3px;">{title}</div>' if title else ""
    html = f"""
    <div class="{type_cls}">
      {title_html}
      <div>{message}</div>
    </div>
    """
    render_html(html)
