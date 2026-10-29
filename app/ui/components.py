"""
components.py - Precision UI components for CellGuard Battery Management System.
Features high-precision tachometer gauge, prismatic cell graphic, and executive KPI cards.
"""

import os
import json
import streamlit as st

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


def render_topbar(active_page: str = "Overview"):
    """Enterprise top bar for CellGuard with instant dual-navigation links."""
    nav_items = [
        ("Overview", "/", "🏠"),
        ("Predict Cycle", "/page_predict_view", "⚡"),
        ("Battery Life & RUL", "/page_life_rul_view", "📈"),
        ("Gap Filler", "/page_gap_filler_view", "🔬"),
        ("Model Performance", "/page_performance_view", "📊"),
        ("EE Background", "/page_ee_bg_view", "🔌"),
    ]

    links_html = []
    for label, url, icon in nav_items:
        is_active = (label.lower() in active_page.lower()) or (active_page.lower() in label.lower())
        active_cls = "cg-nav-item active" if is_active else "cg-nav-item"
        links_html.append(
            f'<a href="{url}" class="{active_cls}" target="_self"><span style="font-size: 0.82rem;">{icon}</span> <span>{label}</span></a>'
        )
    links_markup = "\n".join(links_html)

    topbar_html = f"""
    <div class="cg-topbar">
      <div style="display: flex; align-items: center; gap: 12px;">
        <div style="width: 28px; height: 28px; background: rgba(34, 211, 238, 0.1); border: 1px solid rgba(34, 211, 238, 0.3); border-radius: 6px; display: flex; align-items: center; justify-content: center;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22D3EE" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="2" y="7" width="16" height="12" rx="2"></rect>
            <path d="M20 11v4"></path>
            <path d="M7 13l2-2 2 4 2-2"></path>
          </svg>
        </div>
        <div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-weight: 700; font-size: 0.95rem; letter-spacing: 0.05em; color: #FFFFFF;">CELLGUARD</span>
            <span style="font-size: 0.65rem; font-family: 'JetBrains Mono', monospace; color: #22D3EE; background: rgba(34, 211, 238, 0.1); border: 1px solid rgba(34, 211, 238, 0.25); padding: 1px 6px; border-radius: 4px; font-weight: 600;">PROGNOSTIC SUITE</span>
          </div>
          <div style="font-size: 0.70rem; color: #94A3B8; font-weight: 400;">BMS Estimation of SOH, Impedance &amp; RUL</div>
        </div>
      </div>
      
      <!-- Instant Topbar Navigation Bar (always accessible whether side panel is open or closed) -->
      <nav class="cg-top-nav">
        {links_markup}
      </nav>

      <div style="display: flex; align-items: center; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); padding: 4px 10px; border-radius: 6px; font-size: 0.70rem; font-family: 'JetBrains Mono', monospace; color: #10B981; font-weight: 600;">
          <span style="width: 6px; height: 6px; border-radius: 50%; background: #10B981; box-shadow: 0 0 6px #10B981;"></span>
          ONLINE &bull; NASA B0005/6/7/18
        </div>
      </div>
    </div>
    """
    render_html(topbar_html)


def render_footer(metrics: dict = None):
    """Clean technical footer."""
    versions = "scikit-learn 1.6.1 | numpy 2.1.3 | pandas 2.2.3 | scipy 1.16.3"
    if metrics and "environment_versions" in metrics:
        v = metrics["environment_versions"]
        versions = f"scikit-learn {v.get('scikit_learn', '1.6.1')} | numpy {v.get('numpy', '2.1.3')} | pandas {v.get('pandas', '2.2.3')} | scipy {v.get('scipy', '1.16.3')}"

    footer_html = f"""
    <div class="cg-footer">
      <div>
        <span style="font-weight: 600; color: #E2E8F0;">CellGuard BMS Diagnostics</span> &bull; NASA Ames Prognostics Data Repository (Saha &amp; Goebel)
      </div>
      <div>
        <span style="font-family: 'JetBrains Mono', monospace; color: #64748B;">{versions}</span>
      </div>
      <div>
        Controlled Lab Baseline: 24&deg;C ambient, 2.0 A constant-current discharge
      </div>
    </div>
    """
    render_html(footer_html)


def status_chip_html(status_type: str, label: str = None) -> str:
    """Status chip with crisp typography."""
    status_lower = str(status_type).lower()
    if "good" in status_lower or "healthy" in status_lower:
        lbl = label or "Optimal"
        return f'<span class="cg-status-chip cg-status-healthy">&bull; {lbl}</span>'
    elif "degrad" in status_lower:
        lbl = label or "Degrading"
        return f'<span class="cg-status-chip cg-status-degrading">&bull; {lbl}</span>'
    else:
        lbl = label or "EOL Reached"
        return f'<span class="cg-status-chip cg-status-eol">&bull; {lbl}</span>'


def kpi_card_html(
    label: str,
    value: str,
    unit: str = "",
    error_band: str = None,
    subtext: str = "",
    status: str = None,
    tooltip: str = "",
) -> str:
    """Render a clean precision engineering KPI tile."""
    status_markup = status_chip_html(status) if status else ""
    error_markup = (
        f'<span class="cg-error-band">&plusmn; {error_band} {unit}</span>'
        if error_band
        else ""
    )
    title_tooltip = f'title="{tooltip}"' if tooltip else ""

    return f"""
    <div class="cg-kpi" {title_tooltip}>
      <div class="cg-kpi-label">
        <span>{label}</span>
        {status_markup}
      </div>
      <div class="cg-kpi-value">
        {value}<span class="cg-kpi-unit">{unit}</span>
      </div>
      <div class="cg-kpi-sub">
        {error_markup}
        <span style="color: #94A3B8; font-size: 0.74rem;">{subtext}</span>
      </div>
    </div>
    """


def render_kpi(label: str, value: str, unit: str = "", error_band: str = None, subtext: str = "", status: str = None, tooltip: str = ""):
    """Render KPI card into Streamlit."""
    render_html(kpi_card_html(label, value, unit, error_band, subtext, status, tooltip))


def tachometer_soh_gauge_svg(soh: float, capacity_ah: float = None, size: int = 210) -> str:
    """Precision aerospace tachometer gauge for SOH with calibrated radial tick marks."""
    import math

    clamped_soh = max(0.0, min(100.0, float(soh)))
    radius = size * 0.38
    cx = size / 2.0
    cy = size / 2.0

    # 260 degree arc from 140 deg to 400 deg
    full_circ = 2 * math.pi * radius
    arc_len = full_circ * (260.0 / 360.0)
    stroke_offset = arc_len * (1.0 - clamped_soh / 100.0)

    if clamped_soh >= 80.0:
        stroke_color = "#10B981"
        status_label = "HEALTHY"
        status_color = "#10B981"
    elif clamped_soh >= 70.0:
        stroke_color = "#F59E0B"
        status_label = "DEGRADING"
        status_color = "#F59E0B"
    else:
        stroke_color = "#EF4444"
        status_label = "EOL REACHED"
        status_color = "#EF4444"

    # Generate radial tick marks
    ticks_svg = []
    for pct in range(0, 101, 10):
        deg = 140.0 + (pct / 100.0) * 260.0
        rad = math.radians(deg)
        is_major = pct in [0, 50, 70, 80, 100]
        t_len = 8 if is_major else 4
        t_color = stroke_color if pct <= clamped_soh else "rgba(255,255,255,0.15)"
        if pct == 70:
            t_color = "#EF4444"
        elif pct == 80:
            t_color = "#F59E0B"

        x1 = cx + (radius + 10) * math.cos(rad)
        y1 = cy + (radius + 10) * math.sin(rad)
        x2 = cx + (radius + 10 + t_len) * math.cos(rad)
        y2 = cy + (radius + 10 + t_len) * math.sin(rad)
        ticks_svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{t_color}" stroke-width="{1.5 if is_major else 1}" />')

    ticks_markup = "\n".join(ticks_svg)
    cap_markup = f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.85rem; color: #94A3B8; margin-top: 2px;">{capacity_ah:.3f} Ah</div>' if capacity_ah else ""

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative;">
      <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <defs>
          <linearGradient id="tachoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#22D3EE" />
            <stop offset="100%" stop-color="{stroke_color}" />
          </linearGradient>
        </defs>
        
        <!-- Outer Tick Ring -->
        {ticks_markup}

        <!-- Background Track -->
        <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="8"
                stroke-dasharray="{arc_len} {full_circ}" stroke-linecap="round"
                transform="rotate(140 {cx} {cy})" />
                
        <!-- Active SOH Arc -->
        <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="url(#tachoGrad)" stroke-width="8"
                stroke-dasharray="{arc_len} {full_circ}" stroke-dashoffset="{stroke_offset}" stroke-linecap="round"
                transform="rotate(140 {cx} {cy})" />
      </svg>
      <div style="position: absolute; top: 48%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
        <div style="font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #64748B;">STATE OF HEALTH</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 2.1rem; font-weight: 800; color: #FFFFFF; line-height: 1;">
          {clamped_soh:.1f}<span style="font-size: 1.05rem; color: #94A3B8;">%</span>
        </div>
        {cap_markup}
        <div style="font-size: 0.72rem; font-weight: 700; color: {status_color}; letter-spacing: 0.06em; margin-top: 4px;">
          &bull; {status_label}
        </div>
      </div>
    </div>
    """


def prismatic_battery_svg(soh: float, width: int = 120, height: int = 210) -> str:
    """High-tech prismatic cell graphic with terminal posts and internal energy level."""
    clamped_soh = max(0.0, min(100.0, float(soh)))
    fill_ratio = clamped_soh / 100.0

    if clamped_soh >= 80.0:
        fill_color = "#10B981"
    elif clamped_soh >= 70.0:
        fill_color = "#F59E0B"
    else:
        fill_color = "#EF4444"

    body_x = 16
    body_y = 28
    body_w = width - 32
    body_h = height - 42

    inner_pad = 5
    inner_h = body_h - (2 * inner_pad)
    curr_fill_h = inner_h * fill_ratio
    fill_y = body_y + inner_pad + (inner_h - curr_fill_h)
    fill_w = body_w - (2 * inner_pad)
    fill_x = body_x + inner_pad

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
      <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" fill="none">
        <defs>
          <linearGradient id="cellFillGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="{fill_color}" stop-opacity="0.9" />
            <stop offset="100%" stop-color="{fill_color}" stop-opacity="0.5" />
          </linearGradient>
        </defs>

        <!-- Anode & Cathode Posts -->
        <rect x="{width * 0.28}" y="14" width="14" height="14" rx="2" fill="#E2E8F0" />
        <rect x="{width * 0.60}" y="14" width="14" height="14" rx="2" fill="#94A3B8" />

        <!-- Casing Shell -->
        <rect x="{body_x}" y="{body_y}" width="{body_w}" height="{body_h}" rx="8" fill="#0C1019" stroke="rgba(255,255,255,0.12)" stroke-width="1.5" />

        <!-- Internal Charge Strata -->
        <rect x="{fill_x}" y="{fill_y}" width="{fill_w}" height="{curr_fill_h}" rx="4" fill="url(#cellFillGrad)" />

        <!-- Calibrated Level Ticks -->
        <line x1="{fill_x}" y1="{body_y + inner_pad + inner_h * 0.20}" x2="{fill_x + 8}" y2="{body_y + inner_pad + inner_h * 0.20}" stroke="rgba(255,255,255,0.3)" />
        <line x1="{fill_x}" y1="{body_y + inner_pad + inner_h * 0.40}" x2="{fill_x + 8}" y2="{body_y + inner_pad + inner_h * 0.40}" stroke="rgba(255,255,255,0.3)" />
        <line x1="{fill_x}" y1="{body_y + inner_pad + inner_h * 0.60}" x2="{fill_x + 8}" y2="{body_y + inner_pad + inner_h * 0.60}" stroke="rgba(255,255,255,0.3)" />
        <line x1="{fill_x}" y1="{body_y + inner_pad + inner_h * 0.80}" x2="{fill_x + 8}" y2="{body_y + inner_pad + inner_h * 0.80}" stroke="rgba(255,255,255,0.3)" />

        <!-- Center Li-ion Emblem -->
        <text x="{width/2}" y="{body_y + body_h/2 + 4}" fill="rgba(255,255,255,0.4)" font-family="'JetBrains Mono', monospace" font-size="10" font-weight="700" text-anchor="middle">18650</text>
      </svg>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600; color: #94A3B8; margin-top: 4px;">
        {clamped_soh:.1f}% &bull; {fill_color}
      </div>
    </div>
    """


def alert_box(message: str, alert_type: str = "warn", title: str = None):
    """Clean alert banner."""
    color = "#F59E0B" if alert_type == "warn" else "#EF4444"
    bg = "rgba(245, 158, 11, 0.08)" if alert_type == "warn" else "rgba(239, 68, 68, 0.08)"
    border = "rgba(245, 158, 11, 0.25)" if alert_type == "warn" else "rgba(239, 68, 68, 0.25)"
    t_line = f'<div style="font-weight: 600; margin-bottom: 2px;">{title}</div>' if title else ""

    render_html(f"""
    <div style="background: {bg}; border: 1px solid {border}; border-left: 3px solid {color}; border-radius: 8px; padding: 12px 16px; margin: 10px 0; color: {color}; font-size: 0.84rem; line-height: 1.5;">
      {t_line}
      <div style="color: #E2E8F0;">{message}</div>
    </div>
    """)
