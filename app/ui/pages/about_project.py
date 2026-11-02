"""
about_project.py - Technical Methodology, Electrical Engineering Insights & Dataset Architecture.
Sections:
1. 1st-Order Thevenin Equivalent Circuit Model (ECM) with interactive schematic & governing differential equations
2. Electrochemical degradation mechanisms (SEI layer, lithium plating, loss of lithium inventory)
3. NASA Ames Li-ion Dataset Architecture (B0005, B0006, B0007, B0018, 636 records)
4. Deployed ML Pipeline Architecture (M1-M5 pipelines, feature scaling, zero-leakage partial window)
"""

import os
import streamlit as st
import pandas as pd

from ui.components import (
    render_compact_header,
    render_category_legend,
    render_thevenin_circuit_svg,
    render_callout,
    render_html,
)


def render_about_project(metrics: dict):
    """Render the comprehensive About the Project & Electrical Engineering Insights page."""
    # Ensure default session state
    if "selected_cell" not in st.session_state:
        st.session_state["selected_cell"] = "B0018"

    active_cell = st.session_state["selected_cell"]
    render_compact_header(
        page_title="About the Project & EE Insights",
        selected_battery=active_cell,
        data_source="NASA Ames Prognostics CoE &bull; System Architecture",
        model_status="1st-Order Thevenin ECM &bull; M1-M5 Pipeline Suite",
    )

    render_category_legend()

    tab_ee, tab_dataset, tab_ml_arch = st.tabs([
        "⚡ Electrical Engineering & Thevenin Circuit",
        "📚 NASA Dataset & Testing Protocol",
        "🧠 Machine Learning Pipeline Architecture",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: ELECTRICAL ENGINEERING & THEVENIN ECM
    # -------------------------------------------------------------------------
    with tab_ee:
        st.markdown("""
        <div class="cg-card">
          <div class="cg-card-header">
            <div>
              <h3 class="cg-card-title">1st-Order Thevenin Equivalent Circuit Model (ECM)</h3>
              <p class="cg-card-subtitle">Lumped-parameter electrical model representing cell electrochemical dynamics</p>
            </div>
            <span class="cg-cat-badge cg-cat-calculated">CIRCUIT PHYSICS</span>
          </div>
        """, unsafe_allow_html=True)

        # SVG Schematic
        render_html(render_thevenin_circuit_svg())

        st.markdown("""
        <div style="font-size: 0.84rem; color: #1E293B; line-height: 1.6; margin-top: 14px;">
          <p>
            The 1st-order Thevenin ECM models the lithium-ion cell as an open-circuit voltage source (OCV) dependent on 
            State of Charge (SOC), placed in series with an instantaneous ohmic resistor <code>R₀</code> and a single 
            parallel resistor-capacitor branch <code>(R₁, C₁)</code> modeling charge-transfer polarization and electrochemical double-layer capacitance.
          </p>
          <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px 16px; margin: 10px 0;">
            <strong>Governing Differential Equations:</strong>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #0F172A; margin-top: 6px;">
              1. Terminal Voltage: &nbsp;&nbsp;&nbsp;&nbsp; V_term(t) = OCV(SOC) - I(t) &times; R₀ - V₁(t)<br/>
              2. Polarization State: &nbsp; dV₁(t)/dt = -(1 / (R₁ &times; C₁)) &times; V₁(t) + (1 / C₁) &times; I(t)
            </div>
          </div>
          <p>
            <strong>Electrochemical Parameter Mapping:</strong>
            <ul style="padding-left: 18px; margin: 6px 0;">
              <li><strong>R₀ &asymp; Re (Ohmic Resistance):</strong> Measures bulk electrolyte conductivity, separator porosity, and current collector tabs. Causes instantaneous voltage step at load transition.</li>
              <li><strong>R₁ &asymp; Rct (Charge-Transfer Resistance):</strong> Measures the kinetic overpotential barrier for charge transfer and Li+ intercalation across the electrode-electrolyte interface.</li>
              <li><strong>C₁ (Double-Layer Capacitance):</strong> Models electrical charge accumulation at the Helmholtz double layer.</li>
              <li><strong>Engineering Caveat:</strong> While <code>Re</code> and <code>Rct</code> correspond closely to ECM parameters <code>R₀</code> and <code>R₁</code>, in rigorous electrochemistry they represent frequency-dependent impedances extracted from complex Nyquist arcs.</li>
            </ul>
          </p>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Degradation Mechanisms
        st.markdown("""
        <div class="cg-card">
          <div class="cg-card-header">
            <div>
              <h3 class="cg-card-title">Lithium-Ion Battery Degradation Mechanisms</h3>
              <p class="cg-card-subtitle">Microscopic physical causes of capacity fade and impedance growth</p>
            </div>
            <span class="cg-chip cg-chip-status">CHEMISTRY</span>
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; font-size: 0.83rem; line-height: 1.55;">
            <div>
              <strong style="color: #2563EB;">1. Solid Electrolyte Interphase (SEI) Growth</strong>
              <p style="color: #475569; margin: 4px 0;">
                During cycling, parasitic decomposition of liquid organic electrolyte at the graphite anode continuously forms a passivation layer. 
                Ongoing SEI layer thickening consumes cyclable lithium ions (<strong>Loss of Lithium Inventory, LLI</strong>) and increases <code>Rct</code>.
              </p>
              <strong style="color: #2563EB;">2. Loss of Active Material (LAM)</strong>
              <p style="color: #475569; margin: 4px 0;">
                Repeated volume expansion and contraction during lithium insertion causes micro-cracking and electrical isolation of cathode (LiCoO2) 
                and anode particles, reducing maximum available storage sites.
              </p>
            </div>
            <div>
              <strong style="color: #2563EB;">3. Lithium Plating</strong>
              <p style="color: #475569; margin: 4px 0;">
                Under high charge rates or low temperatures, lithium ions deposit as metallic lithium dendrites rather than intercalating into graphite, 
                risking internal micro-shorts and thermal runaway.
              </p>
              <strong style="color: #2563EB;">4. Electrolyte Depletion &amp; Resistance Rise</strong>
              <p style="color: #475569; margin: 4px 0;">
                Solvent consumption dry-out reduces ionic conductivity in separator pores, manifested as steady ohmic resistance (<code>Re</code>) growth over hundreds of cycles.
              </p>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 2: NASA DATASET & TESTING PROTOCOL
    # -------------------------------------------------------------------------
    with tab_dataset:
        st.markdown("""
        <div class="cg-card">
          <div class="cg-card-header">
            <div>
              <h3 class="cg-card-title">NASA Ames Li-ion Aging Dataset Specification</h3>
              <p class="cg-card-subtitle">Benchmark dataset from the NASA Ames Prognostics Center of Excellence (PCoE)</p>
            </div>
            <span class="cg-cat-badge cg-cat-measured">PCOE BENCHMARK</span>
          </div>
          <div style="font-size: 0.84rem; color: #1E293B; line-height: 1.6;">
            <p>
              The models in CellGuard are trained and validated exclusively on high-precision experimental degradation datasets 
              collected by B. Saha and K. Goebel (2007) at the NASA Ames Prognostics Center of Excellence, Moffett Field, CA.
            </p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_ds1, col_ds2 = st.columns([1.1, 1.2], gap="large")

        with col_ds1:
            st.markdown("##### Cell Experimental Parameters")
            params_df = pd.DataFrame([
                {"Property": "Cell Form Factor", "Value": "18650 Cylindrical"},
                {"Property": "Cathode Chemistry", "Value": "Lithium Cobalt Oxide (LiCoO2)"},
                {"Property": "Anode Chemistry", "Value": "Graphite"},
                {"Property": "Rated Nominal Capacity", "Value": "2.00 Ah"},
                {"Property": "Charge Protocol", "Value": "CC 1.5A to 4.2V + CV to 20mA"},
                {"Property": "Discharge Protocol", "Value": "CC 2.0A to cutoff (2.7V / 2.5V / 2.2V)"},
                {"Property": "Ambient Chamber Temperature", "Value": "24°C nominal"},
                {"Property": "Total Discharge Records", "Value": "636 cycles"},
                {"Property": "EIS-Matched Impedance Records", "Value": "530 cycles"},
            ])
            st.dataframe(params_df, use_container_width=True, hide_index=True)

        with col_ds2:
            st.markdown("##### Cell Cohort Breakdown")
            cohort_df = pd.DataFrame([
                {
                    "Cell ID": "B0005",
                    "Cycles": "168",
                    "Cutoff Voltage": "2.7 V",
                    "Start Capacity": "1.856 Ah",
                    "EOL Cycle (1.4 Ah)": "124",
                    "Role": "Training Fold 1",
                },
                {
                    "Cell ID": "B0006",
                    "Cycles": "168",
                    "Cutoff Voltage": "2.5 V",
                    "Start Capacity": "2.035 Ah",
                    "EOL Cycle (1.4 Ah)": "89 (Accelerated)",
                    "Role": "Training Fold 2",
                },
                {
                    "Cell ID": "B0007",
                    "Cycles": "168",
                    "Cutoff Voltage": "2.2 V",
                    "Start Capacity": "1.891 Ah",
                    "EOL Cycle (1.4 Ah)": "Not reached (Deep cutoff)",
                    "Role": "Training Fold 3",
                },
                {
                    "Cell ID": "B0018",
                    "Cycles": "132",
                    "Cutoff Voltage": "2.5 V",
                    "Start Capacity": "1.855 Ah",
                    "EOL Cycle (1.4 Ah)": "96",
                    "Role": "Primary Held-Out Benchmark",
                },
            ])
            st.dataframe(cohort_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 3: MACHINE LEARNING PIPELINE ARCHITECTURE
    # -------------------------------------------------------------------------
    with tab_ml_arch:
        st.markdown("""
        <div class="cg-card">
          <div class="cg-card-header">
            <div>
              <h3 class="cg-card-title">Production Inference Pipeline Architecture</h3>
              <p class="cg-card-subtitle">Zero data leakage, deterministic feature engineering, and robust pipelines</p>
            </div>
            <span class="cg-cat-badge cg-cat-predicted">PIPELINES</span>
          </div>
          <div style="font-size: 0.84rem; color: #1E293B; line-height: 1.6;">
            <p>
              CellGuard executes five production machine learning pipelines trained strictly on non-destructive partial-window features.
            </p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        models_arch_df = pd.DataFrame([
            {
                "Pipeline": "M1: Capacity (Ah) & SOH (%)",
                "Algorithm": "Ridge Regression (alpha=100.0)",
                "Input Features": "18 Partial-window features (4.0V -> 3.4V)",
                "Preprocessors": "SimpleImputer(mean) -> StandardScaler",
                "Validation MAE": "0.0464 Ah (2.32% SOH)",
                "Engineering Rationale": "Ridge extrapolates to lower unseen capacities (MAE 0.0155 Ah vs GradientBoost 0.0985 Ah under 60/40 split).",
            },
            {
                "Pipeline": "M2: Ohmic Resistance Re (Ω)",
                "Algorithm": "Ridge Regression (alpha=10.0)",
                "Input Features": "18 Partial features + predicted capacity + disc_idx",
                "Preprocessors": "SimpleImputer(median) -> StandardScaler",
                "Validation MAE": "0.0066 Ω (6.6 mΩ)",
                "Engineering Rationale": "Linear impedance tracking across cycles.",
            },
            {
                "Pipeline": "M3: Charge-Transfer Rct (Ω)",
                "Algorithm": "Ridge Regression (alpha=10.0)",
                "Input Features": "18 Partial features + predicted capacity + disc_idx",
                "Preprocessors": "SimpleImputer(median) -> StandardScaler",
                "Validation MAE": "0.0055 Ω (5.5 mΩ)",
                "Engineering Rationale": "Accurately captures non-linear interfacial degradation.",
            },
            {
                "Pipeline": "M4: Remaining Useful Life (RUL)",
                "Algorithm": "Random Forest Regressor (n_est=150)",
                "Input Features": "18 Features + capacity + disc_idx + cap_slope_5 + cap_slope_10",
                "Preprocessors": "SimpleImputer(median) -> StandardScaler",
                "Validation MAE": "15.23 cycles (7.71 on B0018)",
                "Engineering Rationale": "Ensemble captures non-linear aging knee point without overfitting.",
            },
            {
                "Pipeline": "M5: Usable Discharge Energy (Wh)",
                "Algorithm": "Ridge Regression (alpha=50.0)",
                "Input Features": "18 Partial-window features",
                "Preprocessors": "SimpleImputer(mean) -> StandardScaler",
                "Validation MAE": "0.1476 Wh (~2.4% relative)",
                "Engineering Rationale": "Models terminal voltage sag impact on deliverable energy.",
            },
        ])
        st.dataframe(models_arch_df, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        render_callout(
            "<strong>Zero Label Leakage Guarantee:</strong> "
            "Total discharge duration, cutoff minimum voltage (V_min), and total integrated energy to cutoff are strictly excluded from all training features. "
            "All model inputs can be recorded within the first 5 minutes of discharge during vehicle operation.",
            alert_type="success",
            title="Defensibility Checklist",
        )
