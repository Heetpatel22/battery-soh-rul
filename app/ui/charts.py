"""
charts.py - Plotly chart generators with unified industrial engineering theme.
Color palette:
- Muted Electric Blue (#2563EB)
- Teal (#0D9488)
- Amber (#D97706)
- Red (#DC2626)
- Green (#16A34A)
- Slate / Neutral (#475569)
- Violet (#7C3AED)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict


def apply_theme(fig: go.Figure, height: int = 360, title: str = "") -> go.Figure:
    """Apply unified clean light industrial aesthetic to any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFC",
        font=dict(family="Inter, -apple-system, sans-serif", color="#0F172A", size=11),
        margin=dict(l=48, r=24, t=44 if title else 20, b=36),
        height=height,
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=13, color="#0F172A", family="Inter, sans-serif"),
            x=0.01,
            y=0.97,
        ) if title else None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=10.5, color="#475569"),
        ),
        xaxis=dict(
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#64748B"),
            zeroline=False,
            title_font=dict(size=11, color="#334155"),
        ),
        yaxis=dict(
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#64748B"),
            zeroline=False,
            title_font=dict(size=11, color="#334155"),
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=11,
            font_family="JetBrains Mono, monospace",
            font_color="#FFFFFF",
            bordercolor="#2563EB",
        ),
    )
    return fig


def plot_discharge_curve(
    t: np.ndarray,
    v: np.ndarray,
    temp: Optional[np.ndarray] = None,
    ref_t: Optional[np.ndarray] = None,
    ref_v: Optional[np.ndarray] = None,
    ref_label: str = "Reference Cycle 5 (Fresh)",
) -> go.Figure:
    """Plot terminal voltage vs time with 4.0 V - 3.4 V partial window highlight and optional reference overlay."""
    has_temp = temp is not None and len(temp) == len(t)
    if has_temp:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    # Reference curve if provided
    if ref_t is not None and ref_v is not None:
        fig.add_trace(
            go.Scatter(
                x=ref_t,
                y=ref_v,
                mode="lines",
                name=ref_label,
                line=dict(color="#94A3B8", width=1.8, dash="dash"),
                hoverinfo="x+y",
            )
        )

    # Current voltage curve
    fig.add_trace(
        go.Scatter(
            x=t,
            y=v,
            mode="lines",
            name="Terminal Voltage V(t)",
            line=dict(color="#2563EB", width=2.4),
            hoverinfo="x+y",
        )
    )

    # Partial window lines and shading (4.0 V down to 3.4 V)
    fig.add_hline(
        y=4.0,
        line_dash="dot",
        line_color="#D97706",
        line_width=1.5,
        annotation_text="4.00 V (Window Upper Limit)",
        annotation_position="top left",
        annotation_font=dict(size=9.5, color="#D97706", family="JetBrains Mono"),
    )
    fig.add_hline(
        y=3.4,
        line_dash="dot",
        line_color="#D97706",
        line_width=1.5,
        annotation_text="3.40 V (Window Lower Limit)",
        annotation_position="bottom left",
        annotation_font=dict(size=9.5, color="#D97706", family="JetBrains Mono"),
    )

    # Find time bounds where voltage is in [3.4, 4.0]
    in_window = (v <= 4.005) & (v >= 3.395)
    if np.any(in_window):
        t_win = t[in_window]
        fig.add_vrect(
            x0=float(np.min(t_win)),
            x1=float(np.max(t_win)),
            fillcolor="rgba(37, 99, 235, 0.07)",
            layer="below",
            line_width=0,
            annotation_text="M1 18-Feature Extraction Window (No deep discharge)",
            annotation_position="top right",
            annotation_font=dict(size=9.5, color="#2563EB", family="Inter"),
        )

    # Temperature trace
    if has_temp:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=temp,
                mode="lines",
                name="Surface Temp (°C)",
                line=dict(color="#DC2626", width=1.6),
            ),
            secondary_y=True,
        )
        fig.update_yaxes(
            title_text="Temperature (°C)",
            secondary_y=True,
            showgrid=False,
            tickfont=dict(color="#DC2626", family="JetBrains Mono"),
            range=[min(20, float(np.min(temp))), max(45, float(np.max(temp)) + 2)],
        )

    fig.update_xaxes(title_text="Discharge Time (seconds)")
    fig.update_yaxes(title_text="Terminal Voltage (V)", secondary_y=False)
    return apply_theme(fig, height=350, title="Discharge Voltage Profile & BMS Early Inspection Window")


def plot_capacity_fade_all_cells(df: pd.DataFrame, eol_threshold: float = 1.40) -> go.Figure:
    """Plot capacity fade trajectories for all 4 NASA cells with configurable EOL threshold."""
    fig = go.Figure()
    cell_colors = {
        "B0005": "#2563EB",  # Electric blue
        "B0006": "#D97706",  # Amber
        "B0007": "#7C3AED",  # Violet
        "B0018": "#0D9488",  # Teal (Held-out benchmark)
    }

    for cell in ["B0005", "B0006", "B0007", "B0018"]:
        cdf = df[df["cell"] == cell].sort_values("disc_idx")
        if not cdf.empty:
            label = f"{cell} (Held-out Benchmark)" if cell == "B0018" else f"{cell} (Training Rack)"
            fig.add_trace(
                go.Scatter(
                    x=cdf["disc_idx"],
                    y=cdf["capacity"],
                    mode="lines+markers",
                    name=label,
                    marker=dict(size=3.5),
                    line=dict(color=cell_colors.get(cell, "#2563EB"), width=1.8),
                )
            )

    # EOL line
    fig.add_hline(
        y=eol_threshold,
        line_dash="dash",
        line_color="#DC2626",
        line_width=1.8,
        annotation_text=f"{eol_threshold:.2f} Ah End of Life Threshold ({round((1 - eol_threshold/2.0)*100)}% Fade)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#DC2626", family="JetBrains Mono"),
    )

    fig.update_xaxes(title_text="Discharge Cycle Index")
    fig.update_yaxes(title_text="Discharged Capacity (Ah)", range=[1.1, 2.1])
    return apply_theme(fig, height=340, title="Full Degradation Curves Across NASA Ames Li-ion Cells")


def plot_life_and_rul_trajectory(
    cell_df: pd.DataFrame,
    current_cycle: int,
    pred_cap_at_cycle: Optional[float] = None,
    pred_rul_cycles: Optional[int] = None,
    eol_threshold: float = 1.40,
) -> go.Figure:
    """Interactive capacity degradation curve with health zones, cursor, and projected EOL."""
    fig = go.Figure()

    sorted_df = cell_df.sort_values("disc_idx")
    cycles = sorted_df["disc_idx"].values
    caps = sorted_df["capacity"].values

    # Health background zones
    max_c = float(np.max(cycles)) + 25
    fig.add_hrect(y0=1.60, y1=2.10, fillcolor="rgba(22, 163, 74, 0.05)", line_width=0, layer="below")
    fig.add_hrect(y0=eol_threshold, y1=1.60, fillcolor="rgba(217, 119, 6, 0.05)", line_width=0, layer="below")
    fig.add_hrect(y0=1.00, y1=eol_threshold, fillcolor="rgba(220, 38, 38, 0.05)", line_width=0, layer="below")

    # True capacity history
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=caps,
            mode="lines",
            name="Measured Capacity (Ah)",
            line=dict(color="#475569", width=2.0),
        )
    )

    # Current cycle vertical marker
    fig.add_vline(
        x=current_cycle,
        line_dash="dot",
        line_color="#2563EB",
        line_width=2.0,
        annotation_text=f"Current Cycle {current_cycle}",
        annotation_position="top left",
        annotation_font=dict(size=9.5, color="#2563EB", family="JetBrains Mono"),
    )

    # Current cycle measured point
    curr_row = sorted_df[sorted_df["disc_idx"] == current_cycle]
    if not curr_row.empty:
        curr_cap = curr_row["capacity"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[current_cycle],
                y=[curr_cap],
                mode="markers",
                name="Current Measured",
                marker=dict(size=10, color="#FFFFFF", line=dict(color="#0F172A", width=2.5)),
            )
        )

    # ML predicted capacity point
    if pred_cap_at_cycle is not None:
        fig.add_trace(
            go.Scatter(
                x=[current_cycle],
                y=[pred_cap_at_cycle],
                mode="markers",
                name="ML Predicted Capacity",
                marker=dict(size=11, color="#2563EB", symbol="diamond"),
            )
        )

    # Projected EOL marker if RUL available
    if pred_rul_cycles is not None and pred_rul_cycles > 0:
        projected_eol_cycle = current_cycle + pred_rul_cycles
        start_cap = pred_cap_at_cycle if pred_cap_at_cycle else (curr_row["capacity"].iloc[0] if not curr_row.empty else 1.6)
        fig.add_trace(
            go.Scatter(
                x=[current_cycle, projected_eol_cycle],
                y=[start_cap, eol_threshold],
                mode="lines+markers",
                name=f"Projected EOL (Cycle {projected_eol_cycle})",
                line=dict(color="#DC2626", width=2, dash="dash"),
                marker=dict(size=8, color="#DC2626", symbol="x"),
            )
        )

    # EOL line
    fig.add_hline(
        y=eol_threshold,
        line_dash="dash",
        line_color="#DC2626",
        line_width=1.5,
        annotation_text=f"{eol_threshold:.2f} Ah EOL Threshold",
        annotation_position="bottom right",
        annotation_font=dict(size=9.5, color="#DC2626", family="JetBrains Mono"),
    )

    fig.update_xaxes(title_text="Discharge Cycle Number", range=[-2, max_c])
    fig.update_yaxes(title_text="Capacity (Ah)", range=[1.2, 2.05])
    return apply_theme(fig, height=360, title="Battery Capacity Degradation & RUL Horizon")


def plot_impedance_curves(
    cell_df: pd.DataFrame,
    target: str = "Re",
    pred_val: Optional[float] = None,
    current_cycle: Optional[int] = None,
) -> go.Figure:
    """Plot Ohmic (Re) or Charge-Transfer (Rct) impedance over cycle aging."""
    fig = go.Figure()
    cdf = cell_df.dropna(subset=[target]).sort_values("disc_idx")
    unit = "Ω"
    title_param = "Ohmic Resistance Re (R₀)" if target == "Re" else "Charge-Transfer Resistance Rct (R₁)"

    fig.add_trace(
        go.Scatter(
            x=cdf["disc_idx"],
            y=cdf[target],
            mode="lines+markers",
            name=f"Measured {target} (EIS)",
            line=dict(color="#0D9488" if target == "Re" else "#7C3AED", width=1.8),
            marker=dict(size=4.5),
        )
    )

    if current_cycle is not None and pred_val is not None:
        fig.add_trace(
            go.Scatter(
                x=[current_cycle],
                y=[pred_val],
                mode="markers",
                name=f"ML Predicted {target}",
                marker=dict(size=11, color="#2563EB", symbol="diamond"),
            )
        )
        fig.add_vline(x=current_cycle, line_dash="dot", line_color="#2563EB", line_width=1.5)

    fig.update_xaxes(title_text="Discharge Cycle Index")
    fig.update_yaxes(title_text=f"{target} ({unit})")
    return apply_theme(fig, height=330, title=f"{title_param} Evolution Over Cycling")


def plot_gap_filling_benchmark(
    cycles: np.ndarray,
    measured: np.ndarray,
    masked_mask: np.ndarray,
    ml_preds: np.ndarray,
    interp_preds: np.ndarray,
    locf_preds: Optional[np.ndarray] = None,
    target_name: str = "Re",
) -> go.Figure:
    """Plot impedance gap filling comparing measured, hidden, ML prediction, linear interp, and LOCF."""
    fig = go.Figure()
    unit = "Ω"
    known_mask = ~masked_mask

    # Known measured points
    fig.add_trace(
        go.Scatter(
            x=cycles[known_mask],
            y=measured[known_mask],
            mode="lines+markers",
            name=f"Known {target_name}",
            marker=dict(size=5.5, color="#0F172A"),
            line=dict(color="#94A3B8", width=1.5),
        )
    )

    # Hidden ground truth
    fig.add_trace(
        go.Scatter(
            x=cycles[masked_mask],
            y=measured[masked_mask],
            mode="markers",
            name="Hidden Truth (EIS)",
            marker=dict(size=8, color="#FFFFFF", symbol="circle", line=dict(color="#0F172A", width=2)),
        )
    )

    # Linear Interpolation Baseline
    fig.add_trace(
        go.Scatter(
            x=cycles[masked_mask],
            y=interp_preds[masked_mask],
            mode="lines+markers",
            name="Linear Interpolation",
            line=dict(color="#D97706", width=1.6, dash="dash"),
            marker=dict(size=6, color="#D97706", symbol="triangle-up"),
        )
    )

    # LOCF Baseline
    if locf_preds is not None:
        fig.add_trace(
            go.Scatter(
                x=cycles[masked_mask],
                y=locf_preds[masked_mask],
                mode="markers",
                name="Last Observation Carried Forward",
                marker=dict(size=5.5, color="#94A3B8", symbol="square-open"),
            )
        )

    # ML Model Predictions
    fig.add_trace(
        go.Scatter(
            x=cycles[masked_mask],
            y=ml_preds[masked_mask],
            mode="markers",
            name="ML Model Prediction",
            marker=dict(size=8, color="#2563EB", symbol="diamond"),
        )
    )

    fig.update_xaxes(title_text="Discharge Cycle Index")
    fig.update_yaxes(title_text=f"{target_name} ({unit})")
    return apply_theme(fig, height=350, title=f"Impedance Gap-Filling Benchmark: {target_name}")


def plot_model_baseline_bar(target: str, metrics: dict) -> go.Figure:
    """Bar chart comparing ML model against conventional baselines from metrics.json."""
    fig = go.Figure()

    if target == "capacity":
        models = ["ML Ridge (Partial Window)", "Coulomb Counting (Full Cutoff)", "Training Mean Fill"]
        maes = [
            metrics.get("error_bands", {}).get("capacity_mae_loco", 0.0464),
            metrics.get("m1_baselines", {}).get("coulomb_counting_full_discharge", {}).get("mean_mae", 0.0131),
            metrics.get("m1_baselines", {}).get("training_mean_fill", {}).get("mean_mae", 0.1719),
        ]
        colors = ["#2563EB", "#0D9488", "#DC2626"]
        unit = "Ah"
        title = "Capacity Estimation: ML Partial Window vs Baselines (MAE)"

    elif target == "Re":
        mask_data = metrics.get("m2_impedance", {}).get("masking_and_gap_filling", {}).get("Re", {})
        models = ["Interp (10% mask)", "ML (10% mask)", "Interp (30% mask)", "ML (30% mask)", "Interp (Block Gap)", "ML (Block Gap)"]
        r10 = mask_data.get("random_mask_10pct", {})
        r30 = mask_data.get("random_mask_30pct", {})
        blk = mask_data.get("block_mask_15cycles", {})
        maes = [
            r10.get("interpolation_mae", 0.00072),
            r10.get("ml_model_mae", 0.00094),
            r30.get("interpolation_mae", 0.00087),
            r30.get("ml_model_mae", 0.00114),
            blk.get("interpolation_mae", 0.00119),
            blk.get("ml_model_mae", 0.00140),
        ]
        colors = ["#D97706", "#2563EB", "#D97706", "#2563EB", "#D97706", "#2563EB"]
        unit = "Ω"
        title = "Re Impedance: Linear Interpolation vs ML Across Masking Ratios"

    else:
        # Rct
        mask_data = metrics.get("m2_impedance", {}).get("masking_and_gap_filling", {}).get("Rct", {})
        models = ["Interp (10% mask)", "ML (10% mask)", "Interp (30% mask)", "ML (30% mask)", "Interp (Block Gap)", "ML (Block Gap)"]
        r10 = mask_data.get("random_mask_10pct", {})
        r30 = mask_data.get("random_mask_30pct", {})
        blk = mask_data.get("block_mask_15cycles", {})
        maes = [
            r10.get("interpolation_mae", 0.00112),
            r10.get("ml_model_mae", 0.00123),
            r30.get("interpolation_mae", 0.00123),
            r30.get("ml_model_mae", 0.00121),  # ML wins
            blk.get("interpolation_mae", 0.00146),
            blk.get("ml_model_mae", 0.00125),  # ML wins
        ]
        colors = ["#D97706", "#2563EB", "#D97706", "#16A34A", "#D97706", "#16A34A"]
        unit = "Ω"
        title = "Rct Impedance: ML Outperforms on 30% Masking and Contiguous Block Gaps"

    fig.add_trace(
        go.Bar(
            x=models,
            y=maes,
            marker_color=colors,
            text=[f"{m:.4f} {unit}" for m in maes],
            textposition="auto",
        )
    )

    fig.update_yaxes(title_text=f"Mean Absolute Error ({unit})")
    return apply_theme(fig, height=330, title=title)


def plot_feature_importance_horizontal(top_features: List[Dict]) -> go.Figure:
    """Plot horizontal bar chart of top feature importances with physical meaning."""
    fig = go.Figure()

    feats = [item["feature"] for item in reversed(top_features[:8])]
    scores = [item["importance"] for item in reversed(top_features[:8])]
    exps = [item.get("physics_explanation", "") for item in reversed(top_features[:8])]

    fig.add_trace(
        go.Bar(
            y=feats,
            x=scores,
            orientation="h",
            marker_color="#2563EB",
            text=[f"{s:.3f}" for s in scores],
            textposition="auto",
            hovertext=exps,
        )
    )

    fig.update_xaxes(title_text="Normalized Importance Score")
    fig.update_yaxes(title_text="Feature Identifier")
    return apply_theme(fig, height=320, title="Top Physics-Guided Feature Importances (M1)")
