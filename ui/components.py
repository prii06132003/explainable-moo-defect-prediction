"""
DefectIQ UI Design System & Interactive Futuristic Plotly Visualizations.
Adheres strictly to the DefectIQ Dark Futuristic AI Theme:
  - Background: #0B1020
  - Sidebar: #10172A
  - Cards: #151E33
  - Primary Accent: #8B5CF6
  - Secondary Accent: #38BDF8
  - Success: #22C55E
  - Danger: #EF4444
  - Text Primary: #F8FAFC
  - Text Muted: #94A3B8
  - Borders: #26334D
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# Futuristic Dark Palette Constants
BG_APP = "#0B1020"
BG_SIDEBAR = "#10172A"
BG_CARD = "#151E33"
COLOR_PRIMARY = "#8B5CF6"      # Electric Violet
COLOR_SECONDARY = "#38BDF8"    # Cyber Sky Blue
COLOR_SUCCESS = "#22C55E"      # Neon Emerald
COLOR_DANGER = "#EF4444"       # Neon Crimson
COLOR_WARNING = "#F59E0B"      # Amber
COLOR_PINK = "#EC4899"         # Cyber Pink
COLOR_BORDER = "#26334D"
COLOR_TEXT = "#F8FAFC"
COLOR_MUTED = "#94A3B8"


def inject_custom_css():
    """Injects high-end, dark futuristic glassmorphism CSS styling."""
    st.markdown("""
        <style>
        /* Import Google Fonts - Inter */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            color: #F8FAFC;
        }

        /* App Background */
        .stApp {
            background-color: #0B1020;
            background-image: 
                radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(56, 189, 248, 0.06) 0px, transparent 50%);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #10172A !important;
            border-right: 1px solid #26334D !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: #26334D !important;
        }

        /* Futuristic Brand Header */
        .brand-title {
            font-size: 2.3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #F8FAFC 0%, #8B5CF6 50%, #38BDF8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.03em;
            margin-bottom: 0.2rem;
            display: inline-block;
        }
        .brand-tagline {
            font-size: 0.95rem;
            color: #94A3B8;
            font-weight: 400;
            margin-bottom: 1.5rem;
            letter-spacing: 0.01em;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: #151E33;
            border: 1px solid #26334D;
            border-radius: 14px;
            padding: 22px 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            margin-bottom: 16px;
        }
        .glass-card:hover {
            border-color: rgba(139, 92, 246, 0.45);
            box-shadow: 0 8px 30px rgba(139, 92, 246, 0.15);
            transform: translateY(-2px);
        }
        .glass-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.35), transparent);
        }

        /* Metric Cards */
        .metric-card-dark {
            background: #151E33;
            border: 1px solid #26334D;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        .metric-card-dark:hover {
            border-color: #8B5CF6;
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.18);
        }
        .metric-card-dark .card-title {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .metric-card-dark .card-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #F8FAFC;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.1;
        }
        .metric-card-dark .card-sub {
            font-size: 0.78rem;
            color: #38BDF8;
            margin-top: 6px;
            font-weight: 500;
        }

        /* Cyber Badges */
        .badge-clean-dark {
            display: inline-block;
            background: rgba(34, 197, 94, 0.15);
            color: #4ADE80;
            border: 1px solid rgba(34, 197, 94, 0.4);
            padding: 4px 14px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.03em;
        }
        .badge-defective-dark {
            display: inline-block;
            background: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.4);
            padding: 4px 14px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.03em;
        }
        .badge-pill {
            display: inline-block;
            background: rgba(139, 92, 246, 0.12);
            color: #C4B5FD;
            border: 1px solid rgba(139, 92, 246, 0.3);
            padding: 3px 10px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 0.78rem;
            font-family: 'JetBrains Mono', monospace;
            margin: 2px 4px 2px 0;
        }
        .badge-cyan {
            display: inline-block;
            background: rgba(56, 189, 248, 0.12);
            color: #7DD3FC;
            border: 1px solid rgba(56, 189, 248, 0.3);
            padding: 3px 10px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 0.78rem;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Actionable Recommendation Boxes */
        .recom-box-dark {
            background: rgba(139, 92, 246, 0.06);
            border: 1px solid rgba(139, 92, 246, 0.25);
            border-left: 4px solid #8B5CF6;
            border-radius: 0 10px 10px 0;
            padding: 14px 18px;
            margin-bottom: 12px;
            transition: all 0.2s ease;
        }
        .recom-box-dark:hover {
            border-color: #8B5CF6;
            background: rgba(139, 92, 246, 0.1);
        }
        .recom-title {
            font-weight: 700;
            color: #F8FAFC;
            font-size: 0.95rem;
            margin-bottom: 4px;
        }
        .recom-body {
            color: #94A3B8;
            font-size: 0.86rem;
            line-height: 1.5;
        }

        /* Streamlit Input & Widget Dark Overrides */
        div[data-baseweb="select"] > div {
            background-color: #151E33 !important;
            border-color: #26334D !important;
            color: #F8FAFC !important;
        }
        .stTextInput input, .stNumberInput input {
            background-color: #151E33 !important;
            border-color: #26334D !important;
            color: #F8FAFC !important;
        }
        .stButton>button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        .stButton>button[kind="primary"] {
            background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%) !important;
            border: 1px solid #A78BFA !important;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
        }
        .stButton>button[kind="primary"]:hover {
            box-shadow: 0 6px 25px rgba(139, 92, 246, 0.5) !important;
            transform: translateY(-1px);
        }
        </style>
    """, unsafe_allow_html=True)


def render_brand_header(title: str, subtitle: str, badge_text: str = "DefectIQ AI Platform"):
    """Renders the top branding hero container."""
    st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span class="badge-cyan">⚡ {badge_text}</span>
                <span style="color: #64748B; font-size: 0.8rem;">v2.4 Enterprise XAI</span>
            </div>
            <div class="brand-title">{title}</div>
            <div class="brand-tagline">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)


def render_metric_card_dark(title: str, value: Any, subtitle: str = ""):
    """Renders a futuristic dark metric card."""
    st.markdown(f"""
        <div class="metric-card-dark">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
            <div class="card-sub">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)


def get_dark_plotly_layout(height: int = 400, title: Optional[str] = None) -> Dict[str, Any]:
    """Generates standardized dark layout dictionary for all Plotly charts."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#151E33",
        font=dict(color=COLOR_TEXT, family="Inter, system-ui, sans-serif", size=12),
        title=dict(text=title, font=dict(color=COLOR_TEXT, size=14, weight="bold")) if title else None,
        margin=dict(t=50 if title else 25, b=35, l=45, r=25),
        height=height,
        xaxis=dict(
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_MUTED),
            titlefont=dict(color=COLOR_TEXT)
        ),
        yaxis=dict(
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_MUTED),
            titlefont=dict(color=COLOR_TEXT)
        ),
        legend=dict(
            font=dict(color=COLOR_TEXT),
            bgcolor="rgba(16, 23, 42, 0.75)",
            bordercolor=COLOR_BORDER,
            borderwidth=1
        )
    )


def plot_class_distribution_dark(y: pd.Series) -> go.Figure:
    """Plots interactive dark-themed class distribution donut chart."""
    counts = y.value_counts().to_dict()
    labels = ["Clean Modules", "Defective Modules"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = [COLOR_SUCCESS, COLOR_DANGER]

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color=BG_CARD, width=3)),
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        )
    ])
    layout = get_dark_plotly_layout(height=320, title="Module Defect Distribution")
    layout["showlegend"] = True
    fig.update_layout(**layout)
    return fig


def plot_correlation_heatmap_dark(X: pd.DataFrame, max_cols: int = 20) -> go.Figure:
    """Plots correlation matrix heatmap styled with dark gradient."""
    df_sub = X.iloc[:, :max_cols]
    corr = df_sub.corr().round(2)

    fig = px.imshow(
        corr,
        aspect="auto",
        color_continuous_scale="Purples",
        zmin=-1.0,
        zmax=1.0
    )
    layout = get_dark_plotly_layout(height=480, title=f"Metric Inter-Correlation Matrix (Top {min(max_cols, len(X.columns))} Metrics)")
    fig.update_layout(**layout)
    return fig


def plot_ga_pso_convergence_dark(history: List[Dict[str, Any]], title: str = "Metaheuristic Convergence") -> go.Figure:
    """Plots dark convergence curve for GA or BPSO."""
    df_hist = pd.DataFrame(history)
    x_col = "generation" if "generation" in df_hist.columns else "iteration"

    fig = go.Figure()
    if "best_fitness" in df_hist.columns:
        fig.add_trace(go.Scatter(
            x=df_hist[x_col], y=df_hist["best_fitness"],
            mode="lines+markers", name="Best Fitness",
            line=dict(color=COLOR_PRIMARY, width=3),
            marker=dict(size=7, color=COLOR_PRIMARY)
        ))
    elif "gbest_fitness" in df_hist.columns:
        fig.add_trace(go.Scatter(
            x=df_hist[x_col], y=df_hist["gbest_fitness"],
            mode="lines+markers", name="Swarm GBest Fitness",
            line=dict(color=COLOR_PRIMARY, width=3),
            marker=dict(size=7, color=COLOR_PRIMARY)
        ))

    if "avg_fitness" in df_hist.columns:
        fig.add_trace(go.Scatter(
            x=df_hist[x_col], y=df_hist["avg_fitness"],
            mode="lines", name="Population Mean Fitness",
            line=dict(color=COLOR_SECONDARY, width=2, dash="dash")
        ))

    if "features_selected_count" in df_hist.columns:
        fig.add_trace(go.Scatter(
            x=df_hist[x_col], y=df_hist["features_selected_count"],
            mode="lines+markers", name="Selected Features Count",
            yaxis="y2", line=dict(color=COLOR_WARNING, width=2)
        ))

    layout = get_dark_plotly_layout(height=380, title=title)
    layout["xaxis"]["title"] = x_col.capitalize()
    layout["yaxis"]["title"] = "Fitness Score"
    layout["yaxis2"] = dict(
        title="Features Count",
        overlaying="y",
        side="right",
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(color=COLOR_WARNING),
        titlefont=dict(color=COLOR_WARNING)
    )
    fig.update_layout(**layout)
    return fig


def plot_pareto_front_2d_dark(pareto_front: List[Dict[str, Any]], knee_solution: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Plots dark 2D Pareto Front: Performance vs Feature Complexity."""
    df_pf = pd.DataFrame(pareto_front)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_pf["feature_ratio"],
        y=df_pf["raw_metric"],
        mode="markers+lines",
        name="Pareto Optimal Solutions",
        marker=dict(
            size=12,
            color=df_pf["eval_latency_sec"],
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title=dict(text="Latency (s)", font=dict(color=COLOR_TEXT)), tickfont=dict(color=COLOR_MUTED))
        ),
        line=dict(color=COLOR_SECONDARY, width=2, dash="dot"),
        text=[f"Features: {r['n_selected']}<br>Score: {r['raw_metric']}<br>Latency: {r['eval_latency_sec']}s" for _, r in df_pf.iterrows()],
        hoverinfo="text"
    ))

    if knee_solution is not None:
        fig.add_trace(go.Scatter(
            x=[knee_solution["feature_ratio"]],
            y=[knee_solution["raw_metric"]],
            mode="markers+text",
            name="★ Knee-Point Optimal Trade-off",
            marker=dict(size=20, color=COLOR_DANGER, symbol="star", line=dict(color="#FFFFFF", width=1.5)),
            text=["★ Knee Point"],
            textposition="top center",
            textfont=dict(color="#F87171", size=13, weight="bold")
        ))

    layout = get_dark_plotly_layout(height=420, title="NSGA-II 2D Non-Dominated Pareto Frontier")
    layout["xaxis"]["title"] = "Feature Subset Ratio (N_selected / N_total)"
    layout["xaxis"]["range"] = [-0.05, 1.05]
    layout["yaxis"]["title"] = "Classification Metric (F1 / MCC / AUC)"
    layout["yaxis"]["range"] = [0.0, 1.05]
    fig.update_layout(**layout)
    return fig


def plot_pareto_front_3d_dark(pareto_front: List[Dict[str, Any]], knee_solution: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Plots 3D Pareto Front surface with dark background."""
    df_pf = pd.DataFrame(pareto_front)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df_pf["predictive_error"],
        y=df_pf["feature_ratio"],
        z=df_pf["eval_latency_sec"],
        mode="markers",
        name="Pareto Non-Dominated Set",
        marker=dict(
            size=8,
            color=df_pf["raw_metric"],
            colorscale="Viridis",
            colorbar=dict(title=dict(text="Metric Score", font=dict(color=COLOR_TEXT)), tickfont=dict(color=COLOR_MUTED)),
            opacity=0.9
        ),
        text=[f"Error: {r['predictive_error']}<br>Features: {r['n_selected']}<br>Latency: {r['eval_latency_sec']}s" for _, r in df_pf.iterrows()],
        hoverinfo="text"
    ))

    if knee_solution is not None:
        fig.add_trace(go.Scatter3d(
            x=[knee_solution["predictive_error"]],
            y=[knee_solution["feature_ratio"]],
            z=[knee_solution["eval_latency_sec"]],
            mode="markers+text",
            name="★ Knee Point",
            marker=dict(size=14, color=COLOR_DANGER, symbol="diamond"),
            text=["★ Knee Point"],
            textposition="top center"
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_TEXT, family="Inter"),
        title=dict(text="NSGA-II 3D Multi-Objective Optimization Surface", font=dict(color=COLOR_TEXT, size=14, weight="bold")),
        scene=dict(
            xaxis=dict(title="f1: Predictive Error (1 - Score)", backgroundcolor=BG_CARD, gridcolor=COLOR_BORDER),
            yaxis=dict(title="f2: Feature Subset Ratio", backgroundcolor=BG_CARD, gridcolor=COLOR_BORDER),
            zaxis=dict(title="f3: Latency (seconds)", backgroundcolor=BG_CARD, gridcolor=COLOR_BORDER)
        ),
        height=520,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig


def plot_model_radar_dark(comparison_df: pd.DataFrame) -> go.Figure:
    """Plots dark multi-metric radar chart."""
    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig = go.Figure()
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS, COLOR_WARNING, COLOR_PINK]

    for idx, row in comparison_df.iterrows():
        c_idx = idx % len(colors)
        vals = [row[cat] for cat in categories]
        vals.append(vals[0])

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            name=row["Model"],
            line=dict(color=colors[c_idx], width=2.5),
            fill='toself',
            opacity=0.20
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_TEXT, family="Inter"),
        title=dict(text="Multi-Metric Model Evaluation Radar", font=dict(color=COLOR_TEXT, size=14, weight="bold")),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0], gridcolor=COLOR_BORDER, tickfont=dict(color=COLOR_MUTED)),
            angularaxis=dict(gridcolor=COLOR_BORDER, tickfont=dict(color=COLOR_TEXT)),
            bgcolor=BG_CARD
        ),
        height=420,
        margin=dict(t=50, b=40, l=40, r=40)
    )
    return fig


def plot_roc_curves_dark(evaluation_results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Plots interactive dark ROC curves."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Random Guess (AUC = 0.50)",
        line=dict(color="#475569", dash="dash")
    ))

    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS, COLOR_WARNING, COLOR_PINK]
    for idx, (m_name, res) in enumerate(evaluation_results.items()):
        c = colors[idx % len(colors)]
        roc_data = res.get("roc_curve", {})
        auc_score = res.get("roc_auc", 0.0)
        fig.add_trace(go.Scatter(
            x=roc_data.get("fpr", []),
            y=roc_data.get("tpr", []),
            mode="lines",
            name=f"{m_name} (AUC = {auc_score:.3f})",
            line=dict(color=c, width=2.5)
        ))

    layout = get_dark_plotly_layout(height=400, title="Receiver Operating Characteristic (ROC) Spectrum")
    layout["xaxis"]["title"] = "False Positive Rate (1 - Specificity)"
    layout["yaxis"]["title"] = "True Positive Rate (Sensitivity / Recall)"
    fig.update_layout(**layout)
    return fig


def plot_confusion_matrix_dark(cm_dict: Dict[str, int], model_name: str = "Classifier") -> go.Figure:
    """Plots interactive 2x2 confusion matrix styled with purple gradients."""
    tp = cm_dict.get("tp", 0)
    fp = cm_dict.get("fp", 0)
    fn = cm_dict.get("fn", 0)
    tn = cm_dict.get("tn", 0)

    z = [[tn, fp], [fn, tp]]
    text = [
        [f"True Neg (Clean):<br><b>{tn}</b>", f"False Pos (Type I):<br><b>{fp}</b>"],
        [f"False Neg (Type II):<br><b>{fn}</b>", f"True Pos (Buggy):<br><b>{tp}</b>"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Predicted Clean", "Predicted Buggy"],
        y=["Actual Clean", "Actual Buggy"],
        text=text,
        texttemplate="%{text}",
        colorscale=[[0, "#10172A"], [0.5, "#4C1D95"], [1.0, "#8B5CF6"]],
        showscale=False
    ))

    layout = get_dark_plotly_layout(height=300, title=f"Confusion Matrix: {model_name}")
    fig.update_layout(**layout)
    return fig


def plot_shap_global_bar_dark(rankings: List[Dict[str, Any]], top_k: int = 15) -> go.Figure:
    """Plots global SHAP feature importance horizontal bar chart with dark gradient."""
    df_ranks = pd.DataFrame(rankings[:top_k]).iloc[::-1]

    fig = go.Figure(go.Bar(
        x=df_ranks["importance"],
        y=df_ranks["feature"],
        orientation="h",
        marker=dict(
            color=df_ranks["importance"],
            colorscale=[[0, "#38BDF8"], [1, "#8B5CF6"]],
            line=dict(color=COLOR_BORDER, width=1)
        ),
        text=df_ranks["importance"].round(3),
        textposition="outside",
        textfont=dict(color=COLOR_TEXT)
    ))

    layout = get_dark_plotly_layout(height=max(360, len(df_ranks) * 28), title=f"Global SHAP Feature Attribution (Top {len(df_ranks)} Metrics)")
    layout["xaxis"]["title"] = "Mean |SHAP Value| (Impact on Model Defect Output)"
    layout["yaxis"]["title"] = "Software Metric"
    layout["margin"]["l"] = 140
    fig.update_layout(**layout)
    return fig


def plot_shap_local_waterfall_dark(local_res: Dict[str, Any], max_display: int = 10) -> go.Figure:
    """Plots interactive Waterfall chart of local SHAP contributions in dark theme."""
    contributors = local_res.get("top_contributors", [])[:max_display]
    base_val = local_res.get("base_value", 0.5)

    names = ["Base Value"] + [c["feature"] for c in contributors]
    vals = [base_val] + [c["shap_value"] for c in contributors]
    measures = ["absolute"] + ["relative"] * len(contributors)

    fig = go.Figure(go.Waterfall(
        name="SHAP Local Attribution",
        orientation="v",
        measure=measures,
        x=names,
        y=vals,
        connector={"line": {"color": "#475569"}},
        decreasing={"marker": {"color": COLOR_SUCCESS}},
        increasing={"marker": {"color": COLOR_DANGER}},
        totals={"marker": {"color": COLOR_PRIMARY}}
    ))

    prob = local_res.get("defect_probability", 0.5)
    pred = local_res.get("prediction", "Clean")

    layout = get_dark_plotly_layout(height=420, title=f"Local SHAP Explanation: Module Diagnosed as <b>{pred}</b> (P = {prob:.1%})")
    layout["xaxis"]["tickangle"] = -35
    layout["yaxis"]["title"] = "Model Output / Logit Shift"
    layout["margin"]["b"] = 80
    fig.update_layout(**layout)
    return fig


def plot_latency_comparison_dark(comparison_df: pd.DataFrame) -> go.Figure:
    """Plots execution latency comparison between models."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["Train Time (s)"],
        name="Training Latency (s)",
        marker_color=COLOR_PRIMARY
    ))
    fig.add_trace(go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["Inference (s)"] * 1000,
        name="Inference Latency (ms)",
        marker_color=COLOR_SECONDARY
    ))

    layout = get_dark_plotly_layout(height=360, title="Computational Efficiency & Latency Benchmark")
    layout["yaxis"]["title"] = "Latency (seconds / milliseconds)"
    layout["barmode"] = "group"
    fig.update_layout(**layout)
    return fig
