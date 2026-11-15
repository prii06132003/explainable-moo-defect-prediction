"""
UI Components and Interactive Plotly Visualization Helpers for Streamlit.
Provides responsive layouts, sleek custom styling, and publication-ready charts.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# Color palette for consistent scientific visualizations
COLOR_PRIMARY = "#3B82F6"      # Indigo / Blue
COLOR_SECONDARY = "#10B981"    # Emerald Green (Clean)
COLOR_DANGER = "#EF4444"       # Red (Defective)
COLOR_WARNING = "#F59E0B"      # Amber
COLOR_PURPLE = "#8B5CF6"       # Violet
COLOR_CYAN = "#06B6D4"         # Cyan
COLOR_DARK = "#1E293B"         # Slate


def inject_custom_css():
    """Injects modern, polished CSS styling for Streamlit."""
    st.markdown("""
        <style>
        /* Base typography & containers */
        .main-header {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(90deg, #1E40AF 0%, #3B82F6 50%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            letter-spacing: -0.5px;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
            font-weight: 400;
        }
        .metric-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }
        .metric-card:hover {
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.12);
            border-color: #BFDBFE;
        }
        .metric-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.2;
        }
        .metric-sub {
            font-size: 0.78rem;
            color: #94A3B8;
            margin-top: 4px;
        }
        .badge-clean {
            display: inline-block;
            background-color: #DCFCE7;
            color: #166534;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-defective {
            display: inline-block;
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-info {
            display: inline-block;
            background-color: #DBEAFE;
            color: #1E40AF;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 0.8rem;
        }
        .recommendation-box {
            background: #F0FDF4;
            border-left: 4px solid #22C55E;
            padding: 14px 16px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 12px;
        }
        .recommendation-box-danger {
            background: #FEF2F2;
            border-left: 4px solid #EF4444;
            padding: 14px 16px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 12px;
        }
        </style>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: Any, subtitle: str = ""):
    """Renders a custom styled metric card."""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)


def plot_class_distribution(y: pd.Series) -> go.Figure:
    """Plots interactive class distribution donut chart."""
    counts = y.value_counts().to_dict()
    labels = ["Non-Defective (Clean)", "Defective (Buggy)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = [COLOR_SECONDARY, COLOR_DANGER]

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent+value',
            hoverinfo='label+value+percent'
        )
    ])
    fig.update_layout(
        title="Software Defect Class Distribution",
        showlegend=True,
        margin=dict(t=40, b=20, l=20, r=20),
        height=320
    )
    return fig


def plot_correlation_heatmap(X: pd.DataFrame, max_cols: int = 20) -> go.Figure:
    """Plots correlation matrix heatmap of software metrics."""
    df_sub = X.iloc[:, :max_cols]
    corr = df_sub.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=False,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title=f"Metric Inter-Correlation Matrix (Top {min(max_cols, len(X.columns))} features)"
    )
    fig.update_layout(height=480, margin=dict(t=50, b=30, l=50, r=20))
    return fig


def plot_ga_pso_convergence(history: List[Dict[str, Any]], title: str = "Metaheuristic Convergence") -> go.Figure:
    """Plots iteration convergence history for GA / PSO."""
    df_hist = pd.DataFrame(history)
    x_col = "generation" if "generation" in df_hist.columns else "iteration"

    fig = go.Figure()
    if "best_fitness" in df_hist.columns:
        fig.add_trace(go.Scatter(
            x=df_hist[x_col], y=df_hist["best_fitness"],
            mode="lines+markers", name="Best Fitness",
            line=dict(color=COLOR_PRIMARY, width=3)
        ))
    elif "gbest_fitness" in df_hist.columns:
        fig.add_trace(go.Scatter(
            x=df_hist[x_col], y=df_hist["gbest_fitness"],
            mode="lines+markers", name="Global Best Fitness",
            line=dict(color=COLOR_PRIMARY, width=3)
        ))

    if "avg_fitness" in df_hist.columns:
        fig.add_trace(go.Scatter(
            x=df_hist[x_col], y=df_hist["avg_fitness"],
            mode="lines", name="Average Fitness",
            line=dict(color=COLOR_PURPLE, width=2, dash="dash")
        ))

    if "features_selected_count" in df_hist.columns:
        fig.add_trace(go.Scatter(
            x=df_hist[x_col], y=df_hist["features_selected_count"],
            mode="lines+markers", name="Selected Features Count",
            yaxis="y2", line=dict(color=COLOR_WARNING, width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title=x_col.capitalize()),
        yaxis=dict(title="Fitness Score"),
        yaxis2=dict(title="Selected Features Count", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    return fig


def plot_pareto_front_2d(pareto_front: List[Dict[str, Any]], knee_solution: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Plots 2D Pareto Front: Performance vs Feature Complexity."""
    df_pf = pd.DataFrame(pareto_front)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_pf["feature_ratio"],
        y=df_pf["raw_metric"],
        mode="markers+lines",
        name="Pareto Optimal Front",
        marker=dict(
            size=11,
            color=df_pf["eval_latency_sec"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Latency (s)")
        ),
        line=dict(color=COLOR_PRIMARY, width=2, dash="dot"),
        text=[f"Features: {row['n_selected']}<br>Score: {row['raw_metric']}<br>Latency: {row['eval_latency_sec']}s" for _, row in df_pf.iterrows()],
        hoverinfo="text"
    ))

    if knee_solution is not None:
        fig.add_trace(go.Scatter(
            x=[knee_solution["feature_ratio"]],
            y=[knee_solution["raw_metric"]],
            mode="markers+text",
            name="★ Best Trade-off (Knee Point)",
            marker=dict(size=18, color=COLOR_DANGER, symbol="star"),
            text=["★ Knee Point"],
            textposition="top center"
        ))

    fig.update_layout(
        title="NSGA-II 2D Pareto Front (Predictive Score vs Feature Subset Ratio)",
        xaxis=dict(title="Selected Feature Ratio (N_selected / N_total)", range=[0, 1.05]),
        yaxis=dict(title="Classification Metric Score (F1 / MCC / AUC)", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    return fig


def plot_pareto_front_3d(pareto_front: List[Dict[str, Any]], knee_solution: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Plots 3D Pareto Front: Performance Error vs Feature Ratio vs Evaluation Latency."""
    df_pf = pd.DataFrame(pareto_front)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df_pf["predictive_error"],
        y=df_pf["feature_ratio"],
        z=df_pf["eval_latency_sec"],
        mode="markers",
        name="Pareto Solutions",
        marker=dict(
            size=7,
            color=df_pf["raw_metric"],
            colorscale="Plasma",
            colorbar=dict(title="Metric Score"),
            opacity=0.85
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
            marker=dict(size=12, color=COLOR_DANGER, symbol="diamond"),
            text=["★ Knee Point"],
            textposition="top center"
        ))

    fig.update_layout(
        title="NSGA-II 3D Pareto Front (Multi-Objective Optimization Surface)",
        scene=dict(
            xaxis=dict(title="f1: Predictive Error (1 - Score)"),
            yaxis=dict(title="f2: Feature Subset Ratio"),
            zaxis=dict(title="f3: Evaluation Latency (s)")
        ),
        height=500,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig


def plot_model_radar(comparison_df: pd.DataFrame) -> go.Figure:
    """Plots multi-metric radar/spider chart comparing all trained models."""
    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig = go.Figure()

    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_WARNING, COLOR_PURPLE, COLOR_CYAN]

    for idx, row in comparison_df.iterrows():
        c_idx = idx % len(colors)
        vals = [row[cat] for cat in categories]
        vals.append(vals[0])  # close loop

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            name=row["Model"],
            line=dict(color=colors[c_idx], width=2),
            fill='toself',
            opacity=0.18
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0])
        ),
        title="Multi-Metric Model Comparison (Radar Spectrum)",
        showlegend=True,
        height=420,
        margin=dict(t=50, b=40, l=40, r=40)
    )
    return fig


def plot_roc_curves(evaluation_results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Plots interactive ROC curves for evaluated classifiers."""
    fig = go.Figure()
    # Baseline diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Random Guess (AUC = 0.50)",
        line=dict(color="#94A3B8", dash="dash")
    ))

    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_WARNING, COLOR_PURPLE, COLOR_CYAN]
    for idx, (m_name, res) in enumerate(evaluation_results.items()):
        c = colors[idx % len(colors)]
        roc_data = res.get("roc_curve", {})
        auc_score = res.get("roc_auc", 0.0)
        fig.add_trace(go.Scatter(
            x=roc_data.get("fpr", []),
            y=roc_data.get("tpr", []),
            mode="lines",
            name=f"{m_name} (AUC = {auc_score:.3f})",
            line=dict(color=c, width=2)
        ))

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curves",
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02]),
        yaxis=dict(title="True Positive Rate (Sensitivity / Recall)", range=[-0.02, 1.02]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    return fig


def plot_confusion_matrix(cm_dict: Dict[str, int], model_name: str = "Classifier") -> go.Figure:
    """Plots interactive 2x2 confusion matrix heatmap."""
    tp = cm_dict.get("tp", 0)
    fp = cm_dict.get("fp", 0)
    fn = cm_dict.get("fn", 0)
    tn = cm_dict.get("tn", 0)

    z = [[tn, fp], [fn, tp]]
    text = [
        [f"True Neg (Clean):<br>{tn}", f"False Pos (Type I):<br>{fp}"],
        [f"False Neg (Type II):<br>{fn}", f"True Pos (Defective):<br>{tp}"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Predicted Clean", "Predicted Defective"],
        y=["Actual Clean", "Actual Defective"],
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False
    ))

    fig.update_layout(
        title=f"Confusion Matrix: {model_name}",
        height=320,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    return fig


def plot_shap_global_bar(rankings: List[Dict[str, Any]], top_k: int = 15) -> go.Figure:
    """Plots global SHAP feature importance horizontal bar chart."""
    df_ranks = pd.DataFrame(rankings[:top_k]).iloc[::-1]  # reverse for top at top

    fig = go.Figure(go.Bar(
        x=df_ranks["importance"],
        y=df_ranks["feature"],
        orientation="h",
        marker=dict(color=COLOR_PRIMARY),
        text=df_ranks["importance"].round(3),
        textposition="outside"
    ))

    fig.update_layout(
        title=f"Global SHAP Feature Importance (Top {len(df_ranks)} Software Metrics)",
        xaxis=dict(title="Mean |SHAP Value| (Impact on Model Defect Output)"),
        yaxis=dict(title="Software Metric"),
        height=max(360, len(df_ranks) * 26),
        margin=dict(t=50, b=40, l=120, r=40)
    )
    return fig


def plot_shap_local_waterfall(local_res: Dict[str, Any], max_display: int = 10) -> go.Figure:
    """Plots interactive Waterfall chart of local SHAP contributions."""
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
        connector={"line": {"color": "#94A3B8"}},
        decreasing={"marker": {"color": COLOR_SECONDARY}},
        increasing={"marker": {"color": COLOR_DANGER}},
        totals={"marker": {"color": COLOR_PRIMARY}}
    ))

    prob = local_res.get("defect_probability", 0.5)
    pred = local_res.get("prediction", "Clean")

    fig.update_layout(
        title=f"Local SHAP Explanation: Module Diagnosed as <b>{pred}</b> (P = {prob:.1%})",
        xaxis=dict(tickangle=-35),
        yaxis=dict(title="Model Output / Logit Shift"),
        height=420,
        margin=dict(t=60, b=80, l=40, r=40)
    )
    return fig
