"""
Explainable Multi-Objective Metaheuristic Optimization for Software Defect Prediction.
Main Streamlit Application Dashboard.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loader import DatasetLoader, BENCHMARK_CATALOG
from data.preprocessor import DataPreprocessor
from metaheuristics.ga import GeneticAlgorithmFeatureSelector
from metaheuristics.pso import ParticleSwarmFeatureSelector
from metaheuristics.nsga2 import NSGA2FeatureSelector
from models.classifier import ModelTrainer
from explainability.shap_explainer import ShapExplainerEngine
from storage.database import ExperimentDatabase
from ui.components import (
    inject_custom_css, render_metric_card, plot_class_distribution,
    plot_correlation_heatmap, plot_ga_pso_convergence, plot_pareto_front_2d,
    plot_pareto_front_3d, plot_model_radar, plot_roc_curves,
    plot_confusion_matrix, plot_shap_global_bar, plot_shap_local_waterfall
)

# Page configuration
st.set_page_config(
    page_title="Explainable MOO Software Defect Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
inject_custom_css()

# Initialize core services in session state
if "loader" not in st.session_state:
    st.session_state.loader = DatasetLoader()
if "db" not in st.session_state:
    st.session_state.db = ExperimentDatabase()
if "trainer" not in st.session_state:
    st.session_state.trainer = ModelTrainer(random_state=42)

# Session state containers for workflow data
if "raw_X" not in st.session_state:
    st.session_state.raw_X = None
if "raw_y" not in st.session_state:
    st.session_state.raw_y = None
if "current_dataset_name" not in st.session_state:
    st.session_state.current_dataset_name = "None"
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "selected_features" not in st.session_state:
    st.session_state.selected_features = None
if "active_feature_subset" not in st.session_state:
    st.session_state.active_feature_subset = "all"
if "optimizer_history" not in st.session_state:
    st.session_state.optimizer_history = None
if "pareto_front" not in st.session_state:
    st.session_state.pareto_front = None
if "knee_solution" not in st.session_state:
    st.session_state.knee_solution = None
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None
if "explainer_engine" not in st.session_state:
    st.session_state.explainer_engine = None


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ DefectPredict XAI")
    st.markdown("*Multi-Objective Metaheuristics & Explainable AI*")
    st.divider()

    navigation = st.radio(
        "Navigation Menu",
        [
            "🌐 Overview & Methodology",
            "📁 Dataset Studio",
            "⚙️ Preprocessing Studio",
            "🧬 Metaheuristic Feature Selection",
            "🤖 Model Benchmarking",
            "🔍 Explainable AI (SHAP)",
            "🎯 Real-Time Defect Predictor",
            "📜 Experiment History"
        ],
        index=1
    )

    st.divider()
    st.markdown("### 📊 Active Pipeline State")
    if st.session_state.raw_X is not None:
        st.success(f"**Dataset**: {st.session_state.current_dataset_name}")
        st.write(f"- Modules: **{len(st.session_state.raw_X)}**")
        st.write(f"- Total Features: **{st.session_state.raw_X.shape[1]}**")
        if st.session_state.selected_features:
            st.info(f"Selected: **{len(st.session_state.selected_features)}** features")
    else:
        st.warning("No dataset loaded yet. Visit **Dataset Studio**.")


# =============================================================================
# TAB 1: OVERVIEW & METHODOLOGY
# =============================================================================
if navigation == "🌐 Overview & Methodology":
    st.markdown('<div class="main-header">Explainable Multi-Objective Metaheuristic Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An Intelligent Software Defect Prediction Architecture combining NSGA-II, Metaheuristics, and SHAP Explainability</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Benchmark Suites", "4 Suites", "PROMISE, AEEEM, ReLink, MORPH")
    with col2:
        render_metric_card("Metaheuristic Solvers", "3 Solvers", "NSGA-II, GA, BPSO")
    with col3:
        render_metric_card("ML Classifiers", "5 Algorithms", "LR, DT, RF, SVM, GBDT")
    with col4:
        render_metric_card("Explainability", "SHAP XAI", "Global Ranking + Local Waterfall")

    st.markdown("---")
    st.markdown("### 🔬 Scientific Methodology & System Architecture")

    st.markdown("""
    Software Defect Prediction (SDP) aims to identify fault-prone software modules before deployment, enabling software engineering teams to allocate testing resources efficiently. 
    However, industrial defect prediction faces three critical scientific challenges:
    1. **High Dimensionality & Multicollinearity**: Software metrics (McCabe complexity, Halstead tokens, CK object-oriented metrics) exhibit intense inter-correlation and redundancy.
    2. **Multi-Objective Trade-offs**: Real-world engineering requires balancing predictive accuracy against model complexity, selected feature subset size, and inference computational budget.
    3. **Black-Box Opacity**: Traditional ML models output predictions without explaining why a module is defective or how software engineers can refactor the code to mitigate defects.
    """)

    st.markdown("#### 📐 Mathematical Formulation of Multi-Objective Feature Selection (NSGA-II)")
    st.latex(r"""
    \min \mathbf{F}(\mathbf{x}) = \Big( f_1(\mathbf{x}),\; f_2(\mathbf{x}),\; f_3(\mathbf{x}) \Big)
    """)
    st.markdown("""
    where $\mathbf{x} = [x_1, x_2, \dots, x_D] \in \{0, 1\}^D$ represents the binary software metric selection mask:
    - **Objective 1 (Predictive Error)**: $f_1(\mathbf{x}) = 1.0 - \text{Metric}(\mathbf{x})$, where $\text{Metric} \in \{\text{F1-score}, \text{MCC}, \text{ROC-AUC}\}$.
    - **Objective 2 (Feature Complexity)**: $f_2(\mathbf{x}) = \frac{\sum_{i=1}^D x_i}{D}$ (Ratio of selected software metrics).
    - **Objective 3 (Computational Latency)**: $f_3(\mathbf{x}) = t_{\text{eval}}(\mathbf{x})$ (Inference execution latency in seconds).
    """)

    st.info("💡 **Knee-Point Trade-Off Selection**: The knee point on the non-dominated Pareto front represents the optimal balance closest to the ideal vector $(0, 0, 0)$, providing maximum defect detection with minimum software metric overhead.")


# =============================================================================
# TAB 2: DATASET STUDIO
# =============================================================================
elif navigation == "📁 Dataset Studio":
    st.markdown('<div class="main-header">Dataset Management & Benchmark Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select from standard empirical benchmarks (PROMISE, AEEEM, ReLink, MORPH) or upload custom software metric CSVs.</div>', unsafe_allow_html=True)

    data_source = st.radio("Choose Dataset Source", ["Built-in Academic Benchmark Suites", "Upload Custom CSV Dataset"], horizontal=True)

    if data_source == "Built-in Academic Benchmark Suites":
        col_suite, col_dataset = st.columns([1, 2])
        with col_suite:
            suites = st.session_state.loader.get_available_suites()
            selected_suite = st.selectbox("Benchmark Suite", suites, index=0)

        with col_dataset:
            datasets_in_suite = st.session_state.loader.get_datasets_in_suite(selected_suite)
            dataset_options = list(datasets_in_suite.keys())
            selected_dataset_key = st.selectbox(
                "Select Dataset",
                dataset_options,
                format_func=lambda k: f"{k} - {datasets_in_suite[k]['name']}"
            )

        meta = datasets_in_suite[selected_dataset_key]
        st.markdown(f"**Domain**: {meta['domain']} | **Metrics Category**: {meta['metrics_type']}")

        if st.button("📥 Load Benchmark Dataset", type="primary"):
            with st.spinner("Loading benchmark dataset..."):
                X, y, target_col = st.session_state.loader.load_benchmark(selected_suite, selected_dataset_key)
                st.session_state.raw_X = X
                st.session_state.raw_y = y
                st.session_state.current_dataset_name = f"{selected_suite}_{selected_dataset_key}"
                st.session_state.selected_features = None  # reset feature selection
                st.session_state.benchmark_results = None
                st.success(f"Successfully loaded {selected_suite} / {selected_dataset_key} ({len(X)} modules, {X.shape[1]} metrics)!")

    else:
        uploaded_file = st.file_uploader("Upload Software Metrics CSV", type=["csv"])
        if uploaded_file is not None:
            if st.button("📥 Parse and Load Uploaded CSV", type="primary"):
                with st.spinner("Parsing uploaded dataset..."):
                    X, y, target_col = st.session_state.loader.load_custom_csv(uploaded_file)
                    st.session_state.raw_X = X
                    st.session_state.raw_y = y
                    st.session_state.current_dataset_name = uploaded_file.name.replace(".csv", "")
                    st.session_state.selected_features = None
                    st.session_state.benchmark_results = None
                    st.success(f"Successfully parsed {uploaded_file.name} ({len(X)} modules, {X.shape[1]} metrics, target '{target_col}')!")

    # Display dataset exploration if loaded
    if st.session_state.raw_X is not None:
        st.divider()
        st.markdown(f"### 📋 Exploratory Data Analysis: **{st.session_state.current_dataset_name}**")

        summary = st.session_state.loader.get_dataset_summary(
            st.session_state.raw_X, st.session_state.raw_y, st.session_state.current_dataset_name
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            render_metric_card("Total Modules", f"{summary['total_modules']:,}", "Software units")
        with m2:
            render_metric_card("Software Metrics", f"{summary['total_features']}", "Feature columns")
        with m3:
            render_metric_card("Defective Modules", f"{summary['defective_count']:,}", f"{summary['defective_percentage']}% of total")
        with m4:
            render_metric_card("Imbalance Ratio", f"{summary['imbalance_ratio']}:1", "Clean-to-Buggy")
        with m5:
            render_metric_card("Missing Values", f"{summary['missing_values_count']:,}", f"{summary['missing_values_percentage']}% cells")

        col_df, col_dist = st.columns([3, 2])
        with col_df:
            st.markdown("#### Feature Metrics Preview")
            st.dataframe(st.session_state.raw_X.head(8), use_container_width=True)

        with col_dist:
            fig_dist = plot_class_distribution(st.session_state.raw_y)
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("#### Metric Multicollinearity & Correlation Analysis")
        fig_corr = plot_correlation_heatmap(st.session_state.raw_X, max_cols=25)
        st.plotly_chart(fig_corr, use_container_width=True)


# =============================================================================
# TAB 3: PREPROCESSING STUDIO
# =============================================================================
elif navigation == "⚙️ Preprocessing Studio":
    st.markdown('<div class="main-header">Leakage-Free Preprocessing Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Configure missing value imputation, feature normalization, and class imbalance mitigation.</div>', unsafe_allow_html=True)

    if st.session_state.raw_X is None:
        st.warning("⚠️ No dataset loaded. Please navigate to **Dataset Studio** first to select or upload a dataset.")
    else:
        st.info(f"Target Dataset: **{st.session_state.current_dataset_name}** ({len(st.session_state.raw_X)} samples, {st.session_state.raw_X.shape[1]} features)")

        with st.expander("🛠️ Preprocessing Pipeline Configuration", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                impute_strat = st.selectbox(
                    "Missing Value Imputation",
                    ["median", "mean", "most_frequent", "none"],
                    index=0,
                    help="Median is recommended for skewed software engineering metrics."
                )
                remove_dups = st.checkbox("Remove Duplicate Modules", value=True)

            with col2:
                scaler_method = st.selectbox(
                    "Feature Scaling / Normalization",
                    ["standard", "minmax", "robust", "none"],
                    index=0,
                    help="StandardScaler normalizes features to zero mean and unit variance."
                )
                test_ratio = st.slider("Test Partition Ratio", min_value=0.10, max_value=0.40, value=0.20, step=0.05)

            with col3:
                imbalance_tech = st.selectbox(
                    "Class Imbalance Mitigation",
                    ["smote", "adasyn", "random_over", "random_under", "none"],
                    index=0,
                    help="SMOTE synthesizes minority defective modules strictly on training folds to prevent leakage."
                )
                rand_seed = st.number_input("Random Seed", value=42, step=1)

        if st.button("⚡ Execute Leakage-Free Preprocessing Pipeline", type="primary"):
            with st.spinner("Processing data, applying stratified split and resampling..."):
                preprocessor = DataPreprocessor(
                    impute_strategy=impute_strat,
                    remove_duplicates=remove_dups,
                    scaling_method=scaler_method,
                    imbalance_method=imbalance_tech,
                    test_size=test_ratio,
                    random_state=rand_seed
                )

                X_tr, X_te, y_tr, y_te, meta = preprocessor.fit_transform_split(
                    st.session_state.raw_X, st.session_state.raw_y
                )

                st.session_state.preprocessor = preprocessor
                st.session_state.X_train = X_tr
                st.session_state.X_test = X_te
                st.session_state.y_train = y_tr
                st.session_state.y_test = y_te
                st.session_state.benchmark_results = None  # reset models if preprocessed again

                st.success("✅ Preprocessing successfully executed with strict Zero-Data-Leakage guarantee!")

        if st.session_state.X_train is not None:
            st.divider()
            st.markdown("### 🔍 Preprocessing Transformation Verification")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Train Set (Resampled)", f"{len(st.session_state.X_train):,}", "Fit partition")
            with c2:
                render_metric_card("Test Set (Untouched)", f"{len(st.session_state.X_test):,}", "Strict empirical holdout")
            with c3:
                clean_tr = (st.session_state.y_train == 0).sum()
                bug_tr = (st.session_state.y_train == 1).sum()
                render_metric_card("Train Class Balance", f"{clean_tr} : {bug_tr}", "Clean vs Defective")
            with c4:
                clean_te = (st.session_state.y_test == 0).sum()
                bug_te = (st.session_state.y_test == 1).sum()
                render_metric_card("Test Class Balance", f"{clean_te} : {bug_te}", "True empirical ratio")

            st.markdown("""
            > [!NOTE]
            > **Zero-Leakage Assurance**: The test partition was isolated prior to fitting the scaler, imputer, and SMOTE resampler. The test set represents the true empirical class imbalance of production software.
            """)


# =============================================================================
# TAB 4: METAHEURISTIC FEATURE SELECTION
# =============================================================================
elif navigation == "🧬 Metaheuristic Feature Selection":
    st.markdown('<div class="main-header">Metaheuristic Feature Selection Laboratory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Optimize software metric subsets using NSGA-II, Genetic Algorithms, or Binary Particle Swarm Optimization.</div>', unsafe_allow_html=True)

    if st.session_state.X_train is None:
        st.warning("⚠️ Data has not been preprocessed. Please visit **Preprocessing Studio** and execute the pipeline.")
    else:
        st.markdown(f"**Dataset**: `{st.session_state.current_dataset_name}` | Available Metrics: **{st.session_state.X_train.shape[1]}**")

        opt_choice = st.selectbox(
            "Select Metaheuristic Optimization Algorithm",
            [
                "NSGA-II (Multi-Objective: Error + Features + Latency)",
                "Genetic Algorithm (Single-Objective GA)",
                "Particle Swarm Optimization (Binary PSO)"
            ],
            index=0
        )

        with st.expander("⚙️ Optimizer Hyperparameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                pop_size = st.slider("Population Size / Swarm Count", min_value=10, max_value=80, value=30, step=5)
                n_iters = st.slider("Generations / Iterations", min_value=5, max_value=50, value=15, step=5)
            with col2:
                eval_metric = st.selectbox("Objective Metric", ["f1", "mcc", "auc", "accuracy"], index=0)
                cv_folds = st.slider("Internal CV Folds", min_value=2, max_value=5, value=3)
            with col3:
                if "NSGA" in opt_choice:
                    cx_rate = st.slider("Crossover Probability", 0.5, 0.95, 0.85, 0.05)
                    mut_rate = st.slider("Mutation Probability", 0.01, 0.20, 0.05, 0.01)
                elif "Genetic" in opt_choice:
                    alpha_tradeoff = st.slider("Alpha (Performance vs Reduction Weight)", 0.5, 0.95, 0.85, 0.05)
                    cx_rate = st.slider("Crossover Probability", 0.5, 0.95, 0.80, 0.05)
                else:
                    alpha_tradeoff = st.slider("Alpha (Performance vs Reduction Weight)", 0.5, 0.95, 0.85, 0.05)
                    w_max = st.slider("Max Inertia Weight (w_max)", 0.5, 1.0, 0.9, 0.05)

        run_btn = st.button("🚀 Run Metaheuristic Optimization", type="primary")

        if run_btn:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            if "NSGA" in opt_choice:
                def nsga2_callback(gen, total_gen, best_m, avg_f, hv):
                    progress_bar.progress(gen / total_gen)
                    status_text.markdown(f"**Generation {gen}/{total_gen}** | Best Metric ({eval_metric.upper()}): `{best_m:.4f}` | Avg Features: `{avg_f}` | Hypervolume: `{hv:.4f}`")

                optimizer = NSGA2FeatureSelector(
                    population_size=pop_size,
                    n_generations=n_iters,
                    crossover_rate=cx_rate,
                    mutation_rate=mut_rate,
                    metric=eval_metric,
                    cv_folds=cv_folds,
                    random_state=42
                )
                with st.spinner("Executing NSGA-II Multi-Objective Non-Dominated Search..."):
                    optimizer.fit(st.session_state.X_train, st.session_state.y_train, progress_callback=nsga2_callback)

                st.session_state.optimizer_history = optimizer.history
                st.session_state.pareto_front = optimizer.pareto_front
                st.session_state.knee_solution = optimizer.best_tradeoff_solution
                st.session_state.selected_features = optimizer.best_selected_features
                st.session_state.active_feature_subset = "selected"

                progress_bar.progress(1.0)
                status_text.success("🎉 NSGA-II Multi-Objective Optimization Complete!")

            elif "Genetic" in opt_choice:
                def ga_callback(gen, total_gen, best_fit, best_m, n_sel):
                    progress_bar.progress(gen / total_gen)
                    status_text.markdown(f"**Generation {gen}/{total_gen}** | Best Fitness: `{best_fit:.4f}` | Metric ({eval_metric.upper()}): `{best_m:.4f}` | Selected: `{n_sel}` features")

                optimizer = GeneticAlgorithmFeatureSelector(
                    population_size=pop_size,
                    n_generations=n_iters,
                    crossover_rate=cx_rate,
                    alpha=alpha_tradeoff,
                    metric=eval_metric,
                    cv_folds=cv_folds,
                    random_state=42
                )
                with st.spinner("Executing Genetic Algorithm Search..."):
                    optimizer.fit(st.session_state.X_train, st.session_state.y_train, progress_callback=ga_callback)

                st.session_state.optimizer_history = optimizer.history
                st.session_state.pareto_front = None
                st.session_state.selected_features = optimizer.best_selected_features
                st.session_state.active_feature_subset = "selected"

                progress_bar.progress(1.0)
                status_text.success("🎉 Genetic Algorithm Optimization Complete!")

            else:
                def pso_callback(it, total_it, best_fit, best_m, n_sel):
                    progress_bar.progress(it / total_it)
                    status_text.markdown(f"**Iteration {it}/{total_it}** | Swarm Best Fitness: `{best_fit:.4f}` | Metric ({eval_metric.upper()}): `{best_m:.4f}` | Selected: `{n_sel}` features")

                optimizer = ParticleSwarmFeatureSelector(
                    n_particles=pop_size,
                    n_iterations=n_iters,
                    w_max=w_max,
                    alpha=alpha_tradeoff,
                    metric=eval_metric,
                    cv_folds=cv_folds,
                    random_state=42
                )
                with st.spinner("Executing Binary PSO Swarm Search..."):
                    optimizer.fit(st.session_state.X_train, st.session_state.y_train, progress_callback=pso_callback)

                st.session_state.optimizer_history = optimizer.history
                st.session_state.pareto_front = None
                st.session_state.selected_features = optimizer.best_selected_features
                st.session_state.active_feature_subset = "selected"

                progress_bar.progress(1.0)
                status_text.success("🎉 Binary PSO Optimization Complete!")

        # Display results if available
        if st.session_state.selected_features is not None:
            st.divider()
            st.markdown("### 🏆 Optimization Results & Selected Feature Subset")

            n_total = st.session_state.X_train.shape[1]
            n_sel = len(st.session_state.selected_features)
            reduction_pct = (1.0 - n_sel / n_total) * 100

            c1, c2, c3 = st.columns(3)
            with c1:
                render_metric_card("Original Metrics", f"{n_total}", "All input features")
            with c2:
                render_metric_card("Optimized Subset", f"{n_sel}", "Selected software metrics")
            with c3:
                render_metric_card("Dimensionality Reduction", f"{reduction_pct:.1f}%", "Complexity reduced")

            st.markdown(f"**Selected Software Metrics ({n_sel})**:")
            st.markdown(" ".join([f"<span class='badge-info'>{f}</span>" for f in st.session_state.selected_features]), unsafe_allow_html=True)

            # NSGA-II Pareto Front Visualizations
            if st.session_state.pareto_front is not None:
                st.markdown("#### 🌟 NSGA-II Multi-Objective Pareto Optimal Surfaces")
                tab2d, tab3d, tab_table = st.tabs(["2D Pareto Front", "3D Multi-Objective Surface", "Pareto Solutions Table"])

                with tab2d:
                    fig_2d = plot_pareto_front_2d(st.session_state.pareto_front, st.session_state.knee_solution)
                    st.plotly_chart(fig_2d, use_container_width=True)

                with tab3d:
                    fig_3d = plot_pareto_front_3d(st.session_state.pareto_front, st.session_state.knee_solution)
                    st.plotly_chart(fig_3d, use_container_width=True)

                with tab_table:
                    df_pf = pd.DataFrame(st.session_state.pareto_front)[[
                        "solution_index", "n_selected", "feature_ratio", "raw_metric", "predictive_error", "eval_latency_sec", "ideal_distance"
                    ]]
                    st.dataframe(df_pf, use_container_width=True)

            # Convergence plot
            if st.session_state.optimizer_history is not None:
                st.markdown("#### 📈 Iteration Convergence History")
                fig_conv = plot_ga_pso_convergence(st.session_state.optimizer_history)
                st.plotly_chart(fig_conv, use_container_width=True)


# =============================================================================
# TAB 5: MODEL BENCHMARKING
# =============================================================================
elif navigation == "🤖 Model Benchmarking":
    st.markdown('<div class="main-header">Machine Learning Model Benchmarking Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Train, benchmark, and compare 5 classifiers across complete scientific evaluation metrics.</div>', unsafe_allow_html=True)

    if st.session_state.X_train is None:
        st.warning("⚠️ Data has not been preprocessed. Please visit **Preprocessing Studio** first.")
    else:
        # Choose feature subset
        has_metaheuristics = st.session_state.selected_features is not None
        feat_choice = st.radio(
            "Feature Space for Model Training",
            ["Full Feature Set (All Metrics)", "Metaheuristic Selected Subset"] if has_metaheuristics else ["Full Feature Set (All Metrics)"],
            horizontal=True
        )

        if "Metaheuristic" in feat_choice and has_metaheuristics:
            active_feats = st.session_state.selected_features
            st.session_state.active_feature_subset = "selected"
            X_tr = st.session_state.X_train[active_feats]
            X_te = st.session_state.X_test[active_feats]
            st.info(f"Training models using **{len(active_feats)}** metaheuristic-optimized software metrics.")
        else:
            active_feats = list(st.session_state.X_train.columns)
            st.session_state.active_feature_subset = "all"
            X_tr = st.session_state.X_train
            X_te = st.session_state.X_test
            st.info(f"Training models using **all {len(active_feats)}** software metrics.")

        if st.button("🚀 Train & Benchmark All 5 Classifiers", type="primary"):
            with st.spinner("Training Logistic Regression, Decision Tree, Random Forest, SVM, and Gradient Boosting..."):
                trainer = st.session_state.trainer
                comp_df = trainer.train_all_models(X_tr, st.session_state.y_train, X_te, st.session_state.y_test)
                st.session_state.benchmark_results = comp_df

                # Automatically log runs to SQLite database
                for model_name in trainer.get_available_model_names():
                    res = trainer.evaluation_results[model_name]
                    st.session_state.db.log_experiment(
                        dataset_name=st.session_state.current_dataset_name,
                        optimizer="Metaheuristic" if "Metaheuristic" in feat_choice else "None",
                        model_name=model_name,
                        total_features=st.session_state.X_train.shape[1],
                        selected_features_list=active_feats,
                        metrics=res,
                        notes=f"Feature mode: {st.session_state.active_feature_subset}"
                    )

                st.success("✅ All 5 Classifiers evaluated and logged to SQLite audit database!")

        if st.session_state.benchmark_results is not None:
            st.divider()
            st.markdown("### 📊 Comprehensive Model Benchmarking Table")

            st.dataframe(
                st.session_state.benchmark_results.style.highlight_max(
                    subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC"],
                    color="#DCFCE7"
                ),
                use_container_width=True
            )

            col_radar, col_roc = st.columns(2)
            with col_radar:
                fig_radar = plot_model_radar(st.session_state.benchmark_results)
                st.plotly_chart(fig_radar, use_container_width=True)

            with col_roc:
                fig_roc = plot_roc_curves(st.session_state.trainer.evaluation_results)
                st.plotly_chart(fig_roc, use_container_width=True)

            st.markdown("#### 🧩 Confusion Matrices (True/False Positives & Negatives)")
            cm_cols = st.columns(len(st.session_state.trainer.get_available_model_names()))
            for idx, model_name in enumerate(st.session_state.trainer.get_available_model_names()):
                with cm_cols[idx]:
                    cm_dict = st.session_state.trainer.evaluation_results[model_name]["confusion_matrix"]
                    fig_cm = plot_confusion_matrix(cm_dict, model_name=model_name)
                    st.plotly_chart(fig_cm, use_container_width=True)


# =============================================================================
# TAB 6: EXPLAINABLE AI (SHAP)
# =============================================================================
elif navigation == "🔍 Explainable AI (SHAP)":
    st.markdown('<div class="main-header">Explainable AI (SHAP) Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Demystify black-box defect predictions with global feature importance, local waterfall explanations, and actionable developer recommendations.</div>', unsafe_allow_html=True)

    if not st.session_state.trainer.trained_models:
        st.warning("⚠️ No models trained yet. Please navigate to **Model Benchmarking** and train models first.")
    else:
        avail_models = list(st.session_state.trainer.trained_models.keys())
        selected_model_name = st.selectbox("Select Classifier to Explain", avail_models, index=avail_models.index("Random Forest") if "Random Forest" in avail_models else 0)

        clf = st.session_state.trainer.trained_models[selected_model_name]

        # Determine features used
        if st.session_state.active_feature_subset == "selected" and st.session_state.selected_features:
            active_feats = st.session_state.selected_features
        else:
            active_feats = list(st.session_state.X_train.columns)

        X_tr = st.session_state.X_train[active_feats]
        X_te = st.session_state.X_test[active_feats]

        with st.spinner(f"Initializing SHAP Explainer Engine for {selected_model_name}..."):
            explainer_engine = ShapExplainerEngine(clf, X_tr, model_name=selected_model_name)
            st.session_state.explainer_engine = explainer_engine

        tab_global, tab_local = st.tabs(["🌍 Global Feature Importance", "🔬 Local Module Diagnosis & Recommendations"])

        with tab_global:
            st.markdown("#### Global Metric Impact Distribution")
            with st.spinner("Computing global SHAP values..."):
                global_res = explainer_engine.compute_global_importance(X_te, max_eval_samples=100)

            fig_bar = plot_shap_global_bar(global_res["feature_importance_ranking"], top_k=15)
            st.plotly_chart(fig_bar, use_container_width=True)

            with st.expander("Detailed Metric Importance Rankings Table"):
                df_ranks = pd.DataFrame(global_res["feature_importance_ranking"])
                st.dataframe(df_ranks, use_container_width=True)

        with tab_local:
            st.markdown("#### Individual Software Module Diagnosis")
            test_idx = st.slider("Select Test Instance Index", 0, len(X_te) - 1, 0)
            sample_instance = X_te.iloc[[test_idx]]
            actual_label = int(st.session_state.y_test.iloc[test_idx])

            with st.spinner("Calculating SHAP additive local attributions..."):
                local_res = explainer_engine.explain_local_instance(sample_instance)

            c1, c2, c3 = st.columns(3)
            with c1:
                render_metric_card("Actual Ground Truth", "Defective" if actual_label == 1 else "Clean", f"Test row #{test_idx}")
            with c2:
                render_metric_card("Model Prediction", local_res["prediction"], f"Probability: {local_res['defect_probability']:.1%}")
            with c3:
                status_color = "danger" if local_res["prediction"] == "Defective" else "clean"
                st.markdown(f'<div class="metric-card"><div class="metric-title">Verdict</div><div class="metric-value"><span class="badge-{status_color}">{local_res["prediction"]}</span></div></div>', unsafe_allow_html=True)

            st.markdown("#### SHAP Local Attribution Waterfall")
            fig_waterfall = plot_shap_local_waterfall(local_res, max_display=10)
            st.plotly_chart(fig_waterfall, use_container_width=True)

            st.markdown("#### 💡 Actionable Software Engineering Refactoring Guidance")
            recs = local_res.get("actionable_recommendations", [])
            for r in recs:
                st.markdown(f"""
                    <div class="recommendation-box">
                        <b>{r['title']}</b> (Metric <code>{r['feature']}</code> = {r['value']})<br>
                        <span style="color:#15803D;">{r['recommendation']}</span>
                    </div>
                """, unsafe_allow_html=True)


# =============================================================================
# TAB 7: REAL-TIME DEFECT PREDICTOR
# =============================================================================
elif navigation == "🎯 Real-Time Defect Predictor":
    st.markdown('<div class="main-header">Real-Time Defect Prediction Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive module prediction form with live SHAP attribution & batch CSV bulk classification.</div>', unsafe_allow_html=True)

    if not st.session_state.trainer.trained_models:
        st.warning("⚠️ No trained models available. Please train models in **Model Benchmarking** first.")
    else:
        pred_mode = st.radio("Prediction Mode", ["Single Software Module Input", "Batch CSV Bulk Predictor"], horizontal=True)

        selected_model = st.selectbox(
            "Predictive Model",
            list(st.session_state.trainer.trained_models.keys()),
            index=list(st.session_state.trainer.trained_models.keys()).index("Random Forest") if "Random Forest" in st.session_state.trainer.trained_models else 0
        )
        clf = st.session_state.trainer.trained_models[selected_model]

        if st.session_state.active_feature_subset == "selected" and st.session_state.selected_features:
            active_feats = st.session_state.selected_features
        else:
            active_feats = list(st.session_state.X_train.columns)

        if pred_mode == "Single Software Module Input":
            st.markdown("### 📝 Enter Software Module Metrics")

            # Archetype presets
            preset = st.selectbox(
                "Quick Load Metric Archetype",
                ["Custom User Values", "Clean / Low-Risk Module", "High-Complexity Legacy Module", "High-Coupling Controller"]
            )

            # Generate default values based on archetype
            defaults = {}
            for col in active_feats:
                mean_val = float(st.session_state.raw_X[col].mean()) if col in st.session_state.raw_X else 10.0
                std_val = float(st.session_state.raw_X[col].std()) if col in st.session_state.raw_X else 5.0
                if preset == "Clean / Low-Risk Module":
                    defaults[col] = max(0.0, round(mean_val - 0.5 * std_val, 2))
                elif preset == "High-Complexity Legacy Module":
                    defaults[col] = round(mean_val + 2.0 * std_val, 2)
                elif preset == "High-Coupling Controller":
                    defaults[col] = round(mean_val + 1.2 * std_val, 2)
                else:
                    defaults[col] = round(mean_val, 2)

            with st.form("single_predict_form"):
                cols_per_row = 4
                form_cols = st.columns(cols_per_row)
                input_vals = {}

                for idx, feat in enumerate(active_feats):
                    c = form_cols[idx % cols_per_row]
                    input_vals[feat] = c.number_input(
                        f"{feat}",
                        value=float(defaults[feat]),
                        key=f"inp_{feat}"
                    )

                submit_btn = st.form_submit_button("⚡ Predict Defect Risk", type="primary")

            if submit_btn:
                # Build DataFrame and transform with fitted preprocessor
                raw_input_df = pd.DataFrame([input_vals])
                proc_input_df = st.session_state.preprocessor.transform_new_data(raw_input_df)[active_feats]

                prob = float(clf.predict_proba(proc_input_df)[0, 1]) if hasattr(clf, "predict_proba") else float(clf.predict(proc_input_df)[0])
                is_buggy = prob >= 0.5

                st.divider()
                st.markdown("### 🎯 Prediction Results")

                col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
                with col_res1:
                    verdict_text = "DEFECTIVE" if is_buggy else "NON-DEFECTIVE"
                    badge_class = "badge-defective" if is_buggy else "badge-clean"
                    st.markdown(f'<div class="metric-card"><div class="metric-title">Verdict</div><div class="metric-value"><span class="{badge_class}">{verdict_text}</span></div></div>', unsafe_allow_html=True)

                with col_res2:
                    risk_tier = "Critical" if prob > 0.75 else ("Elevated" if prob >= 0.5 else ("Moderate" if prob > 0.25 else "Low"))
                    render_metric_card("Defect Probability", f"{prob:.1%}", f"Risk Tier: {risk_tier}")

                with col_res3:
                    # Gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Defect Risk Score (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#EF4444" if is_buggy else "#10B981"},
                            'steps': [
                                {'range': [0, 50], 'color': "#ECFDF5"},
                                {'range': [50, 100], 'color': "#FEF2F2"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 3},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=200, margin=dict(t=30, b=10, l=20, r=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # Local SHAP explanation
                st.markdown("#### 🔬 Root Cause Attribution (Local SHAP)")
                explainer = ShapExplainerEngine(clf, st.session_state.X_train[active_feats], model_name=selected_model)
                local_diag = explainer.explain_local_instance(proc_input_df)

                fig_wf = plot_shap_local_waterfall(local_diag)
                st.plotly_chart(fig_wf, use_container_width=True)

                st.markdown("#### 🛠️ Refactoring Guidance for this Module")
                for rec in local_diag["actionable_recommendations"]:
                    st.markdown(f"""
                        <div class="recommendation-box">
                            <b>{rec['title']}</b> (Metric: <code>{rec['feature']}</code> = {rec['value']})<br>
                            <span>{rec['recommendation']}</span>
                        </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("### 📦 Batch CSV Software Defect Scoring")
            batch_file = st.file_uploader("Upload Software Modules CSV for Batch Inference", type=["csv"], key="batch_upload")

            if batch_file is not None:
                df_batch = pd.read_csv(batch_file)
                st.write(f"Uploaded **{len(df_batch)}** modules with columns: `{list(df_batch.columns[:6])}...`")

                if st.button("🚀 Run Batch Defect Prediction", type="primary"):
                    with st.spinner("Processing batch modules..."):
                        # Transform through fitted preprocessor
                        proc_batch = st.session_state.preprocessor.transform_new_data(df_batch)[active_feats]

                        if hasattr(clf, "predict_proba"):
                            probs = clf.predict_proba(proc_batch)[:, 1]
                        else:
                            probs = clf.predict(proc_batch).astype(float)

                        preds = (probs >= 0.5).astype(int)

                        # Create annotated DataFrame
                        annotated_df = df_batch.copy()
                        annotated_df["predicted_defect"] = preds
                        annotated_df["defect_label"] = np.where(preds == 1, "Defective", "Clean")
                        annotated_df["defect_probability"] = np.round(probs, 4)
                        annotated_df["risk_tier"] = np.where(
                            probs > 0.75, "Critical",
                            np.where(probs >= 0.5, "Elevated",
                            np.where(probs > 0.25, "Moderate", "Low"))
                        )

                        st.divider()
                        st.markdown("### 📊 Batch Prediction Summary")

                        b1, b2, b3 = st.columns(3)
                        with b1:
                            render_metric_card("Total Scored Modules", f"{len(annotated_df):,}", "Batch size")
                        with b2:
                            n_bugs = int((preds == 1).sum())
                            render_metric_card("Predicted Defective", f"{n_bugs:,}", f"{(n_bugs / len(preds) * 100):.1f}% fault rate")
                        with b3:
                            render_metric_card("Predicted Clean", f"{(len(preds) - n_bugs):,}", "Verified safe")

                        st.dataframe(annotated_df.head(15), use_container_width=True)

                        csv_bytes = annotated_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Annotated Prediction CSV",
                            data=csv_bytes,
                            file_name="software_defect_predictions.csv",
                            mime="text/csv",
                            type="primary"
                        )


# =============================================================================
# TAB 8: EXPERIMENT HISTORY
# =============================================================================
elif navigation == "📜 Experiment History":
    st.markdown('<div class="main-header">SQLite Experiment Tracking & Audit Trail</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Persistent audit log of all model evaluations, metaheuristic runs, and scientific metrics.</div>', unsafe_allow_html=True)

    db = st.session_state.db
    history_df = db.get_all_experiments()

    if history_df.empty:
        st.info("No experiments logged yet. Train models in **Model Benchmarking** to populate the audit database.")
    else:
        st.markdown(f"Total Logged Runs: **{len(history_df)}**")

        # Filters
        c_filter1, c_filter2 = st.columns(2)
        with c_filter1:
            dataset_filter = st.multiselect("Filter by Dataset", options=list(history_df["dataset_name"].unique()))
        with c_filter2:
            model_filter = st.multiselect("Filter by Classifier", options=list(history_df["model_name"].unique()))

        filtered_df = history_df.copy()
        if dataset_filter:
            filtered_df = filtered_df[filtered_df["dataset_name"].isin(dataset_filter)]
        if model_filter:
            filtered_df = filtered_df[filtered_df["model_name"].isin(model_filter)]

        st.dataframe(filtered_df, use_container_width=True)

        # Comparative performance chart across historical runs
        if len(filtered_df) > 1:
            st.markdown("#### 📈 Historical F1-Score & MCC Comparison")
            fig_hist = px.bar(
                filtered_df,
                x="timestamp",
                y=["f1_score", "mcc", "roc_auc"],
                barmode="group",
                color_discrete_sequence=[COLOR_PRIMARY, COLOR_PURPLE, COLOR_SECONDARY],
                title="Historical Performance Metric Benchmarks",
                hover_data=["dataset_name", "model_name", "optimizer", "selected_features_count"]
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

        col_exp1, col_exp2 = st.columns([1, 1])
        with col_exp1:
            csv_exp = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Audit History to CSV",
                data=csv_exp,
                file_name="experiment_audit_history.csv",
                mime="text/csv"
            )

        with col_exp2:
            if st.button("🗑️ Clear Entire Experiment Database", type="secondary"):
                db.clear_all()
                st.success("Database cleared!")
                st.rerun()
