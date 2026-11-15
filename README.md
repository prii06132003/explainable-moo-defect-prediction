# Explainable Multi-Objective Metaheuristic Optimization for Software Defect Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-brightgreen.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, production-grade Software Defect Prediction (SDP) and Explainable AI (XAI) system featuring hybrid metaheuristic optimization, multi-objective feature selection (NSGA-II, GA, BPSO), leakage-free preprocessing, multi-metric classifier benchmarking, and automated developer refactoring recommendations.

---

## 📌 Project Overview

Software Defect Prediction (SDP) allows software quality assurance teams to proactively allocate code review and unit testing resources to fault-prone software modules before shipping code to production. 

This project addresses the core challenges in traditional defect prediction:
1. **High Dimensionality & Multicollinearity**: Software metrics (McCabe cyclomatic complexity, Halstead tokens, CK object-oriented metrics) exhibit intense inter-feature correlations and redundancy.
2. **Multi-Objective Trade-offs**: In real-world software engineering, teams must minimize classification error while simultaneously reducing the number of required code metrics and lowering computational inference budget.
3. **Black-Box Opacity**: Standard predictive models lack explainability. Our system leverages **SHAP (SHapley Additive exPlanations)** to explain both global feature importance and local module predictions, translating metric attributions into concrete **Software Engineering Refactoring Guidance**.

---

## 🏗️ System Architecture

```
                                      +------------------------+
                                      | Software Defect Data   |
                                      | PROMISE / AEEEM /      |
                                      | ReLink / MORPH / CSV   |
                                      +-----------+------------+
                                                  |
                                                  v
                                      +------------------------+
                                      | Leakage-Free Pipeline  |
                                      | Imputation + Scaler +  |
                                      | SMOTE (Train-Only)     |
                                      +-----------+------------+
                                                  |
                                                  v
                         +------------------------------------------------+
                         |      Metaheuristic Feature Selection Lab       |
                         |  +--------------+  +------------+  +--------+  |
                         |  |   NSGA-II    |  | Binary PSO |  |   GA   |  |
                         |  | (Multi-Obj)  |  |  (Swarm)   |  | (Evol) |  |
                         |  +--------------+  +------------+  +--------+  |
                         +------------------------+-----------------------+
                                                  | Optimal Feature Subset
                                                  v
                         +------------------------------------------------+
                         |       5-Classifier Machine Learning Suite      |
                         |  LR | Decision Tree | RF | SVM | Grad. Boosting|
                         +------------------------+-----------------------+
                                                  |
                     +----------------------------+----------------------------+
                     |                                                         |
                     v                                                         v
        +-------------------------+                               +-------------------------+
        |   Explainable AI Studio |                               | Real-Time Prediction    |
        | - Global SHAP Rankings  |                               | - Single Module Scoring |
        | - Local Waterfall Plots |                               | - Batch CSV Processing  |
        | - Refactoring Advice    |                               | - SQLite Audit Database |
        +-------------------------+                               +-------------------------+
```

---

## 🔬 Multi-Objective Formulation (NSGA-II)

The system minimizes three simultaneous conflicting objectives:
$$\min \mathbf{F}(\mathbf{x}) = \Big( f_1(\mathbf{x}),\; f_2(\mathbf{x}),\; f_3(\mathbf{x}) \Big)$$

- **$f_1(\mathbf{x}) = 1.0 - \text{Metric}(\mathbf{x})$**: Predictive classification error (where $\text{Metric} \in \{\text{F1-Score}, \text{MCC}, \text{ROC-AUC}\}$).
- **$f_2(\mathbf{x}) = \frac{\sum_{i=1}^D x_i}{D}$**: Ratio of selected software metrics over total metric space.
- **$f_3(\mathbf{x}) = t_{\text{latency}}(\mathbf{x})$**: Computational inference and evaluation execution latency in seconds.

The **Knee-Point Solution** along the Pareto front is automatically identified using minimum Euclidean distance to the ideal objective vector $(0, 0, 0)$.

---

## 📁 Benchmark Datasets Supported

The repository includes curated, empirical benchmark datasets from standard software engineering repositories:

1. **PROMISE Suite (NASA Metrics Data Program)**:
   - `PC1`, `PC3`, `PC4` (NASA spacecraft and flight instrument software)
   - `KC1`, `KC2` (NASA telemetry reception and storage management)
   - `CM1` (NASA spacecraft scientific instruments)
   - `MW1` (Zero-gravity microgravity combustion experiments)
   - `MC2` (Video guidance system)
2. **AEEEM Suite (D'Ambros et al.)**:
   - `EQ` (Eclipse Equinox OSGi framework)
   - `JDT` (Eclipse Java Development Tools)
   - `LC` (Apache Lucene information retrieval library)
   - `ML` (Eclipse Mylyn task-focused interface)
   - `PDE` (Eclipse Plugin Development Environment)
3. **ReLink Suite (Defect-Corrected Metrics)**:
   - `apache` (Apache HTTP Server)
   - `safe` (J2ME mobile security and password manager)
   - `zxing` (1D/2D barcode image processing library)
4. **MORPH Suite (Turhan et al. / SOFTLAB)**:
   - `ar1`, `ar3`, `ar4`, `ar5`, `ar6` (Embedded software and telecommunications controllers)
5. **Custom CSV Support**: Upload any software metric CSV with automatic target defect column detection.

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/explainable-moo-defect-prediction.git
cd explainable-moo-defect-prediction
```

### 2. Create and Activate Virtual Environment
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
```bash
python -m unittest tests/test_all_modules.py
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Streamlit Dashboard Modules

1. **🌐 Overview & Methodology**: Architecture diagram, mathematical optimization formulations, and academic problem statement.
2. **📁 Dataset Studio**: Interactive dataset selector for PROMISE, AEEEM, ReLink, MORPH, or custom CSV upload, with missing value inspection, class distribution donut charts, and correlation heatmaps.
3. **⚙️ Preprocessing Studio**: Configurable imputer, scaler (Standard, MinMax, Robust), SMOTE / ADASYN class imbalance resampler, and train/test split with zero-leakage guarantee.
4. **🧬 Metaheuristic Feature Selection**: Interactive parameter tuning for NSGA-II, GA, and BPSO, with real-time generation progress, convergence curves, and 2D/3D Plotly Pareto front explorers.
5. **🤖 Model Benchmarking**: Comparative benchmark across all 5 ML models (Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, MCC, latencies), radar charts, and interactive confusion matrices.
6. **🔍 Explainable AI (SHAP) Studio**: Global mean $|SHAP|$ feature rankings and local instance waterfall attributions with automated software engineering refactoring guidance.
7. **🎯 Real-Time Defect Predictor**: Single module metric input form with metric archetype presets ("Low-Risk", "High-Complexity", "High-Coupling") and batch CSV bulk scoring with one-click download.
8. **📜 Experiment History**: SQLite-backed audit log of experimental configurations and comparative performance charts.

---

## 🛠️ Technology Stack

- **Core Language**: Python 3.11
- **Machine Learning**: Scikit-Learn, Imbalanced-Learn
- **Metaheuristics**: Custom Vectorized NSGA-II, Genetic Algorithm, Binary PSO, pymoo, DEAP
- **Explainable AI**: SHAP (SHapley Additive exPlanations)
- **Interactive UI**: Streamlit, Plotly Express & Graph Objects
- **Storage & Audit**: SQLite3, Pandas, NumPy

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
