# DefectIQ — Autonomous Explainable Multi-Objective Software Defect Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-brightgreen.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **DefectIQ** is a state-of-the-art AI-powered research and production platform for **Explainable Multi-Objective Software Defect Prediction (SDP)**. Built with high-performance metaheuristics (NSGA-II, Binary PSO, Genetic Algorithms), leakage-free preprocessing, five classical ML classifier architectures, and game-theoretic Explainable AI (SHAP) that maps empirical metric attributions directly to concrete developer refactoring guidance.

---

## 🎨 Futuristic Dark Research UI Aesthetic

Designed specifically for researchers and software engineering teams:
- **Dark AI Research Palette**:
  - Background: `#0B1020` (Dark Cosmic Navy)
  - Sidebar: `#10172A` (Deep Slate)
  - Cards & Panels: `#151E33` (Translucent Glassmorphism)
  - Primary Accent: `#8B5CF6` (Electric Violet)
  - Secondary Accent: `#38BDF8` (Cyber Sky Blue)
  - Status Accents: `#22C55E` (Clean / Success), `#EF4444` (Defective / Critical)
  - Borders: `#26334D`
- **Interactive Dark Plotly Charts**: Seamlessly themed 2D/3D non-dominated Pareto surfaces, multi-metric radar spectra, ROC/PR curves, and local SHAP waterfall attributions.

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
        | - Refactoring Guidance  |                               | - SQLite Audit Database |
        +-------------------------+                               +-------------------------+
```

---

## 📐 Multi-Objective Optimization Formulation (NSGA-II)

DefectIQ optimizes three conflicting objectives simultaneously using non-dominated sorting and crowding distance:

$$\min \mathbf{F}(\mathbf{x}) = \Big( f_1(\mathbf{x}),\; f_2(\mathbf{x}),\; f_3(\mathbf{x}) \Big)$$

where $\mathbf{x} = [x_1, x_2, \dots, x_D] \in \{0, 1\}^D$ is the binary feature mask:
- **$f_1(\mathbf{x}) = 1.0 - \text{Metric}(\mathbf{x})$**: Predictive classification error (where $\text{Metric} \in \{\text{F1-score}, \text{MCC}, \text{ROC-AUC}\}$).
- **$f_2(\mathbf{x}) = \frac{\sum_{i=1}^D x_i}{D}$**: Ratio of selected software metrics over total metric space.
- **$f_3(\mathbf{x}) = t_{\text{latency}}(\mathbf{x})$**: Model evaluation and inference execution latency in seconds.

The platform automatically computes the **Knee-Point Solution** along the Pareto front via minimum Euclidean distance to the ideal objective vector $(0, 0, 0)$.

---

## 📁 18 Curated Empirical Benchmark Datasets Included

DefectIQ includes **18 offline-ready, genuine academic defect datasets**:

1. **PROMISE Suite (NASA Metrics Data Program)**:
   - `PC1`, `PC3`, `PC4` (NASA spacecraft and flight instrument software)
   - `KC1`, `KC2` (NASA telemetry reception and storage management)
   - `CM1` (NASA spacecraft scientific instruments)
   - `MW1` (Zero-gravity microgravity combustion experiments)
   - `MC2` (Video guidance system)
2. **AEEEM Suite (D'Ambros et al.)**:
   - `EQ` (Eclipse Equinox OSGi runtime framework)
   - `JDT` (Eclipse Java Development Tools)
   - `LC` (Apache Lucene search library)
   - `ML` (Eclipse Mylyn task-focused interface)
   - `PDE` (Eclipse Plugin Development Environment)
3. **ReLink Suite (Defect-Corrected Metrics)**:
   - `apache` (Apache HTTP Server)
   - `safe` (J2ME mobile security and password manager)
   - `zxing` (1D/2D barcode image processing library)
4. **MORPH Suite (Turhan et al. / SOFTLAB)**:
   - `ar1`, `ar3`, `ar4`, `ar5`, `ar6` (Embedded software and telecommunications controllers)
5. **Custom CSV Upload**: Ingest any software metrics CSV with automatic defect target column detection and binarization.

---

## 🧭 Platform Navigation (11 Dedicated Modules)

| # | Module | Key Features |
|---|---|---|
| 1 | **⚡ Overview** | Hero KPI counters, research motivation, platform telemetry, quick action launchers, and empirical findings. |
| 2 | **📁 Dataset Management** | Suite selector (18 benchmarks), custom CSV upload, class distribution donut charts, metric correlation heatmaps, missing value analysis. |
| 3 | **⚙️ Data Preprocessing** | Imputation (median, mean, mode), duplicate removal, scalers (Standard, MinMax, Robust), train-only SMOTE/ADASYN, zero-leakage guarantee. |
| 4 | **🤖 Model Training** | Train all 5 classifiers (LR, DT, RF, SVM, GBDT) on full vs optimized feature subsets; radar chart and latency benchmarks. |
| 5 | **🧬 Optimization Lab** | NSGA-II, Binary PSO, and Genetic Algorithm solvers with real-time animated generation progress and convergence telemetry. |
| 6 | **🌟 Pareto Analysis** | Dedicated deep dive into 2D/3D non-dominated Pareto surfaces, hypervolume analysis, knee-point trade-offs, and 1-click feature adoption. |
| 7 | **🎯 Defect Prediction** | Single-module interactive predictor with presets ("Low-Risk", "High-Complexity", "High-Coupling") and batch CSV scoring with 1-click download. |
| 8 | **🔍 Explainable AI** | Global mean \|SHAP\| feature rankings and local instance waterfall attributions with automated developer refactoring advice. |
| 9 | **📈 Performance Evaluation** | Multi-model radar spectrum, ROC curves with AUC annotations, PR curves, and interactive 2x2 confusion matrix heatmaps. |
| 10 | **📜 Experiment History** | SQLite database viewer, search/filter by dataset and model, historical comparison charts, and CSV/JSON export. |
| 11 | **🛠️ Settings** | Global random seeds, default solver presets, environment telemetry, and session state reset. |

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/prii06132003/explainable-moo-defect-prediction.git
cd explainable-moo-defect-prediction
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Unit Tests
```bash
python -m unittest tests/test_all_modules.py
```
*(All 8 test suites pass verifying end-to-end data loading, preprocessing, metaheuristics, models, SHAP, and SQLite persistence).*

### 4. Launch the Web Application
```bash
# Option A: 1-Click Batch Launcher (Windows)
run_app.bat

# Option B: Direct Terminal Launch
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📄 Academic Project Documentation
The complete comprehensive academic research documentation, mathematical formulations, literature review, and experimental results can be found in:
📖 **[`DOCUMENTATION.md`](DOCUMENTATION.md)**

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
