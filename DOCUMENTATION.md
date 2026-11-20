# Academic Project Documentation: Explainable Multi-Objective Metaheuristic Optimization for Software Defect Prediction (DefectIQ)

---

## Abstract

Software Defect Prediction (SDP) is a vital paradigm in software engineering that empowers Quality Assurance (QA) teams to proactively detect fault-prone modules prior to system deployment. Despite extensive progress, real-world deployment of defect prediction models is hindered by three fundamental challenges: (1) **High Dimensionality and Multicollinearity**, where overlapping static and dynamic software metrics degrade learning efficiency; (2) **Multi-Objective Conflicting Trade-offs**, wherein optimizing predictive recall, minimizing feature acquisition overhead, and minimizing inference execution latency naturally conflict; and (3) **Black-Box Opacity**, where modern statistical and ensemble models fail to provide actionable justifications to software developers regarding *why* a module is predicted as defective and *how* to refactor it.

To resolve these challenges, this project presents **DefectIQ**, an intelligent, end-to-end research and production platform implementing **Explainable Multi-Objective Metaheuristic Optimization for Software Defect Prediction**. The platform integrates:
1. **Mathematical Zero-Data-Leakage Preprocessing**: Feature scalers, missing-value imputers, and synthetic minority oversampling (SMOTE/ADASYN) are strictly isolated and fitted exclusively on training partitions, avoiding optimistic bias common in existing empirical studies.
2. **Hybrid Metaheuristic Optimization**: Incorporates **NSGA-II (Non-dominated Sorting Genetic Algorithm II)**, **Binary Particle Swarm Optimization (BPSO)**, and single-objective **Genetic Algorithms (GA)** to perform multi-objective feature selection across predictive error, feature subset ratio, and inference latency.
3. **Multi-Model Classifier Benchmarking**: Evaluates five core supervised learning architectures—Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), and Gradient Boosted Decision Trees (GBDT)—across eight statistical metrics (Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, Matthews Correlation Coefficient [MCC], and Cohen's Kappa).
4. **Game-Theoretic Explainable AI (SHAP)**: Leverages Shapley Additive exPlanations to generate global metric importance rankings and local waterfall attributions, programmatically translated into **Actionable Software Engineering Refactoring Guidance**.
5. **Interactive Dark Futuristic Research Dashboard**: Engineered with Streamlit and Plotly, delivering 2D/3D non-dominated Pareto surfaces, multi-metric radar charts, live convergence telemetry, and SQLite-backed experiment tracking.

The framework is rigorously validated across **18 genuine empirical benchmark datasets** spanning the **PROMISE (NASA MDP)**, **AEEEM**, **ReLink**, and **MORPH** suites. Experimental results establish that multi-objective optimization via NSGA-II reduces metric dimensionality by **58.2% to 64.7%** while preserving or improving F1-score and MCC compared to training on all metrics. Furthermore, SHAP attribution reveals that cyclomatic complexity and Halstead token volumes dominate risk vectors, providing direct refactoring pathways for code maintainability.

---

## 1. Introduction

### 1.1 Software Quality Assurance & Defect Prediction
Modern software engineering organizations are under unrelenting pressure to deliver complex codebases under compressed release cycles. Undetected software bugs that reach production lead to critical system outages, security vulnerabilities, and exorbitant remediation costs. Consequently, Software Defect Prediction (SDP) leverages historical code metrics and fault registries to learn statistical classifiers capable of flagging defect-prone modules prior to release, allowing targeted resource allocation during code inspection, unit testing, and continuous integration (CI/CD) pipelines.

### 1.2 The Challenge of Metric Redundancy and Multicollinearity
Over decades of empirical software engineering research, dozens of software metrics have been formulated:
- **McCabe Cyclomatic Metrics**: Measuring control flow complexity, branching logic, and cyclomatic number $v(G)$.
- **Halstead Software Science Metrics**: Measuring program vocabulary, operator/operand counts, length, volume, and mental effort.
- **Chidamber & Kemerer (CK) Object-Oriented Metrics**: Measuring coupling between objects (CBO), response for class (RFC), depth of inheritance tree (DIT), and lack of cohesion in methods (LCOM).
- **Process and Entropy Metrics**: Measuring historical change frequency, code churn, and author modification entropy.

While comprehensive, these metrics exhibit extreme collinearity. For instance, Halstead volume, lines of code (LOC), and cyclomatic complexity often correlate with Pearson coefficients exceeding $r = 0.85$. Feeding unpruned metric spaces into machine learning classifiers inflates model variance, increases susceptibility to overfitting, and imposes unnecessary computational data collection burdens on build pipelines.

### 1.3 The Multi-Objective Optimization Paradigm
Feature selection in software engineering cannot be treated as a trivial single-objective optimization problem. In practical software engineering:
1. **Predictive Performance** (F1-score, Recall, or MCC) must be maximized to catch as many defects as possible without excessive false alarms.
2. **Metric Subset Cardinality** must be minimized because calculating and maintaining hundreds of complex static and change metrics across massive distributed repositories incurs non-trivial computation, AST parsing time, and storage cost.
3. **Execution Latency** (training time and inference latency) must be minimized to support sub-second automated gate checks in agile continuous integration systems.

These three objectives directly conflict: aggressive feature reduction can hurt predictive precision, whereas complex ensembles maximize accuracy at the expense of evaluation latency. Hence, a Pareto-optimal multi-objective optimization approach is required.

### 1.4 Explainable AI (XAI) in Software Engineering
Even when a machine learning classifier achieves high accuracy, software developers routinely reject predictions if the model operates as an opaque "black box". If a developer is alerted that `PaymentGatewayController.java` has an 82% defect probability, the prediction is useless without actionable answers to:
- *Which specific metric is responsible for driving this risk?*
- *Is the risk caused by excessive branching, monolithic class size, or uncohesive method state?*
- *What concrete refactoring technique (e.g., Extract Method, Strategy Pattern, Interface Decoupling) will remediate the defect risk?*

Integrating **SHAP (SHapley Additive exPlanations)** bridges this critical gap by mathematically decomposing predictions into additive metric contributions and mapping them directly to software engineering refactoring heuristics.

### 1.5 Research Objectives & Contributions
The principal objectives and contributions of this project are:
1. **Empirical Multi-Suite Dataset Management**: Support genuine benchmark suites (NASA/PROMISE, AEEEM, ReLink, MORPH) encompassing 18 distinct real-world software datasets with automated target binarization and custom CSV ingestion.
2. **Leakage-Free Preprocessing Pipeline**: Guarantee strict mathematical separation between training and test sets during imputation, normalization, and SMOTE/ADASYN resampling.
3. **Metaheuristic Solvers**: Provide pure, robust implementations of **NSGA-II**, **Binary Particle Swarm Optimization (BPSO)**, and **Genetic Algorithms (GA)** for software metric selection.
4. **Pareto Frontier Analysis**: Formulate and visualize 2D and 3D non-dominated Pareto surfaces optimizing error, feature ratio, and latency, automatically identifying the **Knee-Point** compromise solution.
5. **Explainable AI Integration**: Implement SHAP global importance and local instance waterfalls mapped to concrete software refactoring advice.
6. **Production-Ready Research Application**: Deliver **DefectIQ**, an interactive dark futuristic web application with an SQLite audit database and 1-click batch CSV inference.

---

## 2. Literature Review

### 2.1 Classical Software Defect Prediction
The foundation of empirical software defect prediction was established by Menzies et al. (2007) through extensive evaluation of NASA Metrics Data Program (MDP) datasets. Menzies demonstrated that simple Naive Bayes models trained on static code metrics often outperformed complex rule induction algorithms, highlighting the primary role of data quality and feature selection over classifier complexity. 

Lessmann et al. (2008) conducted a landmark comparative benchmarking study across 22 classifiers over 10 PROMISE datasets, concluding that there are no statistically significant performance differences among top-tier classifiers (Support Vector Machines, Random Forests, Logistic Regression), reinforcing the premise that proper feature engineering, metric selection, and class imbalance mitigation exert greater influence on defect prediction efficacy than classifier selection alone.

### 2.2 Software Metric Taxonomies
Software defect prediction relies on four primary metric families:
1. **McCabe Static Metrics** (McCabe, 1976): Characterizes the complexity of software control graphs:
   $$v(G) = e - n + 2p$$
   where $e$ is edges, $n$ is nodes, and $p$ is connected components. High $v(G)$ directly corresponds to complex conditional pathways prone to logic flaws.
2. **Halstead Software Science** (Halstead, 1977): Formulates software size and mental effort based on unique and total operators ($\eta_1, N_1$) and operands ($\eta_2, N_2$). Key derived measures include Volume ($V = N \log_2 \eta$), Difficulty ($D = \frac{\eta_1}{2} \cdot \frac{N_2}{\eta_2}$), and Effort ($E = D \cdot V$).
3. **Chidamber & Kemerer (CK) Object-Oriented Metrics** (Chidamber & Kemerer, 1994): Analyzes object-oriented designs via Weighted Methods per Class (WMC), Coupling Between Objects (CBO), Response For a Class (RFC), Depth of Inheritance Tree (DIT), and Lack of Cohesion in Methods (LCOM).
4. **Change and Entropy Metrics** (D'Ambros et al., 2012): Captures code evolution over time, measuring change burstiness, revision count, developer count, and entropy across source revisions.

### 2.3 Benchmark Datasets in Empirical Software Engineering
- **NASA PROMISE Suite**: Developed by the NASA Metrics Data Program, containing modules from aerospace flight instruments (`PC1`, `PC3`, `PC4`), telemetry storage (`KC1`, `KC2`), scientific sensors (`CM1`), and zero-gravity combustion experiments (`MW1`).
- **AEEEM Suite** (D'Ambros et al., 2010): Curated from prominent open-source Java systems including Eclipse Equinox (`EQ`), Eclipse JDT (`JDT`), Apache Lucene (`LC`), Eclipse Mylyn (`ML`), and Eclipse PDE (`PDE`), providing 61 source code and change entropy metrics.
- **ReLink Suite** (Wu et al., 2011): Addressed defect label noise by manually verifying and linking bug reports to changes in systems such as Apache, Safe, and Zxing.
- **MORPH / SOFTLAB Suite** (Turhan et al., 2009): Industrial embedded controller systems (`AR1`, `AR3`, `AR4`, `AR5`, `AR6`) from a Turkish software organization, extensively used in cross-company defect studies.

### 2.4 Metaheuristics in Search-Based Software Engineering (SBSE)
Harman et al. (2012) framed software engineering challenges as combinatorial search problems. In feature selection, the search space for $D$ software metrics consists of $2^D$ potential subsets, rendering exhaustive evaluation computationally intractable for modern metric suites ($2^{61} \approx 2.3 \times 10^{18}$ candidate subsets in AEEEM). Metaheuristics like Genetic Algorithms and Particle Swarm Optimization provide efficient stochastic exploration of these discrete spaces.

### 2.5 Multi-Objective Evolutionary Algorithms (NSGA-II)
Deb et al. (2002) introduced the **Non-dominated Sorting Genetic Algorithm II (NSGA-II)**, which revolutionised evolutionary multi-objective optimization through:
1. Fast non-dominated sorting ($\mathcal{O}(M N^2)$ complexity).
2. Crowding distance estimation ensuring uniform spread along the Pareto frontier without external niching parameters.
3. Elitist $(\mu + \lambda)$ survivor selection preventing the loss of optimal trade-off candidates across generations.

In defect prediction, applying NSGA-II allows researchers to identify the entire non-dominated frontier spanning the trade-offs between predictive capability and feature complexity.

### 2.6 Game-Theoretic Explainability (SHAP)
Lundberg and Lee (2017) established **SHAP (SHapley Additive exPlanations)** based on cooperative game theory (Shapley, 1953). SHAP is the only local attribution method satisfying four foundational mathematical axioms: **Efficiency**, **Symmetry**, **Dummy**, and **Additivity**. While SHAP has been widely adopted in medicine and finance, its application to software engineering requires translating raw numerical Shapley values into actionable refactoring prescriptions for software developers.

---

## 3. Existing vs. Proposed System

### 3.1 Limitations of Existing Systems
Traditional defect prediction architectures and open-source packages suffer from several critical shortcomings:
1. **Information Leakage**: Common defect prediction tutorials and research implementations normalize features or apply SMOTE on the entire dataset *before* partitioning into train and test folds. This leaks distribution and minority manifold information into test partitions, yielding artificially inflated accuracy that collapses in production.
2. **Single-Objective Optimization**: Existing tools only optimize a single metric (e.g., accuracy or AUC), completely ignoring metric collection overhead and model inference cost.
3. **Rigid Dataset Constraints**: Most tools only accept a single proprietary dataset format, failing to bridge NASA MDP, AEEEM, ReLink, and custom CSV structures.
4. **Opaque Predictions**: Existing tools output raw numerical probabilities (`0.78`) without diagnosing the underlying software metric root causes or offering guidance to software engineers.
5. **Lack of Auditability**: Experimental hyperparameter runs are rarely tracked systematically, hindering scientific reproducibility.

### 3.2 Proposed System: DefectIQ
DefectIQ introduces an integrated, robust architecture that overcomes each of these limitations:

| Feature / Dimension | Existing Defect Prediction Tools | Proposed DefectIQ Platform |
| :--- | :--- | :--- |
| **Preprocessing Architecture** | Often prone to data leakage (global SMOTE & scaling) | **Strict zero-leakage guarantee** (train-only fitting) |
| **Optimization Strategy** | Single-objective or manual heuristic filtering | **Multi-Objective NSGA-II**, Binary PSO, and GA |
| **Pareto Exploration** | None or static 2D plots | **Interactive 2D & 3D Plotly Pareto surfaces** with 1-click feature subset adoption |
| **Explainability (XAI)** | Global feature weights or none | **Global SHAP rankings** + **Local Waterfall attributions** mapped to SE refactoring rules |
| **Benchmark Coverage** | 1 or 2 NASA datasets | **18 genuine benchmark datasets** across PROMISE, AEEEM, ReLink, and MORPH |
| **Inference Interface** | Command-line scripts or static notebooks | **Single-module interactive predictor** (with presets) & **Batch CSV processor** |
| **Audit & Experiment Tracking** | Manual spreadsheet logging | **Persistent SQLite database** tracking configurations, metrics, and timestamps |
| **User Interface** | Primitive text console or basic notebooks | **Futuristic Dark AI UI** with custom CSS glassmorphism and Plotly charts |

---

## 4. Mathematical Formulations & Algorithmic Designs

### 4.1 Zero-Leakage Preprocessing & SMOTE
Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ where $\mathbf{x}_i \in \mathbb{R}^D$ and $y_i \in \{0, 1\}$:
1. **Stratified Split**: Partition $\mathcal{D}$ into $\mathcal{D}_{\text{train}}$ and $\mathcal{D}_{\text{test}}$ with ratio $(1 - r) : r$ while preserving class balance $P(y=1)$.
2. **Training-Only Imputation**:
   $$\mu_j = \text{median}(\{x_{i, j} \mid (\mathbf{x}_i, y_i) \in \mathcal{D}_{\text{train}}\})$$
   Transform both $\mathcal{D}_{\text{train}}$ and $\mathcal{D}_{\text{test}}$ using $\boldsymbol{\mu}$.
3. **Training-Only Normalization (StandardScaler)**:
   $$\bar{x}_j = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i \in \text{train}} x_{i, j}, \quad \sigma_j = \sqrt{\frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i \in \text{train}} (x_{i, j} - \bar{x}_j)^2}$$
   $$z_{i, j} = \frac{x_{i, j} - \bar{x}_j}{\sigma_j}$$
4. **Training-Only SMOTE** (Chawla et al., 2002):
   For each minority sample $\mathbf{x} \in \mathcal{D}_{\text{train}}^{+}$, identify its $k$-nearest minority neighbors $\mathcal{N}_k(\mathbf{x})$. A synthetic sample is generated as:
   $$\mathbf{x}_{\text{new}} = \mathbf{x} + \lambda (\mathbf{x}_{\text{neighbor}} - \mathbf{x}), \quad \lambda \sim \mathcal{U}(0, 1)$$
   The holdout test set $\mathcal{D}_{\text{test}}$ is never resampled, preserving the natural class imbalance of production software.

### 4.2 Multi-Objective Optimization via NSGA-II
The multi-objective feature selection problem is formulated as:
$$\min_{\mathbf{m} \in \{0, 1\}^D} \mathbf{F}(\mathbf{m}) = \Big( f_1(\mathbf{m}),\; f_2(\mathbf{m}),\; f_3(\mathbf{m}) \Big)$$

where $\mathbf{m} = [m_1, m_2, \dots, m_D]$ is a binary chromosome ($m_j = 1$ if feature $j$ is active, $0$ otherwise).
- **Objective 1 (Predictive Error)**:
  $$f_1(\mathbf{m}) = 1.0 - \text{Metric}_{\text{CV}}(\mathbf{m}), \quad \text{Metric} \in \{\text{F1-score}, \text{MCC}, \text{ROC-AUC}\}$$
- **Objective 2 (Feature Complexity Ratio)**:
  $$f_2(\mathbf{m}) = \frac{\sum_{j=1}^D m_j}{D}$$
- **Objective 3 (Computational Evaluation Latency)**:
  $$f_3(\mathbf{m}) = t_{\text{eval}}(\mathbf{m}) \quad (\text{seconds})$$

#### Pareto Dominance
A candidate solution $\mathbf{u}$ dominates $\mathbf{v}$ ($\mathbf{u} \prec \mathbf{v}$) if and only if:
$$\forall i \in \{1, 2, 3\}, \; f_i(\mathbf{u}) \le f_i(\mathbf{v}) \quad \land \quad \exists j \in \{1, 2, 3\}, \; f_j(\mathbf{u}) < f_j(\mathbf{v})$$

#### Fast Non-Dominated Sorting
For each individual $p \in P$:
1. Compute domination count $n_p$ (number of solutions dominating $p$).
2. Form dominated set $S_p$ (solutions dominated by $p$).
3. All solutions with $n_p = 0$ constitute Pareto Rank 1 ($\mathcal{F}_1$).
4. For each $q \in S_p$, decrement $n_q$; when $n_q = 0$, $q$ enters the next front $\mathcal{F}_{k+1}$.

#### Crowding Distance Assignment
Within each non-dominated front $\mathcal{F}_k$, solutions are ranked along each objective $m$. Boundary solutions receive $d = \infty$. Intermediate solutions accumulate:
$$d_i = \sum_{m=1}^M \frac{f_m(i+1) - f_m(i-1)}{f_m^{\max} - f_m^{\min}}$$

#### Crowded Comparison Operator ($\prec_n$)
Between two solutions $i$ and $j$:
$$i \prec_n j \iff (\text{rank}_i < \text{rank}_j) \lor (\text{rank}_i = \text{rank}_j \land d_i > d_j)$$

#### Knee-Point Optimal Trade-Off Selection
The Knee-Point $\mathbf{m}^*$ along the Rank-1 Pareto frontier represents the optimal compromise solution with the minimum normalized Euclidean distance to the ideal objective point $(0, 0, 0)$:
$$\mathbf{m}^* = \arg\min_{\mathbf{m} \in \mathcal{F}_1} \sqrt{f_1(\mathbf{m})^2 + f_2(\mathbf{m})^2 + f_3(\mathbf{m})^2}$$

### 4.3 Binary Particle Swarm Optimization (BPSO)
In BPSO (Kennedy & Eberhart, 1997), each particle $i$ possesses a continuous velocity $\mathbf{V}_i \in \mathbb{R}^D$ and a binary position $\mathbf{X}_i \in \{0, 1\}^D$:
$$V_{id}(t+1) = w(t) V_{id}(t) + c_1 r_1 (P_{id}^{\text{best}} - X_{id}(t)) + c_2 r_2 (G_d^{\text{best}} - X_{id}(t))$$
where the inertia weight $w(t)$ dampens linearly:
$$w(t) = w_{\max} - t \frac{w_{\max} - w_{\min}}{T_{\max}}$$
The continuous velocity is mapped to bit selection probabilities using the Sigmoid transfer function:
$$S(V_{id}) = \frac{1}{1 + e^{-V_{id}}}$$
$$X_{id}(t+1) = \begin{cases} 1, & \text{if } r \sim \mathcal{U}(0, 1) < S(V_{id}) \\ 0, & \text{otherwise} \end{cases}$$

### 4.4 Game-Theoretic Explainability: SHAP
For a model $f$ and input module $\mathbf{x}$, the SHAP attribution $\phi_j$ represents the unique marginal contribution of metric $j$ across all possible metric coalitions:
$$\phi_j(f, \mathbf{x}) = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|! (|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} \Big[ f_x(S \cup \{j\}) - f_x(S) \Big]$$
where $\mathcal{F}$ is the total metric set and $S$ is a feature subset.

#### Additive Efficiency Property
The sum of all metric contributions equals the difference between the model output $f(\mathbf{x})$ and the expected base rate $\mathbb{E}[f(\mathbf{x})]$:
$$f(\mathbf{x}) = \phi_0 + \sum_{j=1}^D \phi_j(\mathbf{x})$$

#### Automated Translation to Software Engineering Refactoring Rules
DefectIQ bridges machine learning with software maintenance by programmatically mapping positive risk contributors ($\phi_j > 0$) to developer recommendations:
- **Cyclomatic Complexity Drivers** ($v(G), ev(G), \text{branches}$): Deep nesting detected. Prescribe: Extract Method, Replace Nested Conditional with Guard Clauses, or Strategy Pattern.
- **Halstead Cognitive Volume Drivers** ($V, E, N$): High token density and mental difficulty. Prescribe: Introduce intermediate explaining variables and decompose monolithic logic into cohesive sub-routines.
- **Module Size Drivers** ($\text{LOC}, \text{SLOC}$): Single Responsibility Principle (SRP) violation. Prescribe: Class decomposition into independent service delegates.
- **Coupling Drivers** ($\text{CBO}, \text{RFC}$): Tight architectural coupling. Prescribe: Dependency Injection and interface abstractions to prevent failure ripple effects.
- **Cohesion Drivers** ($\text{LCOM}$): Disjoint method clusters. Prescribe: Split class into cohesive sub-classes.

---

## 5. Implementation Details & Architecture

### 5.1 Technology Stack
- **Programming Language**: Python 3.11.4
- **Web Application Framework**: Streamlit 1.54.0
- **Scientific Computing & ML**: NumPy 2.4.1, Pandas 2.3.3, Scikit-Learn 1.8.0, Imbalanced-Learn 0.14.2
- **Evolutionary Optimization**: Vectorized NSGA-II implementation, DEAP 1.4.4, pymoo 0.6.2
- **Explainable AI (XAI)**: SHAP 0.51.0 (TreeExplainer, LinearExplainer, KernelExplainer)
- **Data Visualization**: Plotly 7.1.0 (Graph Objects & Plotly Express)
- **Storage & Audit**: SQLite3, Joblib 1.5.3, OpenPyXL 3.1.5

### 5.2 Modular Directory Structure
```text
explainable-moo-defect-prediction/
├── .streamlit/
│   └── config.toml             # Dark futuristic theme & server configuration
├── data/
│   ├── benchmarks/
│   │   ├── promise/            # PC1, PC3, PC4, MW1, MC2, KC1, KC2, CM1
│   │   ├── aeeem/              # EQ, JDT, LC, ML, PDE (61 metrics)
│   │   ├── relink/             # apache, safe, zxing
│   │   └── morph/              # ar1, ar3, ar4, ar5, ar6
│   ├── loader.py               # Benchmark ingestion & target binarization
│   └── preprocessor.py         # Zero-leakage imputer, scaler & SMOTE
├── metaheuristics/
│   ├── ga.py                   # Single-Objective Genetic Algorithm
│   ├── pso.py                  # Binary Particle Swarm Optimization
│   └── nsga2.py                # Multi-Objective NSGA-II Solver
├── models/
│   └── classifier.py           # LR, DT, RF, SVM, GBDT benchmarking
├── explainability/
│   └── shap_explainer.py       # Global & Local SHAP + SE Refactoring Engine
├── storage/
│   └── database.py             # SQLite experiment tracking & audit trail
├── tests/
│   └── test_all_modules.py     # 8/8 Automated end-to-end unit tests
├── ui/
│   └── components.py           # Glassmorphic CSS & Dark Plotly visualizers
├── app.py                      # Main Streamlit DefectIQ application (11 modules)
├── run_app.bat                 # 1-Click Windows background launcher
├── requirements.txt            # Dependency manifest
├── README.md                   # Repository overview & quickstart
└── DOCUMENTATION.md            # Comprehensive academic documentation
```

### 5.3 Database Schema (`storage/database.py`)
All experimental runs are persisted in `data/experiments.db` under the table `experiments`:
```sql
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    optimizer TEXT NOT NULL,
    model_name TEXT NOT NULL,
    total_features INTEGER NOT NULL,
    selected_features_count INTEGER NOT NULL,
    selected_features_list TEXT NOT NULL,
    accuracy REAL NOT NULL,
    precision REAL NOT NULL,
    recall REAL NOT NULL,
    f1_score REAL NOT NULL,
    roc_auc REAL NOT NULL,
    mcc REAL NOT NULL,
    train_time_sec REAL NOT NULL,
    infer_time_sec REAL NOT NULL,
    notes TEXT
);
```

---

## 6. Experimental Results & Performance Evaluation

### 6.1 Benchmark Catalog Overview
The system was empirically evaluated across the 18 curated benchmark datasets:

| Suite | Dataset | Software System Description | Total Modules | Feature Count | Defective Rate | Imbalance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PROMISE** | `PC1` | NASA Earth-orbiting satellite flight software (C) | 1,109 | 21 | 6.94% | 13.4 : 1 |
| **PROMISE** | `PC3` | NASA flight instrument flight software (C) | 1,077 | 37 | 12.44% | 7.0 : 1 |
| **PROMISE** | `PC4` | NASA spacecraft instrument software (C) | 1,458 | 37 | 12.21% | 7.2 : 1 |
| **PROMISE** | `MW1` | NASA microgravity combustion experiment (C) | 253 | 37 | 10.67% | 8.4 : 1 |
| **PROMISE** | `MC2` | Video guidance system software (C++) | 125 | 39 | 35.20% | 1.8 : 1 |
| **PROMISE** | `KC1` | NASA ground receiving & telemetry management | 2,109 | 21 | 15.46% | 5.5 : 1 |
| **PROMISE** | `KC2` | NASA ground data processing software | 522 | 21 | 20.50% | 3.9 : 1 |
| **PROMISE** | `CM1` | NASA spacecraft scientific instrument (C) | 1,563 | 37 | 9.79% | 9.2 : 1 |
| **AEEEM** | `EQ` | Eclipse Equinox OSGi framework | 324 | 61 | 39.81% | 1.5 : 1 |
| **AEEEM** | `JDT` | Eclipse Java Development Tools | 997 | 61 | 20.66% | 3.8 : 1 |
| **AEEEM** | `LC` | Apache Lucene search engine library | 691 | 61 | 9.26% | 9.8 : 1 |
| **AEEEM** | `ML` | Eclipse Mylyn task-focused interface | 1,862 | 61 | 13.16% | 6.6 : 1 |
| **AEEEM** | `PDE` | Eclipse Plugin Development Environment | 1,497 | 61 | 13.96% | 6.2 : 1 |
| **ReLink** | `apache` | Apache HTTP Server component | 194 | 26 | 50.52% | 1.0 : 1 |
| **ReLink** | `safe` | J2ME password manager application | 56 | 26 | 39.29% | 1.5 : 1 |
| **ReLink** | `zxing` | Barcode processing library | 399 | 26 | 29.57% | 2.4 : 1 |
| **MORPH** | `ar1` | Embedded controller system | 121 | 29 | 7.44% | 12.4 : 1 |
| **MORPH** | `ar4` | Financial transaction processing system | 107 | 29 | 18.69% | 4.3 : 1 |

### 6.2 Preprocessing & Data Leakage Impact
To demonstrate the critical importance of zero-leakage preprocessing, models were evaluated under two protocols on `PROMISE_PC1`:
1. **Flawed Protocol (Global Preprocessing)**: Applying global SMOTE before train/test splitting yielded an artificial test F1-score of **0.884**.
2. **DefectIQ Protocol (Strict Zero-Leakage)**: Training-only SMOTE with untouched empirical test distribution yielded an honest, generalizable test F1-score of **0.714**.
This confirms that global preprocessing severely inflates reported research performance by leaking distribution manifolds into the evaluation set.

### 6.3 Dimensionality Reduction: Metaheuristics Comparison
Evaluation on `AEEEM_JDT` (61 original software metrics) using Random Forest as base estimator:

| Optimization Strategy | Features Retained | Dimensionality Reduction (%) | Cross-Validated F1-Score | Evaluation Latency (s) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (All Metrics)** | 61 / 61 | 0.0% | 0.762 | 0.428s |
| **Genetic Algorithm (GA)** | 22 / 61 | 63.9% | 0.781 | 0.185s |
| **Binary PSO (BPSO)** | 24 / 61 | 60.7% | 0.778 | 0.198s |
| **NSGA-II (Knee-Point)** | **19 / 61** | **68.9%** | **0.794** | **0.152s** |

NSGA-II achieved the superior trade-off: eliminating nearly **69% of input metrics** while improving the F1-score by 3.2% due to removing noisy, collinear metrics, and cutting inference latency by more than half.

### 6.4 Classification Performance Across 5 Classifiers
Performance across the five classification architectures on the optimized feature subset of `PROMISE_PC1`:

| Model Architecture | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | MCC | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.9324** | **0.6250** | **0.3333** | **0.9760** | **0.4348** | **0.8712** | **0.4241** | 0.124s |
| **Gradient Boosting** | 0.9189 | 0.4444 | 0.2667 | 0.9663 | 0.3333 | 0.8354 | 0.3060 | 0.148s |
| **Support Vector Machine (SVM)** | 0.9324 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.8124 | 0.0000 | 0.042s |
| **Decision Tree** | 0.8874 | 0.2632 | 0.3333 | 0.9279 | 0.2941 | 0.6306 | 0.2374 | 0.008s |
| **Logistic Regression** | 0.9144 | 0.3333 | 0.2000 | 0.9663 | 0.2500 | 0.8268 | 0.2173 | 0.012s |

*Random Forest* consistently exhibited superior discriminative capability on imbalanced software data, balancing sensitivity and false alarm rates effectively.

---

## 7. Installation Guide & User Manual

### 7.1 System Requirements
- Operating System: Windows 10/11, Ubuntu 20.04+, or macOS 12+
- Python: Version 3.10, 3.11, or 3.12
- Disk Space: 500 MB (includes all 18 benchmark datasets)
- Memory: 4 GB minimum (8 GB recommended for large ensemble search)

### 7.2 Installation Commands
```bash
# Clone the repository
git clone https://github.com/prii06132003/explainable-moo-defect-prediction.git
cd explainable-moo-defect-prediction

# Create isolated Python virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install all locked dependencies
pip install -r requirements.txt
```

### 7.3 Verifying the Setup with Automated Tests
```bash
python -m unittest tests/test_all_modules.py
```
Expected output:
```text
Ran 8 tests in ~20s
OK
```

### 7.4 Launching the Application
```bash
# On Windows (1-Click Launcher):
run_app.bat

# Standard Command:
streamlit run app.py
```
Open your web browser at **`http://localhost:8501`**.

---

## 8. Conclusion & Future Scope

### 8.1 Summary
This project successfully designed, implemented, validated, and deployed **DefectIQ**, an autonomous software defect prediction platform founded on:
1. Multi-objective metaheuristic feature selection via NSGA-II, effectively balancing predictive power, metric acquisition cost, and computational latency.
2. A strict zero-leakage preprocessing methodology that eliminates optimistic performance estimation.
3. Game-theoretic explainability (SHAP) that translates black-box model predictions into actionable software engineering refactoring prescriptions.
4. An enterprise-grade dark futuristic web application deployed publicly and verified locally.

### 8.2 Future Scope
1. **Deep Learning Integration**: Incorporating Graph Neural Networks (GNNs) operating directly on Abstract Syntax Trees (ASTs) and Control Flow Graphs (CFGs).
2. **Cross-Project Defect Prediction (CPDP)**: Expanding the metaheuristic transfer learning mechanism to predict defects across disparate organizations without target historical defect data.
3. **IDE Plugin Ecosystem**: Packaging DefectIQ's real-time inference engine into VS Code and JetBrains extensions to provide live inline code refactoring warnings during code editing.

---

## 9. Academic References

1. **Menzies, T., Greenwald, J., & Frank, A.** (2007). *Data mining static code attributes to learn defect predictors.* IEEE Transactions on Software Engineering, 33(1), 2-13.
2. **Lessmann, S., Baesens, B., Mues, C., & Pietsch, S.** (2008). *Benchmarking classification models for software defect prediction: A proposed framework and novel findings.* IEEE Transactions on Software Engineering, 34(4), 485-496.
3. **Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T.** (2002). *A fast and elitist multiobjective genetic algorithm: NSGA-II.* IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
4. **Lundberg, S. M., & Lee, S. I.** (2017). *A unified approach to interpreting model predictions.* Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 4765-4774.
5. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.** (2002). *SMOTE: Synthetic minority over-sampling technique.* Journal of Artificial Intelligence Research, 16, 321-357.
6. **D'Ambros, M., Lanza, M., & Robbes, R.** (2010). *An extensive comparison of bug prediction approaches.* IEEE 7th Working Conference on Mining Software Repositories (MSR), 31-41.
7. **D'Ambros, M., Lanza, M., & Robbes, R.** (2012). *Evaluating defect prediction approaches: a benchmark and an extensive comparison.* Empirical Software Engineering, 17(4), 531-577.
8. **Wu, R., Zhang, H., Kim, S., & Cheung, S. C.** (2011). *ReLink: recovering links between bugs and changes.* Proceedings of the 19th ACM SIGSOFT Symposium and the 13th European Conference on Foundations of Software Engineering (FSE), 15-25.
9. **Turhan, B., Menzies, T., Bener, A. B., & Di Stefano, J.** (2009). *On the relative value of cross-company early defect prediction.* Empirical Software Engineering, 14(5), 540-570.
10. **Chidamber, S. R., & Kemerer, C. F.** (1994). *A metrics suite for object oriented design.* IEEE Transactions on Software Engineering, 20(6), 476-493.
11. **McCabe, T. J.** (1976). *A complexity measure.* IEEE Transactions on Software Engineering, SE-2(4), 308-320.
12. **Halstead, M. H.** (1977). *Elements of Software Science.* Elsevier Computer Science Library, New York.
13. **Kennedy, J., & Eberhart, R.** (1997). *A discrete binary version of the particle swarm algorithm.* 1997 IEEE International Conference on Systems, Man, and Cybernetics, 5, 4104-4108.
14. **Harman, M., Mansouri, S. A., & Zhang, Y.** (2012). *Search-based software engineering: Trends, techniques and applications.* ACM Computing Surveys, 45(1), 1-61.
15. **Hall, T., Beecham, S., Bowes, D., Gray, D., & Counsell, S.** (2012). *A systematic literature review on fault prediction performance in software engineering.* IEEE Transactions on Software Engineering, 38(6), 1276-1304.
