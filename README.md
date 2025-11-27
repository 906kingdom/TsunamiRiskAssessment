# 🌊 Tsunami Risk Assessment Using Machine Learning

A comprehensive machine learning project for predicting tsunami occurrence based on earthquake characteristics. This project emphasizes **minimizing false negatives** (missed tsunami predictions) as the primary objective, critical for early warning systems and public safety.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Feature Engineering](#-feature-engineering)
- [Installation](#-installation)
- [Usage](#-usage)
- [References](#-references)

---

## 🎯 Overview

This project develops machine learning models to predict whether an earthquake will generate a tsunami. The key challenge is the **high cost of false negatives** — failing to predict a tsunami can have catastrophic consequences. Therefore, the **F2 score** is used as the primary evaluation metric, which emphasizes recall over precision.

### Problem Statement

Given earthquake parameters (magnitude, depth, location, intensity measures), predict whether the earthquake will trigger a tsunami.

### Why F2 Score?

- **False negatives are catastrophic**: Missing a tsunami prediction can cost lives
- **False positives are manageable**: Extra warnings cause inconvenience but save lives
- **F2 score weights recall 4x more than precision**: Optimizes for catching as many tsunamis as possible

---

## 🏆 Key Findings

### Best Performing Model: **LightGBM (Optimized)**

| Metric                        | Value  |
| ----------------------------- | ------ |
| **F2 Score**            | 0.7609 |
| **Recall**              | 82.45% |
| **Accuracy**            | 83.00% |
| **ROC-AUC**             | 0.828  |
| **False Negative Rate** | 17.53% |

### Model Comparison (Ranked by F2 Score)

| Rank | Model                | F2 Score | Recall | Accuracy | False Negative Rate |
| ---- | -------------------- | -------- | ------ | -------- | ------------------- |
| 1    | LightGBM (Optimized) | 0.7609   | 82.45% | 83.00%   | 17.53%              |
| 2    | CatBoost (Optimized) | 0.7477   | 80.54% | 83.00%   | 19.48%              |
| 3    | Neural Network (MLP) | 0.7413   | 82.52% | 80.00%   | 17.53%              |
| 4    | Logistic Regression  | 0.7215   | 81.87% | 77.29%   | 18.18%              |
| 5    | TabPFNClassifier     | 0.6316   | 59.26% | 88.29%   | 40.74%              |

### Key Insights

1. **Magnitude is the strongest predictor** (correlation: 0.459 with tsunami occurrence)
2. **Shallow earthquakes are more tsunamigenic** — depth shows negative correlation (-0.167)
3. **Magnitude × Depth interaction is crucial** — shallow + high magnitude = highest risk
4. **Distance to coastline** is a meaningful engineered feature (correlation: -0.14)
5. **Station coverage features (nst, dmin)** are temporal artifacts and should be excluded

---

## 📁 Project Structure

```
TsunamiRiskAssessment/
├── 📁 data/
│   ├── 📁 raw/                          # Original datasets
│   │   └── 📁 1/
│   │       ├── earthquake_data_tsunami.csv
│   │       └── tsunami_dataset.csv
│   └── 📁 processed/                    # Cleaned & engineered data
│       ├── earthquake_data_tsunami_updated.csv
│       ├── earthquake_data_tsunami_validated.csv
│       └── earthquake_data_tsunami_scaled.csv
│
├── 📁 notebooks/
│   ├── 01_collect_data.ipynb            # Data collection from Kaggle
│   ├── 02_explanatory_data_analysis.ipynb # EDA & visualization
│   ├── 03_feature_engineering.ipynb     # Feature creation & scaling
│   ├── 04_A_linearRegression.ipynb      # Logistic Regression baseline
│   ├── 04_B_TabPFN.ipynb               # TabPFN classifier/regressor
│   ├── 04_C_XGBoost.ipynb              # XGBoost implementation
│   ├── 04_D_LightGBM.ipynb             # LightGBM implementation
│   ├── 04_E_CatBoost.ipynb             # CatBoost implementation
│   ├── 04_F_RandomForest.ipynb         # Random Forest implementation
│   ├── 04_G_NeuralNetwork.ipynb        # PyTorch MLP implementation
│   ├── 05_feature_importance_analysis.ipynb # Permutation importance
│   ├── 06_hyperparameter_optimization.ipynb # RandomizedSearchCV tuning
│   ├── calculate_distance_coastline.py  # Coastline distance utility
│   ├── calculate_ocean.py              # Ocean detection utility
│   └── data_merge.ipynb                # Dataset merging & validation
│
├── 📁 models/
│   ├── best_hyperparameters.json       # Optimized hyperparameters
│   ├── model_results.csv               # All model evaluation results
│   └── 📁 savedModels/                 # Serialized model files
│
├── 📁 reports/                          # Generated analysis reports
│
├── requirements.txt                     # Python dependencies
├── pyproject.toml                       # Project configuration
└── README.md                            # This file
```

---

## 📊 Dataset

### Source

- **Primary**: [Global Earthquake Tsunami Risk Assessment Dataset](https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset) (Kaggle)
- **Secondary**: [Tsunami Dataset](https://www.kaggle.com/datasets/andrewmvd/tsunami-dataset) (Kaggle) — used for validation and data correction

### Features

| Feature       | Description                                                      | Type          |
| ------------- | ---------------------------------------------------------------- | ------------- |
| `magnitude` | Earthquake strength (Richter scale, log scale)                   | Continuous    |
| `cdi`       | Community Decimal Intensity (crowd-sourced felt intensity)       | Ordinal (0-9) |
| `mmi`       | Modified Mercalli Intensity (damage-based intensity)             | Ordinal (1-9) |
| `sig`       | USGS significance score (combines magnitude + felt + impact)     | Continuous    |
| `depth`     | Hypocenter depth in km (shallow events are riskier)              | Continuous    |
| `latitude`  | North-south position (subduction zone patterns)                  | Continuous    |
| `longitude` | East-west position (Pacific Rim patterns)                        | Continuous    |
| `gap`       | Azimuthal station coverage gap (oceanic events have larger gaps) | Continuous    |
| `nst`       | Number of stations used for magnitude calculation                | Discrete      |
| `dmin`      | Distance to nearest seismic station                              | Continuous    |
| `Year`      | Calendar year                                                    | Discrete      |
| `Month`     | Calendar month                                                   | Discrete      |
| `tsunami`   | **Target variable** (0 = No Tsunami, 1 = Tsunami)          | Binary        |

### Engineered Features

| Feature                  | Description                                  | Importance                |
| ------------------------ | -------------------------------------------- | ------------------------- |
| `distance_to_coast_km` | Distance from epicenter to nearest coastline | High (correlation: -0.14) |
| `month_number`         | Absolute month number since 2000             | Low (discarded)           |

### Dataset Statistics

- **Total samples**: 700 earthquakes
- **Class distribution**: 546 non-tsunami (78%) / 154 tsunami (22%)
- **Time range**: 2001-2020
- **Magnitude range**: 6.5-9.1

### Data Quality Improvements

The original dataset had a **data collection artifact** where Year was strongly correlated with tsunami labels (r=0.65). This was corrected through:

1. **Dataset Merging**: Cross-referenced with authoritative tsunami database
2. **Adaptive Tolerance Matching**: Used research-based tolerances for location, magnitude, and depth
3. **Confidence Scoring**: Only high-confidence matches (≥80%) retained
4. **Reverse Validation**: Verified existing tsunami labels against external data

---

## 🔬 Methodology

### 1. Data Collection & Cleaning

- Downloaded datasets from Kaggle using `kagglehub`
- Identified and corrected temporal data artifact
- Merged with authoritative tsunami database using adaptive tolerance matching

### 2. Exploratory Data Analysis

- Analyzed class imbalance (22% positive class)
- Identified key correlations (magnitude: 0.46, depth: -0.17)
- Discovered magnitude × depth interaction effect
- Visualized geographic patterns of tsunami occurrence

### 3. Feature Engineering

- **Created**: `distance_to_coast_km` using coastline distance calculation
- **Dropped**: `nst`, `dmin` (temporal artifacts), `Year`, `Month` (leakage risk), `latitude`, `longitude` (replaced by distance feature)
- **Scaling**: Two-step pipeline (PowerTransformer + StandardScaler) to handle skewed distributions and meaningful zeros

### 4. Model Development

Implemented and compared 7 different algorithms:

| Model                | Approach          | Key Characteristics                  |
| -------------------- | ----------------- | ------------------------------------ |
| Logistic Regression  | Linear            | Interpretable baseline               |
| TabPFN               | Transformer       | Zero-shot learning, foundation model |
| XGBoost              | Gradient Boosting | Industry standard                    |
| LightGBM             | Gradient Boosting | Fast, efficient                      |
| CatBoost             | Gradient Boosting | Handles categorical features         |
| Random Forest        | Ensemble          | Robust to overfitting                |
| Neural Network (MLP) | Deep Learning     | Non-linear patterns                  |

### 5. Handling Class Imbalance

- **Class weights**: `balanced` for logistic regression
- **Scale_pos_weight**: 3.55 (ratio of negative to positive samples) for tree-based models
- **pos_weight**: 3.55 for neural network (BCEWithLogitsLoss)

### 6. Evaluation Strategy

- **5-fold Stratified Cross-Validation**: Preserves class ratio in each fold
- **Primary Metric**: F2 Score (emphasizes recall)
- **Secondary Metrics**: Recall, Accuracy, ROC-AUC, False Negative Rate
- **Train/Test Gap**: Monitored for overfitting

### 7. Hyperparameter Optimization

Used **RandomizedSearchCV** with 50 iterations per model:

```
CatBoost: depth, learning_rate, iterations, l2_leaf_reg, subsample, min_data_in_leaf
LightGBM: max_depth, learning_rate, n_estimators, num_leaves, subsample, colsample_bytree
Logistic Regression: C, penalty, solver, class_weight
Neural Network: hidden_dim, dropout_rate, learning_rate, batch_size, epochs
```

---

## 📈 Models & Results

### Best Model: LightGBM (Optimized)

```python
# Optimized Hyperparameters
{
    'subsample': 0.7,
    'reg_lambda': 0.1,
    'reg_alpha': 0.5,
    'num_leaves': 15,
    'n_estimators': 400,
    'min_child_samples': 20,
    'max_depth': 4,
    'learning_rate': 0.01,
    'colsample_bytree': 0.9
}
```

### Performance Metrics (All Optimized Models)

| Model               | Accuracy | Precision | Recall           | F1    | F2              | ROC-AUC | FN Rate          | Gap             |
| ------------------- | -------- | --------- | ---------------- | ----- | --------------- | ------- | ---------------- | --------------- |
| LightGBM            | 83.00%   | 58.28%    | 82.45%           | 0.682 | **0.761** | 0.828   | **17.53%** | 5.14%           |
| CatBoost            | 83.00%   | 58.39%    | 80.54%           | 0.676 | 0.748           | 0.821   | 19.48%           | 2.93%           |
| Neural Network      | 80.00%   | 53.24%    | **82.52%** | 0.645 | 0.741           | 0.809   | **17.53%** | 1.00%           |
| Logistic Regression | 77.29%   | 49.33%    | 81.87%           | 0.614 | 0.722           | 0.789   | 18.18%           | **0.39%** |

### Feature Importance (via Permutation)

| Rank | Feature                  | Importance |
| ---- | ------------------------ | ---------- |
| 1    | `magnitude`            | 0.285      |
| 2    | `depth`                | 0.142      |
| 3    | `sig`                  | 0.098      |
| 4    | `distance_to_coast_km` | 0.071      |
| 5    | `cdi`                  | 0.056      |
| 6    | `mmi`                  | 0.034      |

---

## 🛠️ Feature Engineering

### Distance to Coastline

A custom feature was engineered to capture the relationship between earthquake location and tsunami potential:

```python
from calculate_distance_coastline import get_distance_to_coast_ddm
from calculate_ocean import is_ocean

# Calculate distance to nearest coastline (0 if in ocean)
df["distance_to_coast_km"] = df.apply(
    lambda r: get_distance_to_coast_ddm(r["latitude"], r["longitude"]) 
    if not is_ocean(r["latitude"], r["longitude"]) else 0, 
    axis=1
)
```

**Result**: Strong negative correlation (-0.14) with tsunami occurrence — earthquakes closer to coast are more likely to generate tsunamis.

### Scaling Pipeline

To handle skewed distributions and meaningful zeros:

```python
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline

feature_pipeline = Pipeline(steps=[
    ('power_transform', PowerTransformer(method='yeo-johnson')),  # Handles zeros
    ('scaler', StandardScaler())  # Centers and scales
])
```

---

## 💻 Installation

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (optional, for neural networks and TabPFN)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/TsunamiRiskAssessment.git
cd TsunamiRiskAssessment
```

2. **Create virtual environment**

```bash
python -m venv .venv
```

3. **Activate virtual environment**

Windows (PowerShell):

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Dependencies

```
kagglehub==0.3.13
pandas==2.3.3
fuzzywuzzy==0.18.0
matplotlib==3.10.7
seaborn==0.13.2
plotly==6.3.1
nbformat==5.10.4
scikit-learn==1.7.2
tabpfn==6.0.6
xgboost==3.1.1
catboost==1.2.8
lightgbm==4.6.0
torch (latest)
python-dotenv
```

---

## 🚀 Usage

### 1. Data Collection

```bash
# Run the data collection notebook
jupyter notebook notebooks/01_collect_data.ipynb
```

### 2. Run Full Pipeline

Execute notebooks in order:

1. `01_collect_data.ipynb` — Download datasets
2. `data_merge.ipynb` — Merge and validate data
3. `02_explanatory_data_analysis.ipynb` — EDA
4. `03_feature_engineering.ipynb` — Feature engineering
5. `04_*.ipynb` — Train individual models
6. `05_feature_importance_analysis.ipynb` — Analyze features
7. `06_hyperparameter_optimization.ipynb` — Optimize models

### 3. Use Pre-trained Models

```python
import lightgbm as lgb
import json

# Load best hyperparameters
with open('models/best_hyperparameters.json', 'r') as f:
    params = json.load(f)

# Create model with optimized parameters
model = lgb.LGBMClassifier(
    **params['lightgbm']['best_params'],
    scale_pos_weight=3.55,
    verbosity=-1
)

# Train on your data
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 📚 References

[1] N. Hollmann, S. Müller, K. Eggensperger, and F. Hutter, "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second," 2022. [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)

[2] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," *Journal of Machine Learning Research*, vol. 15, pp. 1929–1958, 2014.

[3] Kaggle, "Global Earthquake Tsunami Risk Assessment Dataset" and "Tsunami Dataset by Andrew M.V." Available at: [Kaggle Datasets](https://www.kaggle.com/)

