# Lead Scoring Machine Learning Project

## Overview

This project implements a machine learning-based lead scoring system to predict the likelihood of lead conversion for an education company. The system uses XGBoost to classify leads into different categories (Hot, Warm, Cold) based on their conversion probability scores.

## 🎯 Project Goals

- **Lead Classification**: Categorize leads into Hot (>70% conversion probability), Warm (30-70%), and Cold (<30%)
- **Conversion Prediction**: Predict the probability of lead conversion using historical data
- **Business Intelligence**: Provide insights into lead characteristics and patterns
- **Resource Optimization**: Help sales teams prioritize high-value leads

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.x** - Primary programming language
- **Jupyter Notebooks** - Interactive development and analysis
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - Machine learning framework
- **XGBoost** - Gradient boosting algorithm for classification

### Visualization & Analysis
- **Matplotlib & Seaborn** - Static data visualization
- **Plotly** - Interactive visualizations
- **SciPy** - Statistical analysis

### Model Management & Optimization
- **MLflow** - Experiment tracking and model versioning
- **Optuna** - Hyperparameter optimization
- **Joblib** - Model serialization and persistence

## 📁 Project Structure

```
lead-scoring/
├── data/
│   ├── lead_scoring.csv          # Main dataset (2.3MB)
│   └── cold_df.csv              # Cold leads subset (1.2MB)
├── models/
│   ├── xgb_best_lead_scoring_model.joblib    # Optimized XGBoost model
│   ├── xgb_lead_scoring_model.joblib         # Base XGBoost model
│   └── lead_scoring_feature_names.joblib     # Feature names mapping
├── preprocessor/
│   ├── data_preprocessor.py                   # Custom preprocessing pipeline
│   └── lead_scoring_preprocessor.joblib      # Serialized preprocessor
├── mlruns/                                    # MLflow experiment tracking
├── lead_scoring_analysis_cleaning.ipynb      # Data exploration & cleaning
├── lead_scoring_modelling.ipynb              # Model training & optimization
├── lead_scoring_inference_analysis.ipynb     # Model inference & analysis
├── requirements.txt                           # Python dependencies
└── README.md                                 # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter notebook
jupyter notebook
```

### 2. Data Preparation

The project includes a comprehensive data preprocessing pipeline:

- **Data Cleaning**: Handles missing values, outliers, and data inconsistencies
- **Feature Engineering**: Creates new features and transforms existing ones
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Numerical Scaling**: Standardization of numerical features

### 3. Model Training

The project implements a sophisticated machine learning pipeline:

1. **Data Splitting**: 80% training, 20% testing with stratification
2. **Cross-Validation**: 5-fold stratified cross-validation
3. **Hyperparameter Optimization**: Using Optuna with MLflow tracking
4. **Model Evaluation**: Multiple metrics including ROC-AUC, F1-score, Precision, Recall

## 📊 Model Performance

### XGBoost Model Results

| Metric | Value |
|--------|-------|
| **ROC-AUC Score** | 0.8827 |
| **Cross-Validation F1 Mean** | 0.769 |
| **Precision** | 0.887 |
| **Recall** | 0.855 |
| **F1 Score** | 0.871 |

### Lead Classification Distribution

- **Hot Leads** (>70% conversion probability): 2,637 leads
- **Warm Leads** (30-70% conversion probability): 1,693 leads  
- **Cold Leads** (<30% conversion probability): 4,910 leads

## 🔧 Key Features

### Custom Preprocessor (`LeadScoringPreprocessor`)

The project includes a custom preprocessing pipeline that handles:

- **Feature Engineering**: Lead source consolidation, activity categorization
- **Data Cleaning**: Removal of irrelevant columns, handling of 'Select' values
- **Categorical Processing**: One-hot encoding with feature name preservation
- **Numerical Processing**: Standardization and imputation
- **Pipeline Integration**: Seamless integration with scikit-learn pipelines

### Hyperparameter Optimization

Using Optuna for automated hyperparameter tuning:

- **Search Space**: Learning rate, max depth, n_estimators, regularization parameters
- **Optimization Metric**: ROC-AUC score
- **Trials**: 50 optimization trials
- **Best Parameters**:
  - `learning_rate`: 0.032
  - `max_depth`: 4
  - `n_estimators`: 272
  - `colsample_bytree`: 0.3
  - `subsample`: 1.0

## 📈 Model Insights

### Top Features by Importance

The model identifies key factors influencing lead conversion:

1. **Total Time Spent on Website** - Most important feature
2. **Lead Source** - Origin of the lead
3. **Total Visits** - Number of website visits
4. **Page Views Per Visit** - Engagement level
5. **Lead Origin** - How the lead was acquired
6. **Last Activity** - Recent engagement
7. **Specialization** - Course interest area
8. **Current Occupation** - Professional background

### Business Applications

- **Lead Prioritization**: Focus resources on high-scoring leads
- **Campaign Optimization**: Target similar characteristics in marketing campaigns
- **Resource Allocation**: Allocate sales resources based on conversion probability
- **Performance Tracking**: Monitor lead quality over time

## 🔄 Usage Workflow

### 1. Data Loading
```python
import pandas as pd
df = pd.read_csv('data/lead_scoring.csv')
```

### 2. Model Loading
```python
import joblib

# Load trained model and preprocessor
model = joblib.load('models/xgb_best_lead_scoring_model.joblib')
preprocessor = joblib.load('preprocessor/lead_scoring_preprocessor.joblib')
```

### 3. Prediction
```python
# Preprocess new data
X_transformed = preprocessor.transform(new_data)

# Get conversion probabilities
probabilities = model.predict_proba(X_transformed)[:, 1]
lead_scores = probabilities * 100

# Classify leads
lead_types = pd.cut(lead_scores, 
                   bins=[-float('inf'), 30, 70, float('inf')],
                   labels=['Cold', 'Warm', 'Hot'])
```

## 📋 Dependencies

```
pandas
numpy
matplotlib
seaborn
plotly
scipy
scikit-learn
jupyter
xgboost
joblib
optuna
mlflow
```
