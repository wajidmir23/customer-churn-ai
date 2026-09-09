# Customer Churn Prediction & Explainable AI

A portfolio ready Streamlit application that identifies customers likely to leave a telecommunications service and makes every prediction easier to act on with SHAP explanations.

The project is deliberately built around the retention use case: a missed churner can be more costly than a retention offer sent to a customer who stays. The selected model is therefore chosen by held out test recall and F2 score before ROC-AUC and F1—not by accuracy alone.

## What it does

- Explores customer churn patterns through interactive charts.
- Trains and compares Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting classifiers.
- Predicts a manually entered customer's churn probability, churn/stay prediction, and low/medium/high risk level.
- Explains each prediction with factors increasing and decreasing churn risk.
- Displays global SHAP importance to show the most influential features across the dataset.

## Dataset

This project includes the public [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

- 7,043 customer records and 21 source columns
- Target: `Churn` (`Yes` = 1, `No` = 0)
- 19 model features after removing the identifier `customerID`
- Numerical features: `tenure`, `MonthlyCharges`, `TotalCharges`
- Categorical features: service, contract, billing, payment, and demographic fields (including the binary `SeniorCitizen` field)
- Class distribution: 1,869 churners (26.5%) and 5,174 non-churners (73.5%)
- Missing values: 11 blank `TotalCharges` values; all belong to zero tenure customers and are median imputed inside the fitted training pipeline

## Architecture

```text
Telco CSV
   │
   ├── src/data_preprocessing.py ──► leakage-safe ColumnTransformer
   │                                     ├── numeric: median impute + scale
   │                                     └── categorical: mode impute + one-hot encode
   │
   └── src/train_model.py ──► four pipelines ──► held-out evaluation
                                                      │
                                               selected model + metrics + SHAP artifacts
                                                      │
                                                app.py (Streamlit dashboard)
```

The preprocessing object is fitted only on the training split and is stored inside the final scikit-learn pipeline. This means exactly the same learned transformations are applied in training and in the web app, with no data leakage from the test set.

## Machine learning approach

The dataset is split with a stratified 80/20 train/test split (`random_state=42`). The following candidates use the same preprocessing pipeline:

1. Logistic Regression — transparent linear baseline with balanced class weights.
2. Decision Tree — bounded-depth, balanced tree baseline.
3. Random Forest — bagged tree ensemble with balanced class weights.
4. Gradient Boosting — sequential tree ensemble; positive training samples receive a balancing weight.

For every model, the project records accuracy, precision, recall, F1, F2, ROC-AUC, a confusion matrix, and ROC curve points. F2 weights recall more strongly than F1, matching the goal of finding prospective churners.

## Latest reproducible model comparison

These are real results from `python -m src.train_model` on the included data's held out test set using a fixed random seed. Small differences can occur after upgrading scientific Python dependencies.

| Model | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.737 | 0.503 | 0.783 | 0.613 | 0.705 | 0.841 |
| Decision Tree | 0.740 | 0.507 | 0.794 | 0.619 | 0.713 | 0.834 |
| Random Forest | 0.761 | 0.534 | 0.767 | 0.630 | 0.706 | 0.840 |
| **Gradient Boosting (selected)** | **0.742** | **0.509** | **0.797** | **0.621** | **0.716** | **0.846** |

The Gradient Boosting model was selected because it has the highest recall (79.7%) and F2 score (0.716), while also having the best ROC-AUC. Its lower accuracy than Random Forest is an acceptable trade off for catching more customers who actually churn.

## Explainable AI with SHAP

The application calls a SHAP explainer on the selected estimator after applying the fitted preprocessing step. Since one hot encoding expands categorical columns, the code then aggregates SHAP values back to original fields such as `Contract` and `PaymentMethod`.

For an individual customer, the **Explain Prediction** page shows:

- Churn probability and low/medium/high risk level
- Feature effects that increase churn risk
- Feature effects that decrease churn risk
- A signed SHAP bar chart

The global explanation ranks fields by average absolute SHAP impact. In the current trained model, the leading drivers are Contract, tenure, InternetService, OnlineSecurity, TechSupport, PaymentMethod, and MonthlyCharges. SHAP explains the model's learned associations; it does not establish causal effects.

## Project structure

```text
customer-churn-ai/
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Cloud/local dependencies
├── README.md
├── .streamlit/config.toml         # Dashboard theme and headless setting
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── best_churn_model.joblib    # Final fitted preprocessing + model pipeline
│   ├── metadata.json
│   ├── model_comparison.csv
│   ├── roc_curves.json
│   └── global_feature_importance.csv
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── explain.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_prediction.py
└── assets/                        # Optional README screenshots/assets
```

## Install and run locally

Use Python 3.10+ (Python 3.11 or 3.12 is a good choice for broad package compatibility).

```bash
git clone https://github.com/wajidmir23/customer-churn-ai.git
cd customer-churn-ai

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train or retrain the model and generate all dashboard artifacts:

```bash
python -m src.train_model
```

Launch the web application:

```bash
streamlit run app.py
```

Run the automated checks:

```bash
pytest -q
```

## Deployment: Streamlit Community Cloud

1. Push the full project, including `data/` and the generated files in `models/`, to GitHub.
2. In Streamlit Community Cloud, choose the repository, branch, and `app.py` as the entry point.
3. The provided `requirements.txt` is detected automatically. No secrets or external runtime data download are required.
4. Deploy. The saved `best_churn_model.joblib` lets the dashboard start without retraining.

When changing model code, run `python -m src.train_model`, review the refreshed `models/` artifacts, and commit them with the code change.

## Screenshots

After launching the application locally, capture the Dashboard, EDA, Prediction, and Explain Prediction pages and save portfolio screenshots under `assets/`. Suggested filenames are `dashboard.png`, `eda.png`, `prediction.png`, and `explanation.png`.

## Interview talking points

- **Why a Pipeline?** It binds learned preprocessing and the estimator together, preventing train/app transformation drift and leakage.
- **Why not accuracy alone?** The 26.5% churn class is smaller. A model can earn high accuracy by overlooking churners, so recall and F2 are prioritised.
- **Why SHAP after preprocessing?** Tree models consume numeric one hot encoded arrays. SHAP explains those actual inputs, then the project groups category contributions into human-readable source fields.
- **What is the decision threshold?** The current dashboard uses 0.50. In production it should be tuned against campaign capacity, retention offer cost, and the value of saving a customer.
- **What would you improve?** Add threshold/cost optimisation, probability calibration, cross validation, fairness monitoring, feature/data-drift checks, experiment tracking, and a secure batch-scoring workflow.

## Important technical decisions

- `TotalCharges` blanks are converted to missing values, then imputed only after splitting—never dropped silently.
- `OneHotEncoder(handle_unknown="ignore")` permits an unseen category at app inference without crashing.
- The final model is saved with Joblib as one fitted object, so the app does not need to rebuild preprocessing.
- Tests cover cleaning, leakage safe transformer fitting, missing-input rejection, binary predictions, and probability bounds.
