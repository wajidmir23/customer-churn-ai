"""Streamlit dashboard for Customer Churn Prediction & Explainable AI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (  # noqa: E402
    TARGET_COLUMN,
    clean_data,
    get_feature_columns,
    load_raw_data,
    validate_customer_input,
)
from src.explain import (  # noqa: E402
    aggregate_contributions,
    explain_rows,
    transform_features,
)


MODELS_DIR = PROJECT_ROOT / "models"
st.set_page_config(page_title="ChurnScope AI", page_icon="📉", layout="wide")


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_raw_data()


@st.cache_resource(show_spinner=False)
def get_artifacts():
    model_path = MODELS_DIR / "best_churn_model.joblib"
    metadata_path = MODELS_DIR / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        return None
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    return joblib.load(model_path), metadata


@st.cache_data(show_spinner=False)
def get_csv_artifact(name: str) -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / name)


def percentage(value: float) -> str:
    return f"{value:.1%}"


def churn_rate_by(data: pd.DataFrame, column: str) -> pd.DataFrame:
    result = (
        data.assign(churned=data[TARGET_COLUMN].eq("Yes"))
        .groupby(column, observed=False)["churned"]
        .agg(churn_rate="mean", customers="size")
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )
    return result


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.60:
        return "High", "🔴"
    if probability >= 0.35:
        return "Medium", "🟠"
    return "Low", "🟢"


def customer_form(data: pd.DataFrame, feature_columns: list[str], form_key: str) -> pd.DataFrame | None:
    """Render the model's feature fields and return a one-row raw payload."""
    defaults = clean_data(data).drop(columns=TARGET_COLUMN).median(numeric_only=True)
    choices = {
        column: sorted(data[column].dropna().astype(str).unique().tolist())
        for column in feature_columns
        if column not in {"tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"}
    }
    with st.form(form_key):
        st.caption("Enter a customer profile. All fields match the model's training data.")
        left, right = st.columns(2)
        values: dict[str, object] = {}
        for index, column in enumerate(feature_columns):
            container = left if index % 2 == 0 else right
            with container:
                label = " ".join(
                    part.capitalize() for part in column.replace("_", " ").split()
                )
                if column == "tenure":
                    values[column] = st.number_input(label, 0, 72, 12, 1)
                elif column == "MonthlyCharges":
                    values[column] = st.number_input(label, 0.0, 150.0, float(defaults[column]), 0.5)
                elif column == "TotalCharges":
                    values[column] = st.number_input(label, 0.0, 10_000.0, float(defaults[column]), 1.0)
                elif column == "SeniorCitizen":
                    values[column] = 1 if st.selectbox(label, ["No", "Yes"], key=f"{form_key}_{column}") == "Yes" else 0
                else:
                    values[column] = st.selectbox(label, choices[column], key=f"{form_key}_{column}")
        submitted = st.form_submit_button("Analyze churn risk", width="stretch")
    return pd.DataFrame([values]) if submitted else None


def show_prediction(model, metadata: dict, customer: pd.DataFrame) -> tuple[float, int]:
    payload = validate_customer_input(customer, metadata["feature_columns"])
    probability = float(model.predict_proba(payload)[0, 1])
    prediction = int(probability >= metadata["decision_threshold"])
    level, icon = risk_label(probability)
    first, second, third = st.columns(3)
    first.metric("Churn probability", percentage(probability))
    second.metric("Prediction", "Likely to churn" if prediction else "Likely to stay")
    third.metric("Risk level", f"{icon} {level}")
    return probability, prediction


def render_dashboard(data: pd.DataFrame, metadata: dict | None) -> None:
    st.title("Customer Churn Prediction & Explainable AI")
    st.caption("Identify at-risk customers early, then understand the drivers behind each prediction.")
    churn_rate = data[TARGET_COLUMN].eq("Yes").mean()
    total_missing = int(data.replace(r"^\s*$", pd.NA, regex=True).isna().sum().sum())
    one, two, three, four = st.columns(4)
    one.metric("Total customers", f"{len(data):,}")
    two.metric("Overall churn rate", percentage(churn_rate))
    three.metric("Raw features", len(data.columns) - 2)  # Churn and customer ID
    four.metric("Missing values", total_missing)

    left, right = st.columns((1, 1.15))
    with left:
        distribution = data[TARGET_COLUMN].value_counts().rename_axis("Churn").reset_index(name="Customers")
        fig = px.pie(distribution, names="Churn", values="Customers", hole=0.58,
                     color="Churn", color_discrete_map={"Yes": "#e05260", "No": "#38a169"})
        fig.update_layout(title="Churn class distribution", margin=dict(t=50, b=0, l=0, r=0), legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Dataset overview")
        st.write(
            "The IBM Telco Customer Churn dataset includes subscription, service, billing, "
            "and customer-demographic fields. The target is `Churn` (Yes/No)."
        )
        st.dataframe(data.head(8), width="stretch", hide_index=True)
        if metadata:
            comparison = get_csv_artifact("model_comparison.csv")
            best = comparison.loc[comparison["model"] == metadata["best_model"]].iloc[0]
            st.info(
                f"Selected model: **{metadata['best_model']}** · test recall **{best['recall']:.1%}** · "
                f"ROC-AUC **{best['roc_auc']:.3f}**. Selection prioritises finding potential churners."
            )
    cleaned = clean_data(data)
    feature_frame = cleaned.drop(columns=TARGET_COLUMN)
    numerical, categorical = get_feature_columns(feature_frame)
    with st.expander("Data dictionary and quality checks"):
        first, second = st.columns(2)
        with first:
            st.write("**Target:** `Churn` — Yes means the customer left; No means they stayed.")
            st.write("**Numerical columns:** " + ", ".join(f"`{column}`" for column in numerical))
            st.write("**Categorical columns:** " + ", ".join(f"`{column}`" for column in categorical))
        with second:
            missing = cleaned.isna().sum()
            missing = missing[missing > 0]
            if missing.empty:
                st.success("No missing values after standardisation.")
            else:
                st.warning(
                    "Missing values after standardisation: "
                    + ", ".join(f"`{column}` ({count})" for column, count in missing.items())
                    + ". The pipeline imputes these values after splitting."
                )
            st.write(
                f"**Class balance:** {int(data[TARGET_COLUMN].eq('Yes').sum()):,} churners "
                f"({percentage(churn_rate)}) and {int(data[TARGET_COLUMN].eq('No').sum()):,} non-churners."
            )


def render_eda(data: pd.DataFrame) -> None:
    st.title("Exploratory Data Analysis")
    st.caption("Use these patterns to target retention offers and investigate churn drivers.")

    contract = churn_rate_by(data, "Contract")
    payment = churn_rate_by(data, "PaymentMethod")
    service = churn_rate_by(data, "InternetService")
    tenure_data = data.assign(
        tenure_band=pd.cut(data["tenure"], bins=[-1, 6, 12, 24, 48, 72],
                           labels=["0–6", "7–12", "13–24", "25–48", "49–72"])
    )
    charges_data = data.assign(
        # Plotly serialises strings reliably; pandas Interval objects are not JSON serialisable.
        charge_band=pd.qcut(data["MonthlyCharges"], q=5, duplicates="drop").astype(str)
    )

    row_one_left, row_one_right = st.columns(2)
    with row_one_left:
        figure = px.bar(contract, x="Contract", y="churn_rate", text=contract["churn_rate"].map(percentage),
                        color="churn_rate", color_continuous_scale="Reds")
        figure.update_layout(title="Churn rate by contract type", coloraxis_showscale=False, yaxis_tickformat=".0%")
        st.plotly_chart(figure, width="stretch")
    with row_one_right:
        figure = px.bar(churn_rate_by(tenure_data, "tenure_band"), x="tenure_band", y="churn_rate",
                        color="churn_rate", color_continuous_scale="Reds")
        figure.update_layout(title="Churn rate by tenure", coloraxis_showscale=False, yaxis_tickformat=".0%", xaxis_title="Months")
        st.plotly_chart(figure, width="stretch")

    row_two_left, row_two_right = st.columns(2)
    with row_two_left:
        figure = px.bar(churn_rate_by(charges_data, "charge_band"), x="charge_band", y="churn_rate",
                        color="churn_rate", color_continuous_scale="Reds")
        figure.update_layout(title="Churn rate by monthly charges", coloraxis_showscale=False, yaxis_tickformat=".0%", xaxis_title="Monthly charge band")
        st.plotly_chart(figure, width="stretch")
    with row_two_right:
        figure = px.bar(payment, x="PaymentMethod", y="churn_rate", color="churn_rate", color_continuous_scale="Reds")
        figure.update_layout(title="Churn rate by payment method", coloraxis_showscale=False, yaxis_tickformat=".0%", xaxis_title="")
        figure.update_xaxes(tickangle=-25)
        st.plotly_chart(figure, width="stretch")

    figure = px.bar(service, x="InternetService", y="churn_rate", color="churn_rate", color_continuous_scale="Reds")
    figure.update_layout(title="Churn rate by internet service", coloraxis_showscale=False, yaxis_tickformat=".0%")
    st.plotly_chart(figure, width="stretch")
    st.info(
        "Interpret these charts as associations, not proof of causation. The prediction model combines all "
        "available fields for each customer."
    )


def render_prediction(data: pd.DataFrame, model, metadata: dict | None) -> None:
    st.title("Predict Customer Churn")
    if not metadata:
        st.error("No trained model found. Run `python -m src.train_model` from the project root first.")
        return
    customer = customer_form(data, metadata["feature_columns"], "prediction_form")
    if customer is not None:
        show_prediction(model, metadata, customer)
        st.caption("The default decision threshold is 50%. Adjust it only after validating the business cost of retention actions.")


def render_explain_prediction(data: pd.DataFrame, model, metadata: dict | None) -> None:
    st.title("Explain a Prediction")
    st.caption("SHAP values show which fields pushed this individual prediction toward or away from churn.")
    if not metadata:
        st.error("No trained model found. Run `python -m src.train_model` from the project root first.")
        return
    customer = customer_form(data, metadata["feature_columns"], "explanation_form")
    if customer is not None:
        probability, _ = show_prediction(model, metadata, customer)
        background = transform_features(model, clean_data(data).drop(columns=TARGET_COLUMN).sample(200, random_state=42))
        values, transformed_names = explain_rows(model, customer, background=background)
        contributions = aggregate_contributions(values[0], transformed_names, metadata["feature_columns"])
        top = contributions.reindex(contributions.abs().sort_values(ascending=False).head(10).index).sort_values()
        plot = px.bar(top.reset_index(name="SHAP value"), x="SHAP value", y="index", orientation="h",
                      color="SHAP value", color_continuous_scale=["#38a169", "#f7fafc", "#e05260"])
        plot.update_layout(title="Feature effects on this churn prediction", yaxis_title="Feature", coloraxis_showscale=False)
        st.plotly_chart(plot, width="stretch")
        increasing = contributions[contributions > 0].sort_values(ascending=False).head(3)
        decreasing = contributions[contributions < 0].sort_values().head(3)
        left, right = st.columns(2)
        with left:
            st.subheader("Factors increasing churn risk")
            if increasing.empty:
                st.write("No individual feature increased the churn score.")
            else:
                for name, value in increasing.items():
                    st.write(f"• {name} (+{value:.3f})")
        with right:
            st.subheader("Factors decreasing churn risk")
            if decreasing.empty:
                st.write("No individual feature decreased the churn score.")
            else:
                for name, value in decreasing.items():
                    st.write(f"• {name} ({value:.3f})")
        st.caption(
            f"This customer has a {percentage(probability)} estimated churn probability. SHAP explains the model, "
            "not a causal reason the customer will leave."
        )


def render_model_comparison(metadata: dict | None) -> None:
    st.title("Model Comparison")
    if not metadata:
        st.error("No trained model found. Run `python -m src.train_model` from the project root first.")
        return
    comparison = get_csv_artifact("model_comparison.csv")
    metric_columns = ["accuracy", "precision", "recall", "f1", "f2", "roc_auc"]
    display = comparison[["model", *metric_columns]].copy()
    for column in metric_columns:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    st.dataframe(display, width="stretch", hide_index=True)
    chart_data = comparison.melt(id_vars="model", value_vars=metric_columns, var_name="Metric", value_name="Score")
    chart = px.bar(chart_data, x="model", y="Score", color="Metric", barmode="group", range_y=[0, 1])
    chart.update_layout(title="Held-out test performance")
    st.plotly_chart(chart, width="stretch")

    with (MODELS_DIR / "roc_curves.json").open(encoding="utf-8") as file:
        curves = json.load(file)
    roc = go.Figure()
    for name, curve in curves.items():
        auc = comparison.loc[comparison["model"] == name, "roc_auc"].iloc[0]
        roc.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=f"{name} (AUC {auc:.3f})"))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
    roc.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
    st.plotly_chart(roc, width="stretch")
    st.info(f"Selected: **{metadata['best_model']}**. {metadata['selection_rule']}.")

    selected_name = st.selectbox("Inspect a model's confusion matrix", comparison["model"].tolist())
    selected = comparison.loc[comparison["model"] == selected_name].iloc[0]
    confusion = [[selected["true_negatives"], selected["false_positives"]],
                 [selected["false_negatives"], selected["true_positives"]]]
    matrix = go.Figure(
        data=go.Heatmap(
            z=confusion,
            x=["Predicted: stay", "Predicted: churn"],
            y=["Actual: stay", "Actual: churn"],
            colorscale="Blues",
            text=confusion,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    matrix.update_layout(title=f"Confusion matrix — {selected_name}", height=420)
    st.plotly_chart(matrix, width="stretch")

    importance = get_csv_artifact("global_feature_importance.csv").head(15).sort_values("mean_abs_shap")
    importance_chart = px.bar(importance, x="mean_abs_shap", y="feature", orientation="h", color="mean_abs_shap", color_continuous_scale="Blues")
    importance_chart.update_layout(title="Overall feature importance (mean absolute SHAP value)", coloraxis_showscale=False, xaxis_title="Average impact magnitude", yaxis_title="")
    st.plotly_chart(importance_chart, width="stretch")


def main() -> None:
    data = get_data()
    artifacts = get_artifacts()
    model, metadata = artifacts if artifacts else (None, None)
    st.sidebar.title("ChurnScope AI")
    page = st.sidebar.radio("Navigate", ["Dashboard", "EDA", "Prediction", "Explain Prediction", "Model Comparison"])
    st.sidebar.caption("Portfolio project · IBM Telco dataset")
    if page == "Dashboard":
        render_dashboard(data, metadata)
    elif page == "EDA":
        render_eda(data)
    elif page == "Prediction":
        render_prediction(data, model, metadata)
    elif page == "Explain Prediction":
        render_explain_prediction(data, model, metadata)
    else:
        render_model_comparison(metadata)


if __name__ == "__main__":
    main()
