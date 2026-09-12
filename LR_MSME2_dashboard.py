"""Dynamic MSME cashflow dashboard for the LR_MSME2 pipeline.

Run from this folder with:
    streamlit run LR_MSME2_dashboard.py

The dashboard intentionally uses deterministic dummy monthly cashflows so the
UI can be demonstrated without exposing or depending on the source portfolio.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "LR_MSME2_best_model.pkl"

st.set_page_config(
    page_title="LR_MSME2 | Dynamic Cashflow Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #000000 !important; background: #ffffff !important;
    }
    .main .block-container { max-width: 98% !important; padding: 1.5rem 2.5rem 3.5rem !important; }
    h1, h2, h3 { color: #1e3a8a !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
    h1 { font-size: 1.6rem !important; }
    h2 { border-bottom: 2px solid #000000; padding-bottom: 0.4rem; }
    .hero { border: 1.5px solid #000000; border-left: 6px solid #1e3a8a; border-radius: 8px;
        padding: 1.1rem 1.4rem; margin-bottom: 1.2rem; background: #ffffff; }
    .hero h1 { margin: 0 0 0.25rem; }
    .hero p { margin: 0; color: #334155; }
    div[data-testid="stMetric"] { background: #ffffff !important; border: 1.5px solid #000000 !important;
        border-radius: 8px !important; padding: 0.85rem 1.1rem !important; }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] label p {
        color: #1e3a8a !important; font-size: 0.75rem !important; font-weight: 700 !important;
        text-transform: uppercase !important; letter-spacing: 0.05em !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #000000 !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] { background: #ffffff !important; border-right: 2px solid #000000 !important; }
    section[data-testid="stSidebar"] * { color: #000000 !important; }
    div[data-baseweb="select"] > div, input, textarea {
        background: #ffffff !important; color: #000000 !important; border-color: #000000 !important;
    }
    div[data-testid="stDataFrame"], div[data-testid="stTable"], div[data-testid="stExpander"] {
        background: #ffffff !important; border: 1.5px solid #000000 !important; border-radius: 6px !important;
    }
    button[kind="primary"] { background: #1e3a8a !important; border-color: #1e3a8a !important; color: #ffffff !important; }
    hr { border: none !important; border-top: 1.5px solid #000000 !important; }
    .risk-low { color: #166534; font-weight: 700; }
    .risk-high { color: #991b1b; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


RAW_CASHFLOW_COLUMNS = [
    "DOST_iFund_Assistance", "Equity_of_Proponent", "Sales", "Depreciation",
    "Total_Cash_Inflow", "Production_Equipment", "Direct_Labor", "Raw_Materials",
    "Manufacturing_Overhead", "Operating_Expenses", "Total_Cash_Outflow",
    "Net_Operating_Inflow_Outflow", "Beginning_Cash_Balance",
    "Cash_Available_for_Amortization", "Principal", "Interest", "Cash_End_of_Period",
]
BASE_NUMERIC_COLUMNS = [
    "Period", "Month_of_Year", *RAW_CASHFLOW_COLUMNS, "Project_Cost",
]
CATEGORICAL_COLUMNS = ["Province", "Sector", "type_of_ownership", "size_of_enterprise"]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def make_dummy_data(seed: int = 42) -> pd.DataFrame:
    """Create a stable, realistic-looking demonstration portfolio."""
    rng = np.random.default_rng(seed)
    sectors = ["Food Processing", "Manufacturing", "Agriculture", "Services"]
    # Region VI (Western Visayas) provinces only.
    provinces = ["Aklan", "Antique", "Capiz", "Guimaras", "Iloilo", "Negros Occidental"]
    ownership = ["Corporation", "Single Proprietorship", "Partnership"]
    sizes = ["micro", "small", "medium"]
    rows = []
    for beneficiary_number in range(1, 61):
        beneficiary = f"Demo MSME {beneficiary_number:03d}"
        status = "Completed" if beneficiary_number % 5 else "Not Completed"
        sector = sectors[(beneficiary_number - 1) % len(sectors)]
        province = provinces[(beneficiary_number - 1) % len(provinces)]
        enterprise_size = sizes[(beneficiary_number - 1) % len(sizes)]
        cost = float(rng.integers(800_000, 4_500_000))
        months = int(rng.integers(12, 25))
        cash_balance = float(rng.integers(100_000, 500_000))
        for period in range(1, months + 1):
            sales = float(rng.normal(260_000 if status == "Completed" else 180_000, 35_000))
            sales = max(sales, 20_000)
            labor = float(rng.integers(35_000, 80_000))
            materials = float(sales * rng.uniform(0.25, 0.45))
            operating = float(rng.integers(25_000, 70_000))
            inflow = sales + (cost if period == 1 else 0)
            outflow = labor + materials + operating + (cost if period == 1 else 0)
            net = inflow - outflow
            cash_balance = max(cash_balance + net, 0)
            rows.append({
                "Beneficiary_Name": beneficiary, "Period": period, "Month_of_Year": period,
                "Province": province, "Sector": sector, "type_of_ownership": ownership[(beneficiary_number - 1) % 3],
                "size_of_enterprise": enterprise_size, "Project_Cost": cost,
                "DOST_iFund_Assistance": cost if period == 1 else 0.0, "Equity_of_Proponent": 0.0,
                "Sales": sales, "Depreciation": float(rng.integers(8_000, 22_000)),
                "Total_Cash_Inflow": inflow, "Production_Equipment": cost if period == 1 else 0.0,
                "Direct_Labor": labor, "Raw_Materials": materials,
                "Manufacturing_Overhead": float(rng.integers(15_000, 35_000)),
                "Operating_Expenses": operating, "Total_Cash_Outflow": outflow,
                "Net_Operating_Inflow_Outflow": net, "Beginning_Cash_Balance": cash_balance - net,
                "Cash_Available_for_Amortization": max(cash_balance * 0.75, 0),
                "Principal": float(rng.integers(20_000, 70_000)), "Interest": float(rng.integers(1_000, 8_000)),
                "Cash_End_of_Period": cash_balance, "Completion_Status": status,
            })
    return pd.DataFrame(rows)


def add_snapshot_features(data: pd.DataFrame) -> pd.DataFrame:
    snapshot = data.sort_values(["Beneficiary_Name", "Period"]).copy()
    for column in RAW_CASHFLOW_COLUMNS:
        grouped = snapshot.groupby("Beneficiary_Name")[column]
        snapshot[f"{column}_cum"] = grouped.cumsum()
        snapshot[f"{column}_rolling3"] = grouped.transform(lambda values: values.rolling(3, min_periods=1).mean())
    snapshot["Cash_Position"] = snapshot["Cash_End_of_Period"] - snapshot["Principal"] - snapshot["Interest"]
    snapshot["Cumulative_Net_Cashflow"] = snapshot.groupby("Beneficiary_Name")[
        "Net_Operating_Inflow_Outflow"
    ].cumsum()
    snapshot["Cashflow_Submission_Number"] = snapshot.groupby("Beneficiary_Name").cumcount() + 1
    return snapshot


def model_categories(model, column: str, fallback: list[str]) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    for name, transformer, columns in preprocessor.transformers_:
        if name == "cat" and column in columns:
            index = list(columns).index(column)
            return [str(value) for value in transformer.named_steps["encoder"].categories_[index]]
    return fallback


def risk_tier(probability: float) -> tuple[str, str]:
    if probability >= 0.8:
        return "Low", "risk-low"
    if probability >= 0.5:
        return "Moderate", ""
    return "High", "risk-high"


try:
    model = load_model()
    portfolio = make_dummy_data()
    snapshots = add_snapshot_features(portfolio)
except Exception as exc:
    st.error(f"Unable to load the dashboard assets: {exc}")
    st.stop()


def predict_rows(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model_features = list(model.named_steps["preprocessor"].feature_names_in_)
    missing = [column for column in model_features if column not in rows.columns]
    if missing:
        raise ValueError(f"Dashboard feature construction is missing: {', '.join(missing)}")
    prepared = rows[model_features].copy()
    predictions = model.predict(prepared)
    probabilities = model.predict_proba(prepared)
    completed_index = list(model.classes_).index("Completed")
    return predictions, probabilities[:, completed_index]


st.markdown(
    """
    <div class="hero">
        <h1>MSME Dynamic Cashflow Dashboard</h1>
        <p>DOST SETUP / iFund portfolio intelligence | LR_MSME2 | Application assessment plus monthly monitoring</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dashboard controls")
    page = st.radio("Navigate", [
        "Executive Overview", "Predict Cashflow Status", "Monitoring Timeline",
        "Portfolio Explorer", "Model Details",
    ])
    st.caption("This demonstration uses generated dummy monthly cashflows.")


if page == "Executive Overview":
    st.header("Executive Overview")
    predictions, probabilities = predict_rows(snapshots)
    completed = int((portfolio["Completion_Status"] == "Completed").sum())
    cols = st.columns(4)
    cols[0].metric("Demo beneficiaries", f"{portfolio['Beneficiary_Name'].nunique():,}")
    cols[1].metric("Cashflow snapshots", f"{len(portfolio):,}")
    cols[2].metric("Eventual completion rate", f"{completed / len(portfolio):.1%}")
    cols[3].metric("Mean current probability", f"{probabilities.mean():.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Eventual project status")
        counts = portfolio["Completion_Status"].value_counts().rename_axis("Status").reset_index(name="Projects")
        st.plotly_chart(px.bar(counts, x="Status", y="Projects", color="Status"), use_container_width=True)
    with right:
        st.subheader("Predicted completion probability")
        st.plotly_chart(px.histogram(pd.DataFrame({"Probability": probabilities}), x="Probability", nbins=12), use_container_width=True)

    st.subheader("Portfolio snapshot")
    latest = snapshots.sort_values("Period").groupby("Beneficiary_Name").tail(1).copy()
    latest["Predicted_Completion_Probability"] = predict_rows(latest)[1]
    st.dataframe(
        latest[["Beneficiary_Name", "Sector", "size_of_enterprise", "Period",
                "Predicted_Completion_Probability", "Completion_Status"]]
        .sort_values("Predicted_Completion_Probability", ascending=False).head(15),
        use_container_width=True, hide_index=True,
    )

elif page == "Predict Cashflow Status":
    st.header("Predict Cashflow Status")
    st.caption("Choose a dummy MSME and the number of monthly cashflows currently available. Period 1 represents the initial application assessment.")
    beneficiary = st.selectbox("Demo beneficiary", sorted(snapshots["Beneficiary_Name"].unique()))
    history = snapshots[snapshots["Beneficiary_Name"] == beneficiary].copy()
    period = st.slider("Cashflow submission available", 1, int(history["Period"].max()), 1)
    current = history[history["Period"] <= period].tail(1)
    prediction, probability = predict_rows(current)
    completed_probability = float(probability[0])
    tier, tier_class = risk_tier(completed_probability)

    result_cols = st.columns(4)
    result_cols[0].metric("Assessment stage", "Initial application" if period == 1 else f"Month {period} update")
    result_cols[1].metric("Predicted status", str(prediction[0]))
    result_cols[2].metric("Completion probability", f"{completed_probability:.1%}")
    result_cols[3].markdown(f"**Risk tier**<br><span class='{tier_class}'>{tier}</span>", unsafe_allow_html=True)
    st.progress(completed_probability, text=f"Completion probability: {completed_probability:.1%}")

    timeline = history.copy()
    timeline["Completion_Probability"] = predict_rows(timeline)[1]
    st.subheader("Prediction as cashflows arrive")
    st.plotly_chart(
        px.line(timeline, x="Period", y="Completion_Probability", markers=True,
                labels={"Completion_Probability": "Completion probability", "Period": "Submission period"}),
        use_container_width=True,
    )
    st.subheader("Current cashflow snapshot")
    st.dataframe(current[BASE_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS], use_container_width=True, hide_index=True)

elif page == "Monitoring Timeline":
    st.header("Monitoring Timeline")
    st.caption("The aggregate view shows how model assessments change from the application snapshot to later monthly submissions.")
    predictions, probabilities = predict_rows(snapshots)
    monitoring = snapshots[["Period", "Beneficiary_Name", "Completion_Status"]].copy()
    monitoring["Predicted_Status"] = predictions
    monitoring["Completion_Probability"] = probabilities
    summary = monitoring.groupby("Period").agg(
        Beneficiaries=("Beneficiary_Name", "nunique"),
        Mean_Completion_Probability=("Completion_Probability", "mean"),
        Actual_Completion_Rate=("Completion_Status", lambda values: (values == "Completed").mean()),
    ).reset_index()
    cols = st.columns(3)
    cols[0].metric("Initial probability", f"{summary.iloc[0]['Mean_Completion_Probability']:.1%}")
    cols[1].metric("Latest probability", f"{summary.iloc[-1]['Mean_Completion_Probability']:.1%}")
    cols[2].metric("Latest submissions", f"{int(summary.iloc[-1]['Beneficiaries']):,}")
    st.plotly_chart(
        px.line(summary, x="Period", y=["Mean_Completion_Probability", "Actual_Completion_Rate"], markers=True,
                labels={"value": "Rate / probability", "variable": "Measure"}),
        use_container_width=True,
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

elif page == "Portfolio Explorer":
    st.header("Portfolio Explorer")
    sector_filter = st.multiselect("Filter by sector", sorted(portfolio["Sector"].unique()))
    status_filter = st.multiselect("Filter by eventual status", sorted(portfolio["Completion_Status"].unique()))
    filtered = portfolio.copy()
    if sector_filter:
        filtered = filtered[filtered["Sector"].isin(sector_filter)]
    if status_filter:
        filtered = filtered[filtered["Completion_Status"].isin(status_filter)]
    st.metric("Filtered cashflow snapshots", f"{len(filtered):,}")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

else:
    st.header("Model Details")
    st.write("The dashboard loads `LR_MSME2_best_model.pkl` and constructs the same monthly cumulative and rolling features used by the notebook.")
    cols = st.columns(4)
    cols[0].metric("Model", "Logistic Regression")
    cols[1].metric("Penalty", str(model.named_steps["logreg"].penalty))
    cols[2].metric("C", str(model.named_steps["logreg"].C))
    cols[3].metric("Classes", ", ".join(map(str, model.classes_)))
    st.subheader("Model feature contract")
    feature_names = list(model.named_steps["preprocessor"].feature_names_in_)
    st.dataframe(pd.DataFrame({"Feature": feature_names}), use_container_width=True, hide_index=True)
    st.info("The prediction form uses dummy data, while the saved pipeline performs the actual preprocessing and Logistic Regression scoring.")
