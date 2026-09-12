# LR_MSME2 Dynamic Cashflow Dashboard

## Overview

- Please note that the cashflow-related data done in the modeling stage is purely synthetized and may not reflect real-world scenario.


`LR_MSME2_dashboard.py` is a Streamlit dashboard for monitoring MSME project-completion capability over time.

The dashboard represents two related decision stages:

1. **Initial application assessment**  
   The first available monthly cashflow is used to estimate whether the MSME appears capable of completing its project.
2. **Post-acceptance monitoring**  
   Each newly submitted monthly cashflow updates the estimated probability of eventual project completion.

The dashboard uses the trained `LR_MSME2_best_model.pkl` Logistic Regression pipeline. For demonstration purposes, it generates deterministic dummy MSME data instead of loading the source portfolio.

## Files

| File | Purpose |
|---|---|
| `LR_MSME2_dashboard.py` | Streamlit dashboard application |
| `LR_MSME2_best_model.pkl` | Trained Logistic Regression pipeline |
| `LR_MSME2.ipynb` | Model-development notebook |
| `requirements.txt` | Python dependencies |
| `LR_MSME2_dashboard.md` | This documentation |

## Running the dashboard

Open PowerShell in the Season 2 folder and install the dependencies:

```powershell
pip install -r requirements.txt
```

Start Streamlit:

```powershell
streamlit run LR_MSME2_dashboard.py
```

Then open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

## Dashboard pages

### Executive Overview

The overview provides a portfolio-level summary of the generated demonstration data:

- Number of demo beneficiaries
- Number of monthly cashflow snapshots
- Eventual completion rate
- Mean predicted completion probability
- Eventual completion-status chart
- Completion-probability distribution
- Latest portfolio snapshot

### Predict Cashflow Status

This page allows the user to:

1. Select a demo MSME.
2. Select how many monthly cashflows are currently available.
3. Generate an updated completion prediction.

Period 1 represents the initial application assessment. Later periods represent additional monthly cashflow submissions after acceptance.

The page displays:

- Assessment stage
- Predicted completion status
- Completion probability
- Risk tier
- Probability timeline for the selected MSME
- Current cashflow snapshot

### Monitoring Timeline

This page aggregates predictions across the dummy portfolio by submission period. It compares:

- Mean predicted completion probability
- Actual eventual completion rate
- Number of beneficiaries submitting cashflows

This view demonstrates how the model's assessment can change as additional financial information becomes available.

### Portfolio Explorer

The explorer displays the generated monthly cashflow records and supports filtering by:

- Sector
- Eventual completion status

### Model Details

This page displays:

- Model type
- Selected Logistic Regression penalty
- Selected regularization value (`C`)
- Model classes
- The model's complete input-feature contract

## Dummy data

The dashboard intentionally generates dummy data in memory using a fixed random seed. This makes the demonstration repeatable without exposing or depending on the source MSME portfolio.

The dummy portfolio contains:

- 60 demo MSMEs
- 12–24 monthly records per MSME
- Sales and cash inflows
- Operating expenses and cash outflows
- Labor and raw-material costs
- Cash balances
- Principal and interest amounts
- Completed and not-completed eventual outcomes

All generated provinces are from Region VI (Western Visayas):

- Aklan
- Antique
- Capiz
- Guimaras
- Iloilo
- Negros Occidental

To use production data instead, replace the `make_dummy_data()` source with a loader for an approved dataset. The replacement data must provide the fields required by the model feature contract.

## Model purpose

The model predicts the eventual project completion status:

- `Completed`
- `Not Completed`

It does not predict a monthly cashflow amount. Instead, every monthly record is treated as a prediction snapshot. The target is the eventual project outcome, allowing the model to answer:

> Given the cashflow information available up to this month, how likely is the MSME to eventually complete the project?

## Feature engineering

The dashboard recreates the feature engineering used in `LR_MSME2.ipynb`.

### Current-period features

The model uses current monthly values such as:

- Sales
- Total cash inflow
- Total cash outflow
- Net operating inflow/outflow
- Beginning cash balance
- Cash available for amortization
- Principal
- Interest
- Cash end of period
- Project cost
- Province
- Sector
- Type of ownership
- Size of enterprise

### Cumulative features

For each cashflow variable, the dashboard calculates a cumulative value by beneficiary. Examples include:

- `Sales_cum`
- `Total_Cash_Inflow_cum`
- `Total_Cash_Outflow_cum`
- `Net_Operating_Inflow_Outflow_cum`
- `Cash_End_of_Period_cum`

These represent the financial history available up to the current submission.

### Rolling features

The dashboard calculates three-month rolling averages, such as:

- `Sales_rolling3`
- `Total_Cash_Inflow_rolling3`
- `Total_Cash_Outflow_rolling3`
- `Net_Operating_Inflow_Outflow_rolling3`

These help capture recent operating trends and reduce dependence on a single unusual month.

### Additional monitoring features

The dashboard also creates:

- `Cash_Position`  
  `Cash_End_of_Period - Principal - Interest`
- `Cumulative_Net_Cashflow`
- `Cashflow_Submission_Number`

## Model pipeline

The saved model is a Scikit-learn pipeline containing:

1. A `ColumnTransformer`
2. Numeric preprocessing:
   - Median imputation
   - Standardization
3. Categorical preprocessing:
   - Most-frequent imputation
   - One-hot encoding with unknown-category handling
4. Logistic Regression classifier

The trained model uses balanced class weights and supports the two completion-status classes.

The dashboard reads the feature names directly from the saved pipeline and validates that the generated prediction rows contain every required feature before scoring.

## Prediction interpretation

The dashboard uses the completion probability to assign a simple monitoring tier:

| Completion probability | Tier | Interpretation |
|---:|---|---|
| 80% or higher | Low | More closely aligned with completed projects |
| 50% to below 80% | Moderate | Nearer to the decision boundary; monitor closely |
| Below 50% | High | More closely aligned with not-completed projects |

These tiers are dashboard communication aids, not formal approval or funding rules. Thresholds should be reviewed with the program owner before operational use.

## How the risk level is determined

The dashboard's risk level is derived from the model's predicted probability that the project will eventually be completed. It is not calculated from one manually chosen cashflow ratio, and it is not a direct accounting or credit-risk score.

For each available monthly cashflow, the Logistic Regression pipeline:

1. Reads the current-period cashflow and project information.
2. Uses the cashflows available up to that submission to calculate cumulative and rolling features.
3. Applies the same imputation, scaling, and categorical encoding used during model training.
4. Produces a probability for each class:
   - `P(Completed)`
   - `P(Not Completed)`
5. Uses the class with the higher predicted probability as the predicted status.

For a two-class model, the dashboard uses:

```text
Completion probability = P(Completed)
Non-completion probability = P(Not Completed)
```

Because the model has two classes, these probabilities should add up to approximately 100%. The dashboard's risk tier is then assigned from `P(Completed)`:

| Completion probability | Dashboard tier | Operational interpretation |
|---:|---|---|
| `P(Completed) >= 0.80` | Low | The current profile is strongly aligned with completed projects in the training data. Continue standard monitoring. |
| `0.50 <= P(Completed) < 0.80` | Moderate | The model has a positive but not strong completion signal. Use milestone check-ins and review the next cashflow submission. |
| `P(Completed) < 0.50` | High | The current profile is more aligned with non-completed projects. Prioritize review, closer monitoring, and possible early intervention. |

### Basis for the thresholds

The `0.50` boundary is the natural binary-classification decision boundary: the model is more likely to classify the project as `Completed` when `P(Completed)` is at least 50%.

The `0.80` boundary is a conservative communication threshold for a strong positive signal. It separates projects with a high estimated completion probability from projects that are merely above the classification boundary. It is intended to avoid calling a project “low risk” simply because its probability is slightly above 50%.

These thresholds are **business rules layered on top of the model probability**. They were not independently calibrated against a documented loss matrix, intervention budget, or program-approved risk policy. Before production use, the thresholds should be validated using:

- Historical outcomes by probability band
- Calibration curves and reliability checks
- False-negative cost (missing a project that later fails)
- False-positive cost (intervening on a project that would have completed)
- Available monitoring and technical-assistance capacity
- Program-owner definitions of acceptable risk

### Why the risk can change over time

The risk level is recalculated whenever a new monthly cashflow is submitted. It can increase or decrease because the new submission changes the model inputs, including:

- Recent sales trend
- Recent cash inflow and outflow trend
- Cumulative net operating cashflow
- Cash balance and cash position
- Principal and interest obligations
- Number of submitted monthly cashflows

For example, a project may begin with a Moderate tier at the application stage. Consistently positive net cashflow and improving rolling sales can increase `P(Completed)` and move it to Low. Conversely, declining sales, negative net cashflow, or worsening cash position can reduce `P(Completed)` and move it to High.

### What the risk level does not mean

The tier does not mean that:

- A Low-tier project is guaranteed to complete.
- A High-tier project is guaranteed to fail.
- The model has identified fraud or mismanagement.
- The model has calculated a formal loan-default probability.
- The dashboard should automatically approve, reject, suspend, or release funds.

The tier is a prioritization signal for human review. It should be considered together with project milestones, technical reports, procurement status, implementation constraints, and verified financial records.

## Important limitations

- The dashboard currently uses dummy data.
- The generated data is for interface demonstration and is not suitable for operational decisions.
- The model predicts association with historical completion outcomes; it does not establish causation.
- A new production scoring process must apply exactly the same feature engineering as the training process.
- The model should be monitored for data drift, class imbalance, missing fields, and changes in MSME behavior.
- Thresholds and intervention actions should be validated by domain stakeholders.
- Please note that the cashflow-related data done in the modeling stage is purely synthetized and may not reflect real-world scenario.
