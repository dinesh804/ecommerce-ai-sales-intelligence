# AI-Powered E-Commerce Sales Intelligence & Forecasting

Advanced Streamlit portfolio/academic project using the supplied expanded e-commerce dataset.

## Features
- Executive KPI dashboard
- Sales, product, customer, marketing and inventory analytics
- Random Forest, XGBoost, LightGBM, CatBoost and Linear Regression benchmark
- Chronological holdout evaluation with MAE, RMSE and R²
- Recursive multi-day revenue forecasting
- Isolation Forest anomaly detection
- SHAP explainability
- Data-quality center
- CSV downloads for benchmark, forecast and anomaly reports
- Optional Optuna / DuckDB / statsmodels dependencies for further extension
- Dark, clean Streamlit interface

## Run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Important
The dataset is synthetic/demo e-commerce data. It is not live Amazon marketplace data or private seller data.

The benchmark uses a chronological 80/20 holdout rather than a random train/test split for the forecasting task.
