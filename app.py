
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Optional advanced libraries
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    from lightgbm import LGBMRegressor
    LGBM_OK = True
except Exception:
    LGBM_OK = False

try:
    from catboost import CatBoostRegressor
    CAT_OK = True
except Exception:
    CAT_OK = False

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

try:
    import optuna
    OPTUNA_OK = True
except Exception:
    OPTUNA_OK = False

try:
    import duckdb
    DUCKDB_OK = True
except Exception:
    DUCKDB_OK = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="AI E-Commerce Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background:#080b12; color:#e8edf5; }
[data-testid="stSidebar"] { background:#0d111b; border-right:1px solid #202838; }
.block-container { padding-top:1.2rem; max-width:1500px; }
h1,h2,h3 { color:#f4f7fb; }
.small { color:#8995a8; font-size:.85rem; }
.kpi { background:#111722; border:1px solid #202838; border-radius:14px; padding:18px; }
.kpi-label { color:#8d99aa; font-size:.82rem; }
.kpi-value { color:#f5f7fa; font-size:1.55rem; font-weight:700; margin-top:4px; }
.badge { display:inline-block; padding:4px 9px; border-radius:999px; background:#172235; color:#b9c7dc; font-size:.75rem; margin-right:5px; }
div[data-testid="stMetric"] { background:#111722; border:1px solid #202838; padding:12px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = "expanded_ecommerce_sales_forecasting_dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    for c in ["Expected_Delivery_Date", "Actual_Delivery_Date"]:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    numeric = [
        "Quantity","Unit_Price_INR","Gross_Sales_INR","Discount_Percent",
        "Discount_Amount_INR","Tax_Percent","Tax_INR","Shipping_Cost_INR",
        "Net_Revenue_INR","Cost_Price_INR","Profit_INR","Profit_Margin_Percent",
        "Advertising_Spend_INR","Inventory_Level","Reorder_Level","Customer_Age",
        "Product_Rating","Review_Count","Delivery_Days"
    ]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

df = load_data()

# ---------------- HELPERS ----------------
def money(x):
    return f"₹{x:,.0f}"

def metric_card(label, value):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>', unsafe_allow_html=True
    )

def daily_series(data):
    x = data.groupby("Order_Date", as_index=False).agg(
        Revenue=("Net_Revenue_INR","sum"),
        Profit=("Profit_INR","sum"),
        Quantity=("Quantity","sum")
    ).sort_values("Order_Date")
    return x

def feature_frame(data):
    x = data.copy()
    daily = x.groupby("Order_Date").agg(
        Revenue=("Net_Revenue_INR","sum"),
        Profit=("Profit_INR","sum"),
        Quantity=("Quantity","sum"),
        Orders=("Order_ID","nunique"),
        Avg_Discount=("Discount_Percent","mean"),
        Ad_Spend=("Advertising_Spend_INR","sum"),
        Avg_Delivery=("Delivery_Days","mean")
    ).reset_index().sort_values("Order_Date")
    daily["dow"] = daily["Order_Date"].dt.dayofweek
    daily["month"] = daily["Order_Date"].dt.month
    daily["day"] = daily["Order_Date"].dt.day
    daily["weekofyear"] = daily["Order_Date"].dt.isocalendar().week.astype(int)
    for lag in [1,7,14,28]:
        daily[f"lag_{lag}"] = daily["Revenue"].shift(lag)
    for w in [7,14,28]:
        daily[f"roll_{w}"] = daily["Revenue"].shift(1).rolling(w).mean()
    daily = daily.dropna().reset_index(drop=True)
    return daily

@st.cache_resource
def train_models(data):
    d = feature_frame(data)
    if len(d) < 40:
        return {}, d, pd.DataFrame()

    features = [c for c in d.columns if c not in ["Order_Date","Revenue","Profit","Quantity"]]
    X, y = d[features], d["Revenue"]
    split = int(len(d)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=350, max_depth=14, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
    }
    if XGB_OK:
        models["XGBoost"] = XGBRegressor(
            n_estimators=350, max_depth=7, learning_rate=.05,
            subsample=.85, colsample_bytree=.85, objective="reg:squarederror",
            random_state=42, n_jobs=4
        )
    if LGBM_OK:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=350, learning_rate=.05, max_depth=-1,
            num_leaves=31, random_state=42, verbosity=-1
        )
    if CAT_OK:
        models["CatBoost"] = CatBoostRegressor(
            iterations=350, depth=7, learning_rate=.05,
            loss_function="RMSE", verbose=False, random_seed=42
        )

    results, fitted, predictions = [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[name] = model
        predictions[name] = pred
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred)**0.5,
            "R²": r2_score(y_test, pred)
        })
    return {"models":fitted, "features":features, "split":split,
            "X_test":X_test, "y_test":y_test, "predictions":predictions}, d, pd.DataFrame(results).sort_values("RMSE")

@st.cache_data
def anomaly_data(data):
    s = daily_series(data).copy()
    if len(s) < 20:
        s["Anomaly"] = 0
        return s
    cols = ["Revenue","Profit","Quantity"]
    iso = IsolationForest(contamination=0.04, random_state=42)
    s["Anomaly"] = iso.fit_predict(s[cols])
    return s

# ---------------- SIDEBAR ----------------
st.sidebar.title("📈 E-Commerce AI")
st.sidebar.caption("Sales Intelligence & Forecasting Platform")
page = st.sidebar.radio(
    "Navigation",
    ["Executive Overview","Sales Intelligence","Product Intelligence",
     "Customer Intelligence","Marketing Analytics","Inventory & Operations",
     "AI Model Lab","Forecast Center","Anomaly Detection","Explainable AI","Data Quality"]
)

st.sidebar.divider()
st.sidebar.markdown("**Dataset**")
st.sidebar.write(f"{len(df):,} orders")
st.sidebar.write(f"{df['Order_Date'].min().date()} → {df['Order_Date'].max().date()}")
st.sidebar.caption("Synthetic/demo e-commerce dataset — not private or live marketplace data.")

# ---------------- OVERVIEW ----------------
if page == "Executive Overview":
    st.title("AI-Powered E-Commerce Sales Intelligence")
    st.caption("A portfolio-ready analytics, forecasting and ML decision-support dashboard.")

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: metric_card("Net Revenue", money(df["Net_Revenue_INR"].sum()))
    with c2: metric_card("Profit", money(df["Profit_INR"].sum()))
    with c3: metric_card("Orders", f"{df['Order_ID'].nunique():,}")
    with c4: metric_card("Units Sold", f"{df['Quantity'].sum():,}")
    with c5: metric_card("Avg Margin", f"{df['Profit_Margin_Percent'].mean():.1f}%")

    st.subheader("Revenue & Profit Trend")
    ts = daily_series(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.Order_Date,y=ts.Revenue,name="Revenue",mode="lines"))
    fig.add_trace(go.Scatter(x=ts.Order_Date,y=ts.Profit,name="Profit",mode="lines"))
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    a,b = st.columns(2)
    with a:
        cat = df.groupby("Category",as_index=False)["Net_Revenue_INR"].sum().sort_values("Net_Revenue_INR",ascending=False)
        st.plotly_chart(px.bar(cat,x="Net_Revenue_INR",y="Category",orientation="h",
                               template="plotly_dark",title="Revenue by Category"),
                        use_container_width=True)
    with b:
        reg = df.groupby("Customer_Region",as_index=False)["Net_Revenue_INR"].sum().sort_values("Net_Revenue_INR",ascending=False)
        st.plotly_chart(px.bar(reg,x="Customer_Region",y="Net_Revenue_INR",
                               template="plotly_dark",title="Revenue by Region"),
                        use_container_width=True)

# ---------------- SALES ----------------
elif page == "Sales Intelligence":
    st.title("Sales Intelligence")
    ts = daily_series(df)
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(ts,x="Order_Date",y="Revenue",template="plotly_dark",
                                title="Daily Revenue"),use_container_width=True)
    with c2:
        monthly = df.assign(MonthDate=df["Order_Date"].dt.to_period("M").dt.to_timestamp()).groupby("MonthDate",as_index=False)["Net_Revenue_INR"].sum()
        st.plotly_chart(px.bar(monthly,x="MonthDate",y="Net_Revenue_INR",template="plotly_dark",
                               title="Monthly Revenue"),use_container_width=True)
    c1,c2 = st.columns(2)
    with c1:
        dow = df.groupby("Day_of_Week",as_index=False)["Net_Revenue_INR"].sum()
        st.plotly_chart(px.bar(dow,x="Day_of_Week",y="Net_Revenue_INR",template="plotly_dark",
                               title="Revenue by Day"),use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(df.sample(min(4000,len(df)),random_state=42),
                                   x="Discount_Percent",y="Net_Revenue_INR",
                                   size="Quantity",hover_data=["Category","Product_Name"],
                                   template="plotly_dark",title="Discount vs Revenue"),
                        use_container_width=True)

# ---------------- PRODUCTS ----------------
elif page == "Product Intelligence":
    st.title("Product Intelligence")
    top = df.groupby(["Product_ID","Product_Name"],as_index=False).agg(
        Revenue=("Net_Revenue_INR","sum"), Profit=("Profit_INR","sum"), Units=("Quantity","sum")
    ).sort_values("Revenue",ascending=False)
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(top.head(12),x="Revenue",y="Product_Name",orientation="h",
                               template="plotly_dark",title="Top Products by Revenue"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(top,x="Revenue",y="Profit",size="Units",
                                   hover_name="Product_Name",template="plotly_dark",
                                   title="Revenue vs Profit by Product"),
                        use_container_width=True)
    st.dataframe(top.head(30),use_container_width=True,hide_index=True)

# ---------------- CUSTOMERS ----------------
elif page == "Customer Intelligence":
    st.title("Customer Intelligence")
    seg = df.groupby("Customer_Type",as_index=False).agg(
        Customers=("Customer_ID","nunique"), Revenue=("Net_Revenue_INR","sum"),
        Profit=("Profit_INR","sum"), Orders=("Order_ID","nunique")
    )
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(seg,x="Customer_Type",y="Revenue",template="plotly_dark",
                               title="Revenue by Customer Segment"),use_container_width=True)
    with c2:
        age = df.groupby("Customer_Age",as_index=False)["Net_Revenue_INR"].sum()
        st.plotly_chart(px.line(age,x="Customer_Age",y="Net_Revenue_INR",template="plotly_dark",
                                title="Revenue by Customer Age"),use_container_width=True)
    st.dataframe(seg,use_container_width=True,hide_index=True)

# ---------------- MARKETING ----------------
elif page == "Marketing Analytics":
    st.title("Marketing Analytics")
    camp = df.groupby("Campaign",as_index=False).agg(
        Revenue=("Net_Revenue_INR","sum"), Spend=("Advertising_Spend_INR","sum"),
        Orders=("Order_ID","nunique")
    )
    camp["ROAS"] = np.where(camp["Spend"]>0,camp["Revenue"]/camp["Spend"],np.nan)
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(camp.sort_values("Revenue",ascending=False),
                               x="Campaign",y="Revenue",template="plotly_dark",
                               title="Campaign Revenue"),use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(camp,x="Spend",y="Revenue",size="Orders",
                                   hover_name="Campaign",template="plotly_dark",
                                   title="Ad Spend vs Revenue"),use_container_width=True)
    traffic = df.groupby("Traffic_Source",as_index=False)["Net_Revenue_INR"].sum().sort_values("Net_Revenue_INR",ascending=False)
    st.plotly_chart(px.bar(traffic,x="Traffic_Source",y="Net_Revenue_INR",
                           template="plotly_dark",title="Revenue by Traffic Source"),
                    use_container_width=True)
    st.dataframe(camp.sort_values("ROAS",ascending=False),use_container_width=True,hide_index=True)

# ---------------- INVENTORY ----------------
elif page == "Inventory & Operations":
    st.title("Inventory & Operations")
    inv = df.groupby(["Product_Name"],as_index=False).agg(
        Inventory=("Inventory_Level","mean"), Reorder=("Reorder_Level","mean"),
        Units=("Quantity","sum"), Revenue=("Net_Revenue_INR","sum")
    )
    inv["Risk"] = np.where(inv["Inventory"] < inv["Reorder"],"Reorder Required","Healthy")
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.scatter(inv,x="Reorder",y="Inventory",size="Units",
                                   color="Risk",hover_name="Product_Name",
                                   template="plotly_dark",title="Inventory vs Reorder Level"),
                        use_container_width=True)
    with c2:
        returns = df["Return_Reason"].fillna("None").replace("", "None").value_counts().reset_index()
        returns.columns=["Reason","Count"]
        st.plotly_chart(px.pie(returns.head(8),names="Reason",values="Count",
                               hole=.55,template="plotly_dark",title="Return Reasons"),
                        use_container_width=True)
    st.dataframe(inv.sort_values("Inventory").head(30),use_container_width=True,hide_index=True)

# ---------------- MODEL LAB ----------------
elif page == "AI Model Lab":
    st.title("🤖 AI Model Lab")
    st.caption("Chronological holdout evaluation prevents random time leakage in the forecasting benchmark.")

    with st.spinner("Training available models..."):
        pack, d, results = train_models(df)

    if results.empty:
        st.error("Not enough historical data to train the models.")
    else:
        st.subheader("Model Comparison")
        display = results.copy()
        display["MAE"] = display["MAE"].round(2)
        display["RMSE"] = display["RMSE"].round(2)
        display["R²"] = display["R²"].round(4)
        st.dataframe(display,use_container_width=True,hide_index=True)

        best = results.iloc[0]["Model"]
        st.success(f"Lowest holdout RMSE in this benchmark: {best}")

        pred = pack["predictions"][best]
        actual = pack["y_test"].values
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=actual,name="Actual",mode="lines"))
        fig.add_trace(go.Scatter(y=pred,name="Predicted",mode="lines"))
        fig.update_layout(template="plotly_dark",height=400,title=f"{best}: Actual vs Predicted")
        st.plotly_chart(fig,use_container_width=True)

        st.subheader("Feature Importance")
        model = pack["models"][best]
        feats = pack["features"]
        if hasattr(model,"feature_importances_"):
            imp = pd.DataFrame({"Feature":feats,"Importance":model.feature_importances_}).sort_values("Importance",ascending=False).head(15)
            st.plotly_chart(px.bar(imp,x="Importance",y="Feature",orientation="h",
                                   template="plotly_dark",title="Top Predictive Features"),
                            use_container_width=True)
        else:
            st.info("Linear Regression does not expose tree feature importance.")

        st.download_button("Download Model Benchmark CSV",
                           results.to_csv(index=False).encode(),
                           "model_benchmark.csv","text/csv")

        if not (XGB_OK and LGBM_OK and CAT_OK):
            st.info("Optional model status: "
                    f"XGBoost={'ON' if XGB_OK else 'OFF'} · "
                    f"LightGBM={'ON' if LGBM_OK else 'OFF'} · "
                    f"CatBoost={'ON' if CAT_OK else 'OFF'}")

# ---------------- FORECAST ----------------
elif page == "Forecast Center":
    st.title("🔮 Forecast Center")
    horizon = st.slider("Forecast horizon (days)",7,90,30)
    pack,d,results = train_models(df)
    if results.empty:
        st.error("Not enough data.")
    else:
        selected = st.selectbox("Forecast model",results["Model"].tolist())
        model = pack["models"][selected]
        features = pack["features"]
        hist = d.copy()
        future_rows = []
        work = hist.copy()
        for _ in range(horizon):
            next_date = work["Order_Date"].max() + pd.Timedelta(days=1)
            row = {}
            row["Order_Date"] = next_date
            row["dow"] = next_date.dayofweek
            row["month"] = next_date.month
            row["day"] = next_date.day
            row["weekofyear"] = int(next_date.isocalendar().week)
            for lag in [1,7,14,28]:
                row[f"lag_{lag}"] = work["Revenue"].iloc[-lag] if len(work)>=lag else work["Revenue"].mean()
            for w in [7,14,28]:
                row[f"roll_{w}"] = work["Revenue"].tail(w).mean()
            row["Profit"] = work["Profit"].iloc[-1]
            row["Quantity"] = work["Quantity"].iloc[-1]
            row["Orders"] = work["Orders"].iloc[-1]
            row["Avg_Discount"] = work["Avg_Discount"].iloc[-1]
            row["Ad_Spend"] = work["Ad_Spend"].iloc[-1]
            row["Avg_Delivery"] = work["Avg_Delivery"].iloc[-1]
            Xnext = pd.DataFrame([row])[features]
            pred = max(0,float(model.predict(Xnext)[0]))
            row["Revenue"] = pred
            future_rows.append({"Order_Date":next_date,"Forecast_Revenue_INR":pred})
            work = pd.concat([work,pd.DataFrame([row])],ignore_index=True)

        fc = pd.DataFrame(future_rows)
        actual = hist[["Order_Date","Revenue"]].rename(columns={"Revenue":"Actual_Revenue_INR"})
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=actual.Order_Date.tail(90),y=actual.Actual_Revenue_INR.tail(90),name="Historical",mode="lines"))
        chart.add_trace(go.Scatter(x=fc.Order_Date,y=fc.Forecast_Revenue_INR,name="Forecast",mode="lines"))
        chart.update_layout(template="plotly_dark",height=450,title=f"{selected} — {horizon}-Day Revenue Forecast")
        st.plotly_chart(chart,use_container_width=True)
        c1,c2,c3 = st.columns(3)
        with c1: metric_card("Forecast Revenue",money(fc.Forecast_Revenue_INR.sum()))
        with c2: metric_card("Avg Daily Forecast",money(fc.Forecast_Revenue_INR.mean()))
        with c3: metric_card("Peak Forecast Day",fc.loc[fc.Forecast_Revenue_INR.idxmax(),"Order_Date"].strftime("%d %b %Y"))
        st.download_button("Download Forecast CSV",fc.to_csv(index=False).encode(),
                           "sales_forecast.csv","text/csv")

# ---------------- ANOMALIES ----------------
elif page == "Anomaly Detection":
    st.title("🚨 Sales Anomaly Detection")
    st.caption("Isolation Forest flags unusual daily combinations of revenue, profit and units.")
    a = anomaly_data(df)
    anomalies = a[a["Anomaly"]==-1]
    fig = px.line(a,x="Order_Date",y="Revenue",template="plotly_dark",title="Revenue with Anomaly Points")
    fig.add_trace(go.Scatter(x=anomalies.Order_Date,y=anomalies.Revenue,mode="markers",
                             name="Anomaly",marker=dict(size=9)))
    st.plotly_chart(fig,use_container_width=True)
    st.metric("Detected Anomalous Days",len(anomalies))
    st.dataframe(anomalies.sort_values("Revenue",ascending=False),use_container_width=True,hide_index=True)
    st.download_button("Download Anomaly Report",anomalies.to_csv(index=False).encode(),
                       "sales_anomalies.csv","text/csv")

# ---------------- EXPLAINABILITY ----------------
elif page == "Explainable AI":
    st.title("🔍 Explainable AI")
    pack,d,results = train_models(df)
    if results.empty:
        st.error("Train a model first.")
    else:
        model_name = st.selectbox("Model",results["Model"].tolist())
        model = pack["models"][model_name]
        Xtest = pack["X_test"]
        if SHAP_OK and hasattr(model,"predict"):
            try:
                sample = Xtest.tail(min(250,len(Xtest)))
                if model_name == "Linear Regression":
                    explainer = shap.Explainer(model, sample)
                else:
                    explainer = shap.TreeExplainer(model)
                values = explainer(sample)
                mean_abs = np.abs(values.values).mean(axis=0)
                imp = pd.DataFrame({"Feature":sample.columns,"Mean |SHAP|":mean_abs}).sort_values("Mean |SHAP|",ascending=False).head(15)
                st.plotly_chart(px.bar(imp,x="Mean |SHAP|",y="Feature",orientation="h",
                                       template="plotly_dark",title="Global SHAP Feature Impact"),
                                use_container_width=True)
                st.caption("SHAP values describe how model features contribute to predictions; they do not establish causality.")
            except Exception as e:
                st.warning(f"SHAP explanation could not be generated for this model: {e}")
        else:
            st.info("Install SHAP to enable model explainability.")

# ---------------- DATA QUALITY ----------------
elif page == "Data Quality":
    st.title("🧪 Data Quality Center")
    q = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [round(df[c].isna().mean()*100,2) for c in df.columns],
        "Unique Values": [df[c].nunique(dropna=True) for c in df.columns],
        "Dtype": [str(df[c].dtype) for c in df.columns]
    })
    c1,c2,c3 = st.columns(3)
    with c1: metric_card("Rows",f"{len(df):,}")
    with c2: metric_card("Columns",f"{df.shape[1]:,}")
    with c3: metric_card("Duplicate Rows",f"{df.duplicated().sum():,}")
    st.dataframe(q.sort_values("Missing %",ascending=False),use_container_width=True,hide_index=True)
    st.download_button("Download Data Quality Report",q.to_csv(index=False).encode(),
                       "data_quality_report.csv","text/csv")

st.sidebar.divider()
st.sidebar.caption(
    f"Advanced libraries: XGBoost {'✓' if XGB_OK else '—'} · "
    f"LightGBM {'✓' if LGBM_OK else '—'} · CatBoost {'✓' if CAT_OK else '—'} · "
    f"SHAP {'✓' if SHAP_OK else '—'} · Optuna {'✓' if OPTUNA_OK else '—'}"
)
