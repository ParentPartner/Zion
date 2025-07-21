import streamlit as st  # needed for st.error(), UI elements
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
from typing import Optional, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from config import PERFORMANCE_LEVELS


def calculate_performance_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate various performance metrics from delivery data."""
    metrics = {
        "total_orders": 0,
        "total_earnings": 0.0,
        "epm": 0.0,
        "eph": 0.0,
        "performance": "Unknown",
        "type_distribution": {}
    }
    
    if df.empty:
        return metrics
    
    try:
        metrics["total_orders"] = len(df)
        metrics["total_earnings"] = df["order_total"].sum()
        
        if "miles" in df.columns and df["miles"].sum() > 0:
            metrics["epm"] = metrics["total_earnings"] / df["miles"].sum()
        
        if "timestamp" in df.columns and len(df) > 1:
            df = df.sort_values("timestamp")
            time_diff = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 3600
            if time_diff > 0:
                metrics["eph"] = metrics["total_earnings"] / time_diff
        
        if "order_type" in df.columns:
            metrics["type_distribution"] = df["order_type"].value_counts(normalize=True).to_dict()
        
        if "epm" in metrics and "eph" in metrics:
            for level, criteria in PERFORMANCE_LEVELS.items():
                if metrics["epm"] >= criteria["min_epm"] and metrics["eph"] >= criteria["min_eph"]:
                    metrics["performance"] = level
                    break
        
        return metrics
    
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return metrics


def predict_earnings(df: pd.DataFrame, target_date: date) -> Optional[float]:
    """Predict earnings for a target date using machine learning."""
    if df.empty or "date" not in df.columns:
        return None
    
    try:
        # Prepare data
        df_daily = df.groupby("date")["order_total"].sum().reset_index()
        df_daily["date_ordinal"] = df_daily["date"].apply(lambda d: d.toordinal())
        df_daily["day_of_week"] = df_daily["date"].apply(lambda d: d.weekday())
        
        if len(df_daily) < 5:
            return None
        
        # Feature engineering
        X = df_daily[["date_ordinal", "day_of_week"]]
        y = df_daily["order_total"]
        
        # Try more sophisticated model
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
        except:
            # Fall back to linear regression if RF fails
            model = LinearRegression().fit(X, y)
        
        # Make prediction
        target_ordinal = target_date.toordinal()
        target_dow = target_date.weekday()
        prediction = model.predict(np.array([[target_ordinal, target_dow]]))[0]
        
        return max(0, prediction)
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def display_analytics(username: str, df: pd.DataFrame) -> None:
    """Streamlit UI to display analytics for the given user and data."""
    st.header("📊 Performance Analytics")
    
    if df.empty:
        st.info("No delivery data available to analyze.")
        return
    
    metrics = calculate_performance_metrics(df)
    
    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Orders", metrics["total_orders"])
    col2.metric("Total Earnings", f"${metrics['total_earnings']:.2f}")
    col3.metric("Performance Level", metrics["performance"])
    
    st.subheader("Earnings Efficiency")
    col1, col2 = st.columns(2)
    col1.metric("Earnings per Mile (EPM)", f"${metrics['epm']:.2f}")
    col2.metric("Earnings per Hour (EPH)", f"${metrics['eph']:.2f}")
    
    if metrics["type_distribution"]:
        st.subheader("Order Type Distribution")
        order_types = list(metrics["type_distribution"].keys())
        proportions = list(metrics["type_distribution"].values())
        fig = px.pie(
            names=order_types,
            values=proportions,
            title="Order Types"
        )
        st.plotly_chart(fig)
    
    # Show earnings prediction for today and next 7 days
    st.subheader("Earnings Prediction")
    today = date.today()
    predictions = []
    dates = [today + pd.Timedelta(days=i) for i in range(8)]
    for d in dates:
        pred = predict_earnings(df, d)
        predictions.append(pred if pred is not None else 0)
    
    pred_df = pd.DataFrame({
        "Date": dates,
        "Predicted Earnings": predictions
    })
    
    fig2 = px.line(pred_df, x="Date", y="Predicted Earnings", title="Predicted Earnings Over Next 7 Days")
    st.plotly_chart(fig2)
