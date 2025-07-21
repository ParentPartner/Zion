import streamlit as st  # needed for st.error()
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