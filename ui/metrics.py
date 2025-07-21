import streamlit as st
from datetime import date, datetime, timedelta
import pandas as pd
from typing import Tuple
from firebase_helpers import load_user_deliveries

def display_metrics(username: str, today: date) -> Tuple[float, float]:
    """Display key metrics and return earned amount and goal."""
    df_all = load_user_deliveries(username)
    if not df_all.empty and "timestamp" in df_all.columns:
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
        df_all = df_all.dropna(subset=["timestamp"])
        df_all["date"] = df_all["timestamp"].dt.date
        today_df = df_all[df_all["date"] == today]
    else:
        today_df = pd.DataFrame()
    
    daily_checkin = st.session_state.get("daily_checkin", {})
    goal = daily_checkin.get("goal", TARGET_DAILY)
    earned = today_df["order_total"].sum() if not today_df.empty else 0.0
    bonus_earned = today_df["bonus_amount"].sum() if "bonus_amount" in today_df.columns else 0.0
    base_earned = today_df["base_amount"].sum() if "base_amount" in today_df.columns else earned
    perc = min(earned / goal * 100, 100) if goal else 0
    
    # Calculate Earnings Per Hour
    eph = None
    if not today_df.empty:
        today_df = today_df.sort_values("timestamp")
        time_diffs = today_df["timestamp"].diff().dt.total_seconds() / 60  # in minutes
        
        active_minutes = time_diffs[time_diffs <= 15].sum()
        active_hours = active_minutes / 60
        
        if active_hours >= 0.25:
            eph = earned / active_hours
        else:
            total_hours = (today_df["timestamp"].iloc[-1] - today_df["timestamp"].iloc[0]).total_seconds() / 3600
            if total_hours >= 0.25:
                eph = earned / total_hours
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Today's Earnings", 
                 f"${earned:.2f}", 
                 f"{perc:.0f}% of ${goal} goal",
                 delta_color="off" if perc >= 100 else "inverse")
        if bonus_earned > 0:
            st.caption(f"${base_earned:.2f} base + ${bonus_earned:.2f} bonuses")
    with col2:
        st.metric("Orders Completed", 
                 len(today_df) if not today_df.empty else 0)
    with col3:
        if eph is not None:
            realistic_eph = min(eph, 100)
            st.metric("Earnings Per Hour", 
                     f"${realistic_eph:.2f}",
                     "good" if realistic_eph >= 25 else "normal" if realistic_eph >= 20 else "bad")
        else:
            st.metric("Earnings Per Hour", "N/A")
    with col4:
        if not today_df.empty and "miles" in today_df.columns and today_df["miles"].sum() > 0:
            epm = earned / today_df["miles"].sum()
            st.metric("Earnings Per Mile", 
                     f"${epm:.2f}",
                     "good" if epm >= 2.5 else "normal" if epm >= 1.5 else "bad")
    
    return earned, goal