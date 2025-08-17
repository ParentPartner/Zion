# 🚀 Spark Delivery Tracker with Incentives & Tip Baiter Tracking (Complete Edition)
# main.py - Main Streamlit Application

import streamlit as st
from datetime import datetime, date
import pytz
from database import get_user, validate_login, init_user, update_user_data, load_user_deliveries
from ui_components import (
    login_section, daily_checkin, delivery_entry_form, 
    display_metrics, display_ai_insights, display_analytics, 
    delete_entries_section, manage_incentives, tip_baiter_tracker
)
from config import TARGET_DAILY

# === CONFIG & SETUP ===
tz = pytz.timezone("US/Eastern")

def get_current_date() -> date:
    return datetime.now(tz).date()

# === MAIN APP FLOW ===
def main():
    # Check login status
    if "logged_in" not in st.session_state:
        login_section()
        st.stop()

    user = st.session_state["username"]
    today = get_current_date()

    # Check if user has checked in today
    if daily_checkin(user):
        # Main interface
        st.title("📦 Spark Delivery Tracker")
        
        # Display daily notes if available
        daily_notes = st.session_state.get("daily_checkin", {}).get("notes", "")
        if daily_notes:
            with st.expander("📝 Today's Notes"):
                st.write(daily_notes)
        
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tracker", "Incentives", "Tip Baiters", "Analytics", "Settings"])
        
        with tab1:
            # Display current metrics
            earned, goal = display_metrics(user, today)
            
            # Delivery entry form
            delivery_entry_form(user, today)
            
            # AI Insights
            display_ai_insights(user, today)
        
        with tab2:
            manage_incentives(user)
        
        with tab3:
            tip_baiter_tracker(user)
        
        with tab4:
            display_analytics(user)
        
        with tab5:
            delete_entries_section(user)
        
        st.caption("🧠 AI-Powered Spark Tracker v3.0 | Data stays 100% yours.")
    else:
        st.success("🏝️ Enjoy your day off!")
        st.stop()

if __name__ == "__main__":
    main()