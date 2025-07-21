import streamlit as st
import pandas as pd
from datetime import date
from ui.login import login_section
from ui.checkin import daily_checkin
from ui.delivery_form import delivery_entry_form
from ui.metrics import display_metrics, display_ai_insights
from ui.incentives import manage_incentives
from ui.tip_baiters import tip_baiter_tracker
from ui.analytics import display_analytics
from ui.settings import delete_entries_section
from firebase_helpers import get_current_date, get_user
from typing import Dict, Any

# Initialize session state
if "logged_in" not in st.session_state:
    login_section()
    st.stop()

user = st.session_state["username"]
today = get_current_date()

# Get user data once
user_data = get_user(user) or {}

# Convert delivery records to DataFrame
deliveries = user_data.get("deliveries", [])  # Make sure your Firestore structure includes this
df = pd.DataFrame(deliveries) if deliveries else pd.DataFrame(columns=["date", "amount", "type", "notes"])

# Check if user has checked in today
if daily_checkin(user):
    # Main interface
    st.title("📦 Spark Delivery Tracker")

    # Display daily notes if available
    daily_notes = st.session_state.get("daily_checkin", {}).get("notes", user_data.get("today_notes", ""))
    if daily_notes:
        with st.expander("📝 Today's Notes"):
            st.write(daily_notes)

    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tracker", "Incentives", "Tip Baiters", "Analytics", "Settings"])

    with tab1:
        earned, goal = display_metrics(user, today)
        delivery_entry_form(user, today)
        display_ai_insights(user, today)

    with tab2:
        manage_incentives(user)

    with tab3:
        tip_baiter_tracker(user)

    with tab4:
        display_analytics(user, df)  # ✅ FIXED: Now passing the required df

    with tab5:
        delete_entries_section(user)

    st.caption("🧠 AI-Powered Spark Tracker v3.0 | Data stays 100% yours.")
else:
    st.success("🏝️ Enjoy your day off!")
    st.stop()
