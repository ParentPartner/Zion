import streamlit as st
from typing import List, Dict
from firebase_helpers import get_user, save_incentives
from config import ORDER_TYPES

def manage_incentives(username: str) -> None:
    """Add, edit, and remove incentives for the current user."""
    st.subheader("💰 Incentive Management")
    
    # Load current incentives
    user_data = get_user(username)
    current_incentives = user_data.get("incentives", []) if user_data else []
    
    with st.expander("➕ Add New Incentive", expanded=False):
        with st.form("incentive_form"):
            col1, col2 = st.columns(2)
            with col1:
                incentive_name = st.text_input("Incentive Name", help="E.g. 'Lunch Rush Bonus'")
                start_time = st.time_input("Start Time", value=time(11, 0))
            with col2:
                applies_to = st.multiselect("Applies To", ORDER_TYPES, default=ORDER_TYPES)
                end_time = st.time_input("End Time", value=time(14, 0))
            
            amount = st.number_input("Bonus Amount Per Order ($)", value=5.0, step=0.5, min_value=0.1)
            notes = st.text_area("Notes (Optional)")
            
            if st.form_submit_button("Save Incentive"):
                if not incentive_name:
                    st.error("Incentive name is required!")
                elif start_time >= end_time:
                    st.error("End time must be after start time!")
                elif not applies_to:
                    st.error("Must select at least one order type!")
                else:
                    new_incentive = {
                        "name": incentive_name,
                        "start_time": start_time.strftime("%H:%M"),
                        "end_time": end_time.strftime("%H:%M"),
                        "applies_to": applies_to,
                        "amount": amount,
                        "notes": notes,
                        "active": True
                    }
                    current_incentives.append(new_incentive)
                    save_incentives(username, current_incentives)
                    st.success("Incentive saved!")
                    st.rerun()
    
    st.subheader("📋 Your Active Incentives")
    if not current_incentives:
        st.info("No incentives set up yet")
    else:
        for idx, incentive in enumerate(current_incentives):
            with st.expander(f"{incentive['name']} - ${incentive['amount']} per order"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    **Time:** {incentive['start_time']} to {incentive['end_time']}  
                    **Applies to:** {', '.join(incentive['applies_to'])}  
                    **Notes:** {incentive.get('notes', 'None')}
                    """)
                with col2:
                    if st.button("🗑️ Remove", key=f"del_incentive_{idx}"):
                        del current_incentives[idx]
                        save_incentives(username, current_incentives)
                        st.success("Incentive removed!")
                        st.rerun()