import streamlit as st
from datetime import date
from firebase_helpers import get_user, update_user_data, get_current_date
from config import TARGET_DAILY

def daily_checkin(username: str) -> bool:
    """Handle daily check-in process."""
    today = get_current_date()
    user_data = get_user(username) or {}
    last_ci_str = user_data.get("last_checkin_date", "")
    
    if last_ci_str == today.isoformat():
        is_working = user_data.get("is_working", False)
        
        st.session_state.daily_checkin = {
            "working": is_working,
            "goal": user_data.get("today_goal", TARGET_DAILY),
            "notes": user_data.get("today_notes", "")
        }
        
        if not is_working and st.button("🚀 Actually, I want to work today"):
            update_user_data(username, {
                "is_working": True,
                "today_goal": TARGET_DAILY,
                "today_notes": "Changed mind - decided to work"
            })
            st.rerun()
            
        return is_working
    
    st.header("📅 Daily Check‑In")
    working = st.radio("Working today?", ("Yes", "No"), index=0, horizontal=True)
    
    if working == "Yes":
        goal = st.number_input("Today's Goal ($)", value=TARGET_DAILY)
        notes = st.text_area("Notes")
        
        if st.button("Start Tracking"):
            update_user_data(username, {
                "last_checkin_date": today.isoformat(),
                "is_working": True,
                "today_goal": goal,
                "today_notes": notes
            })
            st.session_state.daily_checkin = {
                "working": True,
                "goal": goal,
                "notes": notes
            }
            st.rerun()
    else:
        if st.button("Take the day off"):
            update_user_data(username, {
                "last_checkin_date": today.isoformat(),
                "is_working": False,
                "today_goal": 0,
                "today_notes": "Day off"
            })
            st.session_state.daily_checkin = {
                "working": False,
                "goal": 0,
                "notes": "Day off"
            }
            st.rerun()
    
    st.stop()
    return False