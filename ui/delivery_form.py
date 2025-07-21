import streamlit as st
from datetime import datetime, time, date
from typing import Tuple
from ocr_parser import extract_text_from_image, parse_screenshot_text
from firebase_helpers import add_entry_to_firestore
from config import tz, ORDER_TYPES
from utils import apply_incentives

def delivery_entry_form(username: str, today: date) -> None:
    """Display form for entering delivery information."""
    st.subheader("📝 Order Entry")
    
    # OCR functionality
    uploaded = st.file_uploader("Upload screenshot (optional)", 
                              type=["png", "jpg", "jpeg"],
                              help="Upload a screenshot of your delivery summary to auto-fill details")
    
    parsed = None
    if uploaded:
        with st.spinner("Analyzing with AI..."):
            try:
                text_list = extract_text_from_image(uploaded)
                if text_list:
                    ts, total, ml, order_type = parse_screenshot_text(text_list)
                    parsed = {
                        "timestamp": ts, 
                        "order_total": total,
                        "miles": ml, 
                        "order_type": order_type
                    }
                    st.success(f"✅ AI Analysis: ${total:.2f} | {ml:.1f} mi @ {ts.strftime('%I:%M %p')} | Type: {order_type}")
                else:
                    st.warning("No text found in image")
            except Exception as e:
                st.error(f"Failed to analyze image: {e}")
    
    # Entry form
    with st.form("entry_form", clear_on_submit=True):
        if parsed:
            default_time = parsed["timestamp"].time()
            default_date = parsed["timestamp"].date()
            default_type = parsed["order_type"]
            default_total = parsed["order_total"]
            default_miles = parsed["miles"]
        else:
            now = datetime.now(tz)
            default_time = now.time()
            default_date = today
            default_type = "Delivery"
            default_total = 0.0
            default_miles = 0.0
        
        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.date_input("Date", value=default_date)
        with col2:
            selected_time = st.time_input("Time", value=time(default_time.hour, default_time.minute))
        
        order_type = st.radio("Order Type", ORDER_TYPES, 
                             index=ORDER_TYPES.index(default_type), 
                             horizontal=True)
        
        col3, col4 = st.columns(2)
        with col3:
            base_amount = st.number_input("Base Amount ($)", 
                                       value=default_total, 
                                       step=0.01,
                                       min_value=0.0,
                                       help="Amount before incentives")
        with col4:
            ml = st.number_input("Miles Driven", 
                                value=default_miles if parsed else 0.0, 
                                step=0.1,
                                min_value=0.0)
        
        if st.form_submit_button("Save Delivery", type="primary"):
            try:
                naive_dt = datetime.combine(selected_date, selected_time)
                aware_dt = tz.localize(naive_dt)
                
                # Apply any matching incentives
                total_amount, bonus_amount, applied_incentives = apply_incentives(
                    order_type, selected_time, base_amount, username
                )
                
                entry = {
                    "timestamp": aware_dt.isoformat(),
                    "order_total": float(total_amount),
                    "base_amount": float(base_amount),
                    "bonus_amount": float(bonus_amount),
                    "incentives": applied_incentives,
                    "miles": float(ml),
                    "earnings_per_mile": round(float(total_amount)/float(ml), 2) if ml else 0.0,
                    "hour": selected_time.hour,
                    "username": username,
                    "order_type": order_type
                }
                
                add_entry_to_firestore(entry)
                
                if bonus_amount > 0:
                    incentive_names = ", ".join([i["name"] for i in applied_incentives])
                    st.success(f"✅ Saved {order_type} at {aware_dt.strftime('%I:%M %p')} with ${bonus_amount:.2f} in bonuses ({incentive_names})!")
                else:
                    st.success(f"✅ Saved {order_type} entry at {aware_dt.strftime('%I:%M %p')}!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save entry: {e}")