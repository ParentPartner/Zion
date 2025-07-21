import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from firebase_helpers import add_tip_baiter_to_firestore, load_user_tip_baiters, get_current_date
from config import tz

def tip_baiter_tracker(username: str) -> None:
    st.subheader("🚨 Tip Baiter Tracker")
    
    with st.expander("➕ Add New Tip Baiter", expanded=False):
        with st.form("tip_baiter_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name (Required)", help="Customer name or identifier")
            with col2:
                date_baited = st.date_input("Date", value=get_current_date())
            
            address = st.text_input("Address (Optional)", help="Approximate address or location")
            
            col3, col4 = st.columns(2)
            with col3:
                amount_baited = st.number_input("Amount Baited ($)", value=0.0, step=0.01, min_value=0.0)
            with col4:
                rating = st.slider("Severity (1-5)", 1, 5, 3, 
                                 help="How bad was this tip bait? 1=minor, 5=egregious")
            
            notes = st.text_area("Notes", help="Any additional details about this incident")
            
            if st.form_submit_button("Save Tip Baiter"):
                if not name:
                    st.error("Name is required!")
                else:
                    entry = {
                        "name": name.strip(),
                        "address": address.strip() if address else "",
                        "date": date_baited.isoformat(),
                        "amount": float(amount_baited),
                        "rating": rating,
                        "notes": notes.strip(),
                        "username": username,
                        "timestamp": datetime.now(tz).isoformat()
                    }
                    add_tip_baiter_to_firestore(entry)
                    st.success("Tip baiter saved!")
                    st.experimental_rerun()
    
    st.subheader("📋 Your Tip Baiters")
    tip_baiters_df = load_user_tip_baiters(username)
    
    if not tip_baiters_df.empty:
        tip_baiters_df["date"] = pd.to_datetime(tip_baiters_df["date"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tip Baiters", len(tip_baiters_df))
        with col2:
            st.metric("Total Amount Baited", f"${tip_baiters_df['amount'].sum():.2f}")
        with col3:
            avg_severity = tip_baiters_df['rating'].mean()
            st.metric("Average Severity", f"{avg_severity:.1f}/5")
        
        with st.expander("🔍 Filter Options", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                search_term = st.text_input("Search by name or address")
            with col2:
                date_filter = st.selectbox(
                    "Date Range",
                    ["All", "Today", "Yesterday", "Last 7 days", "Last 30 days", "Last 90 days", "Custom"]
                )
            with col3:
                min_severity = st.slider("Minimum Severity", 1, 5, 1)
        
        if search_term:
            tip_baiters_df = tip_baiters_df[
                tip_baiters_df["name"].str.contains(search_term, case=False) |
                tip_baiters_df["address"].str.contains(search_term, case=False)
            ]
        
        if date_filter != "All":
            today = get_current_date()
            if date_filter == "Today":
                tip_baiters_df = tip_baiters_df[tip_baiters_df["date"].dt.date == today]
            elif date_filter == "Yesterday":
                yesterday = today - timedelta(days=1)
                tip_baiters_df = tip_baiters_df[tip_baiters_df["date"].dt.date == yesterday]
            else:
                days = 7 if date_filter == "Last 7 days" else 30 if date_filter == "Last 30 days" else 90
                cutoff_date = today - timedelta(days=days)
                tip_baiters_df = tip_baiters_df[tip_baiters_df["date"].dt.date >= cutoff_date]
        
        tip_baiters_df = tip_baiters_df[tip_baiters_df["rating"] >= min_severity]
        
        if not tip_baiters_df.empty:
            tip_baiters_df = tip_baiters_df.sort_values(["date", "rating"], ascending=[False, False])
            grouped = tip_baiters_df.groupby(tip_baiters_df["date"].dt.date)
            
            for date_val, group in grouped:
                with st.expander(f"📅 {date_val.strftime('%b %d, %Y')} ({len(group)} incidents)"):
                    for _, row in group.iterrows():
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"**{row['name']}**")
                                st.caption(f"📍 {row['address'] or 'No address'} | 💰 ${row['amount']:.2f} | ⭐ {row['rating']}/5")
                                if row["notes"]:
                                    st.markdown(f"📝 *{row['notes']}*")
                            with col2:
                                if st.button("🗑️", key=f"del_tb_{row['id']}"):
                                    db = get_db()
                                    db.collection("tip_baiters").document(row["id"]).delete()
                                    st.success("Tip baiter removed!")
                                    st.experimental_rerun()
                            st.divider()
        else:
            st.info("No tip baiters match your filters")
    else:
        st.info("No tip baiters recorded yet")
