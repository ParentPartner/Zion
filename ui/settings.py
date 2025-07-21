import streamlit as st
from firebase_helpers import load_user_deliveries, db
from datetime import date
from firebase_helpers import get_current_date


def delete_entries_section(username: str) -> None:
    """Display interface for deleting entries."""
    st.subheader("🗑️ Delete Entries")
    selected_date = st.date_input("Select date to manage entries", 
                                value=get_current_date(), 
                                key="delete_date")
    
    df_all = load_user_deliveries(username)
    if not df_all.empty and "timestamp" in df_all.columns:
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
        df_all["date"] = df_all["timestamp"].dt.date
        entries_to_show = df_all[df_all["date"] == selected_date]
    else:
        entries_to_show = pd.DataFrame()

    if not entries_to_show.empty:
        entries_to_show = entries_to_show.sort_values("timestamp")
        
        for _, row in entries_to_show.iterrows():
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    **🕒 {row['timestamp'].strftime('%I:%M %p')}**  
                    💵 ${row['order_total']:.2f} ({row['order_type']})  
                    🚗 {row.get('miles', 0):.1f} mi (EPM: ${row.get('earnings_per_mile', 0):.2f})
                    """)
                    if row.get("bonus_amount", 0) > 0:
                        st.caption(f"✨ ${row['bonus_amount']:.2f} in bonuses")
                with col2:
                    if st.button("Delete", key=f"del_{row.name}"):
                        # Find the exact document to delete
                        docs = db.collection("deliveries").where("username", "==", username)\
                                                         .where("timestamp", "==", row["timestamp"].isoformat())\
                                                         .where("order_total", "==", row["order_total"])\
                                                         .limit(1).stream()
                        
                        for doc in docs:
                            db.collection("deliveries").document(doc.id).delete()
                            st.success("Entry deleted!")
                            st.rerun()
                st.divider()
    else:
        st.info("No entries found for this date")