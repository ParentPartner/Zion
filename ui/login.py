import streamlit as st
from firebase_helpers import init_user, validate_login

def login_section() -> Optional[str]:
    """Display login interface and return username if successful."""
    st.title("🔐 Zion Delivery Tracker")
    
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username").strip().lower()
    with col2:
        password = st.text_input("Password", type="password")
    
    if st.button("Login", type="primary"):
        if not username or not password:
            st.error("Both username and password are required")
            return None
        
        init_user(username, password)
        if validate_login(username, password):
            st.session_state.update({
                "logged_in": True,
                "username": username
            })
            st.rerun()
        else:
            st.error("Invalid credentials")
    
    st.markdown("---")
    st.markdown("Don't have an account? Just enter a new username and password to create one.")
    return None