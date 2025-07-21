import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
from datetime import datetime, date
from typing import Dict, Optional, List, Any
import pandas as pd
from config import tz

# Define your default goal if not handled elsewhere
TARGET_DAILY = 10

# Initialize Firebase only once
def get_db():
    if not firebase_admin._apps:
        try:
            st.write(f"Firebase secrets keys: {list(st.secrets['firebase'].keys())}")
            cred = credentials.Certificate(dict(st.secrets["firebase"]))
            firebase_admin.initialize_app(cred)
            st.write("✅ Firebase initialized successfully.")
        except Exception as e:
            st.error(f"❌ Failed to initialize Firebase: {e}")
            st.stop()
    return firestore.client()

# Global Firestore DB instance
db = get_db()

def get_current_date() -> date:
    return datetime.now(tz).date()

def get_user(username: str) -> Optional[Dict[str, Any]]:
    try:
        doc = db.collection("users").document(username).get()
        if doc.exists:
            user_data = doc.to_dict()
            return user_data
        st.warning(f"⚠️ User '{username}' not found.")
        return None
    except Exception as e:
        st.error(f"❌ Error accessing user data: {e}")
        return None

def validate_login(username: str, password: str) -> bool:
    user = get_user(username)
    if not user:
        st.warning(f"⚠️ No user found for '{username}'")
        return False
    if user.get("password") != password:
        st.warning(f"🔑 Password mismatch for user '{username}'")
        return False
    return True

def update_user_data(username: str, data: Dict[str, Any]) -> None:
    try:
        db.collection("users").document(username).update(data)
        st.success(f"✅ Updated user data for '{username}'")
    except Exception as e:
        st.error(f"❌ Error updating user data: {e}")

def init_user(username: str, password: str = "password") -> None:
    if not get_user(username):
        try:
            st.info(f"Creating new user: {username}")
            db.collection("users").document(username).set({
                "password": password,
                "last_checkin_date": "",
                "incentives": [],
                "today_goal": TARGET_DAILY,
                "today_notes": "",
                "is_working": False,
                "checkin_time": None,
                "created_at": datetime.now(tz).isoformat()
            })
            st.success(f"✅ User '{username}' created.")
        except Exception as e:
            st.error(f"❌ Error creating user: {e}")

def add_entry_to_firestore(entry: Dict[str, Any]) -> None:
    try:
        entry["created_at"] = datetime.now(tz).isoformat()
        db.collection("deliveries").add(entry)
        st.success("📦 Delivery saved!")
    except Exception as e:
        st.error(f"❌ Error saving delivery: {e}")

def load_user_deliveries(username: str) -> pd.DataFrame:
    try:
        docs = db.collection("deliveries").where("username", "==", username).stream()
        data = [doc.to_dict() for doc in docs]
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading deliveries: {e}")
        return pd.DataFrame()

def add_tip_baiter_to_firestore(entry: Dict[str, Any]) -> None:
    try:
        entry["created_at"] = datetime.now(tz).isoformat()
        db.collection("tip_baiters").add(entry)
        st.success("🎯 Tip baiter added.")
    except Exception as e:
        st.error(f"❌ Error saving tip baiter: {e}")

def load_user_tip_baiters(username: str) -> pd.DataFrame:
    try:
        docs = db.collection("tip_baiters").where("username", "==", username).stream()
        data = []
        for doc in docs:
            entry = doc.to_dict()
            entry["id"] = doc.id
            data.append(entry)
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading tip baiters: {e}")
        return pd.DataFrame()

def save_incentives(username: str, incentives: List[Dict[str, Any]]) -> None:
    try:
        db.collection("users").document(username).update({"incentives": incentives})
        st.success(f"🎁 Incentives saved for '{username}'")
    except Exception as e:
        st.error(f"❌ Error saving incentives: {e}")
