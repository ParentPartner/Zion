# database.py - Database Operations and Data Processing

import streamlit as st
import pandas as pd
import re
from datetime import datetime, date, time, timedelta
import easyocr
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Optional, Tuple
from config import tz, TARGET_DAILY, PERFORMANCE_LEVELS

# Initialize Firebase only once
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        st.stop()

db = firestore.client()

# === FIRESTORE HELPERS ===
def get_user(username: str) -> Optional[Dict]:
    try:
        doc = db.collection("users").document(username).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        st.error(f"Error accessing user data: {e}")
        return None

def validate_login(username: str, password: str) -> bool:
    user = get_user(username)
    if not user:
        return False
    return user.get("password") == password

def update_user_data(username: str, data: Dict) -> None:
    try:
        db.collection("users").document(username).update(data)
    except Exception as e:
        st.error(f"Error updating user data: {e}")

def init_user(username: str, password: str = "password") -> None:
    if not get_user(username):
        try:
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
        except Exception as e:
            st.error(f"Error creating user: {e}")

def add_entry_to_firestore(entry: Dict) -> None:
    try:
        entry["created_at"] = datetime.now(tz).isoformat()
        db.collection("deliveries").add(entry)
    except Exception as e:
        st.error(f"Error saving delivery: {e}")

def load_user_deliveries(username: str) -> pd.DataFrame:
    try:
        docs = db.collection("deliveries").where("username", "==", username).stream()
        data = [doc.to_dict() for doc in docs]
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading deliveries: {e}")
        return pd.DataFrame()

def add_tip_baiter_to_firestore(entry: Dict) -> None:
    try:
        entry["created_at"] = datetime.now(tz).isoformat()
        db.collection("tip_baiters").add(entry)
    except Exception as e:
        st.error(f"Error saving tip baiter: {e}")

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
        st.error(f"Error loading tip baiters: {e}")
        return pd.DataFrame()

def save_incentives(username: str, incentives: List[Dict]) -> None:
    try:
        db.collection("users").document(username).update({"incentives": incentives})
    except Exception as e:
        st.error(f"Error saving incentives: {e}")

# === ENHANCED OCR PARSING ===
def extract_text_from_image(image_file) -> List[str]:
    """Extract text from an image using OCR."""
    try:
        reader = easyocr.Reader(["en"], gpu=False)
        img_bytes = image_file.read()
        image_file.seek(0)
        return reader.readtext(img_bytes, detail=0)
    except Exception as e:
        st.error(f"OCR processing failed: {e}")
        return []

def parse_screenshot_text(text_list: List[str]) -> Tuple[datetime, float, float, str]:
    """Parse OCR text to extract delivery information."""
    joined = " ".join(text_list).lower()
    
    # Default values
    ts = datetime.now(tz)
    total = 0.0
    ml = 0.0
    order_type = "Delivery"
    
    try:
        # Improved amount detection with multiple patterns
        amount_patterns = [
            r"\$?(\d{1,3}(?:,\d{3})*\.\d{2})",  # $1,234.56 or 1,234.56
            r"\$?(\d+\.\d{2})\b",               # $12.34 or 12.34
            r"\$(\d+)\b",                        # $12 (whole dollar amounts)
            r"total.*?(\d+\.\d{2})"              # Total: 12.34
        ]
        
        for pattern in amount_patterns:
            dollar_matches = re.findall(pattern, joined)
            if dollar_matches:
                amounts = [float(amt.replace(',', '')) for amt in dollar_matches]
                total = max(amounts)
                break
        
        # Miles detection with multiple patterns
        mile_patterns = [
            r"(\d+(?:\.\d)?)\s?mi(?:les)?",     # 1.2 mi or 1.2 miles
            r"distance.*?(\d+(?:\.\d)?)",        # Distance: 1.2
            r"(\d+(?:\.\d)?)\s?miles?\b"        # 1.2 mile or 1.2 miles
        ]
        
        for pattern in mile_patterns:
            miles = re.findall(pattern, joined)
            if miles:
                ml = float(miles[0])
                break
        
        # Enhanced time parsing
        time_patterns = [
            r"\b(\d{1,2}):(\d{2})\s?([ap]m)?\b",  # 12:30 PM or 12:30
            r"\b(\d{1,2})\s?([ap]m)\b",           # 12 PM
            r"time.*?(\d{1,2}):(\d{2})"           # Time: 12:30
        ]
        
        for pattern in time_patterns:
            time_match = re.search(pattern, joined, re.IGNORECASE)
            if time_match:
                groups = time_match.groups()
                if len(groups) >= 2:
                    hour = int(groups[0])
                    minute = int(groups[1]) if len(groups) > 1 and groups[1].isdigit() else 0
                    period = groups[-1].lower() if len(groups) > 2 and groups[-1] in ['am', 'pm'] else None
                    
                    if period:
                        if period == "pm" and hour < 12:
                            hour += 12
                        elif period == "am" and hour == 12:
                            hour = 0
                    elif hour < 6 or hour > 21:
                        current_hour = ts.hour
                        if current_hour < 12:  # AM
                            if hour > 11:
                                hour -= 12
                        else:  # PM
                            if hour < 12:
                                hour += 12
                    
                    try:
                        ts = ts.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    except ValueError:
                        pass
                break
        
        # Enhanced order type detection
        type_keywords = {
            "Shop": ["shop", "s&d", "shopping", "scan", "item", "shopping order"],
            "Pickup": ["pickup", "curbside", "pick up", "store pickup", "pick-up"]
        }
        
        for t, keywords in type_keywords.items():
            if any(kw in joined for kw in keywords):
                order_type = t
                break
    
    except Exception as e:
        st.error(f"Error parsing OCR text: {e}")
    
    return ts, total, ml, order_type

# === INCENTIVE PROCESSING ===
def apply_incentives(order_type: str, order_time: time, base_amount: float, username: str) -> Tuple[float, float, List[Dict]]:
    """Apply any matching incentives to the order."""
    user_data = get_user(username)
    incentives = user_data.get("incentives", [])
    bonus_amount = 0.0
    applied_incentives = []
    
    for incentive in incentives:
        if incentive.get("active", True) and order_type in incentive.get("applies_to", []):
            start_time = datetime.strptime(incentive["start_time"], "%H:%M").time()
            end_time = datetime.strptime(incentive["end_time"], "%H:%M").time()
            
            if start_time <= order_time <= end_time:
                bonus_amount += incentive["amount"]
                applied_incentives.append({
                    "name": incentive["name"],
                    "amount": incentive["amount"]
                })
    
    total_amount = base_amount + bonus_amount
    return total_amount, bonus_amount, applied_incentives

# === AI ANALYTICS ===
def calculate_performance_metrics(df: pd.DataFrame) -> Dict:
    """Calculate various performance metrics from delivery data."""
    metrics = {
        "total_orders": 0,
        "total_earnings": 0.0,
        "epm": 0.0,
        "eph": 0.0,
        "performance": "Unknown",
        "type_distribution": {}
    }
    
    if df.empty:
        return metrics
    
    try:
        metrics["total_orders"] = len(df)
        metrics["total_earnings"] = df["order_total"].sum()
        
        if "miles" in df.columns and df["miles"].sum() > 0:
            metrics["epm"] = metrics["total_earnings"] / df["miles"].sum()
        
        if "timestamp" in df.columns and len(df) > 1:
            df = df.sort_values("timestamp")
            time_diff = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 3600
            if time_diff > 0:
                metrics["eph"] = metrics["total_earnings"] / time_diff
        
        if "order_type" in df.columns:
            metrics["type_distribution"] = df["order_type"].value_counts(normalize=True).to_dict()
        
        if "epm" in metrics and "eph" in metrics:
            for level, criteria in PERFORMANCE_LEVELS.items():
                if metrics["epm"] >= criteria["min_epm"] and metrics["eph"] >= criteria["min_eph"]:
                    metrics["performance"] = level
                    break
        
        return metrics
    
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return metrics

def predict_earnings(df: pd.DataFrame, target_date: date) -> Optional[float]:
    """Predict earnings for a target date using machine learning."""
    if df.empty or "date" not in df.columns:
        return None
    
    try:
        # Prepare data
        df_daily = df.groupby("date")["order_total"].sum().reset_index()
        df_daily["date_ordinal"] = df_daily["date"].apply(lambda d: d.toordinal())
        df_daily["day_of_week"] = df_daily["date"].apply(lambda d: d.weekday())
        
        if len(df_daily) < 5:
            return None
        
        # Feature engineering
        X = df_daily[["date_ordinal", "day_of_week"]]
        y = df_daily["order_total"]
        
        # Try more sophisticated model
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
        except:
            # Fall back to linear regression if RF fails
            model = LinearRegression().fit(X, y)
        
        # Make prediction
        target_ordinal = target_date.toordinal()
        target_dow = target_date.weekday()
        prediction = model.predict(np.array([[target_ordinal, target_dow]]))[0]
        
        return max(0, prediction)
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None