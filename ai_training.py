# ai_training.py - AI Training and Model Management

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go

# ML Imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import xgboost as xgb

# Database imports - FIXED IMPORT
from database import db, load_user_deliveries, load_user_tip_baiters
from config import tz, ORDER_TYPES
# Import FieldFilter directly from google.cloud.firestore
from google.cloud.firestore import FieldFilter

class DeliveryAI:
    """Advanced AI system for delivery optimization and predictions."""
    
    def __init__(self, username: str = None):
        self.username = username
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def load_all_data(self) -> pd.DataFrame:
        """Load and prepare all delivery data from the database."""
        try:
            # Load all deliveries from all users if no username specified
            if self.username:
                # FIXED: Use FieldFilter directly instead of db.FieldFilter
                docs = db.collection("deliveries").where(filter=FieldFilter("username", "==", self.username)).stream()
            else:
                docs = db.collection("deliveries").stream()
            
            data = []
            for doc in docs:
                entry = doc.to_dict()
                entry["doc_id"] = doc.id
                data.append(entry)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            return self.prepare_features(df)
            
        except Exception as e:
            st.error(f"Error loading training data: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML training."""
        if df.empty:
            return df
        
        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        
        # Time-based features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["quarter"] = df["timestamp"].dt.quarter
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_rush_hour"] = df["hour"].isin([11, 12, 13, 17, 18, 19]).astype(int)
        df["is_lunch"] = df["hour"].isin([11, 12, 13, 14]).astype(int)
        df["is_dinner"] = df["hour"].isin([17, 18, 19, 20]).astype(int)
        df["is_late_night"] = df["hour"].isin([21, 22, 23, 0]).astype(int)
        
        # Distance and efficiency features
        df["miles"] = pd.to_numeric(df["miles"], errors="coerce").fillna(0)
        df["order_total"] = pd.to_numeric(df["order_total"], errors="coerce").fillna(0)
        df["earnings_per_mile"] = df["order_total"] / df["miles"].replace(0, np.nan)
        df["earnings_per_mile"] = df["earnings_per_mile"].fillna(df["earnings_per_mile"].median())
        
        # Order type encoding
        if "order_type" in df.columns:
            df["order_type"] = df["order_type"].fillna("Delivery")
            for order_type in ORDER_TYPES:
                df[f"is_{order_type.lower()}"] = (df["order_type"] == order_type).astype(int)
        
        # Historical performance features (rolling averages)
        df = df.sort_values("timestamp")
        df["rolling_avg_7d"] = df["order_total"].rolling(window=7, min_periods=1).mean()
        df["rolling_avg_30d"] = df["order_total"].rolling(window=30, min_periods=1).mean()
        df["rolling_std_7d"] = df["order_total"].rolling(window=7, min_periods=1).std().fillna(0)
        
        # Distance categories
        df["distance_category"] = pd.cut(df["miles"], 
                                       bins=[0, 2, 5, 10, float("inf")], 
                                       labels=["short", "medium", "long", "very_long"])
        
        # Earnings categories
        df["earnings_category"] = pd.cut(df["order_total"], 
                                       bins=[0, 10, 20, 30, float("inf")], 
                                       labels=["low", "medium", "high", "very_high"])
        
        return df
    
    def train_earnings_predictor(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train models to predict earnings for orders."""
        if df.empty or len(df) < 50:
            return {"error": "Insufficient data for training (need at least 50 records)"}
        
        # Features for prediction
        feature_cols = [
            "hour", "day_of_week", "month", "is_weekend", "is_rush_hour",
            "is_lunch", "is_dinner", "miles", "rolling_avg_7d", "rolling_avg_30d",
            "is_delivery", "is_shop", "is_pickup"
        ]
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feature_cols].fillna(0)
        y = df["order_total"]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Ridge": Ridge(alpha=1.0)
        }
        
        results = {}
        best_model = None
        best_score = float("-inf")
        
        for name, model in models.items():
            try:
                # Train model
                if name in ["RandomForest", "XGBoost", "GradientBoosting"]:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    "model": model,
                    "mae": mae,
                    "mse": mse,
                    "r2": r2,
                    "predictions": y_pred
                }
                
                if r2 > best_score:
                    best_score = r2
                    best_model = name
                    
            except Exception as e:
                results[name] = {"error": str(e)}
        
        # Store best model and scaler
        if best_model:
            self.models["earnings_predictor"] = results[best_model]["model"]
            self.scalers["earnings_predictor"] = scaler
            self.model_performance["earnings_predictor"] = results
            
            # Feature importance for tree-based models
            if best_model in ["RandomForest", "XGBoost", "GradientBoosting"]:
                importance = results[best_model]["model"].feature_importances_
                self.feature_importance["earnings_predictor"] = dict(zip(feature_cols, importance))
        
        return results
    
    def train_optimal_time_predictor(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train model to predict best times to work."""
        if df.empty:
            return {"error": "No data available"}
        
        # Group by hour and day of week
        hourly_stats = df.groupby(["day_of_week", "hour"]).agg({
            "order_total": ["mean", "sum", "count"],
            "miles": "mean"
        }).reset_index()
        
        # Flatten column names
        hourly_stats.columns = ["_".join(col).strip() if col[1] else col[0] 
                               for col in hourly_stats.columns.values]
        
        # Calculate efficiency metrics
        hourly_stats["avg_earnings"] = hourly_stats["order_total_mean"]
        hourly_stats["total_earnings"] = hourly_stats["order_total_sum"]
        hourly_stats["order_count"] = hourly_stats["order_total_count"]
        hourly_stats["earnings_per_mile"] = (hourly_stats["total_earnings"] / 
                                           hourly_stats["miles_mean"]).fillna(0)
        
        # Score each time slot (0-100)
        scaler = StandardScaler()
        metrics = ["avg_earnings", "order_count", "earnings_per_mile"]
        scores = scaler.fit_transform(hourly_stats[metrics])
        hourly_stats["efficiency_score"] = np.mean(scores, axis=1) * 10 + 50
        hourly_stats["efficiency_score"] = np.clip(hourly_stats["efficiency_score"], 0, 100)
        
        self.models["optimal_times"] = hourly_stats
        return {"success": True, "data": hourly_stats}
    
    def perform_customer_clustering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cluster delivery locations to identify profitable areas."""
        if df.empty or "miles" not in df.columns:
            return {"error": "Insufficient location data"}
        
        try:
            # Create features for clustering
            location_features = df.groupby(df.index // 10).agg({  # Group nearby orders
                "order_total": "mean",
                "miles": "mean",
                "earnings_per_mile": "mean",
                "hour": "mean"
            }).reset_index(drop=True)
            
            # Remove invalid data
            location_features = location_features.dropna()
            if len(location_features) < 10:
                return {"error": "Insufficient data for clustering"}
            
            # Perform clustering
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(location_features)
            
            # Find optimal number of clusters
            n_clusters = min(5, len(location_features) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            
            location_features["cluster"] = clusters
            
            # Analyze clusters
            cluster_analysis = location_features.groupby("cluster").agg({
                "order_total": ["mean", "std"],
                "miles": ["mean", "std"],
                "earnings_per_mile": ["mean", "std"],
                "hour": "mean"
            }).round(2)
            
            self.models["customer_clusters"] = {
                "model": kmeans,
                "scaler": scaler,
                "data": location_features,
                "analysis": cluster_analysis
            }
            
            return {"success": True, "clusters": n_clusters, "analysis": cluster_analysis}
            
        except Exception as e:
            return {"error": f"Clustering failed: {str(e)}"}
    
    def predict_earnings(self, hour: int, day_of_week: int, miles: float, 
                        order_type: str = "Delivery") -> Optional[float]:
        """Predict earnings for a potential order."""
        if "earnings_predictor" not in self.models:
            return None
        
        try:
            # Prepare features
            features = {
                "hour": hour,
                "day_of_week": day_of_week,
                "month": datetime.now().month,
                "is_weekend": 1 if day_of_week in [5, 6] else 0,
                "is_rush_hour": 1 if hour in [11, 12, 13, 17, 18, 19] else 0,
                "is_lunch": 1 if hour in [11, 12, 13, 14] else 0,
                "is_dinner": 1 if hour in [17, 18, 19, 20] else 0,
                "miles": miles,
                "rolling_avg_7d": 15.0,  # Default average
                "rolling_avg_30d": 15.0,  # Default average
                "is_delivery": 1 if order_type == "Delivery" else 0,
                "is_shop": 1 if order_type == "Shop" else 0,
                "is_pickup": 1 if order_type == "Pickup" else 0
            }
            
            X = np.array([list(features.values())]).reshape(1, -1)
            
            # Use appropriate scaler if needed
            model = self.models["earnings_predictor"]
            if "earnings_predictor" in self.scalers:
                X = self.scalers["earnings_predictor"].transform(X)
            
            prediction = model.predict(X)[0]
            return max(0, prediction)  # Ensure non-negative
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def get_optimization_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate actionable recommendations based on data analysis."""
        if df.empty:
            return {"error": "No data available for recommendations"}
        
        recommendations = {
            "optimal_hours": [],
            "avoid_hours": [],
            "best_order_types": [],
            "distance_recommendations": [],
            "general_tips": []
        }
        
        try:
            # Analyze hourly performance
            hourly_perf = df.groupby("hour").agg({
                "order_total": ["mean", "count"],
                "earnings_per_mile": "mean"
            }).round(2)
            
            # Find best and worst hours
            best_hours = hourly_perf["order_total"]["mean"].nlargest(3).index.tolist()
            worst_hours = hourly_perf["order_total"]["mean"].nsmallest(3).index.tolist()
            
            recommendations["optimal_hours"] = [f"{h}:00" for h in best_hours]
            recommendations["avoid_hours"] = [f"{h}:00" for h in worst_hours]
            
            # Analyze order types
            if "order_type" in df.columns:
                type_perf = df.groupby("order_type")["earnings_per_mile"].mean().sort_values(ascending=False)
                recommendations["best_order_types"] = type_perf.index.tolist()
            
            # Distance analysis
            distance_perf = df.groupby(pd.cut(df["miles"], bins=[0, 3, 7, 15, float("inf")], 
                                            labels=["<3mi", "3-7mi", "7-15mi", ">15mi"]))["earnings_per_mile"].mean()
            best_distance = distance_perf.idxmax()
            recommendations["distance_recommendations"] = [
                f"Focus on orders in the {best_distance} range for best efficiency"
            ]
            
            # General tips based on performance
            avg_epm = df["earnings_per_mile"].mean()
            if avg_epm < 2.0:
                recommendations["general_tips"].append("Focus on shorter distance orders to improve earnings per mile")
            if df.groupby("is_weekend")["order_total"].mean().loc[1] > df.groupby("is_weekend")["order_total"].mean().loc[0]:
                recommendations["general_tips"].append("Weekend shifts appear more profitable")
            
            return recommendations
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def save_models(self, filepath: str = "delivery_ai_models.pkl") -> bool:
        """Save trained models to disk."""
        try:
            model_data = {
                "models": self.models,
                "scalers": self.scalers,
                "encoders": self.encoders,
                "feature_importance": self.feature_importance,
                "model_performance": self.model_performance,
                "username": self.username,
                "trained_at": datetime.now().isoformat()
            }
            
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)
            return True
            
        except Exception as e:
            st.error(f"Failed to save models: {e}")
            return False
    
    def load_models(self, filepath: str = "delivery_ai_models.pkl") -> bool:
        """Load trained models from disk."""
        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)
            
            self.models = model_data.get("models", {})
            self.scalers = model_data.get("scalers", {})
            self.encoders = model_data.get("encoders", {})
            self.feature_importance = model_data.get("feature_importance", {})
            self.model_performance = model_data.get("model_performance", {})
            
            return True
            
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            return False

def display_ai_training_interface(username: str) -> None:
    """Streamlit interface for AI training and management."""
    st.subheader("🤖 AI Training Center")
    
    # Initialize AI system
    ai_system = DeliveryAI(username)
    
    # Training options
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🎯 Train AI Models")
        
        if st.button("📄 Load Training Data", type="primary"):
            with st.spinner("Loading all delivery data..."):
                df = ai_system.load_all_data()
                if not df.empty:
                    st.session_state.training_data = df
                    st.success(f"✅ Loaded {len(df)} delivery records for training")
                    
                    # Show data overview
                    with st.expander("📊 Data Overview"):
                        st.write(f"**Date range:** {df['timestamp'].min()} to {df['timestamp'].max()}")
                        st.write(f"**Total earnings:** ${df['order_total'].sum():.2f}")
                        st.write(f"**Average per order:** ${df['order_total'].mean():.2f}")
                        st.write(f"**Total miles:** {df['miles'].sum():.1f}")
                        
                        # Show sample data
                        st.dataframe(df.head())
                else:
                    st.warning("No training data found")
        
        # Training buttons
        if "training_data" in st.session_state:
            df = st.session_state.training_data
            
            if st.button("🎯 Train Earnings Predictor"):
                with st.spinner("Training earnings prediction model..."):
                    results = ai_system.train_earnings_predictor(df)
                    if "error" not in results:
                        st.success("✅ Earnings predictor trained successfully!")
                        
                        # Show model performance
                        best_model = max(results.keys(), key=lambda k: results[k].get("r2", 0))
                        st.write(f"**Best Model:** {best_model}")
                        st.write(f"**R² Score:** {results[best_model]['r2']:.3f}")
                        st.write(f"**Mean Absolute Error:** ${results[best_model]['mae']:.2f}")
                    else:
                        st.error(f"Training failed: {results['error']}")
            
            if st.button("⏰ Analyze Optimal Times"):
                with st.spinner("Analyzing optimal work times..."):
                    results = ai_system.train_optimal_time_predictor(df)
                    if "error" not in results:
                        st.success("✅ Time analysis completed!")
                        
                        # Show heatmap of optimal times
                        data = results["data"]
                        pivot_data = data.pivot(index="day_of_week", columns="hour", values="efficiency_score")
                        
                        fig = px.imshow(pivot_data, 
                                       title="Work Time Efficiency Heatmap",
                                       labels=dict(x="Hour", y="Day of Week", color="Efficiency Score"),
                                       color_continuous_scale="RdYlGn")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Analysis failed: {results['error']}")
            
            if st.button("🎯 Customer Clustering"):
                with st.spinner("Clustering delivery areas..."):
                    results = ai_system.perform_customer_clustering(df)
                    if "error" not in results:
                        st.success(f"✅ Identified {results['clusters']} delivery clusters!")
                        st.dataframe(results["analysis"])
                    else:
                        st.error(f"Clustering failed: {results['error']}")
    
    with col2:
        st.write("### 🔮 AI Predictions & Insights")
        
        # Prediction interface
        if "earnings_predictor" in ai_system.models:
            st.write("#### Earnings Predictor")
            pred_col1, pred_col2 = st.columns(2)
            
            with pred_col1:
                pred_hour = st.slider("Hour", 0, 23, 12)
                pred_miles = st.number_input("Miles", 0.1, 20.0, 3.0, 0.1)
            
            with pred_col2:
                pred_day = st.selectbox("Day of Week", 
                                       ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                        "Friday", "Saturday", "Sunday"])
                pred_type = st.selectbox("Order Type", ORDER_TYPES)
            
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                      "Friday": 4, "Saturday": 5, "Sunday": 6}
            
            prediction = ai_system.predict_earnings(pred_hour, day_map[pred_day], pred_miles, pred_type)
            if prediction:
                st.metric("Predicted Earnings", f"${prediction:.2f}")
                st.metric("Predicted EPM", f"${prediction/pred_miles:.2f}")
        
        # Recommendations
        if "training_data" in st.session_state:
            if st.button("💡 Get AI Recommendations"):
                with st.spinner("Generating recommendations..."):
                    recs = ai_system.get_optimization_recommendations(st.session_state.training_data)
                    if "error" not in recs:
                        st.write("#### 🎯 Optimization Recommendations")
                        
                        if recs["optimal_hours"]:
                            st.success(f"**Best hours to work:** {', '.join(recs['optimal_hours'])}")
                        
                        if recs["avoid_hours"]:
                            st.warning(f"**Consider avoiding:** {', '.join(recs['avoid_hours'])}")
                        
                        if recs["best_order_types"]:
                            st.info(f"**Most profitable order types:** {', '.join(recs['best_order_types'])}")
                        
                        for tip in recs["general_tips"]:
                            st.write(f"💡 {tip}")
    
    # Model management
    st.write("### 💾 Model Management")
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("💾 Save Models"):
            if ai_system.save_models():
                st.success("Models saved successfully!")
    
    with col4:
        if st.button("📁 Load Models"):
            if ai_system.load_models():
                st.success("Models loaded successfully!")
    
    # Feature importance visualization
    if ai_system.feature_importance.get("earnings_predictor"):
        st.write("### 📊 Feature Importance")
        importance_data = ai_system.feature_importance["earnings_predictor"]
        
        fig = px.bar(
            x=list(importance_data.values()),
            y=list(importance_data.keys()),
            orientation='h',
            title="Feature Importance for Earnings Prediction"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)