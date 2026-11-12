"""
Explainable AI Engine using SHAP (SHapley Additive exPlanations).
Provides transparent feature contributions for both XGBoost and Random Forest predictions.
Transforms complex SHAP values into clean, human-readable explanations and chart-ready structures.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import shap


SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


class FraudExplainer:
    def __init__(self):
        self.preprocessor_info = None
        self.xgb_model = None
        self.rf_model = None
        self.xgb_explainer = None
        self.rf_explainer = None
        self._load_artifacts()


    def _load_artifacts(self):
        prep_path = os.path.join(SAVED_MODELS_DIR, "preprocessor.joblib")
        if os.path.exists(prep_path):
            self.preprocessor_info = joblib.load(prep_path)


        xgb_path = os.path.join(SAVED_MODELS_DIR, "xgboost_model.joblib")
        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)


        rf_path = os.path.join(SAVED_MODELS_DIR, "random_forest_model.joblib")
        if os.path.exists(rf_path):
            self.rf_model = joblib.load(rf_path)


        xgb_exp_path = os.path.join(SAVED_MODELS_DIR, "xgb_shap_explainer.joblib")
        if os.path.exists(xgb_exp_path):
            try:
                self.xgb_explainer = joblib.load(xgb_exp_path)
            except Exception:
                self.xgb_explainer = None


        rf_exp_path = os.path.join(SAVED_MODELS_DIR, "rf_shap_explainer.joblib")
        if os.path.exists(rf_exp_path):
            try:
                self.rf_explainer = joblib.load(rf_exp_path)
            except Exception:
                self.rf_explainer = None


    def explain_transaction(
        self,
        transaction_features_df: pd.DataFrame,
        model_name: str = "xgboost"
    ) -> Dict[str, Any]:
        
        """
        Generates individual transaction SHAP explanation for the chosen model.
        Returns top positive (risk-increasing) and negative (risk-decreasing) features.
        """

        if self.preprocessor_info is None:
            self._load_artifacts()


        feature_names = self.preprocessor_info["feature_names"] if self.preprocessor_info else list(transaction_features_df.columns)
        model = self.xgb_model if model_name.lower() == "xgboost" else self.rf_model
        explainer = self.xgb_explainer if model_name.lower() == "xgboost" else self.rf_explainer


        if model is None:
            return {
                "available": False,
                "message": f"Model '{model_name}' artifacts not found. Please train models first.",
                "contributions": []
            }

        try:
            # If explainer is not preloaded or serialized tree differs, build quick TreeExplainer

            if explainer is None:
                explainer = shap.TreeExplainer(model)

            shap_values = explainer(transaction_features_df)


            # Format shap values (handling 2D vs 3D for multiclass/binary)

            values = shap_values.values[0]
            if len(values.shape) > 1 and values.shape[-1] == 2:
                # Class 1 (fraud)
                values = values[:, 1]
            elif len(values.shape) > 1:
                values = values[0]

            base_val = float(shap_values.base_values[0]) if hasattr(shap_values.base_values, "__len__") else float(shap_values.base_values)


            # Build feature breakdown list

            contributions = []
            friendly_names = {
                "amt": "Transaction Amount",
                "amt_log": "Log Transaction Amount",
                "distance_km": "Cardholder-Merchant Distance (km)",
                "hour": "Hour of Day",
                "day_of_week": "Day of Week",
                "is_weekend": "Weekend Indicator",
                "month": "Transaction Month",
                "age": "Cardholder Age",
                "city_pop": "City Population",
                "category_encoded": "Merchant Category",
                "gender_encoded": "Cardholder Gender"
            }


            for i, col in enumerate(feature_names):
                val = float(values[i])
                raw_val = transaction_features_df.iloc[0][col] if col in transaction_features_df.columns else None
                contributions.append({
                    "feature": col,
                    "label": friendly_names.get(col, col),
                    "shap_value": round(val, 4),
                    "impact": "INCREASES_RISK" if val > 0 else "DECREASES_RISK",
                    "magnitude": round(abs(val), 4),
                    "feature_value": round(float(raw_val), 2) if isinstance(raw_val, (int, float)) else str(raw_val)
                })


            # Sort by impact magnitude

            contributions.sort(key=lambda x: x["magnitude"], reverse=True)

            top_risk_drivers = [c for c in contributions if c["shap_value"] > 0][:4]
            top_mitigating_factors = [c for c in contributions if c["shap_value"] < 0][:4]

            # Generate natural language summary

            reasons = []
            for item in top_risk_drivers:
                reasons.append(f"{item['label']} contributed +{item['shap_value']:.2f} toward fraud risk")


            return {
                "available": True,
                "model_name": model_name,
                "base_value": round(base_val, 4),
                "contributions": contributions,
                "top_risk_drivers": top_risk_drivers,
                "top_mitigating_factors": top_mitigating_factors,
                "key_reasons": reasons if reasons else ["Standard low-risk pattern across all features."]
            }


        except Exception as e:
            # Fallback based on model feature importances
            
            importances = getattr(model, "feature_importances_", None)
            fallback_items = []
            if importances is not None:
                for i, col in enumerate(feature_names):
                    fallback_items.append({
                        "feature": col,
                        "label": col.replace("_", " ").title(),
                        "shap_value": round(float(importances[i]), 4),
                        "impact": "GLOBAL_IMPORTANCE",
                        "magnitude": round(float(importances[i]), 4),
                        "feature_value": "N/A"
                    })
                fallback_items.sort(key=lambda x: x["magnitude"], reverse=True)


            return {
                "available": False,
                "error": str(e),
                "model_name": model_name,
                "base_value": 0.5,
                "contributions": fallback_items,
                "top_risk_drivers": fallback_items[:3],
                "top_mitigating_factors": [],
                "key_reasons": ["SHAP calculation approximation used."]
            }