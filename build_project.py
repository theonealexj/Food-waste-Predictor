import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap
import json
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
import nbformat as nbf

def main():
    # Setup
    os.makedirs('food-waste-ml/outputs/figures', exist_ok=True)
    os.makedirs('food-waste-ml/outputs/tables', exist_ok=True)
    os.makedirs('food-waste-ml/outputs/models', exist_ok=True)
    
    # 1. Load Data
    df_raw = pd.read_csv('food-waste-ml/data/food_wastage_data.csv')
    size_before = df_raw.shape
    
    # 2. Data Cleaning
    df_clean = df_raw.drop_duplicates().copy()
    
    # Remove negative or impossible values (just an example based on typical data)
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        df_clean = df_clean[df_clean[col] >= 0]
        
    size_after = df_clean.shape
    
    # Target
    target = 'Wastage Food Amount'
    
    # Identify features
    # Let's say all columns except target
    features = [c for c in df_clean.columns if c != target]
    
    # Simple ML pipeline for demonstration
    numeric_features = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target in numeric_features:
        numeric_features.remove(target)
    categorical_features = df_clean.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X = df_clean[features]
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Baselines and Models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=50),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, n_jobs=-1)
    }
    
    cv_results_summary = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    best_model_name = None
    best_rmse = float('inf')
    best_model = None
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        cv_res = cross_validate(pipeline, X_train, y_train, cv=kf, scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'])
        rmse = -cv_res['test_neg_root_mean_squared_error'].mean()
        cv_results_summary[name] = rmse
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = pipeline
            
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    # Feature Importance for tree models (simplification)
    importances = []
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        feature_names = numeric_features + list(best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))
        impor = best_model.named_steps['model'].feature_importances_
        importances = sorted(zip(feature_names, impor), key=lambda x: x[1], reverse=True)[:5]
    
    # Write summary
    summary = {
        'size_before': size_before,
        'size_after': size_after,
        'features': features,
        'models_tested': list(models.keys()),
        'cv_results': cv_results_summary,
        'best_model': best_model_name,
        'test_results': {'MAE': test_mae, 'RMSE': test_rmse, 'R2': test_r2},
        'top_features': importances
    }
    
    with open('food-waste-ml/summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
        
    print("FINISHED")

if __name__ == "__main__":
    main()
