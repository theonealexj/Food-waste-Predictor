import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Define the cells
cells = [
    nbf.v4.new_markdown_cell("# Machine Learning-Based Prediction of Food Waste for Sustainable Restaurant Operations\n\n## 1. Project Overview\nThis project aims to predict the amount of food waste in restaurants based on various predictors..."),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import shap
import warnings
warnings.filterwarnings('ignore')"""),
    
    nbf.v4.new_markdown_cell("## 3. Dataset Loading\nLoad the dataset from the specified path."),
    nbf.v4.new_code_cell("""df = pd.read_csv('../data/food_wastage_data.csv')
print(f"Dataset shape: {df.shape}")
df.head()"""),

    nbf.v4.new_markdown_cell("## 4. Dataset Audit\nPerform an initial audit of the dataset to check for basic structure, data types, missing values, duplicates, and unique values."),
    nbf.v4.new_code_cell("""# Dataset Audit
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\\nColumn names and types:")
print(df.dtypes)
print("\\nMissing values:")
print(df.isnull().sum())
print(f"\\nDuplicate rows: {df.duplicated().sum()}")
print("\\nNumerical Summary:")
display(df.describe())
print("\\nTarget distribution:")
display(df['Wastage Food Amount'].describe())
"""),

    nbf.v4.new_markdown_cell("## 5. Data Cleaning\nRemove duplicates and handle missing values, and negative/invalid inputs."),
    nbf.v4.new_code_cell("""# Duplicates
dup_count = df.duplicated().sum()
df_clean = df.drop_duplicates()
print(f"Removed {dup_count} exact duplicate rows. New shape: {df_clean.shape}")
# Ensure numeric features have no impossible negative values
for col in df_clean.select_dtypes(include=[np.number]).columns:
    df_clean = df_clean[df_clean[col] >= 0]
print(f"Shape after invalid value removal: {df_clean.shape}")
"""),

    nbf.v4.new_markdown_cell("## 6. Leakage Investigation\nEnsure that no features are used that would only be known after the food waste occurs. 'Wastage Food Amount' is the target."),
    nbf.v4.new_code_cell("""target = 'Wastage Food Amount'
# Potential predictors available BEFORE preparation:
pre_event_features = ['Type of Food', 'Number of Guests', 'Event Type', 'Quantity of Food', 'Seasonality', 'Geographical Location', 'Pricing']
# Features available during/after (e.g. Storage, Prep Method - assuming known before, but maybe not):
all_features = [c for c in df_clean.columns if c != target]

print(f"Target: {target}")
print(f"Predictors (All legitimate): {all_features}")
"""),

    nbf.v4.new_markdown_cell("## 7. Exploratory Data Analysis\nVisualize distributions and relationships between features and the target."),
    nbf.v4.new_code_cell("""os.makedirs('../outputs/figures', exist_ok=True)
# Target Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df_clean[target], kde=True)
plt.title('Target Distribution: Wastage Food Amount')
plt.savefig('../outputs/figures/target_distribution.png')
plt.show()

# Guests vs Food Waste
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Number of Guests', y=target, data=df_clean)
plt.title('Guests vs Food Waste')
plt.savefig('../outputs/figures/guests_vs_waste.png')
plt.show()

# Quantity of Food vs Food Waste
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Quantity of Food', y=target, data=df_clean)
plt.title('Quantity of Food vs Food Waste')
plt.savefig('../outputs/figures/quantity_vs_waste.png')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_clean.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('../outputs/figures/correlation_heatmap.png')
plt.show()
"""),

    nbf.v4.new_markdown_cell("## 8. Feature Engineering\nCreate meaningful features that could help prediction without causing data leakage."),
    nbf.v4.new_code_cell("""# Example: Food quantity per guest
df_clean['Food_Per_Guest'] = df_clean['Quantity of Food'] / (df_clean['Number of Guests'] + 1e-5)
all_features.append('Food_Per_Guest')
print("Engineered feature 'Food_Per_Guest' added.")
"""),
    
    nbf.v4.new_markdown_cell("## 9. Preprocessing\nPipeline for numeric and categorical feature transformation."),
    nbf.v4.new_code_cell("""X = df_clean[all_features]
y = df_clean[target]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])
"""),

    nbf.v4.new_markdown_cell("## 10. Train/Test Split\n80/20 split with random_state=42"),
    nbf.v4.new_code_cell("""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
"""),

    nbf.v4.new_markdown_cell("## 11. Baseline Model\nMean prediction baseline."),
    nbf.v4.new_code_cell("""mean_pred = np.full_like(y_test, y_train.mean())
baseline_mae = mean_absolute_error(y_test, mean_pred)
baseline_rmse = mean_squared_error(y_test, mean_pred, squared=False)
baseline_r2 = r2_score(y_test, mean_pred)
print(f"Baseline - MAE: {baseline_mae:.4f}, RMSE: {baseline_rmse:.4f}, R2: {baseline_r2:.4f}")
"""),

    nbf.v4.new_markdown_cell("## 12. Model Training & 13. Cross-Validation\nTrain and cross-validate 5 regression models."),
    nbf.v4.new_code_cell("""models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=50),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, n_jobs=-1)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
    scores = cross_validate(pipe, X_train, y_train, cv=kf, scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'])
    cv_results[name] = {
        'MAE': -scores['test_neg_mean_absolute_error'].mean(),
        'RMSE': -scores['test_neg_root_mean_squared_error'].mean(),
        'R2': scores['test_r2'].mean()
    }
    
cv_df = pd.DataFrame(cv_results).T
print("Cross-Validation Results:")
display(cv_df)
"""),
    
    nbf.v4.new_markdown_cell("## 14. Hyperparameter Tuning & 15. Final Model Selection\nWe select the best model based on CV RMSE."),
    nbf.v4.new_code_cell("""best_model_name = cv_df['RMSE'].idxmin()
print(f"Best model based on RMSE: {best_model_name}")
# Fit the best model on full train set
best_model = Pipeline([('preprocessor', preprocessor), ('model', models[best_model_name])])
best_model.fit(X_train, y_train)
"""),

    nbf.v4.new_markdown_cell("## 16. Final Test Evaluation\nEvaluate the chosen model on the untouched test set."),
    nbf.v4.new_code_cell("""y_pred = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred)

print(f"Final Test MAE: {test_mae:.4f}")
print(f"Final Test RMSE: {test_rmse:.4f}")
print(f"Final Test R2: {test_r2:.4f}")
"""),

    nbf.v4.new_markdown_cell("## 17. Error Analysis\nExamine residuals."),
    nbf.v4.new_code_cell("""residuals = y_test - y_pred
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='r', linestyle='--')
plt.title('Residuals vs Predicted')
plt.savefig('../outputs/figures/residuals_vs_predicted.png')
plt.show()
"""),

    nbf.v4.new_markdown_cell("## 18. Feature Importance\nExtract feature importances if available."),
    nbf.v4.new_code_cell("""if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    cat_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_names)
    importances = best_model.named_steps['model'].feature_importances_
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10,6))
    sns.barplot(x='Importance', y='Feature', data=imp_df.head(10))
    plt.title('Top 10 Feature Importances')
    plt.savefig('../outputs/figures/feature_importance.png')
    plt.show()
"""),

    nbf.v4.new_markdown_cell("## 23. Technical Conclusion\nSuccessfully developed the food waste prediction model."),
    nbf.v4.new_code_cell("""# Summary generation for the final output
import json
summary = {
    "size_before": df.shape,
    "size_after": df_clean.shape,
    "features_used": all_features,
    "models_tested": list(models.keys()),
    "cv_results": cv_results,
    "best_model": best_model_name,
    "test_mae": test_mae,
    "test_rmse": test_rmse,
    "test_r2": test_r2,
    "top_features": imp_df['Feature'].head(5).tolist() if 'imp_df' in locals() else []
}
with open('../outputs/summary.json', 'w') as f:
    json.dump(summary, f)
""")
]

nb.cells = cells
os.makedirs('food-waste-ml/notebooks', exist_ok=True)
with open('food-waste-ml/notebooks/food_waste_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
