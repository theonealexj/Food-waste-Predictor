# Project Code Walkthrough

This document provides a comprehensive, file-by-file and line-by-line explanation of the entire Machine Learning codebase for the Food Waste Prediction project. This serves as your presentation material for explaining the technical implementation to your academic audience.

---

## 1. `build_project.py`

This script executes the entire machine learning pipeline end-to-end. It cleans data, performs feature engineering, evaluates multiple models using cross-validation, and saves a summary of the results.

### Imports
- `import pandas as pd`, `import numpy as np`: Standard libraries for data manipulation and numerical operations.
- `import matplotlib.pyplot as plt`, `import seaborn as sns`: Plotting libraries for generating visualizations (though most visualizations are generated in the notebook).
- `import os`, `import json`: Built-in libraries for file system operations and exporting the final results as JSON.
- `from sklearn.*`: Imports scikit-learn modules for splitting data (`train_test_split`), cross-validation (`KFold`, `cross_validate`), preprocessing (`StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`), and evaluation metrics (`mean_absolute_error`, `mean_squared_error`, `r2_score`).
- `from sklearn.linear_model, .tree, .ensemble`: Imports the baseline regression algorithms.
- `from xgboost import XGBRegressor`: Imports the advanced gradient boosting algorithm.

### Function `main()`
- **Lines 20-22:** `os.makedirs(...)` creates the output directory structure (`figures`, `tables`, `models`) if they don't already exist.
- **Lines 25-26:** `pd.read_csv(...)` loads the dataset into memory and records the initial row/column count (`size_before`).
- **Lines 29-34:** `df_raw.drop_duplicates().copy()` removes identical rows to prevent duplicate-induced data leakage. A loop iterates through numeric columns ensuring all quantities are `>= 0` (removing invalid negative inputs).
- **Line 36:** Records the cleaned dataset size (`size_after`).
- **Lines 39-43:** Explicitly defines `target = 'Wastage Food Amount'`. Then, creates a list of all `features` by grabbing all column names except the target.
- **Lines 46-49:** Dynamically separates features into numerical (`numeric_features`) and categorical (`categorical_features`) based on their data types.
- **Lines 51-55:** Initializes a `ColumnTransformer`. This is a vital scikit-learn tool that independently routes numerical columns through a `StandardScaler` (making them have a mean of 0 and variance of 1) and categorical columns through a `OneHotEncoder` (converting text categories into 1s and 0s).
- **Lines 57-58:** Separates the inputs (`X`) from the target variable (`y`).
- **Line 60:** `train_test_split(...)` shuffles and splits the data. 80% is kept for training and validation, while 20% is locked away for final testing. `random_state=42` ensures reproducibility.
- **Lines 63-69:** Instantiates a Python dictionary containing five fundamentally different regression algorithms, fulfilling the assignment requirement.
- **Lines 71-76:** Sets up a 5-fold cross-validation strategy (`KFold(n_splits=5)`) and initializes variables to track the best-performing model.
- **Lines 78-86:** The core training loop. For every model:
  1. It builds a `Pipeline` connecting the `preprocessor` and the `model`.
  2. It evaluates the pipeline using `cross_validate`, automatically handling the 5 folds.
  3. It extracts the average Root Mean Squared Error (RMSE).
  4. It records the score and continuously updates the `best_model` if the current model has the lowest RMSE.
- **Lines 88-92:** After selecting the absolute best model, it retrains it on the *entire* 80% training set, then predicts on the untouched `X_test` (the 20% test set). Finally, it calculates standard metrics: Mean Absolute Error (MAE), RMSE, and R-squared.
- **Lines 95-99:** If the winning model is tree-based (like Random Forest or XGBoost), it extracts the built-in feature importance scores, maps them back to the column names, and sorts them to identify what drives food waste.
- **Lines 102-111:** Packages all recorded statistics, data sizes, metrics, and top features into a Python dictionary (`summary`).
- **Lines 113-114:** Writes the dictionary to disk as `outputs/summary.json` so the results can be utilized elsewhere.

---

## 2. `create_notebook.py`

This script programmatically generates a complete Jupyter Notebook (`food_waste_analysis.ipynb`). This allows the project to maintain reproducibility while providing an interactive academic artifact.

### Imports
- `import nbformat as nbf`: A library for reading, writing, and programmatically constructing Jupyter Notebook files.
- `import os`: For file system operations.

### Notebook Generation
- **Line 4:** `nb = nbf.v4.new_notebook()` creates an empty, virtual Jupyter Notebook object in memory.
- **Lines 7-221:** Defines a Python list named `cells`. This list contains a sequence of `new_markdown_cell` (for text and explanations) and `new_code_cell` (for actual Python code).
  - **Cell 1 & 2:** Markdown overview and the block of global imports needed for the notebook to function.
  - **Cell 3 & 4:** Data loading. Demonstrates reading the CSV and previewing the first few rows.
  - **Cell 5 & 6:** Comprehensive dataset audit. Prints data types, missing value counts, and descriptive statistics.
  - **Cell 7 & 8:** Data cleaning block. Explicitly drops duplicates and invalid negative inputs.
  - **Cell 9 & 10:** Identifies the target and the predictors. Contains textual analysis of what features represent pre-event knowledge versus post-event knowledge (preventing leakage).
  - **Cell 11 & 12:** Exploratory Data Analysis (EDA). Contains `seaborn` plotting code to generate distributions, scatter plots comparing variables to waste, and a correlation heatmap. These are automatically saved to `outputs/figures/`.
  - **Cell 13 & 14:** Feature Engineering. Creates a mathematical ratio feature (`Food_Per_Guest`) as an example of deriving new information from existing features.
  - **Cell 15-24:** Contains the interactive replication of the `build_project.py` machine learning pipeline. It builds the preprocessing pipeline, splits the data, runs the mean-prediction baseline, executes cross-validation across all 5 models, and identifies the best performer.
  - **Cell 25-28:** Post-modeling analysis. Calculates final test metrics, plots a residual distribution (Predicted vs Residuals scatterplot) to check for error patterns, and plots a bar chart of the most important features.
- **Lines 223-227:** Attaches the giant list of `cells` to the empty notebook object (`nb`). It ensures the `notebooks` directory exists, and finally uses `nbf.write` to compile everything into the physical file `notebooks/food_waste_analysis.ipynb`.

---

## 3. `summary.json` (Generated Output)

This is a lightweight data file created after running `build_project.py`.

- It is written in JavaScript Object Notation (JSON).
- It strictly acts as a data-store for your final metrics, preventing you from having to rerun the entire ML pipeline just to check what the R-squared score was.
- It proves the data size shrunk from 1782 to 1618 (due to duplicate removal).
- It maps the models to their cross-validation RMSE scores, mathematically proving that Gradient Boosting performed the best.
