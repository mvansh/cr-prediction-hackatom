import mlflow
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap

SHAP_SUBSAMPLE = 50
EXCLUDE_C_PLANT_SOIL = False  

try:
    df = pd.read_csv("cleaned_reduced_dataset.csv")
    print(f"Dataset loaded successfully with {len(df)} rows.")
    print("Column names in dataset:", list(df.columns))
    print("\nSample data (first 5 rows):")
    print(df.head().to_string())
except FileNotFoundError:
    print("Error: Dataset file 'cleaned_reduced_dataset.csv' not found.")
    exit()

def safe_to_float(x):
    if isinstance(x, (int, float)) and not np.isnan(x):
        return float(x)
    if isinstance(x, str):
        if x.startswith('<'):
            try:
                return float(x.replace('<', ''))
            except ValueError:
                return np.nan
        try:
            return float(x)
        except ValueError:
            match = re.search(r"-?\d*\.?\d+[Ee][+-]?\d+", x)
            return float(match.group(0)) if match else np.nan
    return np.nan

numerical_cols = ['CR', 'C_plant', 'C_soil']
for col in numerical_cols:
    if df[col].dtype == 'object' or df[col].isna().any():
        print(f"\nUnique values in {col} before cleaning:", df[col].unique()[:10])
    df[col] = df[col].apply(safe_to_float)
    df[col] = df[col].fillna(df[col].median())
    if df[col].dtype == 'object' or df[col].isna().any():
        print(f"Unique values in {col} after cleaning:", df[col].unique()[:10])

print(f"\nAfter cleaning numeric columns, dataset has {len(df)} rows.")
df = df.dropna(subset=['CR'])
print(f"After dropping rows with missing CR, dataset has {len(df)} rows.")
print("Sample CR values:", df['CR'].head(10).tolist())

if len(df) == 0:
    print("Error: Dataset is empty after cleaning. Check CR column for invalid values.")
    exit()

categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')
    value_counts = df[col].value_counts(normalize=True)
    rare_values = value_counts[value_counts < 0.05].index
    df.loc[df[col].isin(rare_values), col] = 'Other'

exclude_cols = ['CR'] if not EXCLUDE_C_PLANT_SOIL else ['CR', 'C_plant', 'C_soil']
X = pd.get_dummies(df.drop(exclude_cols, axis=1), columns=categorical_cols)
X = X.astype(np.float64)
y = df['CR']
y_log = np.log1p(y)
print(f"\nAfter encoding, dataset has {X.shape[1]} features.")
print("X_train dtypes:", X.dtypes.unique())

try:
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
except ValueError as e:
    print(f"Error during train-test split: {e}")
    exit()

# Start MLflow run
with mlflow.start_run():
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    print("\nTraining XGBoost...")
    model.fit(X_train, y_train_log)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test_log)

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)

    print(f"XGBoost - RMSE (on real CR scale): {rmse:.4f}")
    print(f"XGBoost - R² Score: {r2:.4f}")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = X.columns[sorted_idx]
    print(f"\nTop 5 Most Important Features for XGBoost:")
    print(pd.DataFrame({'Feature': sorted_features[:5], 'Importance': importances[sorted_idx][:5]}))
    mlflow.log_text(
        pd.DataFrame({'Feature': sorted_features[:5], 'Importance': importances[sorted_idx][:5]}).to_csv(),
        "feature_importance.csv"
    )

    print(f"\nComputing SHAP values for XGBoost...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test[:SHAP_SUBSAMPLE].astype(np.float64))

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:SHAP_SUBSAMPLE], show=False)
    plt.title(f"SHAP Summary Plot (XGBoost)")
    plt.tight_layout()
    plt.savefig(f"shap_summary_xgboost.png")
    plt.close()
    mlflow.log_artifact("shap_summary_xgboost.png")
    print(f"SHAP summary plot saved to: shap_summary_xgboost.png")

    results_df = pd.DataFrame({
        "Actual_CR": y_test_actual,
        "Predicted_CR": y_pred,
        "Absolute_Error": np.abs(y_test_actual - y_pred)
    })
    results_df.to_csv(f"CR_predictions_xgboost.csv", index=False)
    mlflow.log_artifact("CR_predictions_xgboost.csv")
    print(f"Predictions saved to: CR_predictions_xgboost.csv")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_actual, y=y_pred, alpha=0.7)
    plt.xlabel("Actual CR")
    plt.ylabel("Predicted CR")
    plt.title(f"Actual vs Predicted CR Values (XGBoost)")
    plt.plot(
        [y_test_actual.min(), y_test_actual.max()],
        [y_test_actual.min(), y_test_actual.max()],
        'r--'
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"actual_vs_predicted_xgboost.png")
    plt.close()
    mlflow.log_artifact("actual_vs_predicted_xgboost.png")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[sorted_idx][:10], y=sorted_features[:10])
    plt.title(f"Feature Importance (XGBoost)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"feature_importance_xgboost.png")
    plt.close()
    mlflow.log_artifact("feature_importance_xgboost.png")

    new_sample = pd.DataFrame(columns=df.drop('CR', axis=1).columns)
    for col, val in {'Element': 'U', 'Radionuclide': 'U-238', 'Common name': 'Cassava', 'Latin name': 'Manihot esculenta', 'Compartment': 'Roots', 'C_plant': 0.41, 'C_soil': 4.2, 'Country': 'Ghana', 'Site': 'Tano basin (CS1)', 'K-G class': 'Am', 'Contamination': 'N', 'Experiment': 'F', 'Soil depth': '0-20'}.items():
        if col in new_sample.columns:
            new_sample[col] = [val]

    for col in numerical_cols:
        if col in new_sample.columns and col != 'CR':
            new_sample[col] = pd.to_numeric(new_sample[col], errors='coerce').fillna(df[col].median())

    for col in categorical_cols:
        if col in new_sample.columns and new_sample[col].iloc[0] not in df[col].unique():
            new_sample[col] = 'Other'

    new_sample_encoded = pd.get_dummies(new_sample).reindex(columns=X.columns, fill_value=0).astype(np.float64)
    print("\nNew sample before encoding:")
    print(new_sample.to_string())
    print("\nNew sample after encoding (first few columns):")
    print(new_sample_encoded.iloc[:, :5].to_string())
    print("New sample dtypes:", new_sample_encoded.dtypes.unique())
    print("New sample shape:", new_sample_encoded.shape)
    print("X_train shape:", X_train.shape)

    X_new_data = new_sample_encoded
    y_pred_log = model.predict(X_new_data)
    y_pred = np.expm1(y_pred_log)
    print(f"\nPredicted CR for new sample (XGBoost): {y_pred[0]:.4f}")
    mlflow.log_metric("new_sample_prediction", y_pred[0])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(new_sample_encoded)
    print(f"SHAP values shape: {np.array(shap_values).shape if shap_values is not None else 'None'}")
    if shap_values is not None and shap_values.shape[1] == new_sample_encoded.shape[1]:
        plt.figure(figsize=(10, 4))
        shap.force_plot(explainer.expected_value, shap_values[0], new_sample_encoded.columns, matplotlib=True, show=False)
        plt.title(f"SHAP Force Plot for New Sample (XGBoost)")
        plt.tight_layout()
        plt.savefig(f"shap_force_xgboost.png")
        plt.close()
        mlflow.log_artifact("shap_force_xgboost.png")
        print(f"SHAP force plot saved to: shap_force_xgboost.png")
    else:
        print(f"SHAP values shape {shap_values.shape if shap_values is not None else 'None'} does not match features shape {new_sample_encoded.shape}")