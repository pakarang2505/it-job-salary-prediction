import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

# === Force pandas to print floats nicely ===
pd.set_option('display.float_format', '{:.2f}'.format)

# === Load processed data ===
train_df = pd.read_csv("train_salary_processed.csv")
test_df = pd.read_csv("test_salary_processed.csv")

# === Target columns ===
target_col_scaled = "Salary_Scaled"        # Already scaled
target_col_orig = "Salary_with_Noise"     # Original noisy salary

# === Clean original salary column ===
for df in [train_df, test_df]:
    if target_col_orig in df.columns:
        df[target_col_orig] = (
            df[target_col_orig]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.strip()
            .replace('', np.nan)
            .astype(float)
        )

# === Drop rows with missing target ===
train_df = train_df.dropna(subset=[target_col_scaled, target_col_orig])
test_df = test_df.dropna(subset=[target_col_scaled, target_col_orig])

# === Prepare features (ignore IT_job_criteria if present) ===
exclude_cols = ["index", target_col_scaled, target_col_orig]
if "IT_job_criteria" in train_df.columns:
    exclude_cols.append("IT_job_criteria")
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

# === Ensure test has same columns as train ===
test_df = test_df.copy()
missing_cols = set(feature_cols) - set(test_df.columns)
for c in missing_cols:
    test_df.loc[:, c] = 0

# === Prepare X, y ===
X_train = train_df[feature_cols]
X_test = test_df[feature_cols].copy()
y_train_orig = train_df[target_col_orig].values
y_test_orig = test_df[target_col_orig].values
y_train_scaled = train_df[target_col_scaled].values  # already scaled

# === Handle missing feature values (if any) ===
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# === Scaler for target (for inverse-transform) ===
scaler_salary = StandardScaler()
scaler_salary.fit(y_train_orig.reshape(-1, 1))  # used only for inverse-transform

# === Define models ===
models = {
    "LinearRegression": LinearRegression(),
    "Lasso_alpha_0.01": Lasso(alpha=0.01, max_iter=10000),
    "Lasso_alpha_0.1": Lasso(alpha=0.1, max_iter=10000),
    "Lasso_alpha_0.001": Lasso(alpha=0.001, max_iter=10000),
    "Ridge": Ridge(alpha=1.0),
    "MLPRegressor_200_100_50": MLPRegressor(hidden_layer_sizes=(200, 100, 50),
                                            max_iter=2000,
                                            random_state=42),
    "MLPRegressor_300_200_100": MLPRegressor(hidden_layer_sizes=(300, 200, 100),
                                             max_iter=2000,
                                             random_state=42),
    "MLPRegressor_300_200_100_reg": MLPRegressor(hidden_layer_sizes=(300, 200, 100),
                                                 alpha=0.01,
                                                 max_iter=5000,
                                                 random_state=42,
                                                 early_stopping=True,
                                                 n_iter_no_change=20,
                                                 learning_rate_init=0.001,
                                                 solver='adam'),
    "XGBoost_default": xgb.XGBRegressor(n_estimators=100,
                                        learning_rate=0.1,
                                        max_depth=5,
                                        random_state=42),
    "XGBoost_deeper_slow": xgb.XGBRegressor(n_estimators=300,
                                            learning_rate=0.05,
                                            max_depth=6,
                                            subsample=0.8,
                                            colsample_bytree=0.8,
                                            random_state=42)
}

# === Train, predict, evaluate ===
predictions = pd.DataFrame()
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    if name.startswith("MLPRegressor") or name in ["LinearRegression", "Lasso_alpha_0.01",
                                                   "Lasso_alpha_0.1", "Lasso_alpha_0.001", "Ridge"]:
        # Train on scaled target
        model.fit(X_train_imputed, y_train_scaled)
        y_pred_train_scaled = model.predict(X_train_imputed)
        y_pred_test_scaled = model.predict(X_test_imputed)
        # Inverse-transform to original salary for evaluation
        y_pred_train = scaler_salary.inverse_transform(y_pred_train_scaled.reshape(-1,1)).ravel()
        y_pred_test = scaler_salary.inverse_transform(y_pred_test_scaled.reshape(-1,1)).ravel()
    else:
        # XGBoost: train on raw features & raw target
        model.fit(X_train_imputed, y_train_orig)
        y_pred_train = model.predict(X_train_imputed)
        y_pred_test = model.predict(X_test_imputed)

    # === Metrics ===
    rmse_train = mean_squared_error(y_train_orig, y_pred_train, squared=False)
    rmse_test = mean_squared_error(y_test_orig, y_pred_test, squared=False)
    r2_train = r2_score(y_train_orig, y_pred_train)
    r2_test = r2_score(y_test_orig, y_pred_test)
    valid_idx = y_test_orig > 0
    pct_error_median = np.median(
        np.abs((y_test_orig[valid_idx] - y_pred_test[valid_idx]) / y_test_orig[valid_idx])
    ) * 100

    print(f"{name} Train RMSE: {rmse_train:.2f}")
    print(f"{name} Test RMSE: {rmse_test:.2f}")
    print(f"{name} Train R^2: {r2_train:.4f}")
    print(f"{name} Test R^2: {r2_test:.4f}")
    print(f"{name} Test Median % Error: {pct_error_median:.2f}%")

    predictions[f"{name}_Pred"] = y_pred_test

    results.append({
        "Model": name,
        "Train RMSE": round(rmse_train,2),
        "Test RMSE": round(rmse_test,2),
        "Train R^2": round(r2_train,4),
        "Test R^2": round(r2_test,4),
        "Test Median % Error": round(pct_error_median,2)
    })

# === Save predictions & summary ===
predictions.to_csv("salary_predictions.csv", index=False)
summary_df = pd.DataFrame(results)
summary_df.to_csv("model_evaluation_summary.csv", index=False)

# === Post-analysis: high % error by job (optional) ===
if "IT_job_criteria" in test_df.columns:
    # pick best model based on lowest median % error
    best_model_name = min(results, key=lambda r: r["Test Median % Error"])["Model"]
    best_preds = predictions[f"{best_model_name}_Pred"].values
    abs_pct_error = np.abs((y_test_orig - best_preds) / y_test_orig) * 100

    error_report_df = pd.DataFrame({
        "Original_Index": test_df["index"] if "index" in test_df.columns else test_df.index,
        "Job_Name": test_df["IT_job_criteria"],
        "True_Salary": y_test_orig,
        "Predicted_Salary": best_preds,
        "Absolute_%_Error": abs_pct_error
    }).sort_values(by="Absolute_%_Error", ascending=False)

    error_report_df.to_csv("high_error_cases_by_job.csv", index=False)

    job_error_summary = error_report_df.groupby("Job_Name")["Absolute_%_Error"].median().sort_values(ascending=False)
    job_error_summary.to_csv("median_error_by_job.csv")

    print("\nTop 10 highest error cases by job:")
    print(error_report_df.head(10))
    print("\nTop 10 jobs with highest median % error:")
    print(job_error_summary.head(10))

# === Save models, scalers, imputer ===
for name, model in models.items():
    joblib.dump(model, f"{name.lower()}_model.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler_salary, "scaler_salary.pkl")

print("\nEvaluation completed.")
print(summary_df)
print("Predictions saved to salary_predictions.csv")
