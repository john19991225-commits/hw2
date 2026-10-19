"""Homework 2: Red Wine Quality Prediction using Multiple Linear Regression and CRISP-DM

Course: Data Mining / Machine Learning
Methodology: CRISP-DM (Cross-Industry Standard Process for Data Mining)
Dataset: Red Wine Quality Dataset (Cortez et al., 2009, UCI / Kaggle)
Primary Model: Multiple Linear Regression (with Ridge and Random Forest comparison)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure stdout handles UTF-8 on Windows command lines
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configure matplotlib headless backend
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "winequality-red.csv"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET_COLUMN = "quality"


# ==============================================================================
# 1. & 2. Data Understanding & Loading
# ==============================================================================
def detect_delimiter_and_load(csv_path: Path) -> tuple[pd.DataFrame, str]:
    """Inspect CSV delimiter, load dataset, and verify features."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline()

    # Determine delimiter
    if ";" in first_line and "," not in first_line:
        delimiter = ";"
    elif "," in first_line and ";" not in first_line:
        delimiter = ","
    else:
        # Fallback heuristic
        delimiter = ";" if first_line.count(";") > first_line.count(",") else ","

    print(f"[Data Understanding] Detected CSV delimiter: '{delimiter}'")
    df = pd.read_csv(csv_path, sep=delimiter)

    # Sanity checks
    print(f"[Data Understanding] Dataset shape: {df.shape}")
    print(f"[Data Understanding] Column count: {len(df.columns)}")
    missing_count = df.isna().sum().sum()
    duplicate_count = df.duplicated().sum()
    print(f"[Data Understanding] Total missing values: {missing_count}")
    print(f"[Data Understanding] Total duplicate rows: {duplicate_count}")

    # Verify 11 predictive features
    features = [c for c in df.columns if c != TARGET_COLUMN]
    if len(features) != 11:
        raise ValueError(f"Expected 11 features, but found {len(features)}: {features}")
    print(f"[Data Understanding] Verified 11 original predictive features: {features}")

    return df, delimiter


def save_correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and save correlation heatmap for all 11 features and target."""
    corr = df.corr()
    cols = df.columns

    fig, ax = plt.subplots(figsize=(11, 9))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation")

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    # Annotate values
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title("Correlation Heatmap: Red Wine Physicochemical Features & Quality", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(ROOT / "correlation_heatmap.png", dpi=150)
    plt.close()
    print("[Data Understanding] Saved correlation_heatmap.png")
    return corr


def save_exploratory_scatter_plots(df: pd.DataFrame) -> None:
    """Generate exploratory scatter plots of the top features against wine quality."""
    top_features = ["alcohol", "volatile acidity", "sulphates", "citric acid"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    rng = np.random.default_rng(RANDOM_STATE)
    jitter = rng.normal(0, 0.08, size=len(df))

    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        ax.scatter(df[feature], df[TARGET_COLUMN] + jitter, alpha=0.45, color="#1f77b4", edgecolors="none", s=25)
        # Add linear trend line for intuition
        m, b = np.polyfit(df[feature], df[TARGET_COLUMN], deg=1)
        x_vals = np.linspace(df[feature].min(), df[feature].max(), 100)
        ax.plot(x_vals, m * x_vals + b, color="#d62728", linewidth=2, label=f"Trend Line (slope: {m:.3f})")
        
        ax.set_title(f"Exploratory Relationship: {feature} vs. quality", fontsize=11)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("quality (with slight jitter)", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(ROOT / "exploratory_scatter_plots.png", dpi=150)
    plt.close()
    print("[Data Understanding] Saved exploratory_scatter_plots.png")


# ==============================================================================
# 3. Data Preparation & Leak-Free Pipeline Construction
# ==============================================================================
def make_linear_pipeline(k: int | str = "all") -> Pipeline:
    """Build leak-free pipeline with imputer, standard scaler, optional selector, and regressor."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if k != "all":
        steps.append(("selector", SelectKBest(score_func=f_regression, k=int(k))))
    steps.append(("regressor", LinearRegression()))
    return Pipeline(steps)


def evaluate_feature_selection(
    X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    """Perform leak-free SelectKBest evaluation within 5-fold cross-validation."""
    total_features = X_train.shape[1]
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for k in range(1, total_features + 1):
        pipe = make_linear_pipeline(k=k)
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=-1,
        )

        # Fit once on X_train to extract selected feature names for reporting
        pipe.fit(X_train, y_train)
        selector = pipe.named_steps["selector"]
        selected_names = list(X_train.columns[selector.get_support()])

        rows.append(
            {
                "Number of Features": k,
                "Selected Features": "; ".join(selected_names),
                "CV RMSE": float(-scores["test_rmse"].mean()),
                "CV RMSE Std": float(scores["test_rmse"].std()),
                "CV MAE": float(-scores["test_mae"].mean()),
                "CV R-squared": float(scores["test_r2"].mean()),
            }
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(ROOT / "feature_selection_results.csv", index=False)
    print("[Data Preparation] Saved feature_selection_results.csv")

    # Plot metrics vs number of features
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(results_df["Number of Features"], results_df["CV RMSE"], "o-", color="#1f77b4", linewidth=2)
    axes[0].set_title("5-Fold CV RMSE vs. Number of Features")
    axes[0].set_xlabel("Number of Features (k)")
    axes[0].set_ylabel("Cross-Validation RMSE (Lower is better)")
    axes[0].set_xticks(range(1, total_features + 1))
    axes[0].grid(alpha=0.3)

    axes[1].plot(results_df["Number of Features"], results_df["CV R-squared"], "s-", color="#2ca02c", linewidth=2)
    axes[1].set_title("5-Fold CV R-squared vs. Number of Features")
    axes[1].set_xlabel("Number of Features (k)")
    axes[1].set_ylabel("Cross-Validation R² (Higher is better)")
    axes[1].set_xticks(range(1, total_features + 1))
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ROOT / "feature_selection_metrics.png", dpi=150)
    plt.close()
    print("[Data Preparation] Saved feature_selection_metrics.png")
    return results_df


# ==============================================================================
# 4. Modeling & Reference Model Comparison
# ==============================================================================
def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Train Multiple Linear Regression, Ridge, and Random Forest models.

    Records CV scores, test metrics, and regression coefficients.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Multiple Linear Regression": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("regressor", LinearRegression())]
        ),
        "Ridge Regression": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("regressor", Ridge(alpha=1.0, random_state=RANDOM_STATE))]
        ),
        "Random Forest Regressor": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("regressor", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))]
        ),
    }

    comparison_rows = []
    trained_pipelines = {}

    for name, pipeline in models.items():
        # 5-fold cross-validation on training data only
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=-1,
        )

        # Fit on entire training set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predictions
        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_r2 = float(r2_score(y_train, train_pred))
        test_r2 = float(r2_score(y_test, test_pred))
        test_mae = float(mean_absolute_error(y_test, test_pred))
        test_mse = float(mean_squared_error(y_test, test_pred))
        test_rmse = float(np.sqrt(test_mse))

        comparison_rows.append(
            {
                "Model": name,
                "CV RMSE Mean": float(-cv_scores["test_rmse"].mean()),
                "CV RMSE Std": float(cv_scores["test_rmse"].std()),
                "CV R-squared": float(cv_scores["test_r2"].mean()),
                "Training R-squared": train_r2,
                "Testing R-squared": test_r2,
                "MAE": test_mae,
                "MSE": test_mse,
                "RMSE": test_rmse,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(ROOT / "model_comparison_metrics.csv", index=False)
    print("[Modeling] Saved model_comparison_metrics.csv")

    # Extract Linear Regression Coefficients
    lr_pipe = trained_pipelines["Multiple Linear Regression"]
    lr_model: LinearRegression = lr_pipe.named_steps["regressor"]
    scaler: StandardScaler = lr_pipe.named_steps["scaler"]

    # Standardized coefficients (from scaled features)
    std_coefs = lr_model.coef_
    # Unstandardized coefficients (raw scale: beta_raw = beta_std / scale)
    raw_coefs = std_coefs / scaler.scale_
    raw_intercept = lr_model.intercept_ - np.sum(std_coefs * scaler.mean_ / scaler.scale_)

    coef_df = pd.DataFrame(
        {
            "Feature": list(X_train.columns),
            "Standardized_Coefficient": std_coefs,
            "Unstandardized_Coefficient": raw_coefs,
        }
    ).sort_values("Standardized_Coefficient", key=abs, ascending=False)

    intercept_row = pd.DataFrame(
        [{"Feature": "(Intercept)", "Standardized_Coefficient": float(lr_model.intercept_), "Unstandardized_Coefficient": float(raw_intercept)}]
    )
    coef_df_full = pd.concat([intercept_row, coef_df], ignore_index=True)
    coef_df_full.to_csv(ROOT / "regression_coefficients.csv", index=False)
    print("[Modeling] Saved regression_coefficients.csv")

    # Plot model comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x_pos = np.arange(len(comparison_df))
    width = 0.35

    # Metrics comparison (RMSE & MAE)
    axes[0].bar(x_pos - width / 2, comparison_df["RMSE"], width, label="Test RMSE", color="#1f77b4")
    axes[0].bar(x_pos + width / 2, comparison_df["MAE"], width, label="Test MAE", color="#ff7f0e")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(comparison_df["Model"], rotation=12, fontsize=9)
    axes[0].set_ylabel("Error Metric (Lower is better)")
    axes[0].set_title("Model Comparison: Test RMSE vs. Test MAE")
    axes[0].legend()
    axes[0].grid(alpha=0.25, axis="y")

    # R2 comparison (Train vs Test)
    axes[1].bar(x_pos - width / 2, comparison_df["Training R-squared"], width, label="Training R²", color="#2ca02c")
    axes[1].bar(x_pos + width / 2, comparison_df["Testing R-squared"], width, label="Testing R²", color="#d62728")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(comparison_df["Model"], rotation=12, fontsize=9)
    axes[1].set_ylabel("R-squared (Higher is better)")
    axes[1].set_title("Generalization Gap: Training R² vs. Testing R²")
    axes[1].legend()
    axes[1].grid(alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(ROOT / "model_comparison.png", dpi=150)
    plt.close()
    print("[Modeling] Saved model_comparison.png")

    return lr_pipe, comparison_df, coef_df_full


# ==============================================================================
# 5. Evaluation & Diagnostic Plots
# ==============================================================================
def save_evaluation_plots(
    lr_pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Generate actual vs predicted, residual plot, and learning curve for Linear Regression."""
    test_pred = lr_pipe.predict(X_test)
    residuals = y_test.to_numpy() - test_pred

    mae = float(mean_absolute_error(y_test, test_pred))
    mse = float(mean_squared_error(y_test, test_pred))
    rmse = float(np.sqrt(mse))
    test_r2 = float(r2_score(y_test, test_pred))
    train_r2 = float(r2_score(y_train, lr_pipe.predict(X_train)))

    # 1. actual_vs_predicted.png
    plt.figure(figsize=(7, 6.5))
    # Add subtle jitter to actual values for visualization because quality is discrete
    rng = np.random.default_rng(RANDOM_STATE)
    jitter = rng.normal(0, 0.05, size=len(y_test))
    plt.scatter(y_test + jitter, test_pred, alpha=0.6, color="#1f77b4", edgecolors="k", s=36, label="Test Samples (with slight jitter)")
    min_val = min(y_test.min(), test_pred.min()) - 0.5
    max_val = max(y_test.max(), test_pred.max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Fit (y = ŷ)")
    plt.xlabel("Actual Quality Rating", fontsize=11)
    plt.ylabel("Predicted Quality Rating (ŷ)", fontsize=11)
    plt.title(f"Multiple Linear Regression: Actual vs. Predicted\nTest R² = {test_r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}", fontsize=11)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "actual_vs_predicted.png", dpi=150)
    plt.close()
    print("[Evaluation] Saved actual_vs_predicted.png")

    # 2. residual_plot.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    # Residuals vs Predicted
    ax1.scatter(test_pred, residuals, alpha=0.65, color="#2b5c8f", edgecolors="k", s=32)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1.8, label="Zero Error Line")
    ax1.set_xlabel("Predicted Quality (ŷ)", fontsize=10)
    ax1.set_ylabel("Residuals (y - ŷ)", fontsize=10)
    ax1.set_title("Residuals vs. Fitted Values", fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Residual Distribution Histogram
    counts, bins, _ = ax2.hist(residuals, bins=25, density=True, alpha=0.65, color="#4a90e2", edgecolor="black")
    # Fit normal distribution curve
    res_mean, res_std = float(np.mean(residuals)), float(np.std(residuals))
    x_grid = np.linspace(bins[0], bins[-1], 100)
    pdf = (1 / (res_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - res_mean) / res_std) ** 2)
    ax2.plot(x_grid, pdf, "r-", linewidth=2, label=f"Normal Fit (μ={res_mean:.2f}, σ={res_std:.2f})")
    ax2.axvline(0, color="black", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("Residual Value", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("Distribution of Residuals", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(ROOT / "residual_plot.png", dpi=150)
    plt.close()
    print("[Evaluation] Saved residual_plot.png")

    # 3. learning_curve.png
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, valid_scores = learning_curve(
        lr_pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.15, 1.0, 8),
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_rmse = -train_scores
    valid_rmse = -valid_scores

    lc_df = pd.DataFrame(
        {
            "training_examples": train_sizes,
            "train_rmse_mean": train_rmse.mean(axis=1),
            "train_rmse_std": train_rmse.std(axis=1),
            "validation_rmse_mean": valid_rmse.mean(axis=1),
            "validation_rmse_std": valid_rmse.std(axis=1),
        }
    )
    lc_df.to_csv(ROOT / "learning_curve.csv", index=False)

    plt.figure(figsize=(8, 5.5))
    plt.plot(train_sizes, lc_df["train_rmse_mean"], "o-", color="#1f77b4", label="Training RMSE")
    plt.plot(train_sizes, lc_df["validation_rmse_mean"], "s-", color="#d62728", label="5-Fold Validation RMSE")
    plt.fill_between(
        train_sizes,
        lc_df["train_rmse_mean"] - lc_df["train_rmse_std"],
        lc_df["train_rmse_mean"] + lc_df["train_rmse_std"],
        alpha=0.15,
        color="#1f77b4",
    )
    plt.fill_between(
        train_sizes,
        lc_df["validation_rmse_mean"] - lc_df["validation_rmse_std"],
        lc_df["validation_rmse_mean"] + lc_df["validation_rmse_std"],
        alpha=0.15,
        color="#d62728",
    )
    plt.xlabel("Number of Training Samples", fontsize=10)
    plt.ylabel("RMSE (Quality scale)", fontsize=10)
    plt.title("Multiple Linear Regression Learning Curve", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "learning_curve.png", dpi=150)
    plt.close()
    print("[Evaluation] Saved learning_curve.png")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R-squared": test_r2,
        "Training R-squared": train_r2,
        "Testing R-squared": test_r2,
    }


# ==============================================================================
# 5b. Statistically Justified 95% Prediction Interval (Split Conformal Prediction)
# ==============================================================================
def save_prediction_interval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Calculate and plot split conformal 95% prediction intervals.

    Theory: Under exchangeability, split conformal prediction provides finite-sample
    distribution-free marginal coverage guarantees P(Y_new in C(X_new)) >= 1 - alpha.
    """
    # Split training set into proper training (75%) and calibration set (25%)
    X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE
    )

    # Train linear regression pipeline on proper training set
    pipe = make_linear_pipeline(k="all")
    pipe.fit(X_prop_train, y_prop_train)

    # Compute calibration nonconformity scores: R_i = |y_i - y_hat_i|
    y_cal_pred = pipe.predict(X_cal)
    cal_scores = np.abs(y_cal.to_numpy() - y_cal_pred)
    n_cal = len(cal_scores)

    # Conformal quantile: ceil((n_cal + 1) * (1 - alpha)) / n_cal
    quantile_level = np.clip(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 0.0, 1.0)
    q = float(np.quantile(cal_scores, quantile_level, method="higher"))

    # Test predictions and intervals
    y_test_pred = pipe.predict(X_test)
    y_test_arr = y_test.to_numpy()
    lower_bound = y_test_pred - q
    upper_bound = y_test_pred + q

    covered = (y_test_arr >= lower_bound) & (y_test_arr <= upper_bound)
    empirical_coverage = float(np.mean(covered))
    interval_width = float(2 * q)

    # Plot sorted prediction intervals
    sort_idx = np.argsort(y_test_pred)
    sorted_pred = y_test_pred[sort_idx]
    sorted_lower = lower_bound[sort_idx]
    sorted_upper = upper_bound[sort_idx]
    sorted_actual = y_test_arr[sort_idx]
    sorted_covered = covered[sort_idx]

    plt.figure(figsize=(12, 6.5))
    x_indices = np.arange(len(sorted_pred))

    # Shaded 95% Prediction Interval
    plt.fill_between(
        x_indices,
        sorted_lower,
        sorted_upper,
        color="#b3cde3",
        alpha=0.6,
        label=f"95% Prediction Interval Band (Width = {interval_width:.3f}, ±{q:.3f})",
    )
    plt.plot(x_indices, sorted_pred, color="#08519c", linewidth=2, label="Point Prediction (ŷ)")

    # Scatter actual observations: in-interval vs out-of-interval
    plt.scatter(
        x_indices[sorted_covered],
        sorted_actual[sorted_covered],
        color="#31a354",
        s=28,
        alpha=0.75,
        edgecolors="k",
        label=f"Covered Actual Quality ({np.sum(covered)}/{len(y_test_arr)})",
        zorder=5,
    )
    if np.sum(~sorted_covered) > 0:
        plt.scatter(
            x_indices[~sorted_covered],
            sorted_actual[~sorted_covered],
            color="#e6550d",
            s=45,
            marker="^",
            edgecolors="k",
            label=f"Out-of-Interval Actual Quality ({np.sum(~covered)}/{len(y_test_arr)})",
            zorder=6,
        )

    plt.xlabel("Test Samples (Sorted by Predicted Quality)", fontsize=10)
    plt.ylabel("Wine Quality Rating", fontsize=10)
    plt.title(
        f"Statistically Justified 95% Prediction Interval (Split Conformal Prediction)\n"
        f"Empirical Test Coverage: {empirical_coverage * 100:.2f}% (Nominal Guarantee: 95.00%, Margin: ±{q:.3f})",
        fontsize=11,
    )
    plt.grid(alpha=0.25)
    plt.legend(loc="upper left", fontsize=9)

    # Descriptive text box highlighting distinction between Prediction vs Confidence interval
    textbox_text = (
        "Statistical Clarification:\n"
        "• PREDICTION INTERVAL: Quantifies uncertainty for INDIVIDUAL observations (y = ŷ + ε).\n"
        "• CONFIDENCE INTERVAL: Only quantifies uncertainty for the conditional MEAN (E[y|x]).\n"
        "• Methodology: Split Conformal Prediction (Calibration set n=320, non-parametric, finite-sample valid)."
    )
    plt.gca().text(
        0.02,
        0.04,
        textbox_text,
        transform=plt.gca().transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(ROOT / "prediction_interval.png", dpi=150)
    plt.close()
    print("[Evaluation] Saved prediction_interval.png")

    return {
        "method": "Split Conformal Prediction",
        "nominal_confidence_level": 1.0 - alpha,
        "n_proper_train": len(X_prop_train),
        "n_calibration": n_cal,
        "n_test": len(X_test),
        "conformal_quantile_q": q,
        "prediction_interval_width": interval_width,
        "empirical_test_coverage": empirical_coverage,
        "test_points_covered": int(np.sum(covered)),
        "test_points_total": len(y_test_arr),
        "statistical_assumptions": "Data exchangeability (i.i.d.), no distributional or homoscedasticity assumption required.",
        "statistical_limitations": "Constant interval width across input space; sample splitting reduces effective training size.",
    }


# ==============================================================================
# 6. Deployment & Serialization
# ==============================================================================
def deploy_and_verify(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Path:
    """Save pipeline to disk, reload, and verify identical predictions."""
    model_path = ROOT / "red_wine_model.pkl.gz"
    joblib.dump(pipeline, model_path, compress=("gzip", 3))
    print(f"[Deployment] Saved model pipeline to {model_path}")

    # Verify reloading
    reloaded_pipeline = joblib.load(model_path)
    sample_X = X_test.head(5)
    orig_preds = pipeline.predict(sample_X)
    reloaded_preds = reloaded_pipeline.predict(sample_X)

    if not np.allclose(orig_preds, reloaded_preds):
        raise RuntimeError("Verification failed: Reloaded model predictions differ from original!")

    print("[Deployment] Model verification passed: Reloaded pipeline predictions match exactly.")
    return model_path


# ==============================================================================
# Main Execution Orchestrator
# ==============================================================================
def main() -> None:
    print("=" * 70)
    print("Red Wine Quality Prediction (CRISP-DM Workflow) - HW2")
    print("=" * 70)

    # 1. & 2. Data Understanding
    df, delimiter = detect_delimiter_and_load(DATA_PATH)
    corr_df = save_correlation_heatmap(df)
    save_exploratory_scatter_plots(df)

    # 3. Data Preparation
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"[Data Preparation] 80/20 Train/Test Split completed:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples:  {X_test.shape[0]}")

    # Feature selection cross-validation
    selection_results = evaluate_feature_selection(X_train, y_train)
    best_fs_row = selection_results.loc[selection_results["CV RMSE"].idxmin()]
    print(f"[Data Preparation] Optimal feature count by CV RMSE: {best_fs_row['Number of Features']}")
    print(f"  Selected features: {best_fs_row['Selected Features']}")
    print(f"  CV RMSE: {best_fs_row['CV RMSE']:.4f}")

    # 4. Modeling & Reference Model Comparison
    lr_pipeline, comparison_df, coef_df = train_and_evaluate_models(
        X_train, y_train, X_test, y_test
    )

    # 5. Evaluation
    eval_metrics = save_evaluation_plots(lr_pipeline, X_train, y_train, X_test, y_test)

    # 5b. Prediction Interval (Split Conformal)
    pi_metrics = save_prediction_interval(X_train, y_train, X_test, y_test, alpha=0.05)

    # 6. Deployment
    saved_model_path = deploy_and_verify(lr_pipeline, X_test, y_test)

    # 7. Save Comprehensive Evaluation Metrics JSON
    final_output = {
        "dataset": {
            "source_url": "https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009",
            "detected_delimiter": delimiter,
            "total_samples": int(df.shape[0]),
            "feature_count": int(X.shape[1]),
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        },
        "linear_regression_metrics": eval_metrics,
        "conformal_prediction_interval": pi_metrics,
        "optimal_feature_selection": {
            "optimal_k": int(best_fs_row["Number of Features"]),
            "selected_features": best_fs_row["Selected Features"].split("; "),
            "cv_rmse": float(best_fs_row["CV RMSE"]),
            "cv_r2": float(best_fs_row["CV R-squared"]),
        },
        "model_comparison": comparison_df.to_dict(orient="records"),
        "coefficients": coef_df.to_dict(orient="records"),
        "serialized_model_path": str(saved_model_path.name),
    }

    with open(ROOT / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print("[Outputs] Saved evaluation_metrics.json")

    print("=" * 70)
    print("Pipeline Execution Completed Successfully!")
    print(f"Linear Regression Test R-squared:   {eval_metrics['Testing R-squared']:.4f}")
    print(f"Linear Regression Test RMSE:        {eval_metrics['RMSE']:.4f}")
    print(f"Linear Regression Test MAE:         {eval_metrics['MAE']:.4f}")
    print(f"95% Prediction Interval Margin:     +/-{pi_metrics['conformal_quantile_q']:.4f}")
    print(f"Empirical Test Coverage:            {pi_metrics['empirical_test_coverage'] * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
