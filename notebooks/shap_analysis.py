import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

# Settings
# -----------------------------
TARGET_COL = "Loading"
TEST_CSV = "test_set.csv"

GBM_MODEL_PATH = "gbm_model.pkl"
GBM_SCALER_PATH = "gbm_scaler.pkl"

RF_MODEL_PATH = "rf_model.pkl"
RF_SCALER_PATH = "rf_scaler.pkl"
DO_RF = False  # set True if you want RF too

# -----------------------------
# Load data
# -----------------------------
test_df = pd.read_csv(TEST_CSV)

# numeric features only (drop target + test-only col)
X_base = test_df.select_dtypes(include=[np.number]).drop(columns=[TARGET_COL], errors="ignore").copy()
X_base = X_base.drop(columns=["ads"], errors="ignore")

# add log_fugacity if raw fugacity exists
if "Fugacity [Pa]" in X_base.columns and "log_fugacity" not in X_base.columns:
    X_base["log_fugacity"] = np.log1p(X_base["Fugacity [Pa]"])

# -----------------------------
# Load GBM + scaler
# -----------------------------
gbm = joblib.load(GBM_MODEL_PATH)
gbm_scaler = joblib.load(GBM_SCALER_PATH)
gbm_feats = list(gbm_scaler.feature_names_in_)

# If mentor wants only log_fugacity, your *model should not expect Fugacity [Pa]*.
# But we won't force-drop here unless the model doesn't expect it.
if "Fugacity [Pa]" in X_base.columns and "Fugacity [Pa]" not in gbm_feats:
    # model doesn't use raw fugacity -> safe to drop
    X_base = X_base.drop(columns=["Fugacity [Pa]"])

# check missing
missing = [c for c in gbm_feats if c not in X_base.columns]
if missing:
    raise ValueError(f"Missing GBM features in X_base: {missing}")

# build aligned dataframe (UNSCALED values, but correct columns/order)
X_gbm_df = X_base[gbm_feats].copy()

# scale -> this is what the model sees
X_gbm_scaled = gbm_scaler.transform(X_gbm_df)

# IMPORTANT: make a dataframe of the SCALED values for SHAP plotting consistency
X_gbm_scaled_df = pd.DataFrame(X_gbm_scaled, columns=gbm_feats, index=X_gbm_df.index)

# -----------------------------
# SHAP for GBM (on scaled inputs)
# -----------------------------
explainer_gbm = shap.TreeExplainer(gbm)
shap_vals_gbm = explainer_gbm.shap_values(X_gbm_scaled_df)
shap_vals_gbm = np.array(shap_vals_gbm)

# 1) beeswarm
plt.figure()
shap.summary_plot(shap_vals_gbm, features=X_gbm_scaled_df, feature_names=gbm_feats, show=False)
plt.tight_layout()
plt.savefig("shap_gbm_beeswarm.png", dpi=300)
plt.show()

# 2) bar
plt.figure()
shap.summary_plot(shap_vals_gbm, features=X_gbm_scaled_df, feature_names=gbm_feats, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_gbm_bar.png", dpi=300)
plt.show()

# 3) grouped physics buckets
abs_mean = np.mean(np.abs(shap_vals_gbm), axis=0)
feat_importance = pd.Series(abs_mean, index=gbm_feats).sort_values(ascending=False)

def group_feature(name: str) -> str:
    if name.startswith("eps_bin_"):
        return "LJ energy (ε)"
    if name.startswith("sigma_bin_"):
        return "LJ size (σ)"
    if name.startswith("q_bin_"):
        return "Electrostatics (q)"

    textural = {"LPD", "PLD", "SA_grav", "VF", "PSSD", "density"}
    if name in textural:
        return "Textural / structural"

    if name in {"Fugacity [Pa]", "log_fugacity"}:
        return "Thermodynamic (fugacity)"

    return "Other"

grouped = feat_importance.groupby(group_feature).sum().sort_values(ascending=False)

plt.figure(figsize=(7,4))
plt.bar(grouped.index, grouped.values)
plt.xticks(rotation=25, ha="right")
plt.ylabel("Sum of mean(|SHAP|)")
plt.title("GBM SHAP importance grouped by physics")
plt.tight_layout()
plt.savefig("shap_gbm_grouped_physics.png", dpi=300)
plt.show()

print("Saved:")
print("  shap_gbm_beeswarm.png")
print("  shap_gbm_bar.png")
print("  shap_gbm_grouped_physics.png")

# -----------------------------
# OPTIONAL: RF SHAP
# -----------------------------
if DO_RF:
    rf = joblib.load(RF_MODEL_PATH)
    rf_scaler = joblib.load(RF_SCALER_PATH)
    rf_feats = list(rf_scaler.feature_names_in_)

    if "Fugacity [Pa]" in X_base.columns and "Fugacity [Pa]" not in rf_feats:
        X_base_rf = X_base.drop(columns=["Fugacity [Pa]"])
    else:
        X_base_rf = X_base

    missing_rf = [c for c in rf_feats if c not in X_base_rf.columns]
    if missing_rf:
        raise ValueError(f"Missing RF features in X_base: {missing_rf}")

    X_rf_df = X_base_rf[rf_feats].copy()
    X_rf_scaled = rf_scaler.transform(X_rf_df)
    X_rf_scaled_df = pd.DataFrame(X_rf_scaled, columns=rf_feats, index=X_rf_df.index)

    explainer_rf = shap.TreeExplainer(rf)
    shap_vals_rf = np.array(explainer_rf.shap_values(X_rf_scaled_df))

    plt.figure()
    shap.summary_plot(shap_vals_rf, features=X_rf_scaled_df, feature_names=rf_feats, show=False)
    plt.tight_layout()
    plt.savefig("shap_rf_beeswarm.png", dpi=300)
    plt.show()

    plt.figure()
    shap.summary_plot(shap_vals_rf, features=X_rf_scaled_df, feature_names=rf_feats, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_rf_bar.png", dpi=300)
    plt.show()

    print("Also saved RF shap plots.")
