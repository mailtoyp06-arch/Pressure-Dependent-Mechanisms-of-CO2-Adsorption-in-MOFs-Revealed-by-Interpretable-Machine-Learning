import re
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

# Input CSVs from your SHAP feature importance outputs
INPUT_FILES = [
    "lowP_full_feature_importance.csv",
    "lowP_hist_fug_feature_importance.csv",
    # add more here if you want
    # "gbm_feature_importance.csv",
]

# How many distance bins were used in the original histogram
# Based on your group's histogram setup:
# 1480 features per histogram group = 20 distance bins * 74 parameter bins
N_DISTANCE_BINS = 20
N_PARAM_BINS = 74

# Distance centers [Å]
# This assumes distances are 0.0, 0.5, 1.0, ..., 9.5
DISTANCE_VALUES = np.arange(0.0, N_DISTANCE_BINS * 0.5, 0.5)

# q histogram values
# This matches the charge scale from -3 to +3
Q_VALUES = np.linspace(-3.0, 3.0, N_PARAM_BINS)

# ------------------------------------------------------------
# IMPORTANT:
# Fill these with the ACTUAL epsilon and sigma bin values
# from your original histogram setup if you have them.
#
# The arrays below are placeholders / defaults.
# Replace them once you confirm the real values from your
# original histogram generation code or source files.
# ------------------------------------------------------------

# Example default epsilon values (monotonic placeholder)
EPS_VALUES = np.linspace(2.52, 333.63, N_PARAM_BINS)

# Example default sigma values (monotonic placeholder)
SIGMA_VALUES = np.linspace(2.00, 4.47, N_PARAM_BINS)

# Output folder prefix behavior
SAVE_TOP_N = 30


# ============================================================
# HELPERS
# ============================================================

def parse_feature(feature_name: str):
    """
    Parse feature like q_bin_823, eps_bin_202, sigma_bin_1283
    Returns:
        hist_type: q / eps / sigma / None
        raw_index: integer index after bin_
    """
    m = re.match(r"^(q|eps|sigma)_bin_(\d+)$", feature_name)
    if not m:
        return None, None
    hist_type = m.group(1)
    raw_index = int(m.group(2))
    return hist_type, raw_index


def get_group_offset(hist_type: str):
    """
    In this reduced feature set, q_bin_*, eps_bin_*, and sigma_bin_*
    already appear to use local indices within each histogram type.

    So:
      local_index = raw_index
    for all histogram types.
    """
    if hist_type in ["q", "eps", "sigma"]:
        return 0
    else:
        raise ValueError(f"Unknown hist_type: {hist_type}")

def decode_local_index(local_index: int, distance_values: np.ndarray, param_values: np.ndarray):
    """
    Decode local index into:
      distance_bin_index
      parameter_bin_index

    Assumes ordering:
      feature_0 = smallest distance + smallest parameter
      feature_1 = same distance + next parameter
      ...
      feature_73 = same distance + max parameter
      feature_74 = next distance + smallest parameter

    So:
      distance_idx = local_index // N_PARAM_BINS
      param_idx    = local_index % N_PARAM_BINS
    """
    distance_idx = local_index // len(param_values)
    param_idx = local_index % len(param_values)

    if distance_idx < 0 or distance_idx >= len(distance_values):
        return None, None, None, None

    if param_idx < 0 or param_idx >= len(param_values):
        return None, None, None, None

    distance_val = distance_values[distance_idx]
    param_val = param_values[param_idx]

    return distance_idx, param_idx, distance_val, param_val


def decode_feature(feature_name: str):
    hist_type, raw_index = parse_feature(feature_name)
    if hist_type is None:
        return {
            "feature": feature_name,
            "hist_type": None,
            "raw_index": None,
            "local_index": None,
            "distance_bin_index": None,
            "parameter_bin_index": None,
            "distance_A": None,
            "parameter_value": None,
        }

    # IMPORTANT FIX:
    # In your reduced feature set, raw_index already appears to be local
    local_index = raw_index

    if hist_type == "q":
        param_values = Q_VALUES
    elif hist_type == "eps":
        param_values = EPS_VALUES
    elif hist_type == "sigma":
        param_values = SIGMA_VALUES
    else:
        param_values = None

    d_idx, p_idx, d_val, p_val = decode_local_index(
        local_index=local_index,
        distance_values=DISTANCE_VALUES,
        param_values=param_values
    )

    return {
        "feature": feature_name,
        "hist_type": hist_type,
        "raw_index": raw_index,
        "local_index": local_index,
        "distance_bin_index": d_idx,
        "parameter_bin_index": p_idx,
        "distance_A": d_val,
        "parameter_value": p_val,
    }

def decode_importance_csv(csv_path: str, top_n: int = 30):
    """
    Read feature importance CSV, decode histogram features,
    and save useful outputs.
    """
    df = pd.read_csv(csv_path)

    if "feature" not in df.columns or "mean_abs_shap" not in df.columns:
        raise ValueError(f"{csv_path} must contain columns: feature, mean_abs_shap")

    # Keep only histogram features
    hist_df = df[
        df["feature"].str.startswith("q_bin_") |
        df["feature"].str.startswith("eps_bin_") |
        df["feature"].str.startswith("sigma_bin_")
    ].copy()

    hist_df = hist_df.sort_values("mean_abs_shap", ascending=False)

    decoded_rows = []
    for _, row in hist_df.iterrows():
        decoded = decode_feature(row["feature"])
        decoded["mean_abs_shap"] = row["mean_abs_shap"]
        decoded_rows.append(decoded)

    decoded_df = pd.DataFrame(decoded_rows)

    base = csv_path.replace(".csv", "")

    # Save all decoded histogram bins
    decoded_df.to_csv(f"{base}_decoded_hist_bins.csv", index=False)
    print(f"Saved: {base}_decoded_hist_bins.csv")

    # Save top-N overall histogram bins
    decoded_df.head(top_n).to_csv(f"{base}_top{top_n}_decoded_hist_bins.csv", index=False)
    print(f"Saved: {base}_top{top_n}_decoded_hist_bins.csv")

    # Save top q / eps / sigma separately
    decoded_df[decoded_df["hist_type"] == "q"].head(top_n).to_csv(
        f"{base}_top{top_n}_decoded_q_bins.csv", index=False
    )
    decoded_df[decoded_df["hist_type"] == "eps"].head(top_n).to_csv(
        f"{base}_top{top_n}_decoded_eps_bins.csv", index=False
    )
    decoded_df[decoded_df["hist_type"] == "sigma"].head(top_n).to_csv(
        f"{base}_top{top_n}_decoded_sigma_bins.csv", index=False
    )

    print(f"Saved: {base}_top{top_n}_decoded_q_bins.csv")
    print(f"Saved: {base}_top{top_n}_decoded_eps_bins.csv")
    print(f"Saved: {base}_top{top_n}_decoded_sigma_bins.csv")

    # Also print a quick summary
    print("\nTop decoded histogram bins from:", csv_path)
    print(decoded_df.head(15).to_string(index=False))
    print("-" * 80)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Distance values used:")
    print(DISTANCE_VALUES)
    print("\nQ values used:")
    print(Q_VALUES[:5], "...", Q_VALUES[-5:])

    print("\nEPS values used:")
    print(EPS_VALUES[:5], "...", EPS_VALUES[-5:])

    print("\nSIGMA values used:")
    print(SIGMA_VALUES[:5], "...", SIGMA_VALUES[-5:])

    print("\nStarting decoding...\n")

    for csv_path in INPUT_FILES:
        decode_importance_csv(csv_path, top_n=SAVE_TOP_N)

    print("\nDone.")


if __name__ == "__main__":
    main()
