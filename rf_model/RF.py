#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib


# -------------------- utilities --------------------

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip, collapse spaces, lower → underscores."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.lower()
    )
    return df

def trailing_int(name: str) -> int:
    m = re.search(r"(\d+)$", str(name))
    return int(m.group(1)) if m else -1

def model_idx_from_text(s: str) -> int:
    """Extract model index from strings like '*model_37*'."""
    m = re.search(r"model[_\- ](\d+)", str(s))
    return int(m.group(1)) if m else -1


# -------------------- loaders --------------------

def load_features_for_datapoint(analysis_csv: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Load the 50×D features for one datapoint.

    Rules:
      - Normalize headers (spaces → underscores, lowercase).
      - Prefer columns from 'confidence_score' onward.
      - If 'confidence_score' missing: drop the first 3 columns by position.
      - Keep only **numeric** columns (coerce non-numerics to NaN then drop).
      - Sort rows by model index parsed from 'model_id' or fallback to row order.
      - Require >= 50 rows; take the first 50.
    """
    # be forgiving with CSV parsing
    df = pd.read_csv(analysis_csv, engine="python", on_bad_lines="skip")
    df = norm_cols(df)

    if "model_id" not in df.columns:
        # try to recover from e.g. 'model' or 'modelid'
        cand = [c for c in df.columns if c.startswith("model")]
        if cand:
            df = df.rename(columns={cand[0]: "model_id"})
        else:
            raise ValueError(f"{analysis_csv.name}: no 'model_id' column")

    # Sort rows by model index if we can parse it, else keep the current order
    idx = df["model_id"].apply(model_idx_from_text)
    if (idx >= 0).any():
        df = df.assign(__model_idx=idx).sort_values("__model_idx")
    else:
        df = df.assign(__model_idx=np.arange(len(df)))

    # Determine feature columns
    cols = list(df.columns)
    if "confidence_score" in cols:
        start = cols.index("confidence_score")
        feature_cols = cols[start:]
    else:
        # drop first 3 columns by position if they exist
        drop_first = min(3, len(cols))
        feature_cols = cols[drop_first:]

    # Coerce to numeric & drop non-numeric columns
    df_num = df.copy()
    for c in feature_cols:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    feature_cols_num = [c for c in feature_cols if df_num[c].dtype.kind in "if"]

    if not feature_cols_num:
        raise ValueError(f"{analysis_csv.name}: no numeric feature columns after filtering")

    # Keep first 50 rows (warn if fewer)
    if len(df_num) < 50:
        raise ValueError(f"{analysis_csv.name}: expected >= 50 rows, got {len(df_num)}")
    df_num = df_num.iloc[:50]

    X = df_num[feature_cols_num].to_numpy(dtype=float)
    return X, feature_cols_num


def build_dataset(combined_csv: Path, features_dir: Path):
    """
    Build X (N×D), y (N,), groups (N,), feature_names, used_datapoint_ids.

    - Reads combined_result.csv (normalize headers).
    - Uses columns named 'rmsd_model_0..49'. If missing, falls back to last 50 numeric columns.
    - For each datapoint row, loads <datapoint_id>_analysis.csv and aligns rows by model index.
    - Skips proteins whose feature CSV is missing or malformed (prints a clear message).
    - Intersects feature sets across proteins to a common set (aligns matrices).
    """
    df = pd.read_csv(combined_csv, engine="python", on_bad_lines="skip")
    df = norm_cols(df)

    if "datapoint_id" not in df.columns:
        raise ValueError("combined_result.csv must have a 'datapoint_id' column (case-insensitive ok)")

    # Detect rmsd columns
    rmsd_cols = [c for c in df.columns if re.fullmatch(r"rmsd_model_\d+", c)]
    if len(rmsd_cols) < 50:
        # fallback: last 50 numeric columns
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) < 50:
            raise ValueError("Could not find 50 RMSD columns. Ensure 'rmsd_model_0..49' exist or that the last 50 columns are numeric.")
        rmsd_cols = num_cols[-50:]
    rmsd_cols = sorted(rmsd_cols, key=trailing_int)

    X_list, y_list, groups = [], [], []
    common_feats: List[str] | None = None
    used_ids: List[str] = []
    skipped = 0

    for prot_idx, (_, row) in enumerate(df.iterrows()):
        dp_id = str(row["datapoint_id"]).strip()
        feat_csv = features_dir / f"{dp_id}_analysis.csv"
        if not feat_csv.exists():
            print(f"[skip] features not found: {feat_csv}")
            skipped += 1
            continue

        try:
            X_50xD, feat_names = load_features_for_datapoint(feat_csv)
        except Exception as e:
            print(f"[skip] {dp_id}: {e}")
            skipped += 1
            continue

        # RMSD 50-vector
        y_50 = row[rmsd_cols].to_numpy(dtype=float)
        if y_50.shape[0] != 50:
            print(f"[skip] {dp_id}: expected 50 RMSD values; got {y_50.shape[0]}")
            skipped += 1
            continue

        # maintain common feature set (intersection) across proteins
        if common_feats is None:
            common_feats = feat_names
            X_cur = X_50xD
        else:
            keep = [c for c in common_feats if c in feat_names]
            if not keep:
                print(f"[skip] {dp_id}: no overlapping feature columns with prior proteins")
                skipped += 1
                continue
            # align previous matrices
            keep_idx_prev = [common_feats.index(c) for c in keep]
            X_list = [X[:, keep_idx_prev] for X in X_list]
            # align current
            X_cur = X_50xD[:, [feat_names.index(c) for c in keep]]
            common_feats = keep

        X_list.append(X_cur)
        y_list.append(y_50)
        groups.extend([prot_idx] * 50)
        used_ids.append(dp_id)

    if not X_list:
        raise RuntimeError("No usable proteins found. Check file names/paths and CSV formats.")

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    groups = np.asarray(groups)

    print(f"\nLoaded dataset:")
    print(f"  proteins used    : {len(used_ids)} (skipped {skipped})")
    print(f"  samples (N)      : {X_all.shape[0]}")
    print(f"  feature dim (D)  : {X_all.shape[1]}")
    return X_all, y_all, groups, common_feats, used_ids


# -------------------- training script --------------------

def main():
    ap = argparse.ArgumentParser(description="Train RF regressor to predict RMSD from Boltz features")
    ap.add_argument("--combined-csv", type=Path, required=True, help="combined_result.csv (one row per protein)")
    ap.add_argument("--features-dir", type=Path, required=True, help="dir containing <datapoint_id>_analysis.csv")
    ap.add_argument("--save-model", type=Path, default=None, help="optional joblib path")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=None)
    args = ap.parse_args()

    X, y, groups, feat_names, used_ids = build_dataset(args.combined_csv, args.features_dir)

    # Group-aware split: split by protein (not by individual samples)
    uniq = np.unique(groups)
    train_ids, test_ids = train_test_split(uniq, test_size=args.test_size, random_state=42, shuffle=True)
    is_train = np.isin(groups, train_ids)
    is_test = np.isin(groups, test_ids)

    X_train, y_train = X[is_train], y[is_train]
    X_test, y_test = X[is_test], y[is_test]

    print(f"\nGroup-aware split:")
    print(f"  train proteins: {len(train_ids)}, samples: {X_train.shape[0]}")
    print(f"  test  proteins: {len(test_ids)}, samples: {X_test.shape[0]}")

    # Scale (optional but convenient if you switch models later)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    y_pred = rf.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Test metrics (RMSD) ===")
    print(f"MAE : {mae:.4f}")
    print(f"R^2 : {r2:.4f}")

    # Feature importances
    imps = rf.feature_importances_
    order = np.argsort(imps)[::-1]
    print("\nTop-20 feature importances:")
    for i in range(min(20, len(order))):
        j = order[i]
        print(f"{i+1:2d}. {feat_names[j]:40s} {imps[j]:.6f}")

    if args.save_model:
        joblib.dump(
            {"model": rf, "scaler": scaler, "feature_names": feat_names},
            args.save_model,
        )
        print(f"\nModel saved to: {args.save_model}")


if __name__ == "__main__":
    main()

