"""Data loading, data-quality checks, modelling and validation helpers.

Everything the dashboard shows is computed here from the supplied CSV.
Nothing in this module hard-codes a finding, score or benchmark.
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

TARGET = "Yield_tons_per_hectare"
CAT = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
BOOL = ["Fertilizer_Used", "Irrigation_Used"]
NUM = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
FEATURES = CAT + BOOL + NUM
ALL_COLUMNS = FEATURES + [TARGET]

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_SAMPLE = 200_000        # rows of the training set used for cross-validation
IMPORTANCE_SAMPLE = 20_000  # test rows used for permutation importance
PLOT_SAMPLE = 5_000
MODEL_VERSION = "v2"        # bump to invalidate cached models


class DataError(Exception):
    """Raised when the CSV is missing, malformed or fails validation."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


def _to_bool(s: pd.Series, name: str) -> pd.Series:
    if s.dtype == bool:
        return s
    mapped = s.astype(str).str.strip().str.lower().map(
        lambda v: True if v in _TRUE else (False if v in _FALSE else np.nan)
    )
    if mapped.isna().any():
        raise DataError(f"Column '{name}' contains values that are not Boolean.")
    return mapped.astype(bool)


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the supplied CSV and return a memory-compact, validated frame."""
    path = Path(path)
    if not path.is_file():
        raise DataError(f"File not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surface any parser problem
        raise DataError(f"Could not read the CSV: {exc}") from exc

    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"CSV is missing required columns: {missing}")
    df = df[ALL_COLUMNS].copy()

    n_read = len(df)
    raw_missing = int(df.isna().sum().sum())          # truly empty cells in the file
    for c in NUM + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # non-numeric text becomes NaN
    if df[TARGET].notna().sum() == 0:
        raise DataError(f"The target column '{TARGET}' is not numeric.")
    for c in CAT:
        df[c] = df[c].astype("string").str.strip()
    df = df.dropna(subset=CAT + BOOL + NUM + [TARGET])  # unusable rows are removed and counted
    df = df[np.isfinite(df[NUM + [TARGET]].to_numpy(np.float64)).all(axis=1)].copy()
    if df.empty:
        raise DataError("No usable records remain after removing missing or non-numeric values.")
    for c in BOOL:
        df[c] = _to_bool(df[c], c)
    for c in CAT:
        df[c] = df[c].astype("category")
    df = df.reset_index(drop=True)
    df.attrs.update({"n_rows_read": n_read, "n_dropped_unusable": n_read - len(df),
                     "raw_missing": raw_missing})

    # Compact dtypes (1M rows stays well under ~100 MB)
    df["Rainfall_mm"] = df["Rainfall_mm"].astype("float32")
    df["Temperature_Celsius"] = df["Temperature_Celsius"].astype("float32")
    df[TARGET] = df[TARGET].astype("float32")
    if df["Days_to_Harvest"].notna().all():
        df["Days_to_Harvest"] = df["Days_to_Harvest"].astype("int16")
    return df


# --------------------------------------------------------------------------- #
# Data-quality report
# --------------------------------------------------------------------------- #
def quality_report(df: pd.DataFrame) -> dict:
    y = df[TARGET].to_numpy(np.float64)
    neg = y < 0
    rain = df["Rainfall_mm"].to_numpy(np.float64)

    rep: dict = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_rows_read": df.attrs.get("n_rows_read", len(df)),
        "n_dropped_unusable": df.attrs.get("n_dropped_unusable", 0),
        "raw_missing": df.attrs.get("raw_missing", 0),
        "n_valid": int((df[TARGET] >= 0).sum()),
        "missing_by_col": df.isna().sum().to_dict(),
        "missing_total": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "categories": {c: sorted(map(str, df[c].cat.categories)) for c in CAT},
        "n_nonfinite": int((~np.isfinite(df[NUM + [TARGET]].to_numpy(np.float64))).sum()),
        "numeric_summary": df[NUM + [TARGET]].describe().T,
        "n_negative": int(neg.sum()),
        "n_zero": int((y == 0).sum()),
    }
    rep["pct_negative"] = 100 * rep["n_negative"] / len(df)

    if rep["n_negative"]:
        rep["neg_rain_mean"] = float(rain[neg].mean())
        rep["neg_rain_max"] = float(rain[neg].max())
        rep["all_rain_mean"] = float(rain.mean())
        rep["neg_yield_min"] = float(y[neg].min())
        rep["neg_yield_mean"] = float(y[neg].mean())
        bands = pd.cut(df["Rainfall_mm"], bins=10)
        tab = (
            pd.DataFrame({"band": bands, "neg": neg})
            .groupby("band", observed=True)["neg"]
            .agg(n="size", negatives="sum")
        )
        tab["negative_rate_pct"] = 100 * tab["negatives"] / tab["n"]
        tab.index = [f"{int(i.left)}–{int(i.right)}" for i in tab.index]
        rep["neg_by_rain_band"] = tab.reset_index(names="Rainfall band (mm)")
        rep["neg_by_cat"] = {
            c: df.loc[neg, c].astype(str).value_counts().to_dict() for c in CAT
        }
    else:
        rep["neg_rain_max"] = None

    # Predictor independence / uniformity: relevant to how far results can transfer
    corr = df[NUM].astype(np.float64).corr().to_numpy()
    off = np.abs(corr[~np.eye(len(NUM), dtype=bool)])
    rep["max_abs_predictor_corr"] = float(off.max())
    rep["cat_balance"] = {
        c: float(df[c].value_counts(normalize=True).max()) for c in CAT
    }
    return rep


def eta_squared(df: pd.DataFrame, col: str) -> float:
    """Share of total yield variance explained by group membership (0-1).
    Rule of thumb: < 0.01 negligible, 0.01-0.06 small, 0.06-0.14 medium, > 0.14 large."""
    y = df[TARGET].astype(np.float64)
    g = y.groupby(df[col], observed=True)
    grand = y.mean()
    ss_between = float((g.size() * (g.mean() - grand) ** 2).sum())
    ss_total = float(((y - grand) ** 2).sum())
    return ss_between / ss_total if ss_total > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Feature frame in a fixed column order; Booleans become 0/1."""
    X = df[FEATURES].copy()
    for c in BOOL:
        X[c] = X[c].astype("int8")
    return X


def _linear_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)],
        remainder="passthrough",
        sparse_threshold=0.0,
    )
    return Pipeline([("pre", pre), ("model", LinearRegression())])


def _hgb_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1, encoded_missing_value=-1), CAT)],
        remainder="passthrough",
        sparse_threshold=0.0,
    )
    model = HistGradientBoostingRegressor(
        categorical_features=[0, 1, 2, 3],  # the CAT block comes first
        max_iter=300, learning_rate=0.1,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
        random_state=SEED,
    )
    return Pipeline([("pre", pre), ("model", model)])


def _baseline_pipeline() -> Pipeline:
    return Pipeline([("model", DummyRegressor(strategy="mean"))])


def _metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _group_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in frame.groupby(by, observed=True):
        m = _metrics(g["actual"], g["pred"])
        rows.append({by: str(key), "n": len(g), **m,
                     "Mean residual": float((g["actual"] - g["pred"]).mean())})
    return pd.DataFrame(rows)


def train_all(df: pd.DataFrame) -> dict:
    """Train baseline + two regressors with a documented, leak-free protocol.

    * 80/20 random split of ALL rows (seed 42) before anything is fitted.
    * Negative-yield rows (physically impossible) are excluded from model
      fitting and from the primary test metrics; a sensitivity evaluation on
      the full test set (negatives included) is also reported.
    * Every transformer lives inside a Pipeline, so it is fitted on training
      data only (also inside each cross-validation fold).
    * The model used by the simulator is chosen by cross-validated RMSE on
      the training data, never by test-set score.
    """
    t0 = time.time()
    n = len(df)
    idx_train, idx_test = train_test_split(np.arange(n), test_size=TEST_SIZE,
                                           random_state=SEED)
    y_all = df[TARGET].to_numpy(np.float64)
    keep = y_all >= 0
    tr = idx_train[keep[idx_train]]
    te = idx_test[keep[idx_test]]
    if len(tr) < 1000 or len(te) < 200:
        raise DataError("Too few valid rows to train and evaluate a model.")

    X = to_model_frame(df)
    X_tr, y_tr = X.iloc[tr], y_all[tr]
    X_te, y_te = X.iloc[te], y_all[te]

    builders = {
        "Baseline (training mean)": _baseline_pipeline,
        "Linear regression": _linear_pipeline,
        "HistGradientBoosting": _hgb_pipeline,
    }
    fitted, test_metrics, fit_seconds = {}, {}, {}
    for name, build in builders.items():
        t1 = time.time()
        pipe = build().fit(X_tr, y_tr)
        fit_seconds[name] = time.time() - t1
        fitted[name] = pipe
        test_metrics[name] = _metrics(y_te, pipe.predict(X_te))

    # ---- Cross-validation on a reproducible subsample of the TRAIN set ----
    rng = np.random.default_rng(SEED)
    cv_idx = rng.choice(len(tr), size=min(CV_SAMPLE, len(tr)), replace=False)
    X_cv, y_cv = X_tr.iloc[cv_idx], y_tr[cv_idx]
    kf = KFold(CV_FOLDS, shuffle=True, random_state=SEED)
    cv_rows = []
    for name, build in builders.items():
        res = cross_validate(
            build(), X_cv, y_cv, cv=kf, n_jobs=1,
            scoring={"MAE": "neg_mean_absolute_error",
                     "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
        )
        cv_rows.append({
            "Model": name,
            "CV MAE": -res["test_MAE"].mean(), "CV MAE sd": res["test_MAE"].std(),
            "CV RMSE": -res["test_RMSE"].mean(), "CV RMSE sd": res["test_RMSE"].std(),
            "CV R2": res["test_R2"].mean(), "CV R2 sd": res["test_R2"].std(),
        })
    cv = pd.DataFrame(cv_rows)

    # ---- Model selection by CV RMSE (candidates only, not the baseline) ----
    cand = cv[cv["Model"] != "Baseline (training mean)"].set_index("Model")
    best_name = cand["CV RMSE"].idxmin()
    best = fitted[best_name]
    pred_te = best.predict(X_te)
    resid = y_te - pred_te

    # ---- Sensitivity: test set INCLUDING negative-yield rows ----
    X_te_all, y_te_all = X.iloc[idx_test], y_all[idx_test]
    sensitivity = _metrics(y_te_all, best.predict(X_te_all))

    # ---- Group performance on the held-out test set ----
    te_frame = df.iloc[te][CAT + BOOL + NUM].copy()
    te_frame["actual"], te_frame["pred"] = y_te, pred_te
    te_frame["Rainfall band"] = pd.qcut(te_frame["Rainfall_mm"], 5, duplicates="drop"
                                        ).astype(str)
    te_frame["Fertilizer"] = np.where(te_frame["Fertilizer_Used"], "Yes", "No")
    te_frame["Irrigation"] = np.where(te_frame["Irrigation_Used"], "Yes", "No")
    groups = {
        "Crop": _group_table(te_frame, "Crop"),
        "Soil_Type": _group_table(te_frame, "Soil_Type"),
        "Region": _group_table(te_frame, "Region"),
        "Weather_Condition": _group_table(te_frame, "Weather_Condition"),
        "Fertilizer": _group_table(te_frame, "Fertilizer"),
        "Irrigation": _group_table(te_frame, "Irrigation"),
        "Rainfall band": _group_table(te_frame, "Rainfall band"),
    }

    # ---- Calibration by predicted decile ----
    dec = pd.qcut(pd.Series(pred_te), 10, duplicates="drop")
    calib = (pd.DataFrame({"pred": pred_te, "actual": y_te, "dec": dec})
             .groupby("dec", observed=True)
             .agg(n=("pred", "size"), mean_pred=("pred", "mean"),
                  mean_actual=("actual", "mean")).reset_index(drop=True))

    # ---- Plot samples ----
    samp = np.random.default_rng(SEED).choice(len(te), min(PLOT_SAMPLE, len(te)),
                                              replace=False)
    plot_df = pd.DataFrame({"actual": y_te[samp], "pred": pred_te[samp],
                            "residual": resid[samp],
                            "Rainfall_mm": df["Rainfall_mm"].to_numpy()[te][samp]})

    # ---- Permutation importance (original columns, test subsample) ----
    k = min(IMPORTANCE_SAMPLE, len(te))
    imp_idx = np.random.default_rng(SEED).choice(len(te), k, replace=False)
    pi = permutation_importance(best, X_te.iloc[imp_idx], y_te[imp_idx],
                                scoring="r2", n_repeats=3, random_state=SEED)
    importance = (pd.DataFrame({"Feature": FEATURES, "R2 drop": pi.importances_mean,
                                "sd": pi.importances_std})
                  .sort_values("R2 drop", ascending=False).reset_index(drop=True))

    # ---- Plausibility of predictions ----
    train_max = float(y_tr.max())
    plaus = {
        "n_test": int(len(te)),
        "pred_negative": int((pred_te < 0).sum()),
        "pred_above_train_max": int((pred_te > train_max).sum()),
        "pred_min": float(pred_te.min()), "pred_max": float(pred_te.max()),
        "train_yield_max": train_max,
    }

    q = np.quantile(resid, [0.05, 0.10, 0.50, 0.90, 0.95])
    neg_rain_max = (float(df.loc[~keep, "Rainfall_mm"].max()) if (~keep).any() else None)

    meta = {
        "trained_at": datetime.now().strftime("%b %d, %Y %I:%M %p"),
        "seed": SEED, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS,
        "cv_sample": int(len(cv_idx)),
        "n_total": n, "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_excluded_train": int(len(idx_train) - len(tr)),
        "n_excluded_test": int(len(idx_test) - len(te)),
        "train_seconds": time.time() - t0,
        "fit_seconds": fit_seconds,
        "sklearn": sklearn.__version__,
        "best_name": best_name,
        "resid_q": dict(zip(["q05", "q10", "q50", "q90", "q95"], map(float, q))),
        "resid_sd": float(resid.std()),
        "neg_rain_max": neg_rain_max,
        "bounds": {c: (float(df[c].min()), float(df[c].max())) for c in NUM},
        "medians": {c: float(df[c].median()) for c in NUM},
        "categories": {c: sorted(map(str, df[c].cat.categories)) for c in CAT},
        "modes": {c: str(df[c].mode().iloc[0]) for c in CAT},
        "hgb_iterations": int(getattr(fitted["HistGradientBoosting"]["model"],
                                      "n_iter_", 0)),
    }
    return {
        "model": best, "models": fitted, "test_metrics": test_metrics, "cv": cv,
        "sensitivity_with_negatives": sensitivity, "groups": groups,
        "calibration": calib, "plot_df": plot_df, "importance": importance,
        "plausibility": plaus, "meta": meta,
    }


def _cache_key(path: Path) -> str:
    st = path.stat()
    raw = f"{path.name}|{st.st_size}|{int(st.st_mtime)}|{MODEL_VERSION}|{SEED}|{sklearn.__version__}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def get_or_train(df: pd.DataFrame, csv_path: str | Path, cache_dir: str | Path) -> dict:
    """Load a previously trained bundle for this exact CSV, else train and save."""
    csv_path, cache_dir = Path(csv_path), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"bundle_{_cache_key(csv_path)}.joblib"
    if f.is_file():
        try:
            return joblib.load(f)
        except Exception:  # noqa: BLE001 - corrupt/incompatible cache: retrain
            f.unlink(missing_ok=True)
    try:
        bundle = train_all(df)
    except DataError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataError(f"Model training failed: {exc}") from exc
    joblib.dump(bundle, f, compress=3)
    return bundle


# --------------------------------------------------------------------------- #
# Scenario helpers
# --------------------------------------------------------------------------- #
def reference_scenario(meta: dict) -> dict:
    """Reference = dataset median weather numerics, most frequent categories,
    and NO fertilizer / NO irrigation (an 'unmanaged' comparison point)."""
    return {
        "Region": meta["modes"]["Region"], "Soil_Type": meta["modes"]["Soil_Type"],
        "Crop": meta["modes"]["Crop"],
        "Weather_Condition": meta["modes"]["Weather_Condition"],
        "Rainfall_mm": meta["medians"]["Rainfall_mm"],
        "Temperature_Celsius": meta["medians"]["Temperature_Celsius"],
        "Days_to_Harvest": int(round(meta["medians"]["Days_to_Harvest"])),
        "Fertilizer_Used": False, "Irrigation_Used": False,
    }


def predict_rows(bundle: dict, rows: list[dict]) -> np.ndarray:
    frame = pd.DataFrame(rows)[FEATURES]
    for c in BOOL:
        frame[c] = frame[c].astype("int8")
    return bundle["model"].predict(frame)


def validate_scenario(meta: dict, row: dict) -> tuple[list[str], bool]:
    """Return (warnings, blocked). Blocked scenarios must not be predicted."""
    warnings, blocked = [], False
    for c in CAT:
        if str(row[c]) not in meta["categories"][c]:
            warnings.append(f"{c} '{row[c]}' was not seen in the data.")
            blocked = True
    for c in NUM:
        lo, hi = meta["bounds"][c]
        v = row[c]
        if v is None or not np.isfinite(v) or v < lo or v > hi:
            warnings.append(f"{c} = {v} is outside the observed range "
                            f"[{lo:.1f}, {hi:.1f}]; the model cannot be trusted there.")
            blocked = True
    nr = meta.get("neg_rain_max")
    if not blocked and nr is not None and row["Rainfall_mm"] <= nr:
        warnings.append(
            f"Rainfall ≤ {nr:.0f} mm is the range where physically impossible "
            "(negative) yields were recorded; predictions here are the least reliable.")
    return warnings, blocked
