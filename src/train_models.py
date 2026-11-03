"""Train CellGuard estimators from the deduplicated NASA Battery .mat collection.

Run from the repository root: python src/train_models.py
Artifacts and a machine-readable validation report are written to models/.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from battery_features import PARTIAL_WINDOW_FEATURE_NAMES, extract_discharge_features

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "Battery_ML" / "5. Battery Data Set"
OUT = ROOT / "models"
BASE_FEATURES = PARTIAL_WINDOW_FEATURE_NAMES
IMP_FEATURES = BASE_FEATURES + ["disc_idx"]


def _items(value):
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return list(value.reshape(-1))
    return [value]


def _f(value):
    try:
        a = np.asarray(value, dtype=float).reshape(-1)
        return float(a[0]) if a.size else np.nan
    except (TypeError, ValueError):
        return np.nan


def collect_rows():
    files = sorted(DATA_ROOT.rglob("*.mat"))
    unique, hashes = [], set()
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).digest()
        if digest not in hashes:
            hashes.add(digest)
            unique.append(path)
    rows = []
    for path in unique:
        mat = sio.loadmat(path, simplify_cells=True)
        cell = mat.get(path.stem)
        if not isinstance(cell, dict) or "cycle" not in cell:
            continue
        cycles = _items(cell["cycle"])
        discharges = [(i, c) for i, c in enumerate(cycles)
                      if isinstance(c, dict) and c.get("type") == "discharge"]
        impedances = [(i, c) for i, c in enumerate(cycles)
                      if isinstance(c, dict) and c.get("type") == "impedance"]
        capacities = [_f(c.get("data", {}).get("Capacity")) for _, c in discharges]
        valid_life = [j for j, q in enumerate(capacities) if np.isfinite(q) and 1.4 < q <= 2.1]
        eol = next((j for j, q in enumerate(capacities)
                    if valid_life and j > min(valid_life) and np.isfinite(q) and 0.5 <= q <= 1.4), None)
        for disc_idx, (raw_idx, cycle) in enumerate(discharges):
            data = cycle.get("data", {})
            t = np.asarray(data.get("Time", []), dtype=float).reshape(-1)
            v = np.asarray(data.get("Voltage_measured", []), dtype=float).reshape(-1)
            cur = np.asarray(data.get("Current_measured", []), dtype=float).reshape(-1)
            temp = np.asarray(data.get("Temperature_measured", []), dtype=float).reshape(-1)
            if min(len(t), len(v), len(cur), len(temp)) < 2:
                continue
            cap = _f(data.get("Capacity"))
            if not np.isfinite(cap) or not (0.5 <= cap <= 2.1):
                continue
            features = extract_discharge_features(t, v, cur, temp)
            row = {"cell": path.stem, "disc_idx": disc_idx, "capacity": cap,
                   "energy": features.get("E_total_Wh", np.nan), **features}
            nearest = min(impedances, key=lambda x: abs(x[0] - raw_idx), default=None)
            if nearest and abs(nearest[0] - raw_idx) <= 3:
                idata = nearest[1].get("data", {})
                row["Re"] = _f(idata.get("Re"))
                row["Rct"] = _f(idata.get("Rct"))
            else:
                row["Re"] = row["Rct"] = np.nan
            row["rul"] = float(eol - disc_idx) if eol is not None and disc_idx <= eol else np.nan
            rows.append(row)
    return pd.DataFrame(rows), len(files), len(unique)


def evaluate_loco(frame, target, features, estimator):
    data = frame.dropna(subset=[target]).copy()
    data = data[np.isfinite(data[target])]
    valid_groups = data.groupby("cell").size()
    data = data[data.cell.isin(valid_groups[valid_groups >= 2].index)].reset_index(drop=True)
    y = data[target].to_numpy(float)
    groups = data.cell.to_numpy()
    pred = np.full(len(data), np.nan)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(data[features], y, groups):
        model = clone(estimator)
        model.fit(data.iloc[tr][features], y[tr])
        pred[te] = model.predict(data.iloc[te][features])
    mask = np.isfinite(pred)
    by_cell = {}
    for name in sorted(data.cell.unique()):
        ix = (data.cell.to_numpy() == name) & mask
        if ix.sum():
            by_cell[name] = {"n": int(ix.sum()), "mae": float(mean_absolute_error(y[ix], pred[ix])),
                             "rmse": float(np.sqrt(mean_squared_error(y[ix], pred[ix]))),
                             "r2": float(r2_score(y[ix], pred[ix])) if ix.sum() > 1 else None}
    summary = {"n": int(mask.sum()), "cells": int(data.cell.nunique()),
               "mae": float(mean_absolute_error(y[mask], pred[mask])),
               "mse": float(mean_squared_error(y[mask], pred[mask])),
               "rmse": float(np.sqrt(mean_squared_error(y[mask], pred[mask]))),
               "r2": float(r2_score(y[mask], pred[mask])), "per_cell": by_cell}
    final = clone(estimator).fit(data[features], y)
    return data, final, summary


def evaluate_forward(frame, target, features, estimator):
    """Evaluate future-cycle estimates using each cell's earlier cycles only."""
    data = frame.dropna(subset=[target]).copy()
    data = data[np.isfinite(data[target])]
    actual, predicted, by_cell, prefix_rows = [], [], {}, []
    for cell, group in data.groupby("cell"):
        group = group.sort_values("disc_idx").reset_index(drop=True)
        if len(group) < 5:
            continue
        split = min(len(group) - 1, max(2, int(len(group) * 0.8)))
        train, test = group.iloc[:split], group.iloc[split:]
        prefix_rows.append(train)
        model = clone(estimator).fit(train[features], train[target])
        pred = model.predict(test[features])
        y = test[target].to_numpy(float)
        actual.extend(y)
        predicted.extend(pred)
        by_cell[cell] = {"n": len(y), "mae": float(mean_absolute_error(y, pred)),
                         "rmse": float(np.sqrt(mean_squared_error(y, pred))),
                         "r2": float(r2_score(y, pred)) if len(y) > 1 else None}
    y_all, p_all = np.asarray(actual), np.asarray(predicted)
    summary = {"n": len(y_all), "cells": len(by_cell),
               "mae": float(mean_absolute_error(y_all, p_all)),
               "mse": float(mean_squared_error(y_all, p_all)),
               "rmse": float(np.sqrt(mean_squared_error(y_all, p_all))),
               "r2": float(r2_score(y_all, p_all)), "per_cell": by_cell,
               "validation_scheme": "Per-cell forward split: first 80% of cycles train, final 20% test"}
    final_data = pd.concat(prefix_rows, ignore_index=True)
    final = clone(estimator).fit(final_data[features], final_data[target])
    return final_data, final, summary


def main():
    OUT.mkdir(exist_ok=True)
    frame, source_files, unique_files = collect_rows()
    if frame.empty:
        raise RuntimeError(f"No usable discharge cycles found under {DATA_ROOT}")
    # Keep familiar column names consumed by the UI and older analysis pages.
    frame["SOH_pct"] = frame["capacity"] / 2.0 * 100.0
    frame["rul_1p4"] = frame["rul"]
    ridge = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    rf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                   ("reg", RandomForestRegressor(n_estimators=120, min_samples_leaf=2,
                                                  max_features=0.8, random_state=42,
                                                  n_jobs=1))])
    re_model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("reg", ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                                     max_features=0.9, random_state=42, n_jobs=1))])
    targets = {
        "capacity": ("capacity", BASE_FEATURES, ridge),
        "Re": ("Re", IMP_FEATURES, re_model),
        "Rct": ("Rct", IMP_FEATURES, Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                ("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])),
        "energy": ("energy", BASE_FEATURES, ridge),
        "rul": ("rul", BASE_FEATURES + ["disc_idx"], rf),
    }
    results = {}
    for key, (target, features, model) in targets.items():
        valid = frame.dropna(subset=[target])
        valid = valid[np.isfinite(valid[target])]
        if key == "Re":
            valid = valid[valid[target].between(0.005, 0.5)]
        elif key == "Rct":
            valid = valid[valid[target].between(0.005, 0.8)]
        if valid.cell.nunique() < 2:
            results[key] = {"error": "Fewer than two cells have labels"}
            continue
        if key == "Re":
            data, fitted, score = evaluate_forward(valid, target, features, model)
        else:
            data, fitted, score = evaluate_loco(valid, target, features, model)
        results[key] = {"target": target, "features": features, "model": fitted,
                        "validation": score, "training_rows": len(data),
                        "training_cells": sorted(data.cell.unique())}
        print(f"{key}: n={score['n']} cells={score['cells']} MAE={score['mae']:.5g} "
              f"RMSE={score['rmse']:.5g} R2={score['r2']:.4f}")
    # Preserve the inference layer's established artifact envelope.
    for key, filename in [("capacity", "model_capacity.pkl"), ("Re", "model_re.pkl"),
                          ("Rct", "model_rct.pkl"), ("rul", "model_rul.pkl"),
                          ("energy", "model_energy.pkl")]:
        result = results.get(key, {})
        if "model" not in result:
            continue
        joblib.dump({"pipeline": result["model"], "features": result["features"],
                     "target": result["target"], "model_type": type(result["model"].named_steps["reg"]).__name__,
                     "training_cells": result["training_cells"]}, OUT / filename)
    feature_table = frame.copy()
    feature_table.to_csv(OUT / "feature_table.csv", index=False)
    schema_path = OUT / "feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else {}
    schema.setdefault("features", {})
    for name in BASE_FEATURES:
        values = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(values):
            old = schema["features"].get(name, {})
            schema["features"][name] = {
                **old,
                "min": float(values.quantile(0.005)),
                "max": float(values.quantile(0.995)),
                "mean": float(values.mean()),
            }
    schema["model_feature_lists"] = {
        "capacity": BASE_FEATURES,
        "re_rct": IMP_FEATURES,
        "rul": BASE_FEATURES + ["disc_idx"],
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    metrics = {
        "dataset": {"source_files": source_files, "unique_files": unique_files,
                    "deduplication": "SHA-256 exact-file de-duplication",
                    "usable_discharge_rows": len(frame), "cells": int(frame.cell.nunique()),
                    "cells_list": sorted(frame.cell.unique()),
                    "capacity_filter_ah": [0.5, 2.1],
                    "rul_label": "First valid observed capacity 0.50-1.40 Ah after an observed capacity > 1.40 Ah"},
        "validation_scheme": "Leave-One-Cell-Out grouped cross-validation, except Re uses per-cell forward validation",
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "model"}
                   | ({"features": v["features"]} if "features" in v else {})
                   for k, v in results.items()},
        "notes": ["Capacity, Rct, energy, and RUL metrics are pooled out-of-cell predictions.",
                  "Re uses a chronological within-cell holdout and measures prediction of later cycles for a battery with prior history; it is not an unseen-cell score.",
                  "Impedance estimators use partial-window features and cycle index; training does not use measured capacity as an input.",
                  "RUL estimates are trained only for cells with an observed 1.40 Ah crossing.",
                  "Use the 4.0 V to 3.4 V input window and check temperature/cutoff applicability before deployment."],
    }
    (OUT / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Keep the app's existing error-band interface aligned with this retrain.
    metrics_path = OUT / "metrics.json"
    existing = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    bands = existing.setdefault("error_bands", {})
    for key, band in [("capacity", "capacity_mae_loco"), ("Re", "re_mae_loco"),
                      ("Rct", "rct_mae_loco"), ("rul", "rul_mae_loco"),
                      ("energy", "energy_mae_loco")]:
        if key in results and "validation" in results[key]:
            bands[band] = results[key]["validation"]["mae"]
    existing["expanded_dataset_validation"] = {
        "metrics_file": "training_metrics.json",
        "validation_scheme": metrics["validation_scheme"],
        "dataset": metrics["dataset"],
    }
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Wrote {OUT / 'training_metrics.json'}; rows={len(frame)}, cells={frame.cell.nunique()}, files={unique_files}/{source_files}")


if __name__ == "__main__":
    main()
