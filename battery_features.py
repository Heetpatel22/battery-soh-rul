"""
battery_features.py - Feature extraction and cell data processing for NASA Battery Dataset.

This module is shared between offline/Colab training pipelines and the runtime
inference interface to ensure strict consistency and avoid training-serving skew.
Pure numpy/scipy/pandas, deterministic and unit-tested.
"""

import os
import numpy as np
import scipy.io as sio
import pandas as pd
from scipy.integrate import trapezoid

# Known operating conditions and specifications
CUTOFF_VOLTAGES = {
    "B0005": 2.7,
    "B0006": 2.5,
    "B0007": 2.2,
    "B0018": 2.5,
}

# 0-based discharge index of End of Life (capacity <= 1.4 Ah)
# Note: B0007 never drops to 1.4000 Ah; min capacity is 1.4005 Ah at index 165 (pseudo EOL)
EOL_DISCHARGE_INDICES = {
    "B0005": 124,
    "B0006": 108,
    "B0007": 165,
    "B0018": 96,
}

RATED_CAPACITY_AH = 2.0
EOL_CAPACITY_AH = 1.4

# Feature sets
PARTIAL_WINDOW_FEATURE_NAMES = [
    "T_40_38",
    "T_38_36",
    "T_36_34",
    "T_40_36",
    "T_38_34",
    "Q_40_36",
    "V_at_60s",
    "V_at_300s",
    "V_at_600s",
    "V_at_1200s",
    "V_start",
    "V_mean_first600s",
    "V_slope_mid",
    "T_start",
    "T_max",
    "T_rise",
    "T_mean",
    "R_dc_proxy",
]

CUTOFF_DEPENDENT_FEATURE_NAMES = [
    "T_total",
    "Q_total",
    "E_total_Wh",
    "V_min",
]

CHARGE_FEATURE_NAMES = [
    "chg_total_t",
    "chg_cc_t",
    "chg_cv_t",
    "chg_Tmax",
    "chg_energy_Wh",
]


def interpolate_crossing_time(t: np.ndarray, v: np.ndarray, v_target: float) -> float:
    """
    Find the time when voltage falls to or crosses v_target using linear interpolation.
    Returns np.nan if v_target is never crossed.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return np.nan

    # If first point is already at or below target
    if v[0] <= v_target:
        return float(t[0])

    for k in range(1, len(v)):
        if v[k] <= v_target and v[k - 1] > v_target:
            dv = v[k] - v[k - 1]
            if abs(dv) < 1e-12:
                return float(t[k])
            frac = (v_target - v[k - 1]) / dv
            return float(t[k - 1] + frac * (t[k] - t[k - 1]))

    return np.nan


def extract_discharge_features(
    t: np.ndarray,
    v: np.ndarray,
    i: np.ndarray,
    temp: np.ndarray,
) -> dict:
    """
    Extract partial-window health indicators, cut-off dependent quantities,
    and electrical resistance proxies from a single discharge cycle.

    Parameters
    ----------
    t : np.ndarray
        Elapsed time array in seconds.
    v : np.ndarray
        Terminal voltage array in Volts.
    i : np.ndarray
        Current array in Amperes (negative during discharge).
    temp : np.ndarray
        Temperature array in degrees Celsius.

    Returns
    -------
    dict
        Dictionary containing extracted electrical and thermal features.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    temp = np.asarray(temp, dtype=float)

    n_samples = len(t)
    if n_samples < 2 or len(v) != n_samples or len(i) != n_samples or len(temp) != n_samples:
        all_features = PARTIAL_WINDOW_FEATURE_NAMES + CUTOFF_DEPENDENT_FEATURE_NAMES
        return {k: np.nan for k in all_features}

    # 1. Identify start of active load (|I| > 0.5 A)
    load_mask = np.abs(i) > 0.5
    if np.any(load_mask):
        k_load = int(np.where(load_mask)[0][0])
        t_load_start = float(t[k_load])
        v_start = float(v[k_load])
        t_start = float(temp[k_load])
    else:
        k_load = 0
        t_load_start = float(t[0])
        v_start = float(v[0])
        t_start = float(temp[0])

    # 2. Resistance proxy R_dc_proxy from initial step
    # If rest samples exist before load step:
    if k_load > 0:
        v_rest = v[k_load - 1]
        i_rest = i[k_load - 1]
        v_loaded = v[k_load]
        i_loaded = i[k_load]
        di = abs(i_loaded - i_rest)
        r_dc_proxy = float((v_rest - v_loaded) / di) if di > 0.1 else np.nan
    else:
        # If load starts at sample 0, check drop across first available points
        if n_samples > 2 and abs(i[1] - i[0]) > 0.1:
            r_dc_proxy = float(abs(v[0] - v[1]) / abs(i[1] - i[0]))
        else:
            r_dc_proxy = np.nan

    # 3. Partial-window crossing times (4.0 V, 3.8 V, 3.6 V, 3.4 V)
    t_40 = interpolate_crossing_time(t, v, 4.0)
    t_38 = interpolate_crossing_time(t, v, 3.8)
    t_36 = interpolate_crossing_time(t, v, 3.6)
    t_34 = interpolate_crossing_time(t, v, 3.4)

    def _diff_time(t_end, t_start_w):
        if np.isfinite(t_end) and np.isfinite(t_start_w) and t_end >= t_start_w:
            return float(t_end - t_start_w)
        return np.nan

    T_40_38 = _diff_time(t_38, t_40)
    T_38_36 = _diff_time(t_36, t_38)
    T_36_34 = _diff_time(t_34, t_36)
    T_40_36 = _diff_time(t_36, t_40)
    T_38_34 = _diff_time(t_34, t_38)

    # 4. Partial-window charge Q_40_36 (Ah)
    if np.isfinite(t_40) and np.isfinite(t_36) and t_36 > t_40:
        # Interpolate current at endpoints
        i_40 = float(np.interp(t_40, t, np.abs(i)))
        i_36 = float(np.interp(t_36, t, np.abs(i)))
        mid_mask = (t > t_40) & (t < t_36)
        t_sub = np.concatenate(([t_40], t[mid_mask], [t_36]))
        i_sub = np.concatenate(([i_40], np.abs(i)[mid_mask], [i_36]))
        Q_40_36 = float(trapezoid(i_sub, t_sub) / 3600.0)
    else:
        Q_40_36 = np.nan

    # 5. Voltages at fixed elapsed times from load start
    def _v_at_elapsed(dt_sec):
        target_t = t_load_start + dt_sec
        if target_t <= t[-1]:
            return float(np.interp(target_t, t, v))
        return np.nan

    V_at_60s = _v_at_elapsed(60.0)
    V_at_300s = _v_at_elapsed(300.0)
    V_at_600s = _v_at_elapsed(600.0)
    V_at_1200s = _v_at_elapsed(1200.0)

    # 6. Mean voltage over first 600 seconds of load
    mask_600 = (t >= t_load_start) & (t <= t_load_start + 600.0)
    if np.any(mask_600):
        V_mean_first600s = float(np.mean(v[mask_600]))
    else:
        V_mean_first600s = float(v_start)

    # 7. Voltage slope around 3.7 V (dV/dt, negative)
    t_375 = interpolate_crossing_time(t, v, 3.75)
    t_365 = interpolate_crossing_time(t, v, 3.65)
    if np.isfinite(t_375) and np.isfinite(t_365) and t_365 > t_375:
        V_slope_mid = float((3.65 - 3.75) / (t_365 - t_375))
    else:
        V_slope_mid = np.nan

    # 8. Thermal features
    T_max = float(np.max(temp))
    T_rise = float(T_max - t_start)
    T_mean = float(np.mean(temp))

    # 9. Cut-off dependent features (for M4 and Coulomb counting baseline, NOT for M1)
    T_total = float(t[-1] - t_load_start)
    Q_total = float(trapezoid(np.abs(i), t) / 3600.0)
    E_total_Wh = float(trapezoid(v * np.abs(i), t) / 3600.0)
    V_min = float(np.min(v))

    return {
        # Partial-window timing
        "T_40_38": T_40_38,
        "T_38_36": T_38_36,
        "T_36_34": T_36_34,
        "T_40_36": T_40_36,
        "T_38_34": T_38_34,
        # Partial-window charge
        "Q_40_36": Q_40_36,
        # Fixed elapsed time voltages
        "V_at_60s": V_at_60s,
        "V_at_300s": V_at_300s,
        "V_at_600s": V_at_600s,
        "V_at_1200s": V_at_1200s,
        # Early-curve voltage & slope
        "V_start": v_start,
        "V_mean_first600s": V_mean_first600s,
        "V_slope_mid": V_slope_mid,
        # Temperature
        "T_start": t_start,
        "T_max": T_max,
        "T_rise": T_rise,
        "T_mean": T_mean,
        # Resistance proxy
        "R_dc_proxy": r_dc_proxy,
        # Cut-off dependent (kept strictly separate from M1 features)
        "T_total": T_total,
        "Q_total": Q_total,
        "E_total_Wh": E_total_Wh,
        "V_min": V_min,
    }


def extract_charge_features(
    t: np.ndarray,
    v: np.ndarray,
    i: np.ndarray,
    temp: np.ndarray,
) -> dict:
    """
    Extract electrical and thermal features from a charge cycle.
    Records with fewer than 50 samples or zero CC duration are flagged as invalid.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    temp = np.asarray(temp, dtype=float)

    if len(t) < 50:
        return {
            "has_valid_chg": False,
            "chg_total_t": np.nan,
            "chg_cc_t": np.nan,
            "chg_cv_t": np.nan,
            "chg_Tmax": np.nan,
            "chg_energy_Wh": np.nan,
        }

    # Remove any isolated NaNs by linear interpolation
    for arr in [t, v, i, temp]:
        if np.isnan(arr).any():
            nan_mask = np.isnan(arr)
            valid_idx = np.where(~nan_mask)[0]
            if len(valid_idx) > 1:
                arr[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, arr[valid_idx])

    chg_total_t = float(t[-1] - t[0])
    chg_Tmax = float(np.max(temp))

    # CC phase: current > 1.3 A and voltage < 4.19 V
    cc_mask = (i > 1.3) & (v < 4.19)
    if np.any(cc_mask):
        t_cc = t[cc_mask]
        chg_cc_t = float(t_cc[-1] - t_cc[0])
    else:
        chg_cc_t = 0.0

    # CV phase: voltage >= 4.19 V and current >= 0.02 A
    cv_mask = (v >= 4.19) & (i >= 0.02)
    if np.any(cv_mask):
        t_cv = t[cv_mask]
        chg_cv_t = float(t_cv[-1] - t_cv[0])
    else:
        chg_cv_t = 0.0

    # Charge energy (integral of positive V*I)
    p_chg = np.maximum(v * i, 0.0)
    chg_energy_Wh = float(trapezoid(p_chg, t) / 3600.0)

    # Invalidate if no CC stage was observed
    if chg_cc_t <= 0.0:
        return {
            "has_valid_chg": False,
            "chg_total_t": np.nan,
            "chg_cc_t": np.nan,
            "chg_cv_t": np.nan,
            "chg_Tmax": np.nan,
            "chg_energy_Wh": np.nan,
        }

    return {
        "has_valid_chg": True,
        "chg_total_t": chg_total_t,
        "chg_cc_t": chg_cc_t,
        "chg_cv_t": chg_cv_t,
        "chg_Tmax": chg_Tmax,
        "chg_energy_Wh": chg_energy_Wh,
    }


def load_cell(mat_path: str) -> pd.DataFrame:
    """
    Load a NASA Li-ion battery .mat file and construct a feature DataFrame
    containing one row per discharge cycle.

    Performs:
    1. Discharge cycle feature extraction.
    2. EIS sweep alignment within +/- 3 raw records.
    3. Preceding charge feature association.
    4. SOH and RUL labeling according to strict project definitions.

    Parameters
    ----------
    mat_path : str
        Path to the .mat file (e.g. 'B0005.mat').

    Returns
    -------
    pd.DataFrame
        Table with all cycle features, ground truth targets, and audit columns.
    """
    cell_name = os.path.basename(mat_path).split(".")[0]
    mat = sio.loadmat(mat_path, simplify_cells=True)

    if cell_name not in mat:
        raise KeyError(f"Top-level cell key '{cell_name}' not found in {mat_path}")

    cycles = mat[cell_name]["cycle"]
    num_records = len(cycles)

    # 1. Map EIS sweeps (raw_idx -> (Re, Rct))
    eis_sweeps = {}
    for idx, c in enumerate(cycles):
        if c.get("type") == "impedance":
            data = c.get("data", {})
            re_val = data.get("Re", np.nan)
            rct_val = data.get("Rct", np.nan)
            try:
                re_f = float(re_val)
            except (ValueError, TypeError):
                re_f = np.nan
            try:
                rct_f = float(rct_val)
            except (ValueError, TypeError):
                rct_f = np.nan
            eis_sweeps[idx] = (re_f, rct_f)

    # 2. Map valid charge records
    charge_features_map = {}
    for idx, c in enumerate(cycles):
        if c.get("type") == "charge":
            cdata = c.get("data", {})
            t = np.asarray(cdata.get("Time", []), dtype=float)
            v = np.asarray(cdata.get("Voltage_measured", []), dtype=float)
            i = np.asarray(cdata.get("Current_measured", []), dtype=float)
            temp = np.asarray(cdata.get("Temperature_measured", []), dtype=float)
            charge_features_map[idx] = extract_charge_features(t, v, i, temp)

    # 3. Process discharge records
    rows = []
    disc_idx = 0
    cutoff_V = CUTOFF_VOLTAGES.get(cell_name, np.nan)
    eol_idx = EOL_DISCHARGE_INDICES.get(cell_name, 0)
    eol_is_pseudo = cell_name == "B0007"

    for raw_idx, c in enumerate(cycles):
        if c.get("type") != "discharge":
            continue

        ddata = c.get("data", {})
        t = np.asarray(ddata.get("Time", []), dtype=float)
        v = np.asarray(ddata.get("Voltage_measured", []), dtype=float)
        i = np.asarray(ddata.get("Current_measured", []), dtype=float)
        temp = np.asarray(ddata.get("Temperature_measured", []), dtype=float)

        # Capacity label
        cap_val = ddata.get("Capacity", np.nan)
        try:
            if isinstance(cap_val, np.ndarray):
                capacity = float(cap_val.ravel()[0]) if cap_val.size > 0 else np.nan
            else:
                capacity = float(cap_val)
        except (ValueError, TypeError):
            capacity = np.nan

        soh_pct = (capacity / RATED_CAPACITY_AH * 100.0) if np.isfinite(capacity) else np.nan

        # Discharge features
        disch_feats = extract_discharge_features(t, v, i, temp)

        # Impedance alignment: find nearest EIS sweep within +/- 3 raw records
        best_dist = 999
        matched_imp_idx = np.nan
        matched_re = np.nan
        matched_rct = np.nan
        for imp_idx, (re_val, rct_val) in eis_sweeps.items():
            dist = abs(imp_idx - raw_idx)
            if dist <= 3 and dist < best_dist:
                best_dist = dist
                matched_imp_idx = imp_idx
                matched_re = re_val
                matched_rct = rct_val

        # Preceding charge alignment (look back up to 3 records)
        matched_chg = {k: np.nan for k in CHARGE_FEATURE_NAMES}
        matched_chg["has_valid_chg"] = False
        for lookback in range(1, 4):
            prev_idx = raw_idx - lookback
            if prev_idx in charge_features_map:
                chg_res = charge_features_map[prev_idx]
                if chg_res.get("has_valid_chg", False):
                    matched_chg = chg_res
                    break

        # RUL: defined as eol_idx - disc_idx (NaN for cycles past EOL)
        rul_1p4 = float(eol_idx - disc_idx) if disc_idx <= eol_idx else np.nan

        row = {
            "cell": cell_name,
            "disc_idx": disc_idx,
            "raw_idx": raw_idx,
            "capacity": capacity,
            "SOH_pct": soh_pct,
            "Re": matched_re,
            "Rct": matched_rct,
            "re_dist": float(best_dist) if best_dist <= 3 else np.nan,
            "imp_raw_idx": float(matched_imp_idx) if best_dist <= 3 else np.nan,
            **disch_feats,
            **matched_chg,
            "rul_1p4": rul_1p4,
            "eol_idx": eol_idx,
            "eol_is_pseudo": eol_is_pseudo,
            "cutoff_V": cutoff_V,
        }

        rows.append(row)
        disc_idx += 1

    df = pd.DataFrame(rows)
    return df


def build_all_features(data_dir: str) -> pd.DataFrame:
    """
    Load all 4 cells from data_dir and build the unified feature table.
    """
    dfs = []
    cells = ["B0005", "B0006", "B0007", "B0018"]
    for cell in cells:
        mat_path = os.path.join(data_dir, f"{cell}.mat")
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Missing required cell file: {mat_path}")
        df_cell = load_cell(mat_path)
        dfs.append(df_cell)

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df
