"""
test_features.py - Unit tests for battery_features.py
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add src to path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from battery_features import (
    extract_discharge_features,
    interpolate_crossing_time,
    load_cell,
    PARTIAL_WINDOW_FEATURE_NAMES,
    CUTOFF_DEPENDENT_FEATURE_NAMES,
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def test_synthetic_discharge_crossing_times():
    """
    Test feature extraction on a synthetic linear discharge curve with exact known crossing points.
    Linear drop from 4.2 V to 3.2 V over 1000 s:
    V(t) = 4.2 - 0.001 * t
    Target crossings:
      4.0 V at t = 200 s
      3.8 V at t = 400 s
      3.6 V at t = 600 s
      3.4 V at t = 800 s
    """
    t = np.linspace(0, 1000, 1001)  # dt = 1 s
    v = 4.2 - 0.001 * t
    # Include an initial rest sample at t=0 then load of 2A
    i = np.full_like(t, -2.0)
    i[0] = 0.0  # rest at t=0
    temp = 25.0 + 0.015 * t  # rises from 25 to 40 C

    feats = extract_discharge_features(t, v, i, temp)

    # Check timing
    assert np.isclose(feats["T_40_38"], 200.0, atol=1e-2), f"Expected 200s, got {feats['T_40_38']}"
    assert np.isclose(feats["T_38_36"], 200.0, atol=1e-2), f"Expected 200s, got {feats['T_38_36']}"
    assert np.isclose(feats["T_36_34"], 200.0, atol=1e-2), f"Expected 200s, got {feats['T_36_34']}"
    assert np.isclose(feats["T_40_36"], 400.0, atol=1e-2), f"Expected 400s, got {feats['T_40_36']}"
    assert np.isclose(feats["T_38_34"], 400.0, atol=1e-2), f"Expected 400s, got {feats['T_38_34']}"

    # Window charge: 2 A * 400 s = 800 As = 800/3600 Ah = 0.2222 Ah
    expected_q = (2.0 * 400.0) / 3600.0
    assert np.isclose(feats["Q_40_36"], expected_q, atol=1e-3)

    # Temperature
    assert np.isclose(feats["T_max"], 40.0, atol=1e-2)
    assert np.isclose(feats["T_rise"], 15.0, atol=0.05)

    # Initial resistance proxy: (V[0] - V[1]) / (I[1] - I[0])
    # V[0] = 4.2, V[1] = 4.199, I[0] = 0, I[1] = -2.0
    expected_r = (4.2 - 4.199) / 2.0  # 0.0005 Ohm
    assert np.isclose(feats["R_dc_proxy"], expected_r, atol=1e-4)


def test_edge_case_truncated_curve():
    """
    Test curve that stops at 3.5 V (never reaches 3.4 V).
    T_36_34 and T_38_34 must return np.nan without raising errors.
    """
    t = np.linspace(0, 500, 501)
    v = 4.1 - 0.001 * t  # ends at 3.6 V
    i = np.full_like(t, -2.0)
    temp = np.full_like(t, 25.0)

    feats = extract_discharge_features(t, v, i, temp)
    assert np.isnan(feats["T_36_34"])
    assert np.isnan(feats["T_38_34"])
    assert np.isfinite(feats["T_40_38"])


def test_real_b0005_discharge_cycle():
    """
    Test on real B0005 cycle 0 data. Verify all partial-window features are finite.
    """
    mat_path = os.path.join(DATA_DIR, "B0005.mat")
    if not os.path.exists(mat_path):
        pytest.skip("B0005.mat not found in data directory")

    df = load_cell(mat_path)
    assert len(df) == 168, f"Expected 168 discharge cycles, got {len(df)}"

    row0 = df.iloc[0]
    for feat in PARTIAL_WINDOW_FEATURE_NAMES:
        assert feat in row0, f"Missing feature {feat}"
        assert np.isfinite(row0[feat]), f"Feature {feat} is NaN on real B0005 cycle 0: {row0[feat]}"

    assert np.isclose(row0["capacity"], 1.8565, atol=1e-2)
    assert row0["eol_idx"] == 124
    assert bool(row0["eol_is_pseudo"]) is False
    assert row0["cutoff_V"] == 2.7


def test_b0007_pseudo_eol_flag():
    """
    Test that B0007 is properly flagged with eol_is_pseudo=True and eol_idx=165.
    """
    mat_path = os.path.join(DATA_DIR, "B0007.mat")
    if not os.path.exists(mat_path):
        pytest.skip("B0007.mat not found in data directory")

    df = load_cell(mat_path)
    assert bool(df["eol_is_pseudo"].all()) is True
    assert df["eol_idx"].iloc[0] == 165

