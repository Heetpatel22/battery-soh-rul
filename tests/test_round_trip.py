"""
test_round_trip.py - Verify predictions against sample_cycles truth table.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference import BatteryPredictor

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_cycles"))

GROUND_TRUTH = {
    "B0018_cycle_005_early.csv": {"cap": 1.8285, "soh": 91.43, "re": None, "rct": None, "idx": 5},
    "B0018_cycle_015_early.csv": {"cap": 1.7712, "soh": 88.56, "re": None, "rct": None, "idx": 15},
    "B0018_cycle_050_middle.csv": {"cap": 1.6664, "soh": 83.32, "re": 0.06336, "rct": 0.08601, "idx": 50},
    "B0018_cycle_065_middle.csv": {"cap": 1.5316, "soh": 76.58, "re": 0.06244, "rct": 0.08553, "idx": 65},
    "B0018_cycle_100_late.csv": {"cap": 1.3849, "soh": 69.25, "re": 0.06545, "rct": 0.08884, "idx": 100},
    "B0018_cycle_115_late.csv": {"cap": 1.3882, "soh": 69.41, "re": 0.06297, "rct": 0.08791, "idx": 115},
}


def test_round_trip_predictions():
    predictor = BatteryPredictor(MODELS_DIR)
    assert predictor.models_loaded, "Models failed to load from models_dir"

    for fname, truth in GROUND_TRUTH.items():
        fpath = os.path.join(SAMPLE_DIR, fname)
        assert os.path.exists(fpath), f"Sample cycle missing: {fpath}"

        df = pd.read_csv(fpath)
        t = df["Time_s"].values
        v = df["Voltage_V"].values
        i = df["Current_A"].values
        temp = df["Temperature_C"].values

        res = predictor.predict_from_raw_discharge(t, v, i, temp, disc_idx=truth["idx"])
        assert res["success"], f"Prediction failed on {fname}: {res['errors']}"

        pred_cap = res["predictions"]["capacity"]["value"]
        true_cap = truth["cap"]

        # Assert prediction is within LOCO bounds (~0.05 Ah)
        assert abs(pred_cap - true_cap) < 0.06, f"{fname} Cap error too large: pred={pred_cap}, true={true_cap}"

        # Check RUL EOL behavior: if capacity <= 1.4 Ah show "End of life reached", do not show RUL number
        if pred_cap <= 1.40:
            assert res["predictions"]["RUL"]["status"] == "End of life reached"
            assert res["predictions"]["RUL"]["value"] is None
        else:
            assert res["predictions"]["RUL"]["status"] == "Active"
            assert isinstance(res["predictions"]["RUL"]["value"], int)
            assert res["predictions"]["RUL"]["value"] >= 0

        # Test held-out model
        res_holdout = predictor.predict_demo_unseen_b0018(res["features_extracted"], disc_idx=truth["idx"])
        assert res_holdout["success"]
        pred_holdout_cap = res_holdout["predictions"]["capacity"]["value"]
        assert abs(pred_holdout_cap - true_cap) < 0.08, f"{fname} Holdout error too large: pred={pred_holdout_cap}, true={true_cap}"

        if pred_holdout_cap <= 1.40:
            assert res_holdout["predictions"]["RUL"]["status"] == "End of life reached"
            assert res_holdout["predictions"]["RUL"]["value"] is None
        else:
            assert res_holdout["predictions"]["RUL"]["status"] == "Active"
            assert isinstance(res_holdout["predictions"]["RUL"]["value"], int)

        # Check impedance predictions for cycles with ground truth
        if truth["re"] is not None:
            pred_re = res["predictions"]["Re"]["value"]
            # LOCO Re error band is ~0.0066 ohm, allow reasonable band
            assert abs(pred_re - truth["re"]) < 0.015, f"{fname} Re error: pred={pred_re}, true={truth['re']}"
        if truth["rct"] is not None:
            pred_rct = res["predictions"]["Rct"]["value"]
            # LOCO Rct error band is ~0.0055 ohm, allow reasonable band
            assert abs(pred_rct - truth["rct"]) < 0.015, f"{fname} Rct error: pred={pred_rct}, true={truth['rct']}"


if __name__ == "__main__":
    test_round_trip_predictions()
    print("All sample cycle round-trip assertions passed!")
