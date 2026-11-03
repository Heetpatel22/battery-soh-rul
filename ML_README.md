# CellGuard ML training and validation

## What the models estimate

CellGuard estimates discharge capacity (Ah), state of health (SOH), ohmic resistance (Re), charge-transfer resistance (Rct), remaining useful life (RUL) in cycles to a 1.40 Ah capacity threshold, and discharge energy (Wh). The capacity and SOH estimates use the measured discharge trace from 4.0 V to 3.4 V. The input trace must cross that whole voltage window.

## Training data and preparation

The expanded NASA Battery Aging collection is under `Battery_ML/5. Battery Data Set/`. The training script de-duplicates files by SHA-256 before reading them. The current run found 38 source files and 34 distinct cells. It retained 2,538 discharge cycles with a finite recorded capacity between 0.50 and 2.10 Ah. This excludes missing, zero, implausibly low, and above-rated capacity records from supervised training. Re and Rct targets have additional broad physical bounds (0.005–0.50 Ω and 0.005–0.80 Ω respectively) to remove corrupted extreme EIS values. Re/Rct are joined to the nearest impedance record within three raw records.

RUL labels are formed only for cells that first have an observed capacity above 1.40 Ah and later have a valid recorded discharge capacity between 0.50 and 1.40 Ah. The 0.50 Ah floor avoids treating zero-capacity data anomalies as end-of-life. The label is the cycle distance to that first valid threshold crossing. This leaves 793 labeled cycles from 17 cells. Cells that never cross the threshold, or begin below it, do not contribute to RUL training.

The data includes different temperatures, cutoff voltages, operating histories, and documented anomalies. The filtering above is a practical first pass, not a claim that all experimental conditions are equivalent. The feature pipeline does not currently add explicit ambient temperature or cutoff voltage as separate model inputs.

## Models and why they were selected

| Target | Estimator | Inputs | Reason |
| --- | --- | --- | --- |
| Capacity | Median imputation, standard scaling, Ridge regression | 18 partial-discharge features | Regularized linear model is compact and stable with correlated voltage-window features, and can extrapolate beyond the training target range. |
| SOH | Derived from predicted capacity: capacity / 2.0 Ah × 100 | Capacity estimate | SOH is a rescaling of capacity for the nominal 2.0 Ah cell rating, not a separate fitted model. |
| Re | Median imputation and Extra Trees regression | 18 partial-discharge features plus discharge-cycle index | Captures nonlinear within-cell changes and supports prediction of later cycles from earlier history. Its validation is chronological within known cells, not a test on entirely unseen cells. |
| Rct | Median imputation, standard scaling, Ridge regression | 18 partial-discharge features plus discharge-cycle index | Same compact estimator; grouped validation shows useful pooled fit, though cell-level performance varies. |
| RUL | Median imputation and Random Forest regression (120 trees, minimum leaf size 2) | 18 partial-discharge features plus cycle index | A tree ensemble can model nonlinear degradation patterns without assuming one linear capacity-fade rate. |
| Discharge energy | Median imputation, standard scaling, Ridge regression | 18 partial-discharge features | A compact baseline for a continuous target, using only early-trace features at inference. |

## Validation protocol and results

Capacity, Rct, energy, and RUL use **Leave-One-Cell-Out (LOCO)**: all cycles from one cell are held out together. Re uses a **chronological within-cell holdout**: the first 80% of each cell's cycles train the estimator and the last 20% are predicted. This measures later-cycle prediction for a battery with an established history; it does not measure performance on a never-seen battery. MAE and RMSE use target units; R² is unitless. Validation MSE is mean squared error (squared-error loss), in squared target units. Ridge uses L2 regularization; Extra Trees averages randomized trees; the Random Forest uses squared-error splitting.

| Target | Validation rows / cells | MAE | Validation MSE (loss) | RMSE | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Capacity | 2,538 / 34 | 0.1042 Ah | 0.02180 Ah² | 0.1477 Ah | 0.7635 |
| Re | 424 / 33 | 0.00266 Ω | 0.0000341 Ω² | 0.00584 Ω | 0.9407 |
| Rct | 2,055 / 34 | 0.01483 Ω | 0.0004391 Ω² | 0.02096 Ω | 0.8336 |
| Discharge energy | 2,538 / 34 | 0.3483 Wh | 0.23811 Wh² | 0.4880 Wh | 0.7807 |
| RUL | 793 / 17 | 19.31 cycles | 709.40 cycles² | 26.63 cycles | 0.3709 |

**Interpretation:** Capacity and energy show useful pooled cross-cell predictive signal, with materially higher errors than the earlier four-cell results. Re achieves R² 0.9407 for later-cycle prediction within cells that have prior history; this must not be presented as unseen-cell performance. Rct has a strong pooled R², but should still be checked by cell before operational use. RUL has moderate fit and a wide error (about 19.3 cycles MAE); use as a rough planning estimate, not a precise service-life guarantee. A positive pooled score does not imply equally good performance on every held-out cell. Detailed per-cell metrics and model metadata are in `models/training_metrics.json`.

## Reproduce training

From the repository root, with dependencies in `requirements.txt` installed:

```bash
python src/train_models.py
```

The script writes the deployable pickles in `models/`, the extracted `models/feature_table.csv`, and the full LOCO report to `models/training_metrics.json`. It also refreshes the app's uncertainty bands in `models/metrics.json`. Scikit-learn model pickles should be loaded with a compatible scikit-learn version.

## User inputs and model limits

The diagnostics page accepts either a NASA reference cell/cycle or a CSV of raw discharge telemetry with columns `Time_s`, `Voltage_V`, `Current_A`, and `Temperature_C`. Current should be negative during discharge. The trace must cross 4.0 V down through 3.4 V. The app derives the 18 features; users do not need to calculate them manually. The included data-backed reference selector covers B0005, B0006, B0007, and B0018, while training uses the deduplicated 34-cell collection.

RUL depends on the NASA dataset's first measured crossing of 1.40 Ah and is available only for represented degradation histories. Inference uses the entered cycle number (or a default cycle index for a new cell), so RUL for an unrelated chemistry, current profile, temperature, or cycle numbering scheme is out of distribution. The current interface is a research/demo tool and should not be treated as a safety-certified BMS.
