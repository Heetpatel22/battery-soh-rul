# CellGuard: Machine Learning-Based Battery Internal Resistance (Re & Rct) Estimation
## Comprehensive Final Year EEE Project Guide & Viva Defense Companion

---

## 1. Executive Summary & Project Purpose

### What is CellGuard?
**CellGuard** is a battery diagnostic and prognostics platform designed for Lithium-ion batteries (specifically 18650 cylindrical cells). It uses machine learning to estimate a battery's **internal health parameters**—specifically **Ohmic Resistance ($R_e$)** and **Charge-Transfer Resistance ($R_{ct}$)**—directly from basic operational data (voltage, current, temperature, and cycle number) during normal battery discharge.

### In Simple Words:
Normally, checking a battery's internal health requires stopping the vehicle, taking the battery to an advanced laboratory, and connecting it to a \$50,000 spectrometer for hours. **CellGuard replaces that expensive laboratory test with smart software.** It watches how the voltage and temperature change during standard use, and immediately predicts the internal resistance with over **96% accuracy**.

---

## 2. The Core Engineering Problem We Are Solving

### Why Do Batteries Age?
As a Lithium-ion battery charges and discharges over hundreds of cycles, physical and chemical degradation takes place inside the cell:
1. **Electrolyte degradation & contact wear:** The liquid electrolyte gradually loses conductivity, and physical tab connections degrade. This increases **Ohmic Resistance ($R_e$)**.
2. **Solid Electrolyte Interphase (SEI) Layer Growth:** Parasitic chemical reactions between the lithium ions and the organic electrolyte create a resistive coating on the graphite anode surface. As this coating thickens, it becomes harder and harder for lithium ions to pass across the interface. This increases **Charge-Transfer Resistance ($R_{ct}$)**.

### Why Is This a Problem in Electric Vehicles (EVs) & Energy Storage Systems?
- **Voltage Drop Under Load:** According to Ohm's Law, when an electric vehicle accelerates and draws high current ($I$), the internal resistance causes an immediate voltage drop:
  $$\Delta V = I \cdot (R_e + R_{ct})$$
  Higher resistance causes the battery to hit the low-voltage cutoff limit prematurely, stranding the vehicle even when chemical energy remains.
- **Joule Heating & Fire Hazard:** Power lost as waste heat inside the battery is:
  $$P_{\text{loss}} = I^2 \cdot (R_e + R_{ct})$$
  As $R_e$ and $R_{ct}$ increase, heat generation grows exponentially, placing huge stress on the cooling system and risking **thermal runaway**.

### The Real-World Hardware Dilemma:
The gold standard for measuring $R_e$ and $R_{ct}$ is **Electrochemical Impedance Spectroscopy (EIS)**. 
However, **EIS cannot be used in commercial EVs or active solar battery banks** because:
1. It requires the battery to be in an idle rest state for hours at strict temperature equilibrium (24°C).
2. It requires injecting complex sinusoidal AC currents from 0.1 Hz to 5,000 Hz.
3. The measurement hardware is bulky, fragile, expensive, and sensitive to automotive electrical noise.

### CellGuard's Solution:
CellGuard extracts **19 non-invasive physical features** from the observable partial discharge window (from 4.0 V down to 3.4 V). A trained **Extra Trees machine learning ensemble** then maps these operational voltage and temperature signatures directly to $R_e$ and $R_{ct}$ in real time—**without needing any EIS hardware.**

---

## 3. Which Machine Learning Models Are Used & Why

We evaluated four different machine learning algorithms representing distinct mathematical paradigms:

| Model | Architecture Type | What It Does / How It Works | Fit Accuracy ($R^2$) | Average Error (MAE) | Role in Project |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Extra Trees Regressor** | Extremely Randomized Trees Ensemble | Builds 150 randomized decision trees. Cuts the feature space at random orthogonal thresholds and averages the predictions. | **$R_e: 97.01\%$**<br/>**$R_{ct}: 96.83\%$** | **$1.13\text{ m}\Omega$**<br/>**$1.28\text{ m}\Omega$** | **Primary Production Model (Winner)** |
| **XGBoost Regressor** | Gradient Boosted Decision Trees | Iteratively builds sequential trees where each new tree corrects the residual errors made by the previous trees. | **$R_e: 96.02\%$**<br/>**$R_{ct}: 95.83\%$** | $1.24\text{ m}\Omega$<br/>$1.47\text{ m}\Omega$ | Strong Alternative Engine |
| **Random Forest Regressor** | Bagging Ensemble of Decision Trees | Builds 150 standard decision trees using bootstrap data samples and finds the optimal split per feature. | **$R_e: 95.68\%$**<br/>**$R_{ct}: 95.41\%$** | $1.25\text{ m}\Omega$<br/>$1.50\text{ m}\Omega$ | Robust Benchmark Model |
| **Ridge Regression** | L2-Regularized Linear Model | Fits a linear hyperplane ($y = \mathbf{w}^T \mathbf{x} + b$) with a penalty on large weights to prevent overfitting. | **$R_e: 90.29\%$**<br/>**$R_{ct}: 90.60\%$** | $2.28\text{ m}\Omega$<br/>$2.44\text{ m}\Omega$ | Linear Baseline Model |

### Why Extra Trees Outperforms Other Models:
1. **Non-Linear Dynamics:** The relationship between battery overpotential and internal resistance is fundamentally non-linear. Linear models (Ridge) cannot capture how resistance sharply bends as the battery degrades.
2. **Noise Immunity:** Extra Trees randomizes both the cut-points and feature subsets, making it much more robust against sensor noise and voltage spikes than a single tree or linear regression.
3. **Execution Speed:** Extra Trees trains faster than standard Random Forest because it does not exhaustively search for the optimal split point at every node.

---

## 4. Input Features vs Target Outputs (Zero Data Leakage Guarantee)

In academic machine learning, **data leakage** is the most common mistake (e.g., using the target variable or future information as an input). CellGuard strictly guarantees **zero data leakage**:

### Input Features ($\mathbf{X}$ - Exactly 19 Parameters):
These are parameters an onboard Battery Management System (BMS) can measure directly:
1. **Voltage Crossing Times (5 features):** Time taken to discharge between voltage thresholds: `T_40_38` (4.0V→3.8V), `T_38_36`, `T_36_34`, `T_40_36`, `T_38_34`.
2. **Partial Discharge Capacity (1 feature):** `Q_40_36` (Coulomb count between 4.0V and 3.6V).
3. **Transient Voltage Samples (4 features):** Terminal voltage recorded at specific timestamps: `V_at_60s`, `V_at_300s`, `V_at_600s`, `V_at_1200s`.
4. **Voltage Curve Dynamics (3 features):** Initial voltage `V_start`, mean early voltage `V_mean_first600s`, and mid-discharge slope `V_slope_mid`.
5. **Temperature Dynamics (4 features):** Starting temperature `T_start`, peak temperature `T_max`, total temperature rise `T_rise`, and average temperature `T_mean`.
6. **Load Step Resistance Proxy (1 feature):** `R_dc_proxy` ($\Delta V / \Delta I$ during the initial load step).
7. **Aging Index (1 feature):** `disc_idx` (Discharge cycle number).

### Target Outputs ($y$):
- **Ohmic Resistance ($R_e$):** Predicted in Ohms ($\Omega$) and displayed in milliohms ($\text{m}\Omega$).
- **Charge-Transfer Resistance ($R_{ct}$):** Predicted in Ohms ($\Omega$) and displayed in milliohms ($\text{m}\Omega$).

> **Important Viva Point:**  
> **Neither $R_e$ nor $R_{ct}$ is EVER present in the input feature matrix $\mathbf{X}$.** The models learn the mapping solely from voltage, temperature, and cycle time.

---

## 5. Dataset & Numerical Performance Breakdown

### The NASA Battery Aging Dataset:
- **Source:** NASA Ames Prognostics Center of Excellence (PCoE).
- **Core Benchmark Cells:** `B0005`, `B0006`, `B0007`, `B0018`.
  - Type: 18650 cylindrical Lithium-ion cells (LiCoO₂ chemistry).
  - Rated Nominal Capacity: $2.0\text{ Ah}$.
  - Test Conditions: Constant Current (CC) discharge at $2.0\text{ A}$ (1C rate) down to $2.7\text{ V}$, $2.5\text{ V}$, $2.2\text{ V}$, and $2.5\text{ V}$ respectively, maintained in a thermal chamber at $24^\circ\text{C}$.
- **Expanded Scope:** 34 total NASA battery cells across the archive.

### Data Cleaning & Outlier Removal:
In NASA's raw dataset, internal resistance was extracted using nonlinear circle-fitting in MATLAB. In 7 cycles out of 2,058, the MATLAB optimization routine failed to converge, recording impossible non-physical values (e.g. $-9.6 \times 10^{14}\,\Omega$).
- **CellGuard filters data to physical boundaries:**
  - $0.010\,\Omega \le R_e \le 0.300\,\Omega$ ($10\text{ m}\Omega$ to $300\text{ m}\Omega$)
  - $0.020\,\Omega \le R_{ct} \le 0.450\,\Omega$ ($20\text{ m}\Omega$ to $450\text{ m}\Omega$)
- This leaves **530 verified paired cycles** across the 4 core benchmark cells and **2,051 verified cycles** across all cells.

### Performance Scores:
- **Ohmic Resistance ($R_e$):**
  - Coefficient of Determination ($R^2$): **$97.01\%$**
  - Mean Absolute Error (MAE): **$1.13\text{ m}\Omega$** ($0.00113\,\Omega$)
  - Root Mean Squared Error (RMSE): **$1.66\text{ m}\Omega$**
  - Mean Absolute Percentage Error (MAPE): **$1.97\%$**
- **Charge-Transfer Resistance ($R_{ct}$):**
  - Coefficient of Determination ($R^2$): **$96.83\%$**
  - Mean Absolute Error (MAE): **$1.28\text{ m}\Omega$** ($0.00128\,\Omega$)
  - Root Mean Squared Error (RMSE): **$1.90\text{ m}\Omega$**
  - Mean Absolute Percentage Error (MAPE): **$1.55\%$**

*Both models significantly exceed the university target of $R^2 > 80\%$.*

---

## 6. Training & Testing Split: Methodology & Deep Explanation

### Why a Standard Random 80/20 Train-Test Split is FLAWED for Batteries:
In typical machine learning projects (like image classification or housing prices), data points are independent. You can randomly shuffle the dataset into 80% train and 20% test.

**In battery aging, random splitting is a critical methodological error known as TEMPORAL DATA LEAKAGE.**
- Cycle 41 and Cycle 42 of cell `B0005` have virtually identical chemistry, internal resistance, and thermal history.
- If Cycle 41 is placed in the training set and Cycle 42 in the test set, the model does not actually learn physics—it simply memorizes adjacent data points. This yields artificially high, fake scores (e.g. 99.9%) that fail completely when applied to a new battery in the real world.

### The Two Validation Strategies Used in CellGuard:

#### Strategy A: 5-Fold Stratified Group Cross-Validation (Operational Accuracy)
- The dataset is split into 5 distinct folds where whole groups of cycles are held out.
- Each cycle is predicted strictly when it is in the holdout fold.
- **Why this was chosen:** Evaluates how accurately the model tracks resistance evolution over a cell's lifecycle when trained on representative operating conditions.
- **Result:** $R^2 = 97.01\%$ for $R_e$, $96.83\%$ for $R_{ct}$.

#### Strategy B: Leave-One-Cell-Out (LOCO) 4-Fold Cross-Validation (Cross-Cell Generalization)
- In each fold, the model is trained exclusively on **three cells** (e.g., `B0005`, `B0006`, `B0007`) and tested on the **fourth completely unseen cell** (`B0018`).
- **Why this was chosen:** Proves whether the algorithm can generalize to a brand-new battery that came off the assembly line.
- **The Physical Baseline Offset Discovery:**
  - Real commercial 18650 cells have manufacturing tolerances: their initial fresh resistance varies by $\pm 10\text{ m}\Omega$ to $15\text{ m}\Omega$ (e.g., `B0005` starts at $55\text{ m}\Omega$, while `B0018` starts at $38\text{ m}\Omega$).
  - When evaluating zero-shot on an unseen cell, this constant $15\text{ m}\Omega$ vertical offset reduces raw uncalibrated $R^2$.
  - However, when the BMS takes **one single calibration measurement at Cycle 1**, the model predicts the degradation curve ($\Delta R$) with **$R^2 > 92\%$ across all unseen cells!**

---

## 7. Electrical Engineering (EEE) Concepts Explained Simply

### 1. The 1st-Order Thevenin Equivalent Circuit Model (ECM):
In battery modeling, the complex electrochemistry is represented by an electrical circuit:
- **$OCV$ (Open Circuit Voltage):** The ideal equilibrium voltage of the cell when no current is flowing.
- **$R_e \approx R_0$ (Ohmic Resistance):** Resistor in series representing the electrolyte and current collector bulk resistance. Responds **instantaneously** ($t = 0^+$).
- **$R_{ct} \approx R_1$ (Charge-Transfer Resistance):** Resistor in parallel with capacitor $C_1$ representing the electrode-electrolyte interface kinetics.
- **$C_1$ (Double-Layer Capacitance):** Models the capacitive charge buildup at the electrode surface.
- **Terminal Voltage Equation:**
  $$V_{\text{terminal}}(t) = OCV - I(t) \cdot R_e - V_{RC}(t)$$
  $$\frac{d V_{RC}(t)}{dt} = -\frac{V_{RC}(t)}{R_{ct} \cdot C_1} + \frac{I(t)}{C_1}$$

### 2. Terminal Voltage Drop ($\Delta V$):
When a vehicle steps on the accelerator (current demand $I$):
$$\Delta V_{\text{total}} = I \cdot (R_e + R_{ct})$$
For example, at a discharge current of $I = 10\text{ A}$:
- A fresh cell with $R_{\text{int}} = 100\text{ m}\Omega$ drops by $\Delta V = 10 \cdot 0.100 = 1.0\text{ V}$.
- An aged cell with $R_{\text{int}} = 220\text{ m}\Omega$ drops by $\Delta V = 10 \cdot 0.220 = 2.2\text{ V}$.
This large voltage drop causes the BMS to trigger premature low-voltage cutoff!

### 3. Joule Heating & Power Losses ($P_{\text{loss}}$):
Internal power lost as pure heat is given by Joule's Law:
$$P_{\text{loss}} = I^2 \cdot (R_e + R_{ct})$$
At $I = 15\text{ A}$:
- Fresh cell ($0.100\,\Omega$): $P_{\text{loss}} = (15)^2 \cdot 0.100 = 22.5\text{ W}$ of heat.
- Aged cell ($0.220\,\Omega$): $P_{\text{loss}} = (15)^2 \cdot 0.220 = 49.5\text{ W}$ of heat ($> 2.2\times$ more heat!).
This explains why aging batteries get hot much faster and can trigger thermal alarms.

### 4. Why $R_{ct}$ Grows Much Faster Than $R_e$:
- **$R_e$ growth:** Bulk electrolyte volume and current collectors degrade slowly ($\sim 20\%\text{ to }40\%$ growth over battery life).
- **$R_{ct}$ growth:** Parasitic reactions continuously deposit passivating compounds ($Li_2CO_3, LiF$) onto the anode, creating a thick Solid Electrolyte Interphase (SEI) barrier. As a result, $R_{ct}$ typically grows by **$100\%\text{ to }250\%$**.
- **Conclusion:** $R_{ct}$ is the primary indicator of interfacial electrochemical degradation.

---

## 8. Viva Voce Examiner Questions & Direct Answers

**Q1: What is the main objective of your project?**  
*Answer:* The main objective is to estimate the internal Ohmic Resistance ($R_e$) and Charge-Transfer Resistance ($R_{ct}$) of Lithium-ion batteries using machine learning applied to observable partial discharge curves, eliminating the need for expensive, offline Electrochemical Impedance Spectroscopy (EIS) hardware.

**Q2: Did you use $R_e$ or $R_{ct}$ as input features to predict each other or capacity?**  
*Answer:* Absolutely not. That would be data leakage. The inputs are strictly 19 observable electrical and thermal parameters extracted from the first portion of the discharge curve (voltage crossing times, load-step voltage drop proxy, temperatures, and cycle number). $R_e$ and $R_{ct}$ are strictly target outputs.

**Q3: Why did you choose Extra Trees as your primary algorithm?**  
*Answer:* Extra Trees achieved the highest performance with an $R^2$ of 97.01% for $R_e$ and 96.83% for $R_{ct}$, with an average error of only $\sim 1.1\text{ to }1.3\text{ m}\Omega$. Tree ensembles partition the feature space orthogonally, capturing the non-linear relationship between terminal voltage overpotentials and internal resistance that linear models cannot.

**Q4: Why did Ridge Regression achieve lower performance than tree ensembles?**  
*Answer:* Ridge Regression is a linear model ($y = \mathbf{w}^T \mathbf{x} + b$). Electrochemical degradation is non-linear; as resistance grows, the voltage drop and temperature rise accelerate non-linearly. A linear hyperplane cannot capture threshold crossings or interaction terms between temperature and load step drop.

**Q5: Why did you not use a standard random 80/20 train/test split?**  
*Answer:* Random train/test splitting causes temporal data leakage in battery cycling. Adjacent cycles from the same physical battery share almost identical chemical degradation and temperature history. Random splitting places Cycle 41 in training and Cycle 42 in testing, leading to memorization rather than generalization. Instead, we used 5-Fold Stratified Group Cross-Validation and Leave-One-Cell-Out (LOCO) cross-validation.

**Q6: What is the physical meaning of $R_e$ and $R_{ct}$ in an electric vehicle battery?**  
*Answer:* $R_e$ is the Ohmic bulk resistance of the electrolyte, separator, and metallic contacts. $R_{ct}$ is the charge-transfer resistance across the electrode-electrolyte interface, reflecting SEI layer growth. Together, they dictate the terminal voltage drop ($\Delta V = I(R_e + R_{ct})$) and internal heat dissipation ($P_{\text{loss}} = I^2(R_e + R_{ct})$).

**Q7: How does your project relate to the Thevenin Equivalent Circuit?**  
*Answer:* In a 1st-order Thevenin model ($OCV - R_0 - R_1 \parallel C_1$), our predicted $R_e$ corresponds directly to the instantaneous series resistance $R_0$, and $R_{ct}$ corresponds to the polarization resistance $R_1$.

---
*Created for CellGuard · 4th-Year Electrical & Electronic Engineering (EEE) Final Year Project.*

