# Pressure-Dependent Mechanisms of CO₂ Adsorption in MOFs

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20SHAP-orange)](https://scikit-learn.org/)

## Executive Summary (Engineering Takeaways)
* **High-Fidelity Surrogate Modeling:** Built a Gradient Boosting (GBM) surrogate model trained on Grand Canonical Monte Carlo (GCMC) simulation data, predicting CO₂ uptake across thousands of MOF candidate structures with R^2 = 0.982 and RMSE = 2.38 mol/kg.
* **Pressure-Dependent Mechanics Uncovered:** Used SHAP diagnostics to prove that global pore descriptors (void fraction, pore limiting diameter) govern high-pressure capacity (98.1 predictive variance), whereas local atomic interaction histograms jump to 27 importance at low pressure (P < 1 bar).
* **Physically-Informed Material Design:** Decoded abstract atomistic histogram features back into physical parameters (sigma, varepsilon, at specific radial distances), providing actionable design criteria for high-efficiency carbon capture materials.

---

## Overview & Research Intent

Screening novel Metal-Organic Frameworks (MOFs) for carbon capture requires evaluating gas-solid adsorption equilibria across wide pressure regimes. While Grand Canonical Monte Carlo (GCMC) simulations provide high-fidelity thermodynamic data, running full structural parameter sweeps is computationally intensive.

This project implements an **interpretable machine learning framework** to accelerate material evaluation while uncovering the exact physical and chemical features governing pressure-dependent CO₂ capture.

---

## Feature Representation

To evaluate both macro-scale geometry and micro-scale atomic environments, MOFs were parameterized into two feature groups:

### 1. Global / Textural Descriptors
* **Operating Environment:** Gas Fugacity
* **Pore Geometry:** Largest Pore Diameter (LPD), Pore Limiting Diameter (PLD), Surface Area, Void Fraction, Material Density, Pore Size Distribution Descriptor (PSSD)

### 2. Local Atomistic Interaction Histograms
Distance-dependent 2D interaction histograms capturing local spatial and thermodynamic force-field environments between CO₂ and host MOF atoms:
* **$\sigma$ (Sigma):** Lennard-Jones steric/size parameter
* **$\varepsilon$ (Epsilon):** Lennard-Jones well-depth/interaction-strength parameter
* **$q$ (Charge):** Local atomic partial charges

---

## Machine Learning & Performance

Gradient Boosting (GBM) and Random Forest (RF) models were trained on GCMC simulation data at $298\text{ K}$.

| Model Metric | Performance | Physical Meaning |
| :--- | :--- | :--- |
| **Test $R^2$** | **0.982** | Explains $98.2\%$ of variance in test-set CO₂ uptake values |
| **Test RMSE** | **2.38 mol/kg** | Average deviation scale relative to high-fidelity GCMC baselines |

---

## Visual Insights & SHAP Diagnostics

### 1. Model Parity Plot
Predicted CO₂ adsorption vs. true GCMC simulation benchmarks (R^2 = 0.982).

<img width="571" height="575" alt="image" src="https://github.com/user-attachments/assets/2da56092-01b9-4884-863d-e2f13f22ec3c" />


### 2. Pressure-Dependent Adsorption Regimes
SHAP analysis revealed a fundamental shift in controlling physics based on operating pressure:

* **High Pressure Regime:** Gas fugacity and global pore descriptors (void fraction, pore volume) dominate predictions, reflecting bulk pore filling dynamics.
* **Low Pressure Regime ($P < 1\text{ bar}$):** Atomistic interaction histograms become primary drivers, as sparse gas molecules interact directly with specific local binding sites.

<img width="1132" height="449" alt="image" src="https://github.com/user-attachments/assets/4f3b59cf-c489-43e5-b97a-6bbe93259d60" />


---

## Decoding Atomistic Histogram Features

To connect ML predictions to physical MOF synthesis criteria, high-importance histogram bins were mapped back to physical radial coordinates and interaction scales:

| Feature | Radial Distance ($r$) | Physical Parameter | Physical Interpretation |
| :--- | :---: | :---: | :--- |
| `sigma_bin_1283` | 8.5 Å | sigma approx 2.85 | Medium-range steric framework packing |
| `epsilon_bin_202` | 1.0 Å | varepsilon approx 247.45 K | Short-range strong interaction site |
| `epsilon_bin_203` | 1.0 Å | varepsilon approx 251.99\text{ K}$ | Primary electrostatic/dispersion binding pocket |
| `sigma_bin_1121` | 7.5 Å | $\sigma \approx 2.37$ | Interstitial pore throat boundary |
| `q_bin_823` | 5.5 Å | $q \approx -2.26\text{ e}$ | Polarized framework site driving low-P adsorption |

---

## Repository Structure

```text
├── data/
│   └── mof_co2_adsorption_dataset.csv  <-- GCMC simulation data
├── notebooks/
│   ├── 01_model_training.ipynb         <-- GBM & RF training pipelines
│   ├── 02_shap_analysis.ipynb          <-- Pressure-stratified SHAP evaluation
│   └── 03_histogram_decoding.ipynb     <-- Mapping bins to physical parameters (r, σ, ε, q)
├── figures/
│   ├── prediction_vs_true.png          <-- Parity plot
│   └── shap_pressure_dependence.png   <-- SHAP regime comparison
├── requirements.txt                    <-- Python dependencies
└── README.md
