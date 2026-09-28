# Supervised-ML-Portfolio

Welcome to my machine learning showcase. Below are five complete supervised learning projects covering **regression**, **binary classification**, and **multi-class classification**. Each project ships a runnable script, the original exploratory notebook, diagnostic plots, and a written account of what the model can and cannot be trusted to do.

**Stack:** Python 3.9+ · scikit-learn · LightGBM · pandas · NumPy · Matplotlib · Streamlit

---

## Project 1: Hospital 30-Day Readmission Risk — Diabetic Patients

- **Problem Type:** Binary classification
- **Algorithms Used:** Logistic Regression (balanced) vs LightGBM (300 trees, validation-tuned)
- **Tools Used:** Python (pandas, scikit-learn, LightGBM, SHAP, Streamlit)
- **Dataset:** 101,766 clinical encounters · 71,518 patients · 130 US hospitals (1999–2008)
- **Live Demo:** [Streamlit App](Add URL here)
- **Directory:** [`01-hospital-readmission/`](./01-hospital-readmission)

### Core Insights Summary:

- **Patient-Level Split Prevented Leakage:** A standard row-level train/test split would have placed 6,697 patients on both sides — the model would have trained on one encounter and evaluated on another from the same person. The split was performed on unique patient IDs instead, giving zero overlap and an honest test set of 13,998 patients.
- **Operating Point Chosen on Validation, Applied Once:** The classification threshold (0.157, top 20% of scores) was selected on a held-out validation set carved from training patients, before the test set was touched. At this cutoff the model flags 18.8% of patients, catches 38.1% of actual readmissions, and produces roughly 3.6 false alarms per correct flag — all stated plainly rather than buried.
- **Prior Inpatient Stays Dominate by a Wide Margin:** SHAP analysis on 3,000 test patients shows `number_inpatient` (mean |SHAP| 0.29) is 2.7× more influential than the next feature. Readmission rate climbs from 8.6% for patients with no prior inpatient stays to 37.1% for those with five or more — a pattern consistent across the full dataset and confirmed by the SHAP beeswarm.

---

## Project 2: HomeVista Automated Residential Valuation Engine

- **Problem Type:** Regression (continuous target)
- **Algorithms Used:** Linear Regression, with a standardised-feature control arm
- **Tools Used:** Python (pandas, scikit-learn, NumPy, Matplotlib)
- **Dataset:** 2,919 residential property records, 12 predictors
- **Directory:** [`02-house-price-regression/`](./02-house-price-regression)

### Core Insights Summary:

- **Explanatory Ceiling Identified:** The fitted model explains 37.4% of sale price variance (R² = 0.374, MAE $30,830, MAPE 18.7%). Rather than presenting this as a success, the project isolates the cause — the twelve available predictors carry no measure of above-grade living area, bathroom count, or neighbourhood, the three variables that dominate residential pricing. The constraint is the feature set, not the algorithm.
- **Scaling Ruled Out as a Remedy:** A second model trained on standardised features returned an R² identical to fourteen decimal places (difference: 2.2e-14). This is the expected result, since ordinary least squares has a closed-form solution invariant to linear input transformation. It is retained in the pipeline as a deliberate negative control that eliminates one hypothesis before more expensive ones are tested.
- **Structural Missingness Handled Transparently:** The source file is a concatenated train/test export in which 1,459 of 2,919 target values are blank by construction. The pipeline imputes rather than silently discards, and documents how filling half a target column with a constant compresses variance and drags R² downward.

---

## Project 3: TalentCore Employee Attrition Risk Classifier

- **Problem Type:** Binary classification
- **Algorithms Used:** Logistic Regression — baseline vs L1 (Lasso) vs L2 (Ridge)
- **Tools Used:** Python (pandas, scikit-learn, NumPy, Matplotlib)
- **Dataset:** 900 employee records, 15 features including 2 engineered terms
- **Directory:** [`03-employee-turnover-classification/`](./03-employee-turnover-classification)

### Core Insights Summary:

- **Regularization Delivered a Measured, Not Dramatic, Gain:** L1 regularization lifted accuracy from 85.9% to 87.0% and F1 on the leaver class from 0.84 to 0.86, while L2 returned results identical to the unpenalised baseline. On a 270-row test set that improvement is three employees classified differently — reported as the modest gain it is rather than inflated into a headline.
- **Feature Selection as the Real L1 Payoff:** The value of the Lasso arm was not the accuracy delta but the shrinkage of weak coefficients to exactly zero, which prunes the engineered `Annual_Bonus_Squared` and interaction terms where they earn nothing. The pipeline exports a side-by-side coefficient comparison so the selection decision is auditable rather than asserted.
- **Recall Prioritised Over Headline Accuracy:** Model ranking is driven by recall on the leaver class, on the reasoning that a missed resignation costs a full replacement-hire cycle while a false alarm costs a retention conversation. At 82–83% recall roughly one leaver in six still escapes the flag, and the README states this in the deployment section rather than burying it.

---

## Project 4: Iris Species Classification Benchmark

- **Problem Type:** Multi-class classification (3 species)
- **Algorithms Used:** K-Nearest Neighbours (k=5), Logistic Regression, Gaussian Naive Bayes
- **Tools Used:** Python (pandas, scikit-learn, NumPy, Matplotlib)
- **Dataset:** 150 specimens, 4 morphological measurements, perfectly balanced
- **Directory:** [`04-iris-species-classification/`](./04-iris-species-classification)

### Core Insights Summary:

- **Target Leakage Detected and Root-Caused:** The first-pass notebook returned 100% accuracy for Logistic Regression. Investigation traced this to the `Id` column being retained as a predictor in a file sorted by species — rows 1–50 setosa, 51–100 versicolor, 101–150 virginica — making the row counter a perfect proxy for the label. Removing `Id` moved honest held-out accuracy to 94.7%. The leaked result is preserved in the repository beside the corrected one, because catching the error is the finding.
- **Evaluation Protocol Rebuilt:** The brief specified training on 50% of the data and scoring on 100% of it, which scores every model partly on rows it had memorised. The pipeline reports that figure as requested, then adds a clean held-out score and a 5-fold cross-validation figure, and quantifies the optimism gap between them.
- **Algorithm Choice Reframed as an Engineering Decision:** After correction all three classifiers converged to 94.7% held-out accuracy with a 0.0 percentage point spread, every error falling in the single
