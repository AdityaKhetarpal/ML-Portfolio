Welcome to my machine learning showcase. Below are three complete supervised learning projects covering **regression**, **binary classification**, and **multi-class classification**. Each project ships a runnable script, the original exploratory notebook, diagnostic plots, and a written account of what the model can and cannot be trusted to do.

The theme running through all three is model scepticism. Every project reports where its own results fall short — because a portfolio where every model scores 99% is a portfolio nobody stress-tested.

**Stack:** Python 3.9+ · scikit-learn · pandas · NumPy · Matplotlib

---

## Project 1: HomeVista Automated Residential Valuation Engine

- **Problem Type:** Regression (continuous target)
- **Algorithms Used:** Linear Regression, with a standardised-feature control arm
- **Tools Used:** Python (pandas, scikit-learn, NumPy, Matplotlib)
- **Dataset:** 2,919 residential property records, 12 predictors

### Core Insights Summary:

- **Explanatory Ceiling Identified:** The fitted model explains 37.4% of sale price variance (R² = 0.374, MAE $30,830, MAPE 18.7%). Rather than presenting this as a success, the project isolates the cause — the twelve available predictors carry no measure of above-grade living area, bathroom count, or neighbourhood, the three variables that dominate residential pricing. The constraint is the feature set, not the algorithm.
- **Scaling Ruled Out as a Remedy:** A second model trained on standardised features returned an R² identical to fourteen decimal places (difference: 2.2e-14). This is the expected result, since ordinary least squares has a closed-form solution invariant to linear input transformation. It is retained in the pipeline as a deliberate negative control that eliminates one hypothesis before more expensive ones are tested.
- **Structural Missingness Handled Transparently:** The source file is a concatenated train/test export in which 1,459 of 2,919 target values are blank by construction. The pipeline imputes rather than silently discards, and documents how filling half a target column with a constant compresses variance and drags R² downward.

---

## Project 2: TalentCore Employee Attrition Risk Classifier

- **Problem Type:** Binary classification
- **Algorithms Used:** Logistic Regression — baseline vs L1 (Lasso) vs L2 (Ridge)
- **Tools Used:** Python (pandas, scikit-learn, NumPy, Matplotlib)
- **Dataset:** 900 employee records, 15 features including 2 engineered terms

### Core Insights Summary:

- **Regularization Delivered a Measured, Not Dramatic, Gain:** L1 regularization lifted accuracy from 85.9% to 87.0% and F1 on the leaver class from 0.84 to 0.86, while L2 returned results identical to the unpenalised baseline. On a 270-row test set that improvement is three employees classified differently — reported as the modest gain it is rather than inflated into a headline.
- **Feature Selection as the Real L1 Payoff:** The value of the Lasso arm was not the accuracy delta but the shrinkage of weak coefficients to exactly zero, which prunes the engineered `Annual_Bonus_Squared` and interaction terms where they earn nothing. The pipeline exports a side-by-side coefficient comparison so the selection decision is auditable rather than asserted.
- **Recall Prioritised Over Headline Accuracy:** Model ranking is driven by recall on the leaver class, on the reasoning that a missed resignation costs a full replacement-hire cycle while a false alarm costs a retention conversation. At 82–83% recall roughly one leaver in six still escapes the flag, and the README states this in the deployment section rather than burying it.

---

## Project 3: Iris Species Classification Benchmark

- **Problem Type:** Multi-class classification (3 species)
- **Algorithms Used:** K-Nearest Neighbours (k=5), Logistic Regression, Gaussian Naive Bayes
- **Tools Used:** Python (pandas, scikit-learn, NumPy, Matplotlib)
- **Dataset:** 150 specimens, 4 morphological measurements, perfectly balanced

### Core Insights Summary:

- **Target Leakage Detected and Root-Caused:** The first-pass notebook returned 100% accuracy for Logistic Regression. Investigation traced this to the `Id` column being retained as a predictor in a file sorted by species — rows 1–50 setosa, 51–100 versicolor, 101–150 virginica — making the row counter a perfect proxy for the label. Removing `Id` moved honest held-out accuracy to 94.7%. The leaked result is preserved in the repository beside the corrected one, because catching the error is the finding.
- **Evaluation Protocol Rebuilt:** The brief specified training on 50% of the data and scoring on 100% of it, which scores every model partly on rows it had memorised. The pipeline reports that figure as requested, then adds a clean held-out score and a 5-fold cross-validation figure, and quantifies the optimism gap between them.
- **Algorithm Choice Reframed as an Engineering Decision:** After correction all three classifiers converged to 94.7% held-out accuracy with a 0.0 percentage point spread, every error falling in the single region where versicolor and virginica petal dimensions overlap. With no statistical separation between candidates, the recommendation rests on inference cost, deployment footprint and interpretability instead of a metric that cannot distinguish them.

---

## Project 4: SmartShop Online Purchase Intent Classifier

- **Problem Type:** Binary classification
- **Algorithm Used:** Decision Tree Classifier with scikit-learn Pipeline
- **Tools Used:** Python (pandas, scikit-learn, NumPy)
- **Dataset:** 12,330 online shopping sessions, 17 behavioural and technical features

### Core Insights Summary:

- **Class Imbalance Handled at the Model Level:** Only 15.5% of sessions end in a purchase (382 buyers vs 2,084 non-buyers). Rather than resampling the data, the pipeline uses `class_weight="balanced"` inside the Decision Tree, which adjusts the split criterion to weight minority-class errors more heavily. This lifted recall on the buyer class to 83% — the model catches 8 in 10 actual purchasers.
- **Recall Over Precision as the Business-Correct Choice:** The tuned model achieves 83% recall at 50% precision on the buyer class. In a retargeting context that trade-off is correct — a false alarm costs an ad impression, a miss costs a lost sale. The project frames model selection around this asymmetry rather than reporting headline accuracy (85%), which a "nobody buys" baseline would nearly match.
- **GridSearchCV Showed a Shallower Tree Generalises Better:** The baseline was trained at `max_depth=6`. A 5-fold cross-validated grid search across depths [4, 6, 8] and minimum leaf sizes [20, 30, 50] found the best cross-validated F1 at `max_depth=4, min_samples_leaf=50` — a counter-intuitive result that documents the overfitting risk of deep trees on a small minority class and justifies the conservative final configuration.

---

## Results at a Glance

| Project | Task | Best Model | Headline Metric |
|---|---|---|---|
| HomeVista Valuation | Regression | Linear Regression | R² 0.374 · MAE $30,830 |
| TalentCore Attrition | Binary classification | L1 / Lasso | Accuracy 87.0% · Recall 0.83 |
| Iris Benchmark | Multi-class classification | Logistic Regression | Held-out accuracy 94.7% |
| SmartShop Intent | Binary classification | Decision Tree (tuned) | Recall 0.83 · F1 0.63 |

---

## Repository Structure

```
ML-Portfolio/
├── 01-house-price-regression/
│   ├── README.md                     # full write-up
│   ├── house_price_predictor.py      # runnable pipeline
│   ├── notebooks/                    # original exploratory notebook
│   ├── data/                         # dataset (see data/README.md)
│   └── outputs/                      # metrics, coefficients, plots
├── 02-employee-turnover-classification/
│   └── ... same layout
├── 03-iris-species-classification/
│   └── ... same layout (dataset included — runs out of the box)
├── 04-smartshop-purchase-intent/
│   └── ... same layout
├── requirements.txt
└── README.md
```

---

## Contact

**Aditya Khetarpal**

[LinkedIn](https://linkedin.com/in/adityakhetarpal) · [X](https://x.com/0xAditya_k) · [Data Analytics Portfolio](https://github.com/AdityaKhetarpal/Data-Analytics-Portfolio) · [Substack](https://degenfinds.substack.com)
