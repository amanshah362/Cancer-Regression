# Cancer Mortality Rate Prediction (Regression)

Predicting county-level cancer death rates in the US from socio-economic, demographic, healthcare-coverage, and cancer-incidence data. The model is an **XGBoost regressor tuned with Optuna**, wrapped in a leakage-safe scikit-learn pipeline.

**Final result: R² = 0.678 on the held-out test set.**

---

## Problem Statement

Cancer mortality varies a lot between US counties. The goal is to predict `target_deathrate` (cancer deaths per 100,000 people) for each county and to understand which factors are most associated with it.

## Dataset

- **Source:** [Cancer Regression dataset on Kaggle](https://www.kaggle.com/datasets/varunraskar/cancer-regression) (`cancer_reg.csv`)
- **Size:** 3,047 counties, 33 columns
- **Target:** `target_deathrate`

| Group | Example features |
|---|---|
| Cancer statistics | `avganncount`, `avgdeathsperyear`, `incidencerate` |
| Economic | `medincome`, `povertypercent`, `binnedinc` |
| Demographics | `popest2015`, `medianage`, `percentmarried`, `birthrate` |
| Education | `pcths25_over`, `pctbachdeg25_over`, `pctnohs18_24` |
| Employment | `pctemployed16_over`, `pctunemployed16_over` |
| Health coverage | `pctprivatecoverage`, `pctpubliccoverage`, `pctempprivcoverage` |
| Race | `pctwhite`, `pctblack`, `pctasian`, `pctotherrace` |

## Project Workflow

### 1. Exploratory Data Analysis
- Checked shape, dtypes, duplicates, and missing values
- Validated that percentage columns stay within 0 to 100 and that no numeric column has negative values
- Found impossible `medianage` values (above 100) and removed those rows
- Built a 9-panel dashboard: top counties by death rate, target distribution, income / poverty / age / incidence / insurance coverage vs. death rate, and death rate by income group

### 2. Data Cleaning
- Dropped `pctsomecol18_24` (about 75% missing)
- Dropped `binnedinc` (redundant with `medincome`)
- Dropped `geography` (nearly unique per row)
- Removed rows with invalid `medianage`

### 3. Preprocessing Pipeline
All steps live inside a scikit-learn `Pipeline`, so they are fit only on training folds and there is no data leakage during cross-validation.

```
IterativeImputer -> RobustScaler -> PowerTransformer (Yeo-Johnson)
```

- **IterativeImputer** fills the remaining missing values (`pctemployed16_over`, `pctprivatecoveragealone`) using the other features
- **RobustScaler** limits the influence of outliers
- **PowerTransformer** reduces skew in features such as population and cancer counts

### 4. Modeling and Hyperparameter Tuning
- Train/test split: **75% / 25%** (`random_state=42`)
- Model: **XGBRegressor** (`tree_method='hist'`)
- Tuning: **Optuna** (TPE sampler), **50 trials**, **5-fold shuffled KFold** CV, scoring on R²
- Tuned parameters: `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`
- Inspected the search with Optuna's optimization history, parameter importance, parallel coordinate, and slice plots

### 5. Evaluation
The best parameters were used to refit the pipeline on the full training set, and the model was evaluated **once** on the untouched test set.

| Metric | Score |
|---|---|
| R² (test set) | **0.678** |

## Tech Stack

- **Language:** Python
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly (via Optuna)
- **Modeling:** scikit-learn, XGBoost
- **Tuning:** Optuna
- **Environment:** Kaggle Notebook

## Repository Structure

```
.
├── cancer-regression.ipynb   # Full EDA, preprocessing, tuning, evaluation
└── README.md
```

## How to Run

```bash
git clone <your-repo-url>
cd <your-repo-name>

pip install numpy pandas matplotlib seaborn scikit-learn xgboost optuna plotly kagglehub jupyter

jupyter notebook cancer-regression.ipynb
```

Download `cancer_reg.csv` from the Kaggle link above and update the path in the `pd.read_csv(...)` cell if you are not running on Kaggle.

## Key Takeaways

- County-level mortality data is noisy, so an R² in the high 0.6s is a solid result for this dataset
- Putting imputation and scaling inside the pipeline keeps cross-validation honest
- Optuna with a TPE sampler found good XGBoost settings in only 50 trials

## Future Improvements

- Extract `state` from `geography` and encode it to capture regional patterns
- Add per-capita features (for example cases per population)
- Compare against LightGBM, CatBoost, and Ridge, then try a stacking ensemble
- Test whether removing population-driven columns such as `avgdeathsperyear` changes performance
- Add SHAP analysis for model interpretability

## Author

**Your Name**
[LinkedIn](https://www.linkedin.com/in/your-profile) | [GitHub](https://github.com/your-username)
