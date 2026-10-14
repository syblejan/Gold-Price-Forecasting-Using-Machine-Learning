# Gold Price Prediction — Monthly Time-Series Forecasting

**Python | Pandas | NumPy | Scikit-learn | Ridge Regression | Lag Features | Time-Series Evaluation**

An educational machine learning project for **monthly gold-price forecasting** using 12 previous months of recorded prices. It preserves the original six-model forecasting notebook and adds a reproducible Ridge-versus-naive-baseline evaluation with a chronological train/test split.

> **Disclaimer:** This project is for research and education only, **not trading or financial advice**. Historical prediction accuracy does not guarantee future market performance.

## 1. Project Highlights

| Component | Details |
|---|---|
| Problem | One-month-ahead regression using historical monthly gold prices |
| Dataset | 2,312 observations, January 1833–August 2025 |
| Variables | `Date`, `Price` |
| Input features | Previous 12 monthly prices |
| Split | Chronological 80% training / 20% testing after lag creation |
| Corrected model | Scaled Ridge Regression (`alpha=1`) |
| Baseline | Previous month's price |
| Evaluation | MAE, RMSE, R² on an untouched chronological holdout |
| Multi-step output | Illustrative recursive forecast for seven **months** |

**Key learning:** A strong R² on a trending price series is not by itself proof of useful forecasting. Compare against a simple previous-month baseline, and check MAE/RMSE.

## 2. Dataset and Preparation

The repository contains `data/monthly.csv` with 2,312 monthly `Date,Price` records. The observed dates span **1833-01 through 2025-08** and are monthly, not daily.

Preparation steps:

- Parse dates as months (`YYYY-MM`) and sort chronologically.
- Validate missing/duplicate dates and missing price observations.
- Create 12 lagged features (`lag_1` through `lag_12`), with each input using **only prices known before the target month**.
- Drop the first 12 rows, leaving 2,300 supervised observations.
- Split chronologically: **1,840 train** and **460 test** observations.

The source, currency/units and licensing of the historical CSV have not been independently verified; do not assume today's market quote or purchasing value.

## 3. Original Notebook and Historical Results

The original notebook is preserved at `notebooks/original_gold_price_predication.ipynb`. It compared:

- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- Support Vector Regressor
- Decision Tree Regressor

It used 12-month lag features, an 80/20 chronological split, `TimeSeriesSplit(n_splits=5)`, GridSearchCV and metrics MAE/RMSE/R².

**Original notebook's recorded test RMSE** (historical outputs, not rerun here):

| Model | Recorded test RMSE |
|---|---:|
| Ridge | 39.7759 |
| Lasso | 39.8282 |
| ElasticNet | 40.9491 |
| Decision Tree | 703.6071 |
| Random Forest | 726.0433 |
| SVR | 1021.8773 |

The notebook also printed a projected next price of **3323.97** from its older forecasting code. That projection must not be interpreted as a real-time or verified future price.

### Original limitations

Some charts were constructed using loop variables outside their original loop. More importantly, the original multi-step function labeled its output **7-day forecast** despite the underlying dataset being **monthly**. Its next-period lag alignment also required review.

The updated code fixes the calendar interpretation, applies an explicit next-month feature row, and reports a simple baseline. The original notebook is retained without editing.

## 4. Corrected Evaluation Design

```text
Monthly Date, Price observations
               |
               v
        Sort & Validate
               |
               v
      Lag_1 ... Lag_12
               |
               v
 Chronological 80/20 split
        /              \
       v                v
 Training months     Test months
       |                |
       v                |
 StandardScaler +       |
 Ridge Regression       |
       |                |
       +----------------+
               |
               v
 One-month-ahead evaluation
  (observed historical lags)
               |
               v
 Compare against previous-month baseline
               |
               v
       MAE / RMSE / R²
```

The preprocessing scaler is fitted **on training data only**. Model evaluation uses the next observed month as the target, with earlier **observed** prices as inputs. This is different from a recursively generated multi-month forecast, where earlier forecast errors can accumulate.

After holdout evaluation, a model is refitted on all available supervised historical data to produce an illustrative future seven-month sequence.

## 5. Results and Interpretation

The original notebook reported Ridge RMSE about **39.78**. A separate review of a fixed-alpha Ridge model on the 460-month holdout recorded:

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Previous-month baseline | ~24.98 | ~40.06 | ~0.9964 |
| Ridge (`alpha=1`) | ~24.63 | ~40.29 | ~0.9964 |

These figures describe the supplied historical dataset, **not live price estimates**. Ridge slightly improved MAE in this comparison but did **not** outperform the naive baseline on RMSE. This is why the baseline is an important part of the project rather than a minor detail.

Run `src/evaluate.py` locally to regenerate exact machine-readable metrics and per-month predictions under your installed dependency versions.

**Do not call R² × 100 an accuracy percentage for regression.** Here, MAE and RMSE are errors in the source CSV's price units, not classification accuracies.

## 6. Seven-Month Forecast

The updated script optionally produces a **recursive forecast of seven future months**, starting after the most recent recorded observation. It:

1. Takes the latest 12 known monthly prices.
2. Predicts the following month.
3. Appends that forecast and uses it to predict the next month.
4. Repeats for the requested number of future months.

Because future months beyond the first depend on earlier forecasts, errors can compound. The multi-step forecasts are **illustrative and not independently backtested**; no investment conclusions should be drawn from them.

## 7. Repository Structure

```text
Gold-Price-Prediction-/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── monthly.csv
├── src/
│   └── evaluate.py
└── notebooks/
    ├── original_gold_price_predication.ipynb
    └── monthly_gold_forecasting.ipynb
```

Running the updated script also creates local `results/` files, including `evaluation.json`, `holdout_predictions.csv` and `future_monthly_forecast.csv`. The results folder may be absent on a fresh clone until the script is run.

## 8. Installation and Reproduction

On Windows:

```bat
git clone https://github.com/syblejan/Gold-Price-Prediction-.git
cd Gold-Price-Prediction-
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/evaluate.py
```

Optional: `python src/evaluate.py --months 12` produces 12-month recursive outputs. The CLI accepts 1–24 months.

Open `notebooks/monthly_gold_forecasting.ipynb` in Jupyter from the repository root for an interactive version.

## 9. Technologies, Limitations and Future Work

**Technologies:** Python, Pandas, NumPy, Scikit-learn, Ridge Regression, StandardScaler, Jupyter, regression metrics.

**Limitations:**

- Historical monthly prices only; no macroeconomic features or transaction costs.
- Currency and dataset provenance/licensing need clear documentation.
- Backtest uses historical lagged actual prices; it is **not** a realistic multi-step rolling-origin backtest.
- Very long historical periods cover major regime shifts.
- No uncertainty intervals, production monitoring, or trading validation.

**Future improvements:** Rolling-origin evaluation; stronger seasonal/naive baselines; time-series cross-validation for hyperparameter tuning; regime-aware analysis; forecast intervals; improved data provenance documentation.

## 10. Author

**Syble Jan** — BS Computer Science student, aspiring AI/ML Engineer.

[GitHub profile](https://github.com/syblejan)

---

**Status:** Corrected educational monthly time-series evaluation and forecasting code added. No real-world financial accuracy guarantee.
