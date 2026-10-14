"""Honest time-series model comparison: select only on development folds.

The untouched chronological holdout is never used to choose the model.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from evaluate import prepare

MODELS = {
    "ridge_a0.1": lambda: Ridge(alpha=0.1),
    "ridge_a1": lambda: Ridge(alpha=1),
    "ridge_a10": lambda: Ridge(alpha=10),
    "lasso_a0.1": lambda: Lasso(alpha=0.1, max_iter=30000),
    "lasso_a1": lambda: Lasso(alpha=1, max_iter=30000),
    "elasticnet_a0.1_r0.5": lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=30000),
    "elasticnet_a1_r0.5": lambda: ElasticNet(alpha=1, l1_ratio=0.5, max_iter=30000),
}

def scores(y, predicted):
    return {"mae": float(mean_absolute_error(y, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y, predicted)))}

def fit_model(name, train, columns):
    estimator = Pipeline([("scaler", StandardScaler()), ("regressor", MODELS[name]())])
    return estimator.fit(train[columns], train["Price"])

def run(data_path, out_dir):
    _, data = prepare(data_path)
    cols = [f"lag_{i}" for i in range(1, 13)]
    split = int(len(data) * 0.8)
    development, holdout = data.iloc[:split], data.iloc[split:]
    folds = [(i, i + 240) for i in range(600, len(development) - 240 + 1, 240)]
    records = []
    for name in ["previous_month", *MODELS.keys()]:
        for start, stop in folds:
            train = development.iloc[:start]
            test = development.iloc[start:stop]
            if name == "previous_month":
                predicted = test["lag_1"].to_numpy()
            else:
                predicted = fit_model(name, train, cols).predict(test[cols])
            records.append({"model": name, "train_rows": start,
                            "start": test.index[0].strftime("%Y-%m"),
                            "end": test.index[-1].strftime("%Y-%m"),
                            **scores(test["Price"], predicted)})
    details = pd.DataFrame(records)
    summary = details.groupby("model", as_index=False).agg(
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"))
    summary = summary.sort_values(["mean_rmse", "model"]).reset_index(drop=True)
    selected = summary.iloc[0]["model"]
    holdout_results = []
    for name in dict.fromkeys(["previous_month", selected]):
        if name == "previous_month":
            predicted = holdout["lag_1"].to_numpy()
        else:
            predicted = fit_model(name, development, cols).predict(holdout[cols])
        holdout_results.append({"model": name, **scores(holdout["Price"], predicted)})
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    details.to_csv(output / "model_comparison_folds.csv", index=False)
    summary.to_csv(output / "model_comparison_summary.csv", index=False)
    report = {
        "methodology": "Expanding-window model selection on development data only",
        "development_rows": len(development), "holdout_rows": len(holdout),
        "development_folds": len(folds),
        "selection_metric": "Mean fold RMSE; historical price scales differ",
        "selected": selected, "holdout": holdout_results
    }
    (output / "model_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(summary.to_string(index=False))
    return report

if __name__ == "__main__":
    import argparse
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=root / "data" / "monthly.csv")
    parser.add_argument("--out", default=root / "results")
    args = parser.parse_args()
    run(args.data, args.out)
