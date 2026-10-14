"""Leakage-aware monthly gold price forecasting (educational, not financial advice)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LAGS = 12

def prepare(csv_path):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
    df["Price"] = pd.to_numeric(df["Price"], errors="raise")
    if df[["Date", "Price"]].isna().any().any() or df["Date"].duplicated().any():
        raise ValueError("Missing prices, dates, or duplicate dates")
    df = df.sort_values("Date").set_index("Date")
    expected = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    if not expected.equals(df.index):
        raise ValueError("Expected continuous monthly observations")
    series = df[["Price"]].copy()
    for i in range(1, LAGS + 1):
        series[f"lag_{i}"] = series["Price"].shift(i)
    return df, series.dropna()

def score(y, pred):
    return {"mae":float(mean_absolute_error(y,pred)),
            "rmse":float(np.sqrt(mean_squared_error(y,pred))),
            "r2":float(r2_score(y,pred))}

def rolling_backtest(data, features, initial_train=600, window=240, step=240):
    """Expanding training windows, non-overlapping forward test blocks.

    Predictions use observed lagged historical values, not recursive simulation.
    """
    records = []
    for end_train in range(initial_train, len(data) - window + 1, step):
        train = data.iloc[:end_train]
        test = data.iloc[end_train:end_train + window]
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        model.fit(train[features], train["Price"])
        ridge_pred = model.predict(test[features])
        baseline_pred = test["lag_1"].to_numpy()
        records.append({
            "start": test.index.min().strftime("%Y-%m"),
            "end": test.index.max().strftime("%Y-%m"),
            "n": len(test),
            "ridge": score(test["Price"], ridge_pred),
            "baseline": score(test["Price"], baseline_pred)
        })
    return records


def run(data_path, out_dir, horizon=7):
    if not 1 <= horizon <= 24:
        raise ValueError("horizon must be between 1 and 24 months")
    df, data = prepare(data_path)
    n = int(0.8 * len(data))
    tr, te = data.iloc[:n], data.iloc[n:]
    features = [f"lag_{i}" for i in range(1, LAGS+1)]
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model.fit(tr[features],tr["Price"])
    pred = model.predict(te[features])
    baseline = te["lag_1"].to_numpy()
    metrics = {"frequency":"monthly","samples":len(df),"train_samples":len(tr),
               "test_samples":len(te),"test_start":te.index.min().strftime("%Y-%m"),
               "baseline_previous_month":score(te["Price"],baseline),
               "ridge_alpha_1":score(te["Price"],pred),
               "note":"One-step test predictions use observed prior months; recursively produced future forecasts are not backtested."}
    rolling = rolling_backtest(data, features)
    metrics["rolling_backtest"] = {
        "protocol": "expanding training windows, 240-month fixed test blocks, 240-month step; Ridge alpha=1 fixed in advance",
        "windows": rolling,
        "ridge_better_rmse_windows": sum(w["ridge"]["rmse"] < w["baseline"]["rmse"] for w in rolling),
        "total_windows": len(rolling)
    }
    # Refit for future predictions only after holdout metrics have been calculated.
    model.fit(data[features],data["Price"])
    history=list(df["Price"].to_numpy())
    future=[]
    for date in pd.date_range(df.index[-1]+pd.offsets.MonthBegin(1),periods=horizon,freq="MS"):
        row=pd.DataFrame([[history[-i] for i in range(1,LAGS+1)]],columns=features)
        forecast=float(model.predict(row)[0])
        future.append({"month":date.strftime("%Y-%m"),"forecast_price":forecast})
        history.append(forecast)
    output=Path(out_dir)
    output.mkdir(parents=True,exist_ok=True)
    (output/"evaluation.json").write_text(json.dumps(metrics,indent=2))
    pd.DataFrame({"month":te.index.strftime("%Y-%m"),"actual":te["Price"].to_numpy(),
                  "ridge_prediction":pred,"previous_month_baseline":baseline}).to_csv(output/"holdout_predictions.csv",index=False)
    pd.DataFrame(future).to_csv(output/"future_monthly_forecast.csv",index=False)
    print(json.dumps(metrics,indent=2))
    print("Forecast months:",future)
    return metrics

if __name__=="__main__":
    import argparse
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default=str(root/"data"/"monthly.csv"))
    parser.add_argument("--out",default=str(root/"results"))
    parser.add_argument("--months",type=int,default=7)
    args=parser.parse_args()
    run(args.data,args.out,args.months)
