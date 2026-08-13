# -*- coding: utf-8 -*-
"""
demand.py
수도권 시간대별 전력수요 예측 모델 (LightGBM 회귀).

- 시계열이므로 랜덤 K-fold가 아니라 '마지막 N일'을 테스트셋으로 분리
- lag/rolling 피처를 쓰기 때문에, 실제 운영 시 T+1시간 예측은 문제없으나
  T+24시간 이상 먼 미래를 예측하려면 재귀적(recursive) 예측이 필요함
  (recursive_forecast 함수 참고)
- 기온 피처(TA_capital_avg) 포함 여부는 feature_cols 인자로 결정되며,
  build_feature_frame()이 반환하는 feature_cols를 그대로 넘기면 된다.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from . import config


def time_based_split(df: pd.DataFrame, test_days: int = config.TEST_HOLDOUT_DAYS):
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    train = df[df["timestamp"] <= cutoff].reset_index(drop=True)
    test = df[df["timestamp"] > cutoff].reset_index(drop=True)
    return train, test


def train_demand_model(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "capital_demand_mw",
    save_model: bool = True,
    model_filename: str = "capital_demand_lgbm.joblib",
):
    train, test = time_based_split(feature_df)

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    pred = model.predict(X_test)
    metrics = evaluate(y_test.values, pred)

    print("[모델 성능 - 테스트셋 / 최근 {}일] 피처: {}".format(
        config.TEST_HOLDOUT_DAYS, feature_cols))
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    if save_model:
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = config.MODEL_DIR / model_filename
        joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
        print(f"[OK] 모델 저장: {model_path}")

    result_df = test[["timestamp", target_col]].copy()
    result_df["predicted_mw"] = pred
    result_df["error_mw"] = result_df["predicted_mw"] - result_df[target_col]

    return model, metrics, result_df


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE(MW)": mae, "RMSE(MW)": rmse, "MAPE(%)": mape}


def feature_importance(model, feature_cols: list[str], top_n: int = 10) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp.head(top_n)


def recursive_forecast(
    model,
    history_df: pd.DataFrame,
    feature_cols: list[str],
    horizon_hours: int,
    target_col: str = "capital_demand_mw",
    weather_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    학습에 쓴 것과 동일한 피처 구조로, 마지막 시점 이후 horizon_hours 만큼
    한 시간씩 재귀적으로 예측한다.
    (lag_1h/24h/168h, rolling 값을 매 스텝 갱신)

    weather_df: 미래 시점의 기온 예보가 있으면 전달 (timestamp, TA_capital_avg).
    없으면 TA_capital_avg가 feature_cols에 포함된 경우 마지막 관측값으로 고정한다
    (단기 예측 시 근사로는 무방하나, 장기 예측일수록 오차가 커짐에 유의).
    """
    from .features import add_calendar_features, add_cyclical_features

    use_weather = "TA_capital_avg" in feature_cols
    work = history_df[["timestamp", target_col]].copy().sort_values("timestamp")
    work = work.reset_index(drop=True)

    last_known_ta = None
    if use_weather and weather_df is None:
        # fallback: 원 데이터에 기온이 있었다면 그 마지막 값을 그대로 사용
        if "TA_capital_avg" in history_df.columns:
            last_known_ta = history_df["TA_capital_avg"].iloc[-1]

    forecasts = []
    last_ts = work["timestamp"].iloc[-1]

    for step in range(1, horizon_hours + 1):
        next_ts = last_ts + pd.Timedelta(hours=step)
        row = pd.DataFrame({"timestamp": [next_ts]})
        row = add_calendar_features(row)
        row = add_cyclical_features(row)

        values = pd.concat(
            [work[target_col], pd.Series([np.nan])], ignore_index=True
        )

        def get_lag(h):
            idx = len(values) - 1 - h
            return values.iloc[idx] if idx >= 0 else np.nan

        row["lag_1h"] = get_lag(1)
        row["lag_24h"] = get_lag(24)
        row["lag_168h"] = get_lag(168)

        recent_24 = values.iloc[-24:].astype(float)
        row["roll_mean_24h"] = recent_24.mean()
        row["roll_max_24h"] = recent_24.max()
        row["roll_std_24h"] = recent_24.std()

        if use_weather:
            if weather_df is not None:
                match = weather_df.loc[weather_df["timestamp"] == next_ts, "TA_capital_avg"]
                row["TA_capital_avg"] = match.iloc[0] if len(match) else last_known_ta
            else:
                row["TA_capital_avg"] = last_known_ta

        X_next = row[feature_cols]
        pred = model.predict(X_next)[0]

        forecasts.append({"timestamp": next_ts, "predicted_mw": pred})
        work = pd.concat(
            [work, pd.DataFrame({"timestamp": [next_ts], target_col: [pred]})],
            ignore_index=True,
        )

    return pd.DataFrame(forecasts)


if __name__ == "__main__":
    from .data_prep import load_or_build_capital_dataset
    from .features import build_feature_frame
    from .weather import load_weather_if_exists

    capital_df = load_or_build_capital_dataset()
    weather_df = load_weather_if_exists()
    feat_df, feature_cols = build_feature_frame(capital_df, weather_df=weather_df)
    model, metrics, result_df = train_demand_model(feat_df, feature_cols)

    print("\n[Feature Importance Top 10]")
    print(feature_importance(model, feature_cols))

    print("\n[향후 48시간 예측 예시]")
    future = recursive_forecast(model, capital_df, feature_cols, horizon_hours=48)
    print(future.head(10))
