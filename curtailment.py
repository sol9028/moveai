# -*- coding: utf-8 -*-
"""(a) 호남권 태양광 출력제어 예측 — 통합 모델.

출력제어 실적은 비공개이므로, 기상 기반 이론발전량과 실제 발전량의 차이(gap)에
경부하 조건을 결합해 제어 이벤트를 역추정한 뒤 2-stage 모델로 학습한다.

[통합안 반영]
  - 기상: 호남 23개 관측소, 변수별 유효 지점만 평균 (일사는 14곳만 관측)
  - 피처: 전운량(CA_TOT) 포함 — 흐린 날 오판 감소
  - 기준 발전량: 물리식 (유효용량 × 일사비율). 학습 기반은 제어된 값을 학습해 과소평가
  - 판정: 상대 gap ≥20% AND 경부하 / −1.5σ 통계 임계는 보조 지표로 병기
  - 학습 연도: 2025 단일 (설비 급증으로 연도 혼합 시 이론발전량 스케일 왜곡)

[실행]
    python forecast/curtailment.py
→ output/ 에 CSV 저장 (supply_forecast.csv 가 최적화팀 전달본). 시각화는 notebooks/curtailment_viz.ipynb 에서.

[입력 파일] 프로젝트 루트에 위치
    1. 한국전력거래소_시간별 전국 전력수요량_2025.csv
    2. 한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_2025.csv
    5. KPX전력수급실적_2025.csv
    honam_asos24_2025.csv   (없으면 .env의 KMA_SERVICE_KEY로 자동 수집)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import unquote

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit

# --------------------------------------------------------------------------
# 경로 · 상수
# --------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)
ASOS_FILE = BASE / "honam_asos24_2025.csv"

RNG = 42
GAP_REL_TH = 0.20        # 이론발전량 대비 상대 gap 임계
THEO_MIN_MWH = 800       # 저일사(흐린 날·아침저녁) 오판 방지
SIGMA_TH = -1.5          # 보조 지표 (통계 임계)
LIGHT_MONTHS = [3, 4, 5, 10, 11]

HOLIDAYS_2025 = pd.to_datetime([
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30",
    "2025-03-01", "2025-03-03", "2025-05-05", "2025-05-06",
    "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03",
    "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08",
    "2025-10-09", "2025-12-25",
])

# 호남 태양광 누적 보급용량 (2024년말, 한국에너지공단) — MW
CAPACITY_MW = {"전남": 7177, "전북": 5755, "광주": 489}
CAP_TOTAL = sum(CAPACITY_MW.values())

HONAM_REGIONS = ["전라남도", "전라북도", "전북특별자치도", "광주시"]
MJ_TO_SUN_RATIO = 1e6 / 3600 / 1000      # 1 MJ/m²/h → STC(1000W/m²) 대비 비율

# 충전지점 가중치 (설비용량 스냅샷 기준)
NODE_W = {
    "장성": (CAPACITY_MW["전남"] + CAPACITY_MW["광주"]) / CAP_TOTAL,
    "동산": CAPACITY_MW["전북"] / 3 / CAP_TOTAL,
    "남원": CAPACITY_MW["전북"] / 3 / CAP_TOTAL,
    "새만금": CAPACITY_MW["전북"] / 3 / CAP_TOTAL,
}
# 지점별 대표 관측소 (일사 미관측 지점은 전주로 대체)
NODE_STN = {
    "장성": {"기온": "광주", "일사": "광주"},
    "동산": {"기온": "전주", "일사": "전주"},
    "남원": {"기온": "남원", "일사": "전주"},
    "새만금": {"기온": "군산", "일사": "전주"},
}

FEATURE_CANDIDATES = [
    "일사_호남평균", "일사_광주", "일사_전주", "일사_목포",
    "기온_호남평균", "풍속_호남평균", "전운량_호남평균",
    "month", "hour", "dayofweek", "is_offday", "is_holiday",
    "capacity_mw", "reserve_pct_d1", "demand_lag24",
]

# --------------------------------------------------------------------------
# ASOS 수집 (파일이 없을 때만)
# --------------------------------------------------------------------------
ENDPOINT = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
STATIONS = {
    "140": "군산", "146": "전주", "243": "부안", "244": "임실", "245": "정읍",
    "247": "남원", "248": "장수", "254": "순창",
    "156": "광주", "165": "목포", "168": "여수", "169": "흑산도", "170": "완도",
    "172": "고창", "174": "순천", "251": "고창군", "252": "영광군",
    "258": "보성군", "259": "강진군", "260": "장흥", "261": "해남",
    "262": "고흥", "268": "진도군",
}
ASOS_KEEP = {"tm": "dt", "ta": "기온", "ws": "풍속",
             "icsr": "일사", "dc10Tca": "전운량"}
MONTH_ENDS = {f"{m:02d}": d for m, d in zip(
    range(1, 13), ["31", "28", "31", "30", "31", "30",
                   "31", "31", "30", "31", "30", "31"])}


def _service_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE / ".env")
    except ImportError:
        pass
    key = os.environ.get("KMA_SERVICE_KEY", "")
    if not key:
        raise SystemExit(
            "KMA_SERVICE_KEY 가 없습니다. 프로젝트 루트에 .env 파일을 만들고\n"
            "  KMA_SERVICE_KEY=발급받은키\n"
            "형태로 저장하세요 (.gitignore 에 .env 추가 필수).")
    return unquote(key)


def _fetch_month(key: str, stn_id: str, start_dt: str, end_dt: str) -> list:
    import requests
    rows, page = [], 1
    while True:
        r = requests.get(ENDPOINT, timeout=30, params={
            "serviceKey": key, "pageNo": page, "numOfRows": 999,
            "dataType": "JSON", "dataCd": "ASOS", "dateCd": "HR",
            "startDt": start_dt, "startHh": "01",
            "endDt": end_dt, "endHh": "23", "stnIds": stn_id})
        r.raise_for_status()
        body = r.json()["response"]["body"]
        items = body["items"].get("item", [])
        if isinstance(items, dict):
            items = [items]
        rows.extend(items)
        if page * 999 >= int(body["totalCount"]) or not items:
            break
        page += 1
        time.sleep(0.25)
    return rows


def collect_asos() -> pd.DataFrame:
    """호남 23개소 2025년 시간자료 수집 (약 5~8분)."""
    key = _service_key()
    frames = []
    for stn_id, name in STATIONS.items():
        print(f"  [{name}]", end=" ", flush=True)
        rows = []
        for mm, last in MONTH_ENDS.items():
            try:
                rows += _fetch_month(key, stn_id, f"2025{mm}01", f"2025{mm}{last}")
                print(mm, end="", flush=True)
            except Exception:
                print("x", end="", flush=True)
            time.sleep(0.25)
        print()
        if not rows:
            continue
        d = pd.DataFrame(rows)
        d = d[[c for c in ASOS_KEEP if c in d.columns]].rename(columns=ASOS_KEEP)
        for c in d.columns:
            if c != "dt":
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d["stn"] = name
        frames.append(d)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["stn", "dt"])
    out.to_csv(ASOS_FILE, index=False, encoding="utf-8-sig")
    return out


# --------------------------------------------------------------------------
# 로딩 · 전처리
# --------------------------------------------------------------------------
def _hour_ending(date, hour) -> pd.Series:
    """시간 규약: 구간 종료 시각 (1시 = 00~01시 구간 → ts=01:00)."""
    return pd.to_datetime(date) + pd.to_timedelta(hour, unit="h")


def load_asos_raw() -> pd.DataFrame:
    if not ASOS_FILE.exists():
        print("[기상] 수집 시작 (23개소 × 12개월)")
        collect_asos()
    raw = pd.read_csv(ASOS_FILE, encoding="utf-8-sig")
    raw["ts"] = pd.to_datetime(raw["dt"])
    return raw


def observed_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """지점별 변수 관측 여부 (연중 10% 이상 값 존재 = 관측)."""
    return raw.groupby("stn")[["기온", "풍속", "일사", "전운량"]].apply(
        lambda g: g.notna().mean() > 0.1)


def build_weather(raw: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """변수별 유효 지점만 평균한 호남 대표 기상.

    일사·전운량은 관측 지점만으로 평균한다(미관측 지점의 NaN이 0으로 섞여
    평균을 끌어내리는 것을 방지 — 군산 사례의 일반화).
    """
    parts = {}
    for var in ["기온", "풍속", "전운량"]:
        valid = obs.index[obs[var]]
        p = raw[raw["stn"].isin(valid)].pivot_table(
            index="ts", columns="stn", values=var)
        parts[f"{var}_호남평균"] = p.mean(axis=1)

    valid_si = obs.index[obs["일사"]]
    p_si = raw[raw["stn"].isin(valid_si)].pivot_table(
        index="ts", columns="stn", values="일사")
    parts["일사_호남평균"] = p_si.fillna(0.0).mean(axis=1)
    for stn in ["광주", "전주", "목포"]:
        if stn in p_si.columns:
            parts[f"일사_{stn}"] = p_si[stn].fillna(0.0)

    w = pd.DataFrame(parts).sort_index()
    for c in ["기온_호남평균", "풍속_호남평균", "전운량_호남평균"]:
        w[c] = w[c].interpolate(limit=6)
    print(f"[기상] 유효 지점 — 기온 {int(obs['기온'].sum())}, "
          f"전운량 {int(obs['전운량'].sum())}, 일사 {int(obs['일사'].sum())} "
          f"({', '.join(valid_si)})")
    return w


def load_honam_solar() -> pd.DataFrame:
    """파일 2 → 호남 3개 시도 태양광 시간별 거래량 합계 (계량 기준)."""
    df = pd.read_csv(
        BASE / "2. 한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_2025.csv",
        encoding="cp949")
    m = df["지역"].isin(HONAM_REGIONS) & (df["연료원"] == "태양광")
    g = df[m].groupby(["거래일", "거래시간"])["전력거래량(MWh)"].sum().reset_index()
    g["ts"] = _hour_ending(g["거래일"], g["거래시간"])
    return (g[["ts", "전력거래량(MWh)"]]
            .rename(columns={"전력거래량(MWh)": "solar_mwh"})
            .set_index("ts").sort_index())


def load_national_demand() -> pd.DataFrame:
    """파일 1 → 전국 시간별 수요 (경부하 판별용)."""
    df = pd.read_csv(BASE / "1. 한국전력거래소_시간별 전국 전력수요량_2025.csv",
                     encoding="cp949")
    long = df.melt(id_vars="날짜", var_name="hour", value_name="demand_mwh")
    long["hour"] = long["hour"].str.replace("시", "").astype(int)
    long["ts"] = _hour_ending(
        long["날짜"].astype(str).str.replace(".", "-", regex=False), long["hour"])
    return long[["ts", "demand_mwh"]].dropna().sort_values("ts").set_index("ts")


def load_kpx_daily() -> pd.DataFrame:
    """파일 5 → 일별 수급실적 (예비율)."""
    df = pd.read_csv(BASE / "5. KPX전력수급실적_2025.csv", encoding="cp949")
    df["date"] = pd.to_datetime(df[["년", "월", "일"]].rename(
        columns={"년": "year", "월": "month", "일": "day"}))
    return df.set_index("date").sort_index()


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    eff = idx - pd.Timedelta(hours=1)     # 24시(익일 00:00)는 전날 구간
    f["month"] = eff.month
    f["hour"] = idx.hour.where(idx.hour != 0, 24)
    f["dayofweek"] = eff.dayofweek
    f["is_weekend"] = pd.Series(eff.dayofweek.isin([5, 6]).astype(int), index=idx)
    f["is_holiday"] = pd.Series(eff.normalize(), index=idx).isin(
        HOLIDAYS_2025).astype(int)
    f["is_offday"] = ((f["is_weekend"] + f["is_holiday"]) > 0).astype(int)
    return f


# --------------------------------------------------------------------------
# 이론발전량 · 라벨 재구성
# --------------------------------------------------------------------------
def build_dataset(raw: pd.DataFrame, obs: pd.DataFrame) -> tuple:
    solar = load_honam_solar()
    weather = build_weather(raw, obs)
    demand = load_national_demand()
    kpx = load_kpx_daily()

    df = solar.join(weather, how="inner").join(demand, how="left")
    df = df.join(calendar_features(df.index))

    # 월별 유효용량 프록시: 맑은 한낮 implied capacity 98분위, 단조 증가 강제
    irr = df["일사_호남평균"] * MJ_TO_SUN_RATIO
    clear = irr > 0.55
    implied = df.loc[clear, "solar_mwh"] / irr[clear]
    proxy = (implied.groupby(df.loc[clear, "month"]).quantile(0.98)
             .reindex(range(1, 13)).interpolate(limit_direction="both").cummax())
    pr = float(proxy.iloc[0] / CAP_TOTAL)
    eff_cap = df["month"].map(proxy)

    df["capacity_mw"] = eff_cap / pr
    df["theo_mwh"] = eff_cap * irr
    df["gap_mwh"] = (df["theo_mwh"] - df["solar_mwh"]).clip(lower=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["gap_rel"] = np.where(df["theo_mwh"] > 0,
                                 df["gap_mwh"] / df["theo_mwh"], 0.0)

    # 주 라벨: 상대 gap + 경부하 AND
    hour_pctl = df.groupby("hour")["demand_mwh"].rank(pct=True)
    light_load = (df["month"].isin(LIGHT_MONTHS)
                  & ((df["is_offday"] == 1) | (hour_pctl < 0.25))
                  & df["hour"].between(9, 17))
    df["curtail_label"] = ((df["gap_rel"] >= GAP_REL_TH)
                           & (df["theo_mwh"] >= THEO_MIN_MWH)
                           & light_load).astype(int)
    df["curtail_mwh_label"] = np.where(df["curtail_label"] == 1, df["gap_mwh"], 0.0)

    # 보조 지표: −1.5σ 통계 임계 (교차검증용)
    day_mask = df["theo_mwh"] >= THEO_MIN_MWH
    resid = (df["solar_mwh"] - df["theo_mwh"])[day_mask]
    z = (resid - resid.mean()) / resid.std()
    df["sigma_label"] = 0
    df.loc[z.index, "sigma_label"] = (z < SIGMA_TH).astype(int)

    n_main = int(df["curtail_label"].sum())
    n_sig = int(df["sigma_label"].sum())
    n_both = int(((df["curtail_label"] == 1) & (df["sigma_label"] == 1)).sum())
    print(f"[교차검증] 주 방법 {n_main}h / −1.5σ {n_sig}h / 동시 지목 {n_both}h")
    print("  ※ −1.5σ는 계통 조건 필터가 없어 국지적 흐림·설비 이슈가 섞임.")
    print(f"[라벨] 제어 {n_main}h ({n_main / len(df) * 100:.1f}%), "
          f"총 {df['curtail_mwh_label'].sum():,.0f} MWh")

    # 예측 시점에 알 수 있는 정보만 (전일 예비율, 전일 동시간 수요)
    eff_date = (df.index - pd.Timedelta(hours=1)).normalize()
    df["reserve_pct_d1"] = pd.Series(
        kpx["공급예비율(%)"].reindex(eff_date - pd.Timedelta(days=1)).values,
        index=df.index)
    df["demand_lag24"] = df["demand_mwh"].shift(24)
    return df, proxy, pr


# --------------------------------------------------------------------------
# 2-stage 모델
# --------------------------------------------------------------------------
def train_predict(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in FEATURE_CANDIDATES if c in df.columns]
    data = df.dropna(subset=["demand_lag24", "reserve_pct_d1"]).copy()

    # 일 단위 그룹 분리 — 같은 날이 train/test에 섞이는 누수 방지
    days = (data.index - pd.Timedelta(hours=1)).normalize()
    tr_i, te_i = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=RNG).split(data, groups=days))
    tr, te = data.iloc[tr_i], data.iloc[te_i]

    clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                             min_child_samples=30, subsample=0.9,
                             colsample_bytree=0.9, random_state=RNG, verbose=-1)
    clf.fit(tr[feats], tr["curtail_label"])
    prob_te = clf.predict_proba(te[feats])[:, 1]
    pred_ev = (prob_te >= 0.5).astype(int)
    print(f"[분류] AUC={roc_auc_score(te['curtail_label'], prob_te):.3f}  "
          f"P={precision_score(te['curtail_label'], pred_ev, zero_division=0):.3f}  "
          f"R={recall_score(te['curtail_label'], pred_ev, zero_division=0):.3f}")

    reg = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                            min_child_samples=15, subsample=0.9,
                            colsample_bytree=0.9, random_state=RNG, verbose=-1)
    ev_tr = tr[tr["curtail_label"] == 1]
    reg.fit(ev_tr[feats], ev_tr["curtail_mwh_label"])
    ev_te = te[te["curtail_label"] == 1]
    if len(ev_te):
        mae = mean_absolute_error(ev_te["curtail_mwh_label"],
                                  reg.predict(ev_te[feats]).clip(min=0))
        print(f"[회귀] 이벤트 MAE={mae:,.0f} MWh")

    prob = clf.predict_proba(data[feats])[:, 1]
    amount = reg.predict(data[feats]).clip(min=0)
    out = pd.DataFrame({
        "curtail_prob": prob.round(4),
        "curtail_mwh_if_event": amount.round(1),
        "expected_curtail_mwh": (prob * amount).round(1),
        "label_curtail": data["curtail_label"].values,
        "label_curtail_mwh": data["curtail_mwh_label"].round(1).values,
        "sigma_label": data["sigma_label"].values,
        "theo_mwh": data["theo_mwh"].round(1).values,
        "actual_solar_mwh": data["solar_mwh"].values,
    }, index=data.index)
    out.index.name = "ts"

    # 시각화용 ROC·피처중요도 재현을 위해 테스트셋 예측도 저장
    roc_df = pd.DataFrame({"y_true": te["curtail_label"].values,
                           "y_prob": prob_te}, index=te.index)
    roc_df.index.name = "ts"
    roc_df.to_csv(OUT / "model_test_predictions.csv", encoding="utf-8-sig")
    pd.Series(clf.feature_importances_, index=feats,
              name="importance").to_csv(OUT / "model_feature_importance.csv",
                                        encoding="utf-8-sig")
    return out


# --------------------------------------------------------------------------
# 산출물
# --------------------------------------------------------------------------
def export_hourly_daily(pred: pd.DataFrame) -> pd.DataFrame:
    pred.to_csv(OUT / "pred_curtailment_hourly.csv", encoding="utf-8-sig")
    daily = (pred.assign(date=(pred.index - pd.Timedelta(hours=1)).normalize())
             .groupby("date").agg(
                 expected_curtail_mwh=("expected_curtail_mwh", "sum"),
                 max_prob=("curtail_prob", "max"),
                 label_curtail_mwh=("label_curtail_mwh", "sum")))
    daily.to_csv(OUT / "pred_curtailment_daily.csv", encoding="utf-8-sig")
    print(f"[저장] hourly {len(pred):,}행 / daily {len(daily)}행")
    return daily


def export_node_file(pred: pd.DataFrame, raw: pd.DataFrame) -> None:
    """상위 7일 × 4개 충전지점 인계 파일 2종을 저장한다.

    supply_forecast.csv (최적화팀 전달용, 5컬럼)
        date, node, hour, solar_mwh, chargeable_mwh
          - solar_mwh      : 시간대별 재생에너지 발전량 (역별 안분)
          - chargeable_mwh : 출력제어 예상량 = 역별 충전 가능 에너지 (충전 상한)

    supply_forecast_detail.csv (검증·발표용, 10컬럼)
        위 + TA, SI, theo_mwh_node, curtail_prob, curtail_mwh_if_event

    한계: KPX 발전량이 시도 단위까지만 공개되어 역별 실측이 불가능하다.
    총량은 보존되나 지점 간 배분은 고정 가중치이므로, 결과 해석 시
    "총량·시간대"는 유효하되 "어느 역"은 잠정치로 다룬다.
    """
    ta = raw.pivot_table(index="ts", columns="stn", values="기온")
    si = raw.pivot_table(index="ts", columns="stn", values="일사")

    tmp = pred.copy()
    tmp["date"] = (tmp.index - pd.Timedelta(hours=1)).date
    tmp["h"] = (tmp.index - pd.Timedelta(hours=1)).hour
    top7 = tmp.groupby("date")["expected_curtail_mwh"].sum().nlargest(7).index.tolist()
    sub = tmp[tmp["date"].isin(top7)]

    rows = []
    for node, w in NODE_W.items():
        t = sub[["date", "h"]].copy()
        t["node"] = node
        t["TA"] = (ta[NODE_STN[node]["기온"]].reindex(sub.index)
                   .interpolate(limit=3).round(1).values)
        t["SI"] = (si[NODE_STN[node]["일사"]].reindex(sub.index)
                   .fillna(0.0).round(2).values)
        # 역별 안분 (야간 결측은 0 — 태양광 발전이 없는 시간)
        t["solar_mwh"] = (sub["actual_solar_mwh"].fillna(0).values * w).round(1)
        t["theo_mwh_node"] = (sub["theo_mwh"].fillna(0).values * w).round(1)
        t["chargeable_mwh"] = (sub["expected_curtail_mwh"].values * w).round(1)
        t["curtail_prob"] = sub["curtail_prob"].values
        t["curtail_mwh_if_event"] = sub["curtail_mwh_if_event"].values
        rows.append(t.rename(columns={"h": "hour"}))

    detail = (pd.concat(rows)[["date", "node", "hour", "TA", "SI",
                               "solar_mwh", "theo_mwh_node", "chargeable_mwh",
                               "curtail_prob", "curtail_mwh_if_event"]]
              .sort_values(["date", "node", "hour"]))
    detail.to_csv(OUT / "supply_forecast_detail.csv",
                  index=False, encoding="utf-8-sig")

    main_df = detail[["date", "node", "hour", "solar_mwh", "chargeable_mwh"]]
    main_df.to_csv(OUT / "supply_forecast.csv",
                   index=False, encoding="utf-8-sig")

    print(f"[저장] supply_forecast.csv ({len(main_df)}행 = "
          f"{len(top7)}일 × {len(NODE_W)}지점 × 24h) — 최적화팀 전달용")
    print(f"[저장] supply_forecast_detail.csv ({len(detail)}행) — 검증·발표용")
    print(f"  상위 7일: {[str(d) for d in top7]}")


def main() -> None:
    raw = load_asos_raw()
    obs = observed_matrix(raw)
    obs.to_csv(OUT / "asos_station_coverage.csv", encoding="utf-8-sig")

    df, proxy, pr = build_dataset(raw, obs)
    print(f"[캘리브레이션] 1월 유효용량 {proxy.iloc[0]:,.0f} MW "
          f"/ 설비 스냅샷({CAP_TOTAL:,} MW) 대비 {pr:.3f}")

    # 시각화용 원본 시계열 (이론 vs 실제)
    df[["solar_mwh", "theo_mwh", "gap_mwh", "gap_rel",
        "일사_호남평균", "기온_호남평균", "전운량_호남평균",
        "curtail_label", "sigma_label"]].to_csv(
            OUT / "theo_vs_actual_hourly.csv", encoding="utf-8-sig")

    pred = train_predict(df)
    export_hourly_daily(pred)
    export_node_file(pred, raw)


if __name__ == "__main__":
    main()
