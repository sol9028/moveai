# 수도권 전력수요 / 부족량 / ESS 방전필요량 예측 모듈

E-Train RE:LOOP 프로젝트의 `forecast/demand.py`에 해당하는 예측 파트 구현입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
# (선택) 실제 기상데이터 먼저 받기 — 없으면 합성데이터로 배관만 검증하고 진행됨
export KMA_AUTH_KEY="발급받은키"
python -m forecast.weather --start 2024-01-01 --end 2025-12-31

# 전체 파이프라인
python -m forecast.run_pipeline
```

한 번 실행하면:
1. `data/한국전력거래소_시간별_전국_전력수요량_2024.csv`, `..._2025.csv`를 읽어 정제
2. 전국 → 수도권 비중(기본 40%, `config.py`에서 조정) 적용해 수도권 수요 생성
3. `data/capital_weather_hourly.csv`가 있으면 로드, 없으면 배관 검증용 합성 기온 데이터 사용
4. 캘린더/주기성/lag/기온 피처 생성 (기온은 있을 때만 자동 포함)
5. LightGBM으로 시간대별 수도권 전력수요 예측 모델 학습
6. 부족량(baseline 방식 기본) → ESS 방전필요량 → 안중/서화성 50:50 분배 계산
7. **결과를 두 파일로 분리 저장**:
   - `outputs/capital_demand_forecast.csv` — 전체 수요 예측치만
   - `outputs/shortage_forecast.csv` — 부족량/ESS방전필요량/거점분배만

## 파일 구조

```
forecast/
├─ config.py       모든 가정치(비중, baseline/capacity 파라미터 등)와 경로를 한 곳에 모음
├─ data_prep.py     CSV 정제 (날짜 포맷 통일, wide→long, 전국→수도권 비중)
├─ weather.py        KMA ASOS API에서 수도권(서울·인천·수원 평균) 기온 수집
├─ features.py       캘린더/주기성/lag/rolling/기온 피처 생성 (기온은 선택적)
├─ demand.py          LightGBM 학습·평가·재귀예측(recursive_forecast)
├─ shortage.py        부족량(baseline 또는 capacity 방식) → ESS 방전필요량 → 거점 분배
└─ run_pipeline.py   전체 파이프라인 실행 + 시각화 + 결과 파일 분리 저장
```

## 부족량 산정 방식 두 가지 (`config.SHORTAGE_METHOD`)

| 방식 | 정의 | 의미 | 언제 쓰나 |
|---|---|---|---|
| **baseline** (기본값) | 예측수요 − (최근 7일 일별 하위10%의 롤링 중앙값) | "그 시간대가 야간 저부하 대비 얼마나 높은가" — **ESS 방전 기회 신호**에 가까움. 실제 정전위기와는 다른 개념 | 공급능력 실데이터가 없는 지금 단계 |
| **capacity** (레거시) | 예측수요 − (최근 30일 최대수요×예비율×가용률) | "실제 공급 여력 대비 부족한가" — 계통 위기에 가까운 개념 | KPX 발전설비 데이터 확보 후 |

⚠️ baseline 방식은 특성상 하루 중 상당 시간(주간~저녁)에 "부족"이 잡힙니다. 이는 버그가 아니라 "야간 대비 부하 변동폭"을 측정하는 방식의 자연스러운 결과입니다. `ESS_COVERAGE_RATIO`(기본 30%)가 이 중 ESS가 실제로 커버하는 비율을 조절합니다.

## 현재 모델 성능 (2024~2025 데이터, 최근 60일 홀드아웃)

| 지표 | 값 |
|---|---|
| MAE | 약 239 MW |
| RMSE | 약 344 MW |
| MAPE | 약 0.93% |

(수도권 수요를 "전국 수요 × 고정비율"로 만든 프록시 타깃이라 정확도가
매우 높게 나옵니다 — 실제 지역별 수요 데이터로 교체하면 이 숫자는
달라질 수 있습니다. 아래 "다음 단계" 참고.)

## ⚠️ 지금 들어있는 '가정치' 목록 — 실데이터 확보 시 교체 필요

| 가정치 | 위치 | 현재 값 | 교체할 실제 데이터 |
|---|---|---|---|
| 전국→수도권 비중 | `config.CAPITAL_REGION_RATIO` | 0.40 (고정) | 한전 데이터포털 지역별(시도별) 판매량 |
| baseline 부족량 파라미터 | `config.BASELINE_QUANTILE/LOOKBACK_DAYS` | 하위10% / 7일 | 실공급 데이터 확보 시 `capacity` 방식으로 전환 검토 |
| 공급능력(capacity, 레거시) | `shortage.estimate_supply_capacity` | 최근 30일 최대수요×1.1×0.95 | KPX 발전설비 현황 + 정비계획 |
| ESS 커버 비율 | `config.ESS_COVERAGE_RATIO` | 0.30 | 실제 ESS 설비용량/SOC 데이터 |
| 안중/서화성 분배 | `config.DISCHARGE_SITE_SPLIT` | 50:50 고정 | 거점별 실수요 비중 or MILP가 직접 최적화 |
| 날씨(기온) | `forecast/weather.py` 구현 완료, **실행은 사용자 환경에서 authKey로 해야 함** | 미실행 시 합성데이터로 배관만 검증 | `python -m forecast.weather --start ... --end ...` 실행 후 `data/capital_weather_hourly.csv` 생성 |
| SMP(가격, 부족량 프록시) | 아직 미연동 | - | KPX EPSIS SMP 시계열 |

## ⚠️ 기상 데이터 관련 중요 사항

`forecast/weather.py`는 Claude 샌드박스에서 **직접 실행이 안 됩니다** (apihub.kma.go.kr 접근 제한). 코드는 완성되어 있으니, 사용자 로컬/Jupyter 환경에서 아래처럼 실행해서 실제 데이터를 만들어야 합니다:

```bash
export KMA_AUTH_KEY="실제authKey"
python -m forecast.weather --start 2024-01-01 --end 2025-12-31
```

이렇게 만들어진 `data/capital_weather_hourly.csv`를 이 폴더에 넣고 `run_pipeline.py`를 다시 돌리면 자동으로 실제 기온이 반영됩니다. (지금 제출된 성능 지표는 **합성 기온** 기준이므로 실데이터로 교체 후 반드시 재검증하세요.)

## 다음 단계 제안

1. **실제 기상 데이터로 교체 후 재검증**: 위 방법으로 `capital_weather_hourly.csv` 생성 → 재실행 → MAE/MAPE 재확인
2. **부족량 방식 재검토**: baseline 방식은 "ESS 방전 기회"에 가깝고 capacity 방식은 "실제 위기"에 가까움. 발표/문서에서 어떤 의미로 쓸지 명확히 할 것
3. **재귀예측(recursive_forecast) 정확도 검증**: lag 피처를 예측값으로 계속 채워나가는 구조라 먼 미래로 갈수록 오차가 누적될 수 있음 → 실제 운영 시 Rolling Horizon으로 짧은 주기(예: 6~24시간)마다 재학습/재예측 권장
4. **MILP 모듈과 연결**: `shortage_forecast.csv`의 `ess_need_안중_mw`, `ess_need_서화성_mw`를 `optimization/milp_v4.py`의 입력으로 사용
5. **SMP/재생에너지 출력제어 데이터 착수**: "상행" 쪽(호남 재생에너지) 예측 모듈은 아직 미착수
