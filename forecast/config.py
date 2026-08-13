# -*- coding: utf-8 -*-
"""
E-Train RE:LOOP - 수도권 전력수요/부족량/ESS 방전필요량 예측
설정값 모음

주의: 아래 수치들 중 '가정치'로 표시된 값은 실제 데이터가 확보되기 전까지
쓰는 임시값입니다. 실증/상용화 단계에서 실제 통계로 교체해야 합니다.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# 경로
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

RAW_FILES = [
    DATA_DIR / "한국전력거래소_시간별_전국_전력수요량_2024.csv",
    DATA_DIR / "한국전력거래소_시간별_전국_전력수요량_2025.csv",
]

PROCESSED_LONG_PATH = DATA_DIR / "national_demand_long.csv"
CAPITAL_DEMAND_PATH = DATA_DIR / "capital_region_demand.csv"
WEATHER_PATH = DATA_DIR / "capital_weather_hourly.csv"

# 최종 산출물 (수요예측 / 부족량은 별도 파일로 분리)
DEMAND_FORECAST_OUTPUT = OUTPUT_DIR / "capital_demand_forecast.csv"
SHORTAGE_FORECAST_OUTPUT = OUTPUT_DIR / "shortage_forecast.csv"

# ----------------------------------------------------------------------
# 기상 데이터 (KMA API)
# ----------------------------------------------------------------------
# 수도권 커버용 ASOS 지점 (지점번호: 지점명)
# 서울/인천/수원 3개 평균으로 '수도권 대표기온' 근사.
# 필요시 지점 추가/조정 가능 (예: 동두천98, 파주99 등)
WEATHER_STATIONS = {
    "108": "서울",
    "112": "인천",
    "119": "수원",
}
KMA_WEATHER_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
# authKey는 코드에 하드코딩하지 않고 환경변수나 함수 인자로 전달할 것
KMA_AUTH_KEY_ENV = "KMA_AUTH_KEY"

# ----------------------------------------------------------------------
# [가정치] 전국 수요 -> 수도권 수요 변환 비중
# 근거: 한전 통계연보 기준 수도권(서울/인천/경기) 전력판매량 비중은
# 대략 35~40% 수준으로 알려져 있음. 정확한 최신 수치는
# 한전 데이터포털/전력통계정보시스템(EPSIS)에서 검증 필요.
# ----------------------------------------------------------------------
CAPITAL_REGION_RATIO = 0.40

# ----------------------------------------------------------------------
# 부족량(shortage/deficit) 산정 방식 — 두 가지 지원
# ----------------------------------------------------------------------
# "baseline": 일별 저부하 시간대 대비 초과분 방식 (기본값)
#   - 실제 발전설비/공급능력 데이터가 없어도 계산 가능
#   - baseline(t) = 최근 BASELINE_LOOKBACK_DAYS일 '일별 하위분위수'의 롤링 중앙값
#   - 하루짜리 노이즈에 덜 민감하도록 다일(多日) 롤링으로 개선
#     (특정 하루의 15%ile만 쓰면 그날 자체가 이상치일 때 baseline이 왜곡됨)
#   - shortage(t) = max(0, demand(t) - baseline(t))
# "capacity": 공급능력 가정 기반 방식 (구 버전, 실데이터 확보 시 유용)
#   - capacity(t) = 최근 30일 최대수요 * (1+예비율) * 가용률
#   - shortage(t) = max(0, demand(t) - capacity(t))
SHORTAGE_METHOD = "baseline"     # "baseline" | "capacity"

# --- baseline 방식 파라미터 ---
BASELINE_QUANTILE = 0.10          # 일별 하위 10% 지점을 그날의 '기저부하'로 봄
BASELINE_LOOKBACK_DAYS = 7        # 최근 7일 기저부하의 중앙값을 baseline으로 사용

# --- capacity 방식 파라미터 (레거시) ---
TARGET_RESERVE_MARGIN = 0.10     # 목표 예비율 10% 가정
SUPPLY_SAFETY_FACTOR = 0.95      # 실공급 가용률 95% 가정 (정비/고장 등 반영)

# ----------------------------------------------------------------------
# ESS 방전 관련 가정
# ----------------------------------------------------------------------
# 부족량 중 ESS가 커버하는 비율 (나머지는 다른 예비자원으로 대응한다고 가정)
ESS_COVERAGE_RATIO = 0.30

# 안중 / 서화성 방전거점 분배 비율 (현재는 50:50 고정, 추후 MILP에서
# 동적 배분으로 확장 가능)
DISCHARGE_SITE_SPLIT = {
    "안중": 0.5,
    "서화성": 0.5,
}

# ----------------------------------------------------------------------
# 학습/평가 관련
# ----------------------------------------------------------------------
TEST_HOLDOUT_DAYS = 60     # 마지막 60일을 테스트셋으로 분리 (시계열이므로 랜덤 분리 X)
RANDOM_STATE = 42

# 시간별 wide 컬럼명 (1시~24시)
HOUR_COLS = [f"{h}시" for h in range(1, 25)]


TOP_SHORTAGE_DAYS_OUTPUT = OUTPUT_DIR / "top_shortage_days_hourly.csv"
TOP_SHORTAGE_DAYS_OUTPUT_10TO18 = OUTPUT_DIR / "top_shortage_days_hourly_10to18.csv"
TOP_SHORTAGE_DAYS_N = 7
OPERATING_HOUR_RANGE = (10, 18)   # ESS 방전 운영 가능 시간대


# ----------------------------------------------------------------------
# 단기예보 API (미래 기온 예보용 — weather.py의 과거관측 API와 다름)
# ----------------------------------------------------------------------
KMA_FORECAST_URL = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"

# 단기예보는 지점번호가 아니라 격자좌표(nx, ny)를 씁니다.
# 아래 값은 일반적으로 알려진 값이며, 정확도가 중요하면
# 기상청 "격자좌표 변환" 페이지에서 직접 재확인 권장:
# https://www.kma.go.kr/DFSROOT/POINT/DATA/latlon_dfs_grid.zip 등
WEATHER_FORECAST_GRID = {
    "108": {"name": "서울", "nx": 60, "ny": 127},
    "112": {"name": "인천", "nx": 55, "ny": 124},
    "119": {"name": "수원", "nx": 60, "ny": 121},
}

FORECAST_WEATHER_PATH = DATA_DIR / "capital_weather_forecast.csv"


# ----------------------------------------------------------------------
# 최신 실측 수요 API (data.go.kr, KPX 오늘전력수급현황조회)
# ----------------------------------------------------------------------
KPX_LATEST_DEMAND_URL = "https://openapi.kpx.or.kr/openapi/sukub5mToday/getSukub5mToday"
KPX_AUTH_KEY_ENV = "KPX_AUTH_KEY"
LATEST_DEMAND_PATH = DATA_DIR / "latest_national_demand.csv"