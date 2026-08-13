# E-TRAIN RE:LOOP

**AI 기반 양방향 에너지·배터리 순환 철도 운영 최적화 MVP**

호남의 출력제어 재생에너지를 ESS 화차로 수도권에 운송하고, 하행에서는 방전된 ESS 회송과 블랙매스·사용후 EV 배터리·배터리 제조스크랩 운송을 같은 20량 편성 안에서 최적화합니다.

> 실제 데이터가 없는 현재 단계에서는 가상 데이터를 사용하되, 실데이터가 들어왔을 때 `상태 → 예측 → MILP → 시뮬레이션 → 의사결정 → 브리핑 → 재최적화`가 그대로 작동하도록 계산 구조를 분리했습니다.

## AI LOOP

1. **Sense** — `digital_twin.SystemState`가 ESS SOC, 재고, 화물 backlog, 열차/선로 상태를 읽습니다.
2. **Predict** — `forecast/`에서 출력제어·수도권 수요·Battery Circular Cargo를 예측합니다.
3. **Optimize** — `optimization/milp_v4.py`가 ESS 회송량, 화종별 화차 수, 에너지량, 추가 하행편 여부를 MILP로 계산합니다.
4. **Simulate** — `simulation/`이 지연·ESS 고장·수요/출력제어 변동을 Monte Carlo 시나리오로 검증합니다.
5. **Decide** — `agent/orchestrator.py`가 balanced / energy priority / cargo priority 대안을 위험조정 점수로 비교합니다.
6. **Explain** — 선택 이유와 버린 대안의 이유를 `agent/briefing.py`가 관제자용 문장으로 만듭니다.
7. **Re-optimize** — 실제 운영에서는 새로운 스냅샷마다 같은 루프를 Rolling Horizon으로 반복합니다.

## 프로젝트 구조

```text
re-etrain-loop/
├─ digital_twin/
│  ├─ state.py
│  ├─ train.py
│  ├─ ess.py
│  ├─ station.py
│  ├─ cargo.py
│  └─ rail.py
├─ simulation/
│  ├─ engine.py
│  ├─ events.py
│  └─ scenarios.py
├─ optimization/
│  └─ milp_v4.py
├─ forecast/
│  ├─ curtailment.py
│  ├─ demand.py
│  └─ cargo.py
├─ agent/
│  ├─ orchestrator.py
│  └─ briefing.py
├─ api/
│  └─ server.py
├─ dashboard/
│  ├─ index.html
│  ├─ backend_bridge.js
│  └─ digital_twin_reference.html
├─ tests/
│  └─ test_core.py
├─ run_mvp.py
└─ requirements.txt
```

## 두 기존 버전에서 가져온 장점

### 실행형 ZIP 버전에서 유지
- 실제 시간축 시뮬레이션 엔진
- 이벤트 큐와 선로 지연/고장 시나리오
- Monte Carlo 시나리오 반복 실행
- 테스트 가능한 프로젝트 구조
- 데모 실행 진입점

### 세밀한 digital twin 버전에서 강화
- ESS 화차 개별 SOC, 충·방전율, 위치, 고장, 회송 기준
- CargoBatch / CargoWagon / backlog 분리
- 역의 플랫폼·ESS dock·Cargo dock·충방전 용량 제약
- RailSegment 그래프, 다익스트라 경로, 시간대별 rail slot 예약
- Train은 객체 대신 wagon ID를 보유해 모듈 결합도를 낮춤

## MILP 결정변수

현재 MVP의 주요 결정변수는 다음과 같습니다.

- 하행 ESS 회송 화차 수
- `black_mass` 화차 수
- `used_ev_battery` 화차 수
- `manufacturing_scrap` 화차 수
- 추가 하행편 투입 여부(binary)
- 상행 에너지 운송량(MWh)
- 호남 ESS 부족 slack
- 수도권 ESS 버퍼 초과 slack
- 고SOC ESS 회송 slack

기본 하행은 20량이며 **추가 하행편은 화물수익을 더 얻기 위해서가 아니라, ESS 재고 병목을 20량으로 해소할 수 없을 때만 허용**합니다.

## 블랙매스 실제 운영 가정 반영

현재 MVP에는 새로 확인된 **블랙매스 주 2회 × 회당 약 4량** 조건을 공급 제약으로 반영했습니다.

- 화차 1량 적재용량 가정: **20t**
- 블랙매스 1회 출고 가능량: **최대 4량 = 80t**
- 주간 명목 최대량: **2회 × 80t = 160t/week**
- MILP는 현재 ready backlog가 많더라도 한 dispatch cycle에서 블랙매스를 4량보다 많이 선택하지 못합니다.
- 실제 출고 요일은 아직 확인되지 않았으므로 데모에서는 **월/목**을 사용합니다. 이 요일은 `simulation/scenarios.py`의 `CargoSupplySchedule` 설정값이며 실제 운영요일 확인 즉시 교체할 수 있습니다.
- 사용후 EV 배터리와 제조스크랩 물량은 실데이터가 아직 없으므로 기존 **가상값**입니다.

Digital Twin의 `CargoSupplySchedule`이 주기·회당 화차수·다음 공급시각을 보유하고, `forecast/cargo.py`가 이를 예측 입력으로 넘기며, `optimization/milp_v4.py`가 이를 화종별 상한으로 사용합니다.

## 데모에서 확인되는 세 편성안

현재 공급제약을 적용하면 정책별 해는 다음과 같습니다.

- **Balanced:** ESS 12 + Circular Cargo 8 = 20량  
  - Black Mass 4 + Used EV Battery 2 + Manufacturing Scrap 2
- **Energy priority:** ESS 17 + Circular Cargo 3 = 20량
- **Cargo priority:** ESS 8 + Circular Cargo 8 = **16량**

Cargo priority가 16량인 이유는 **운송할 화물이 실제로 8량만 준비되어 있는데 고SOC ESS까지 억지로 회송하거나 20량을 채우지 않도록** 모델링했기 때문입니다. 따라서 기존의 `ESS 8 + Black Mass 12` 같은 편성은 블랙매스만으로는 더 이상 나올 수 없습니다. 사용후배터리·제조스크랩 등 다른 순환자원이 실제로 충분할 때만 총 Cargo가 8량을 넘어갈 수 있습니다.

Agent는 세 안을 각각 미래 시나리오에서 돌린 뒤 기본 상태에서는 Balanced를 최종 선택합니다.

## 실행

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_mvp.py
```

테스트:

```bash
python -m unittest discover -s tests -v
```

대시보드 + Python API:

```bash
python -m api.server
```

브라우저에서 `http://127.0.0.1:8000`을 열고 상단 **PYTHON AI LOOP** 버튼을 누르면 브라우저의 기존 mock 계산이 아니라 Python Digital Twin → MILP → Scenario Simulation 결과가 UI에 반영됩니다.

## 실데이터 연결 시 교체할 부분

현재 `forecast/*`는 가상 예측기입니다. 실제 상용화 시 다음 데이터 소스로 대체하면 됩니다.

- 재생에너지 발전량 / 출력제어 예측
- 수도권 시간대별 전력수요
- 화차 GPS·SOC·충방전 상태
- 업체별 블랙매스/사용후 배터리/제조스크랩 출고 가능량
- 역 처리용량 및 재활용 거점 처리 가능량
- 열차/기관차 가용성
- 실시간 선로 장애·지연·운행 slot

`digital_twin`, `optimization`, `simulation`, `agent` 인터페이스는 그대로 유지하는 것이 목표입니다.
