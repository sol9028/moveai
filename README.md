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

## 데모에서 확인되는 세 편성안

기본 가상 데이터에서는 정책 가중치에 따라 다음 해가 생성됩니다.

- **Balanced:** ESS 12 + Circular Cargo 8
- **Energy priority:** ESS 17 + Circular Cargo 3
- **Cargo priority:** ESS 8 + Circular Cargo 12

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
