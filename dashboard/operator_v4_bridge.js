(() => {
  // 이 파일은 operator_v4.html(디자인 목업)을 실제 Python AI Loop 결과와 연결한다.
  // plans/proposal은 원래 하드코딩된 값이었는데, 여기서 /api/day-schedule, /api/scenario를
  // 호출해서 실제 forecast/MILP 계산 결과로 그 자리를 채운다.

  async function loadDaySchedule() {
    try {
      const res = await fetch('/api/day-schedule', { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (typeof plans === 'undefined' || !Array.isArray(data.plans)) return;

      plans.length = 0;
      data.plans.forEach((p) => plans.push(p));

      if (typeof renderPlans === 'function') renderPlans();
      if (typeof renderNetwork === 'function') renderNetwork();
      if (typeof renderHistory === 'function') renderHistory();
      if (typeof renderCompare === 'function') renderCompare();

      if (typeof toast === 'function') {
        toast('실제 Python AI Loop 결과로 오늘 운행계획을 갱신했습니다.');
      }
    } catch (err) {
      console.error('day-schedule fetch failed:', err);
      if (typeof toast === 'function') {
        toast('백엔드 연결 실패: 디자인 목업 데이터로 표시 중입니다.');
      }
    }
  }

  async function calcProposalReal() {
    const curt = +byId('sCurt').value;
    const demand = +byId('sDemand').value;
    const cargoTon = +byId('sCargo').value;
    const delay = +byId('sDelay').value;

    try {
      const url = `/api/scenario?curt=${curt}&demand=${demand}&cargo_ton=${cargoTon}&delay=${delay}`;
      const res = await fetch(url, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const real = await res.json();

      // eslint-disable-next-line no-global-assign
      proposal = {
        energy: real.energy,
        ess: real.ess,
        cargo: real.cargo,
        time: real.time,
        delay: real.delay,
        cargoTon: cargoTon,
        curt: curt,
        demand: demand,
        profit: real.profit,
      };

      byId('riskStock').textContent = real.ess <= 9 ? '중간' : '낮음';
      byId('riskHonam').textContent = curt >= 25 && real.ess < 16 ? '중간' : '낮음';
      byId('trackStatus').textContent = delay ? `지연 +${delay}분` : '정상';
      byId('trackStatus').className = 'status ' + (delay >= 60 ? 'warn' : 'good');

      if (typeof renderCompare === 'function') renderCompare();

      byId('optProfit').textContent = real.profit;
      byId('optEnergy').textContent = real.energy + 'MWh';
      byId('optEss').textContent = real.ess + '량';
      byId('optCargo').textContent = real.cargo + '량';
      byId('optRisk').textContent = real.risk;

      if (typeof animateWorkflow === 'function') animateWorkflow('workflow');
      if (typeof runDecisionTrace === 'function') runDecisionTrace();

      setTimeout(
        () => byId('liveSimDetails')?.scrollIntoView({ behavior: 'smooth', block: 'center' }),
        120
      );
      if (typeof toast === 'function') {
        toast('실제 forecast/MILP 계산 결과로 후보안을 갱신했습니다.');
      }
    } catch (err) {
      console.error('scenario fetch failed:', err);
      if (typeof toast === 'function') toast('백엔드 계산 실패: 콘솔을 확인하세요.');
    }
  }

  // 페이지의 기존 inline <script>는 이 파일보다 먼저 실행되므로
  // plans/byId/renderPlans 등은 이미 전역에 정의돼 있다.
  loadDaySchedule();

  const runBtn = typeof byId === 'function' ? byId('scenarioRun') : null;
  if (runBtn) runBtn.onclick = calcProposalReal;
})();
