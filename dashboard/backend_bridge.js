(() => {
  const actions = document.querySelector('.actions');
  if (!actions) return;
  const btn = document.createElement('button');
  btn.className = 'btn primary';
  btn.id = 'backendRunBtn';
  btn.textContent = 'PYTHON AI LOOP';
  btn.title = 'Python Digital Twin → Forecast → MILP → Scenario Simulation → Decision 실행';
  actions.appendChild(btn);

  btn.addEventListener('click', async () => {
    const before = btn.textContent;
    btn.textContent = 'OPTIMIZING…';
    btn.disabled = true;
    try {
      const res = await fetch('/api/run-loop', { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const selected = data.selected;
      const p = selected.plan;
      const f = data.forecasts;
      const snap = data.state_snapshot;
      const sim = selected.scenario_summary;

      if (typeof state !== 'undefined') {
        state.curtailment = Math.round(f.curtailment.current_mwh);
        state.demand = Math.round(f.demand.current_mwh);
        state.honam = snap.regions.honam.ess_inventory;
        state.capital = snap.regions.metro.ess_inventory;
        state.max = snap.stations.METRO_MAIN.ess_capacity;
        state.need = selected.optimization.diagnostics.required_honam_ess_next;
        state.bm = Math.round(f.cargo.current_ton.black_mass || 0);
        state.other = Math.round((f.cargo.current_ton.used_ev_battery || 0) + (f.cargo.current_ton.manufacturing_scrap || 0));
        state.bmMaxCars = f.cargo.current_dispatch_limit_wagons?.black_mass ?? 4;
        state.source = 'PYTHON DIGITAL TWIN + MILP';
      }

      if (typeof plan !== 'undefined') {
        plan.ret = p.return_ess_count;
        plan.bmCars = p.cargo_wagons_by_type.black_mass || 0;
        plan.otherCars = (p.cargo_wagons_by_type.used_ev_battery || 0) + (p.cargo_wagons_by_type.manufacturing_scrap || 0);
        plan.cargoCars = plan.bmCars + plan.otherCars;
        plan.energy = Math.round(p.energy_target_mwh);
        plan.profit = `₩${selected.optimization.objective_million_krw.toFixed(1)}M`;
        const riskPct = Math.round(100 * Math.max(sim.honam_ess_shortage_probability, sim.capital_bottleneck_probability));
        plan.risk = `${riskPct <= 5 ? 'LOW' : riskPct <= 20 ? 'MEDIUM' : 'HIGH'} · ${riskPct}%`;
        plan.type = p.policy.toUpperCase();
        plan.depart = new Date(p.departure_time).toLocaleTimeString('ko-KR', {hour:'2-digit', minute:'2-digit', hour12:false});
      }

      if (typeof render === 'function') render();
      if (typeof addEvent === 'function') addEvent(`<b>Python MILP</b> ESS ${p.return_ess_count} + Cargo ${plan.cargoCars} (BM ${plan.bmCars}) 선택`, 'AI LOOP');
      if (typeof toast === 'function') toast(`<b>${p.policy}</b> 안을 실제 Python 최적화 결과로 반영했습니다.`);
      console.log('E-TRAIN RE:LOOP backend result', data);
    } catch (err) {
      console.error(err);
      if (typeof toast === 'function') toast('Python API 연결을 확인해주세요. <b>python -m api.server</b>');
    } finally {
      btn.textContent = before;
      btn.disabled = false;
    }
  });
})();
