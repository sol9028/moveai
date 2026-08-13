from __future__ import annotations

import json
from pathlib import Path

from agent import ETrainAgent
from simulation import build_demo_state


def main() -> dict:
    state = build_demo_state()
    result = ETrainAgent(scenario_count=12, seed=42).run_once(state).to_dict()

    print("\n=== E-TRAIN RE:LOOP · AI LOOP MVP ===")
    print(json.dumps({
        "selected_policy": result["selected"]["policy"],
        "selected_plan": result["selected"]["plan"],
        "scenario_summary": result["selected"]["scenario_summary"],
        "briefing": result["briefing"],
    }, ensure_ascii=False, indent=2))

    output = Path(__file__).parent / "dashboard" / "latest_result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
