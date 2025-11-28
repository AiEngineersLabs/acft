from __future__ import annotations

from typing import List, Dict, Any, Tuple


def run_stability_eval(
    engine,
    dataset: List[Dict[str, Any]],
    debug: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run ACFT over a dataset of prompts and summarize stability.

    dataset: list of {"prompt": "..."} dicts
    engine: any object with .run(prompt, debug=...) returning ACFTResult

    Returns:
      summary: aggregated metrics (avg stability, decision counts, etc.)
      details: per-prompt metrics and decisions
    """
    details: List[Dict[str, Any]] = []

    for item in dataset:
        prompt = item["prompt"]
        result = engine.run(prompt, debug=debug)
        details.append(
            {
                "prompt": prompt,
                "decision": result.decision,
                "stability": result.metrics.stability,
                "grad_norm": result.metrics.grad_norm,
                "osc_norm": result.metrics.osc_norm,
            }
        )

    if not details:
        return {
            "count": 0,
            "avg_stability": 0.0,
            "low_stability_fraction": 0.0,
            "decisions_count": {},
        }, details

    count = len(details)
    avg_stability = sum(d["stability"] for d in details) / count
    low_stability_fraction = sum(
        1 for d in details if d["stability"] < 0.5
    ) / count

    decisions_count: Dict[str, int] = {}
    for d in details:
        decisions_count[d["decision"]] = decisions_count.get(d["decision"], 0) + 1

    summary = {
        "count": count,
        "avg_stability": avg_stability,
        "low_stability_fraction": low_stability_fraction,
        "decisions_count": decisions_count,
    }

    return summary, details