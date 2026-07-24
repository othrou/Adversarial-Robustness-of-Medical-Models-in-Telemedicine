"""Merge several single-repeat run reports into one report with mean/std stats.

Running each seed as its own ``simulation.py --repeats 1 --report ...`` makes a long
sweep crash-safe: every seed's numbers hit disk the moment it finishes. This tool
then combines those per-seed reports into a single report in the *same* schema the
built-in ``--repeats N`` path emits -- each attack's headline metric becomes the
mean across seeds, with a ``stats`` block (mean/std/repeats) so ``analysis.plots``
can draw error bars, plus averaged ``pii_metrics`` for the signature attack.

    python -m analysis.merge results/run_s0.json results/run_s1.json results/run_s2.json \
        -o results/run_merged.json

Population std (divisor N), matching ``simulation.py`` -- see docs/05-experiments.md s3.1.
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Dict, List

# Headline scalar metrics carried on every attack result.
_SCALAR_KEYS = [
    "adversarial_turns", "benign_turns", "attack_success_rate",
    "defense_success_rate", "over_refusal_rate",
    "mean_harm_score", "max_harm_score",
    "attacker_return", "defender_return",
]


def _agg(values: List[float]) -> Dict:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {}
    return {
        "mean": round(statistics.mean(vals), 4),
        "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "repeats": len(vals),
    }


def merge_reports(paths: List[str]) -> dict:
    reports = []
    for p in paths:
        with open(p) as f:
            reports.append(json.load(f))

    attacks = sorted({a for r in reports for a in r.get("results", {})})
    merged_results: Dict[str, dict] = {}
    for atk in attacks:
        blocks = [r["results"][atk] for r in reports if atk in r.get("results", {})]
        entry: dict = {}
        stats: dict = {}
        for key in _SCALAR_KEYS:
            vals = [b[key] for b in blocks if key in b]
            if not vals:
                continue
            entry[key] = round(statistics.mean(vals), 4)
            stats[key] = _agg(vals)
        if stats:
            entry["stats"] = stats
        # Per-goal harm grades: average element-wise across seeds (goal order),
        # for the attack x goal harm heatmap.
        grade_lists = [b["harm_grades"] for b in blocks if isinstance(b.get("harm_grades"), list)]
        if grade_lists:
            n_goals = min(len(g) for g in grade_lists)
            entry["harm_grades"] = [
                round(statistics.mean(g[i] for g in grade_lists), 2) for i in range(n_goals)
            ]
        # Average the signature PII metrics (each stored as {metric: {mean,...}}).
        pii_blocks = [b["pii_metrics"] for b in blocks if "pii_metrics" in b]
        if pii_blocks:
            pii_keys = pii_blocks[0].keys()
            entry["pii_metrics"] = {
                k: _agg([pb[k]["mean"] for pb in pii_blocks if k in pb and "mean" in pb[k]])
                for k in pii_keys
            }
        merged_results[atk] = entry

    base_cfg = dict(reports[0].get("config", {}))
    base_cfg["repeats"] = len(reports)
    base_cfg["merged_from"] = [p.split("/")[-1] for p in paths]
    base_cfg["seeds"] = [r.get("config", {}).get("seed") for r in reports]
    return {
        "timestamp": reports[-1].get("timestamp"),
        "config": base_cfg,
        "results": merged_results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge per-seed reports into one stats report")
    ap.add_argument("reports", nargs="+", help="single-repeat report JSONs")
    ap.add_argument("-o", "--out", required=True, help="output merged report path")
    args = ap.parse_args()
    merged = merge_reports(args.reports)
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    n = len(args.reports)
    print(f"[merge] combined {n} report(s) -> {args.out}")
    for atk, block in merged["results"].items():
        asr = block.get("stats", {}).get("attack_success_rate", {})
        harm = block.get("stats", {}).get("mean_harm_score", {})
        print(f"  {atk:<10} ASR={block.get('attack_success_rate', 0):.3f} "
              f"(std {asr.get('std', 0):.3f})  harm={block.get('mean_harm_score', 0):.2f}/10 "
              f"(std {harm.get('std', 0):.2f}, n={asr.get('repeats', n)})")


if __name__ == "__main__":
    main()
