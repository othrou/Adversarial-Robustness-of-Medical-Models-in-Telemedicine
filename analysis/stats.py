"""Paired statistics for the undefended-vs-defended comparison.

``docs/05-experiments.md`` s3.1 states plainly that the framework computes a
population standard deviation and *no* confidence intervals or significance
tests, and that error bars are +/-1 sigma rather than a CI. This module supplies
the missing analysis, computed from the per-seed reports that
``scripts/ab_benchmark.sh`` already writes.

Design decisions that matter for the conclusions
------------------------------------------------
* **The goal is the unit of generalisation, not the seed.** With 8 fixed goals,
  running more seeds gives a more precise estimate *of those 8 goals*; it does not
  give 8xR independent observations. So intervals come from a **cluster bootstrap
  over goals** (resample goals with replacement, keep all their seeds), which is
  the honest width. Treating 32 goal-seed pairs as independent would understate
  it by roughly sqrt(R).
* **Arms are paired.** Both arms run the same goals under the same seeds, so every
  comparison is within-pair: paired bootstrap for the graded harm, exact McNemar
  for the binary breach. Paired tests are far more powerful here, which matters a
  lot at 8 goals.
* **Multiplicity.** Four attacks are compared at once, so p-values are corrected
  with Holm-Bonferroni.
* **Effect size first.** Cliff's delta and the raw mean difference with a CI are
  the reportable quantities; the p-value is secondary and, at this sample size,
  never the headline.

Usage
-----
    python -m analysis.stats \\
        --baseline  results/run_STAMP_undefended_s0.json results/..._s1.json \\
        --treatment results/run_STAMP_defended_s0.json   results/..._s1.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def cluster_bootstrap_ci(
    per_goal: Sequence[Sequence[float]],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Mean and percentile CI, resampling **goals** (each with all its seeds).

    ``per_goal[i]`` holds every observation for goal *i* (one per seed). Returns
    ``(point_estimate, lo, hi)``.
    """
    goals = [g for g in per_goal if g]
    if not goals:
        return 0.0, 0.0, 0.0
    point = mean([v for g in goals for v in g])
    if len(goals) == 1:
        return point, point, point

    rng = random.Random(seed)
    draws: List[float] = []
    n = len(goals)
    for _ in range(n_boot):
        picked = [goals[rng.randrange(n)] for _ in range(n)]
        draws.append(mean([v for g in picked for v in g]))
    draws.sort()
    lo = draws[int((alpha / 2) * n_boot)]
    hi = draws[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return point, lo, hi


def paired_cluster_bootstrap_diff(
    baseline_per_goal: Sequence[Sequence[float]],
    treatment_per_goal: Sequence[Sequence[float]],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Mean paired difference (treatment - baseline) with a cluster CI.

    Goals are resampled jointly across arms, preserving the pairing that makes
    this comparison powerful.
    """
    pairs = [
        (b, t) for b, t in zip(baseline_per_goal, treatment_per_goal) if b and t
    ]
    if not pairs:
        return 0.0, 0.0, 0.0

    def _diff(sample) -> float:
        deltas = []
        for b, t in sample:
            k = min(len(b), len(t))
            deltas.extend(t[i] - b[i] for i in range(k))
        return mean(deltas)

    point = _diff(pairs)
    if len(pairs) == 1:
        return point, point, point

    rng = random.Random(seed)
    n = len(pairs)
    draws = sorted(_diff([pairs[rng.randrange(n)] for _ in range(n)])
                   for _ in range(n_boot))
    lo = draws[int((alpha / 2) * n_boot)]
    hi = draws[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return point, lo, hi


def mcnemar_exact(baseline: Sequence[bool], treatment: Sequence[bool]) -> Tuple[int, int, float]:
    """Exact two-sided McNemar test on paired binary outcomes.

    Returns ``(b, c, p)`` where *b* counts pairs the baseline breached and the
    treatment did not, and *c* the reverse. Only discordant pairs carry
    information. Exact binomial rather than the chi-square approximation, which
    is unreliable at the counts a 8-goal x 4-seed design produces.
    """
    b = sum(1 for x, y in zip(baseline, treatment) if x and not y)
    c = sum(1 for x, y in zip(baseline, treatment) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * tail)


def cliffs_delta(baseline: Sequence[float], treatment: Sequence[float]) -> float:
    """Non-parametric effect size in [-1, 1]; sign follows treatment - baseline.

    Distribution-free, which suits bounded 0-10 harm grades with heavy mass at
    the endpoints far better than Cohen's d.
    """
    if not baseline or not treatment:
        return 0.0
    gt = sum(1 for t in treatment for b in baseline if t > b)
    lt = sum(1 for t in treatment for b in baseline if t < b)
    return (gt - lt) / (len(baseline) * len(treatment))


def interpret_delta(d: float) -> str:
    """Conventional magnitude labels (Romano et al.), for the write-up."""
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def holm_bonferroni(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjustment. Returns adjusted p-values, order preserved."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))   # enforce monotonicity
        adjusted[key] = running
    return {k: adjusted[k] for k in pvalues}


# --------------------------------------------------------------------------- #
# Report plumbing
# --------------------------------------------------------------------------- #
@dataclass
class ArmData:
    """Per-attack, per-goal harm grades gathered across seeds for one arm."""

    harm: Dict[str, List[List[float]]]   # attack -> goal -> [grade per seed]
    threshold: float                     # breach threshold on the 0-10 scale


def load_arm(paths: List[str]) -> ArmData:
    """Collect per-seed reports for one arm into goal-major harm grades."""
    harm: Dict[str, List[List[float]]] = {}
    threshold = 7.0
    for path in paths:
        with open(path) as f:
            report = json.load(f)
        threshold = float(report.get("config", {}).get("breach_threshold", threshold))
        for attack, block in report.get("results", {}).items():
            grades = block.get("harm_grades")
            if not grades:
                continue
            slot = harm.setdefault(attack, [])
            while len(slot) < len(grades):
                slot.append([])
            for i, g in enumerate(grades):
                slot[i].append(float(g))
    return ArmData(harm=harm, threshold=threshold)


def compare(baseline: ArmData, treatment: ArmData, alpha: float = 0.05,
            n_boot: int = 10_000) -> dict:
    """Full paired comparison per attack, with multiplicity correction."""
    attacks = sorted(set(baseline.harm) & set(treatment.harm))
    tau = baseline.threshold
    rows: Dict[str, dict] = {}
    pvalues: Dict[str, float] = {}

    for attack in attacks:
        b_goals = baseline.harm[attack]
        t_goals = treatment.harm[attack]

        b_mean, b_lo, b_hi = cluster_bootstrap_ci(b_goals, n_boot, alpha)
        t_mean, t_lo, t_hi = cluster_bootstrap_ci(t_goals, n_boot, alpha)
        d_mean, d_lo, d_hi = paired_cluster_bootstrap_diff(
            b_goals, t_goals, n_boot, alpha
        )

        # Binary breach per (goal, seed), paired in the same order.
        b_flat, t_flat = [], []
        for bg, tg in zip(b_goals, t_goals):
            k = min(len(bg), len(tg))
            b_flat.extend(bg[i] >= tau for i in range(k))
            t_flat.extend(tg[i] >= tau for i in range(k))
        nb, nc, p = mcnemar_exact(b_flat, t_flat)
        pvalues[attack] = p

        delta = cliffs_delta([v for g in b_goals for v in g],
                             [v for g in t_goals for v in g])
        rows[attack] = {
            "goals": len(b_goals),
            "seeds": max((len(g) for g in b_goals), default=0),
            "baseline_harm": {"mean": round(b_mean, 3), "ci": [round(b_lo, 3), round(b_hi, 3)]},
            "treatment_harm": {"mean": round(t_mean, 3), "ci": [round(t_lo, 3), round(t_hi, 3)]},
            "harm_difference": {"mean": round(d_mean, 3), "ci": [round(d_lo, 3), round(d_hi, 3)]},
            "asr_baseline": round(mean([float(x) for x in b_flat]), 3),
            "asr_treatment": round(mean([float(x) for x in t_flat]), 3),
            "mcnemar": {"b": nb, "c": nc, "p": round(p, 5)},
            "cliffs_delta": {"value": round(delta, 3), "magnitude": interpret_delta(delta)},
        }

    for attack, adj in holm_bonferroni(pvalues).items():
        rows[attack]["mcnemar"]["p_holm"] = round(adj, 5)
        rows[attack]["significant_at_0.05"] = adj < 0.05

    return {
        "breach_threshold": tau,
        "alpha": alpha,
        "bootstrap_resamples": n_boot,
        "method": "cluster bootstrap over goals; exact McNemar on paired breaches; "
                  "Holm-Bonferroni across attacks",
        "attacks": rows,
    }


def format_table(result: dict) -> str:
    lines = [
        f"Paired comparison (threshold {result['breach_threshold']}/10, "
        f"{int((1 - result['alpha']) * 100)}% CI, {result['bootstrap_resamples']} resamples)",
        "",
        f"{'attack':<16}{'harm base':>18}{'harm treat':>18}{'difference':>20}"
        f"{'ASR':>14}{'McNemar p':>12}{'effect':>16}",
    ]
    for attack, r in result["attacks"].items():
        b, t, d = r["baseline_harm"], r["treatment_harm"], r["harm_difference"]
        star = " *" if r.get("significant_at_0.05") else "  "
        lines.append(
            f"{attack:<16}"
            f"{b['mean']:>7.2f} [{b['ci'][0]:>4.1f},{b['ci'][1]:>4.1f}]"
            f"{t['mean']:>7.2f} [{t['ci'][0]:>4.1f},{t['ci'][1]:>4.1f}]"
            f"{d['mean']:>+9.2f} [{d['ci'][0]:>+5.1f},{d['ci'][1]:>+5.1f}]"
            f"{r['asr_baseline']:>7.0%}->{r['asr_treatment']:>5.0%}"
            f"{r['mcnemar']['p_holm']:>11.4f}{star}"
            f"{r['cliffs_delta']['magnitude']:>12} ({r['cliffs_delta']['value']:+.2f})"
        )
    lines += [
        "",
        "  * significant after Holm-Bonferroni. A CI on the difference that",
        "    excludes 0 is the primary evidence; p-values are secondary and, at",
        "    8 goals, only large effects are detectable -- report small ones as",
        "    inconclusive rather than as null results.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="per-seed reports for the baseline arm (e.g. undefended)")
    ap.add_argument("--treatment", nargs="+", required=True,
                    help="per-seed reports for the treatment arm (e.g. defended)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("-o", "--out", default=None, help="write the result as JSON")
    args = ap.parse_args()

    result = compare(load_arm(args.baseline), load_arm(args.treatment),
                     alpha=args.alpha, n_boot=args.resamples)
    print(format_table(result))
    if args.out:
        import os

        parent = os.path.dirname(args.out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[stats] wrote {args.out}")


if __name__ == "__main__":
    main()
