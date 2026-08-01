"""Game-theoretic analysis of a measured attacker-vs-defender payoff matrix.

The rest of the analysis package reports *marginal* metrics: ASR per attack, harm
per arm. That describes the experiment but never uses the Markov-game framing --
the game formulation stays decorative, and a reader is entitled to ask what it
bought. This module makes it load-bearing.

Every sweep measures one payoff per (attack strategy, defence configuration)
cell. That is exactly the normal-form matrix of the stage game the Markov game
reduces to when both players commit to a strategy for a whole episode. So the
standard solution concepts apply, computed from measured numbers rather than
assumed ones:

* **Dominance** -- an attack whose payoff beats another's in *every* defence
  column is strictly dominant; the dominated row is never played by a rational
  attacker, whatever the defender does.
* **Minimax / Nash value** -- for a two-player zero-sum matrix game the
  equilibrium is a pair of mixed strategies solvable by linear programming, and
  the value ``V`` is the ASR a rational attacker guarantees against the
  defender's best mixture.
* **Off-equilibrium effects** -- an ASR reduction on a row the equilibrium never
  plays does not lower ``V``. This is the distinction the marginal tables cannot
  express, and it is the point of the exercise.

ASR is the default payoff because it is unit-free and comparable across attacks.
Judge harm is **not** comparable: ``signature`` reports a programmatic severity
(``judge_scored = False``), so pooling it with judge grades mixes units. The
``--payoff harm`` mode exists for the appendix and warns when signature is in
the matrix.

Usage
-----
    python -m analysis.game \\
        --baseline  results/run_STAMP_undefended_s0.json results/..._s1.json \\
        --treatment results/run_STAMP_defended_s0.json   results/..._s1.json \\
        --labels raw guarded --outdir results/figures_game
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")   # headless, as in analysis.plots
import matplotlib.pyplot as plt
import numpy as np

#: Wong (2011) colour-blind-safe palette, shared with analysis.plots.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

#: Attacks whose ``best_harm`` is a structural severity rather than a judge
#: grade. Safe in an ASR matrix, never poolable in a harm matrix.
NON_JUDGE_SCORED = {"signature"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_arm(paths: Sequence[str]) -> Dict[str, dict]:
    """Average each attack's metrics across the per-seed reports of one arm.

    Per-seed reports are used rather than a merged one on purpose:
    ``analysis.merge`` drops ``asr_curve``, ``leak_rate`` and the validity
    fields, all of which this module reports.
    """
    acc: Dict[str, Dict[str, list]] = {}
    for path in paths:
        with open(path) as f:
            report = json.load(f)
        for attack, block in report.get("results", {}).items():
            slot = acc.setdefault(attack, {})
            for key in ("attack_success_rate", "leak_rate", "mean_harm_score",
                        "attacker_return", "defender_return"):
                if key in block:
                    slot.setdefault(key, []).append(float(block[key]))
            curve = block.get("asr_curve")
            if curve:
                slot.setdefault("asr_curve_seeds", []).append(
                    [float(curve[str(t)]) for t in range(1, 11)]
                )
            slot.setdefault("valid", []).append(bool(block.get("valid", True)))

    out: Dict[str, dict] = {}
    for attack, slot in acc.items():
        entry = {k: float(np.mean(v)) for k, v in slot.items()
                 if k not in ("asr_curve_seeds", "valid")}
        if "asr_curve_seeds" in slot:
            seeds = np.array(slot["asr_curve_seeds"])
            entry["asr_curve"] = seeds.mean(axis=0)
            # Keep the per-seed spread: at 2 seeds a mean line alone hides the
            # entire sample, and the band is the only honest error indication.
            entry["asr_curve_lo"] = seeds.min(axis=0)
            entry["asr_curve_hi"] = seeds.max(axis=0)
        entry["seeds"] = len(slot.get("valid", []))
        entry["all_valid"] = all(slot.get("valid", [True]))
        out[attack] = entry
    return out


def build_matrix(arms: List[Dict[str, dict]], metric: str
                 ) -> Tuple[List[str], np.ndarray]:
    """Attacker payoff matrix: rows = attacks, columns = defence arms."""
    attacks = sorted(set.intersection(*(set(a) for a in arms)))
    matrix = np.array([[arms[j][a][metric] for j in range(len(arms))]
                       for a in attacks], dtype=float)
    return attacks, matrix


# --------------------------------------------------------------------------- #
# Solution concepts
# --------------------------------------------------------------------------- #
def strict_dominance(attacks: Sequence[str], payoff: np.ndarray) -> Dict[str, list]:
    """Rows that strictly beat other rows in *every* column.

    A strictly dominant row makes the game trivial: the attacker plays it
    regardless of the defender, so the defender's choice cannot change the
    outcome. Reported explicitly because it is a far stronger statement than any
    single-cell ASR difference.
    """
    out: Dict[str, list] = {}
    for i, a in enumerate(attacks):
        dominated = [attacks[k] for k in range(len(attacks))
                     if k != i and np.all(payoff[i] > payoff[k])]
        if dominated:
            out[a] = dominated
    return out


def solve_zero_sum(payoff: np.ndarray) -> dict:
    """Nash equilibrium of a two-player zero-sum matrix game, by LP.

    The row player (attacker) maximises; the column player (defender) minimises
    the same quantity. Returns both mixed strategies and the value ``V``.

    A constant is added before solving and removed afterwards so the LP is posed
    on a strictly positive matrix, which keeps it bounded regardless of the
    payoff's sign convention.
    """
    from scipy.optimize import linprog

    m, n = payoff.shape
    shift = float(payoff.min()) - 1.0
    A = payoff - shift                       # strictly positive

    # Attacker: max v  s.t.  x^T A >= v (all columns), sum x = 1, x >= 0.
    c = np.zeros(m + 1)
    c[-1] = -1.0                             # linprog minimises
    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    bounds = [(0.0, 1.0)] * m + [(None, None)]
    row = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds)

    # Defender: min w  s.t.  A y <= w (all rows), sum y = 1, y >= 0.
    c2 = np.zeros(n + 1)
    c2[-1] = 1.0
    A_ub2 = np.hstack([A, -np.ones((m, 1))])
    b_ub2 = np.zeros(m)
    A_eq2 = np.zeros((1, n + 1))
    A_eq2[0, :n] = 1.0
    bounds2 = [(0.0, 1.0)] * n + [(None, None)]
    col = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=[1.0], bounds=bounds2)

    if not (row.success and col.success):
        raise RuntimeError(f"LP failed: {row.message} / {col.message}")

    return {
        "value": float(row.x[-1] + shift),
        "attacker_mix": [float(v) for v in np.round(row.x[:m], 6)],
        "defender_mix": [float(v) for v in np.round(col.x[:n], 6)],
    }


def analyse(attacks: List[str], payoff: np.ndarray, arm_labels: List[str],
            metric: str) -> dict:
    """Full solution: dominance, equilibrium, and the same with rows removed.

    The restricted solve is what separates "the guardrail does nothing" from
    "the guardrail does nothing *the equilibrium notices*": if a dominant row is
    excluded, the defender's choice can matter again, and the drop in ``V``
    quantifies by how much.
    """
    dominance = strict_dominance(attacks, payoff)
    full = solve_zero_sum(payoff)
    full["attacker_mix"] = dict(zip(attacks, full["attacker_mix"]))
    full["defender_mix"] = dict(zip(arm_labels, full["defender_mix"]))

    result = {
        "payoff_metric": metric,
        "arms": arm_labels,
        "attacks": attacks,
        "payoff_matrix": {a: dict(zip(arm_labels, map(float, payoff[i])))
                          for i, a in enumerate(attacks)},
        "strict_dominance": dominance,
        "equilibrium": full,
    }

    dominant = [a for a, beaten in dominance.items()
                if len(beaten) == len(attacks) - 1]
    result["dominant_strategy"] = dominant[0] if dominant else None

    if dominant:
        keep = [i for i, a in enumerate(attacks) if a not in dominant]
        if len(keep) >= 2:
            sub_attacks = [attacks[i] for i in keep]
            sub = solve_zero_sum(payoff[keep])
            sub["attacker_mix"] = dict(zip(sub_attacks, sub["attacker_mix"]))
            sub["defender_mix"] = dict(zip(arm_labels, sub["defender_mix"]))
            result["equilibrium_without_dominant"] = sub
            result["value_drop_if_dominant_removed"] = round(
                full["value"] - sub["value"], 4)
    return result


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_payoff_matrix(attacks, payoff, arm_labels, solution, out, metric):
    """Heatmap of the measured payoff matrix, annotated with the solution."""
    fig, ax = plt.subplots(figsize=(1.9 * len(arm_labels) + 4.2,
                                    0.62 * len(attacks) + 2.6))
    vmax = 1.0 if metric != "mean_harm_score" else 10.0
    im = ax.imshow(payoff, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")

    dominant = solution.get("dominant_strategy")
    for i, a in enumerate(attacks):
        for j in range(len(arm_labels)):
            v = payoff[i, j]
            txt = f"{v:.0%}" if metric != "mean_harm_score" else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    fontweight="bold" if a == dominant else "normal",
                    color="black" if v < 0.6 * vmax else "white")
        if a == dominant:
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(arm_labels), 1,
                                       fill=False, edgecolor="#0072B2",
                                       linewidth=3, zorder=5))

    ax.set_xticks(range(len(arm_labels)))
    ax.set_xticklabels(arm_labels)
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels([f"$\\bf{{{a}}}$" if a == dominant else a
                        for a in attacks])
    ax.set_xlabel("defender strategy")
    ax.set_ylabel("attacker strategy")
    fig.colorbar(im, ax=ax, label=metric.replace("_", " "))

    v = solution["equilibrium"]["value"]
    title = f"Measured payoff matrix — game value V = {v:.0%}" \
        if metric != "mean_harm_score" else \
        f"Measured payoff matrix — game value V = {v:.2f}"
    if dominant:
        title += f"\n'{dominant}' strictly dominates: defender's choice is off-equilibrium"
    ax.set_title(title, fontsize=10)
    return _save(fig, out)


def plot_asr_curves(arms, arm_labels, attacks, out, reported_tau: Optional[float] = None):
    """ASR as a function of the breach threshold tau, per attack, per arm.

    Removes the arbitrariness of a single ``--breach-threshold``: an effect that
    exists only at one tau is a threshold artifact, one that holds across the
    whole range is not. The data is already in every report -- nothing is re-run.

    Three things are drawn deliberately:

    * the **per-seed min-max band**, because at two seeds a mean line hides the
      entire sample and would imply a precision the run does not have;
    * a marker at the tau actually **reported**, so a reader can see where the
      headline number sits on the curve rather than taking it on trust;
    * overlapping arms are annotated, since two identical curves render as one
      line and silently look like a missing series.
    """
    have = [a for a in attacks if "asr_curve" in arms[0].get(a, {})]
    if not have:
        return None
    ncol = min(3, len(have))
    nrow = int(np.ceil(len(have) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.0 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    taus = np.arange(1, 11)

    for k, attack in enumerate(have):
        ax = axes[k // ncol][k % ncol]
        curves = []
        for j, (arm, label) in enumerate(zip(arms, arm_labels)):
            entry = arm.get(attack, {})
            curve = entry.get("asr_curve")
            if curve is None:
                continue
            curves.append(np.asarray(curve))
            colour = PALETTE[j % len(PALETTE)]
            lo, hi = entry.get("asr_curve_lo"), entry.get("asr_curve_hi")
            if lo is not None and hi is not None:
                ax.fill_between(taus, lo, hi, color=colour, alpha=0.18,
                                linewidth=0, zorder=1)
            ax.plot(taus, curve, marker="o", markersize=3.8, color=colour,
                    label=label, zorder=3)

        if reported_tau is not None:
            ax.axvline(reported_tau, color="0.35", linestyle=":", linewidth=1.2,
                       zorder=2)

        # Identical arms overlap into a single visible line; say so.
        if len(curves) == 2 and np.allclose(curves[0], curves[1]):
            ax.text(0.5, 0.5, "arms identical\nat every $\\tau$", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="0.25",
                    fontweight="bold")

        ax.set_title(attack, fontsize=10)
        ax.set_ylim(-0.04, 1.06)
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.grid(alpha=0.3)
        if k % ncol == 0:
            ax.set_ylabel("ASR")
        if k // ncol == nrow - 1:
            ax.set_xlabel(r"breach threshold $\tau$ (0–10)")
    for k in range(len(have), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if reported_tau is not None:
        handles.append(plt.Line2D([], [], color="0.35", linestyle=":", linewidth=1.2))
        labels.append(rf"reported $\tau$ = {reported_tau:g}")
    axes[0][0].legend(handles, labels, frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(r"ASR($\tau$) — is the guardrail effect a threshold artifact?"
                 "\nshaded band = per-seed min–max", fontsize=11)
    return _save(fig, out)


def plot_zero_sum_plane(arms, arm_labels, attacks, out):
    """Attacker return vs defender return -- how zero-sum the game really is.

    A strictly zero-sum game puts every point on the anti-diagonal. This game is
    only *near* zero-sum by construction: the attacker pays a per-query cost and
    the defender is scored on benign utility turns the attacker never plays. The
    vertical distance from the anti-diagonal is precisely that non-zero-sum
    surplus, i.e. the safety/utility term -- which is the reason "refuse
    everything" is not an optimal defence.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for j, (arm, label) in enumerate(zip(arms, arm_labels)):
        for i, attack in enumerate(attacks):
            e = arm.get(attack, {})
            if "attacker_return" not in e or "defender_return" not in e:
                continue
            ax.scatter(e["attacker_return"], e["defender_return"],
                       s=95, color=PALETTE[j % len(PALETTE)],
                       marker=markers[i % len(markers)],
                       edgecolor="black", linewidth=0.6, zorder=3,
                       label=attack if j == 0 else None)

    lo = min(ax.get_xlim()[0], -ax.get_ylim()[1])
    hi = max(ax.get_xlim()[1], -ax.get_ylim()[0])
    ax.plot([lo, hi], [-lo, -hi], "--", color="0.45", linewidth=1.2, zorder=1,
            label="strict zero sum")
    ax.axhline(0, color="black", linewidth=0.6, zorder=1)
    ax.axvline(0, color="black", linewidth=0.6, zorder=1)
    ax.set_xlabel("attacker return")
    ax.set_ylabel("defender return")
    ax.set_title("Near-zero-sum structure\n(distance from the dashed line = "
                 "utility/over-refusal surplus)", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    for j, label in enumerate(arm_labels):
        ax.scatter([], [], color=PALETTE[j % len(PALETTE)], s=95,
                   edgecolor="black", linewidth=0.6, label=label)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    return _save(fig, out)


def _save(fig, out: str) -> str:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_solution(solution: dict) -> str:
    metric = solution["payoff_metric"]
    pct = metric != "mean_harm_score"

    def fmt(v: float) -> str:
        return f"{v:.1%}" if pct else f"{v:.2f}"

    arms = solution["arms"]
    lines = [
        f"Payoff matrix ({metric}); rows = attacker, columns = defender",
        "",
        "  " + f"{'attack':<16}" + "".join(f"{a:>14}" for a in arms),
    ]
    for attack, row in solution["payoff_matrix"].items():
        lines.append("  " + f"{attack:<16}"
                     + "".join(f"{fmt(row[a]):>14}" for a in arms))

    eq = solution["equilibrium"]
    lines += ["", f"Nash equilibrium (zero-sum, LP):  V = {fmt(eq['value'])}"]
    for who, mix in (("attacker", eq["attacker_mix"]), ("defender", eq["defender_mix"])):
        played = {k: v for k, v in mix.items() if v > 1e-6}
        lines.append(f"  {who:<9} " + ", ".join(f"{k}={v:.3f}" for k, v in played.items()))

    dom = solution.get("dominant_strategy")
    if dom:
        lines += [
            "",
            f"'{dom}' STRICTLY DOMINATES every other attack in every defence column.",
            "  A rational attacker plays it whatever the defender does, so the",
            "  defender's configuration cannot change the outcome. Every ASR",
            "  reduction measured on the other rows is off the equilibrium path.",
        ]
        sub = solution.get("equilibrium_without_dominant")
        if sub:
            lines += [
                "",
                f"Restricted game (without '{dom}'):  V = {fmt(sub['value'])}"
                f"   [drop {fmt(solution['value_drop_if_dominant_removed'])}]",
            ]
            for who, mix in (("attacker", sub["attacker_mix"]),
                             ("defender", sub["defender_mix"])):
                played = {k: v for k, v in mix.items() if v > 1e-6}
                lines.append(f"  {who:<9} "
                             + ", ".join(f"{k}={v:.3f}" for k, v in played.items()))
            lines += [
                "  i.e. the guardrails DO lower the value of the game once the",
                "  dominant retrieval-layer attack is removed from the strategy set.",
            ]
    else:
        lines += ["", "No strictly dominant attack: the equilibrium is a genuine mixture."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="per-seed reports for the first defence arm")
    ap.add_argument("--treatment", nargs="+", required=True,
                    help="per-seed reports for the second defence arm")
    ap.add_argument("--labels", nargs=2, default=["raw", "guarded"])
    ap.add_argument("--payoff", default="attack_success_rate",
                    choices=["attack_success_rate", "leak_rate", "mean_harm_score"],
                    help="attacker payoff. ASR is unit-free and comparable across "
                         "attacks; harm is NOT (signature reports a programmatic "
                         "severity, not a judge grade).")
    ap.add_argument("--outdir", default="results/figures_game")
    ap.add_argument("-o", "--out", default=None, help="write the solution as JSON")
    args = ap.parse_args()

    arms = [load_arm(args.baseline), load_arm(args.treatment)]
    attacks, payoff = build_matrix(arms, args.payoff)

    invalid = [a for a in attacks if not all(arm[a]["all_valid"] for arm in arms)]
    if invalid:
        print(f"[game] WARNING: attacks with a degenerate attacker in some seed: "
              f"{', '.join(invalid)}. Their payoffs measure the attacker, not the "
              f"defence; the equilibrium below inherits that.")
    if args.payoff == "mean_harm_score" and NON_JUDGE_SCORED & set(attacks):
        print(f"[game] WARNING: {', '.join(sorted(NON_JUDGE_SCORED & set(attacks)))} "
              f"reports a programmatic severity, not a judge harm grade. This "
              f"matrix mixes units -- use --payoff attack_success_rate for the "
              f"headline result.")

    solution = analyse(attacks, payoff, args.labels, args.payoff)
    print(format_solution(solution))

    written = [
        plot_payoff_matrix(attacks, payoff, args.labels, solution,
                           os.path.join(args.outdir, "payoff_matrix.png"),
                           args.payoff),
        plot_zero_sum_plane(arms, args.labels, attacks,
                            os.path.join(args.outdir, "zero_sum_plane.png")),
    ]
    with open(args.baseline[0]) as f:
        reported_tau = json.load(f).get("config", {}).get("breach_threshold")
    curves = plot_asr_curves(arms, args.labels, attacks,
                             os.path.join(args.outdir, "asr_curves.png"),
                             reported_tau=reported_tau)
    if curves:
        written.append(curves)

    print(f"\n[game] wrote {len(written)} figure(s):")
    for w in written:
        print(f"  - {w}")

    if args.out:
        parent = os.path.dirname(args.out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(solution, f, indent=2)
        print(f"[game] wrote {args.out}")


if __name__ == "__main__":
    main()
