"""Generate statistical figures from Markov-game evaluation reports.

Reads one or more run reports (the JSON that ``simulation.py --report`` writes,
see ``results/eval_all_20260722.json``) and renders PNG figures with matplotlib.

Two modes:

* **Single run** (one report) -> per-attack figures:
    - ``rates``   : ASR / DSR / over-refusal bars (with std error bars if the run
                    used ``--repeats``);
    - ``returns`` : attacker vs defender cumulative return;
    - ``pii``     : the signature attack's PII leak / refusal / target-accuracy.

* **Benchmark comparison** (>=2 reports) -> one grouped bar chart per metric,
  with each run a series. Use this to see how a metric moves when you **change
  the model or the prompt** -- the report's ``config`` (attacker/judge model,
  defender, thresholds) labels each series, so a regression is visible at a glance.

Usage
-----
    # figures for a single run
    python -m analysis.plots results/eval_all_20260722.json --outdir results/figures

    # compare several runs (e.g. before/after a prompt change)
    python -m analysis.plots results/run_baseline.json results/run_newprompt.json \
        --outdir results/figures --labels baseline newprompt

Design: never reads anything but the report files passed on the command line, and
writes only into ``--outdir``; it imports nothing from the game loop, so drawing a
figure can never perturb an experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: works on a server with no display
import matplotlib.pyplot as plt

# The three headline rates every run reports, in [0, 1].
RATE_METRICS = ["attack_success_rate", "defense_success_rate", "over_refusal_rate"]
RATE_LABELS = {
    "attack_success_rate": "ASR",
    "defense_success_rate": "DSR",
    "over_refusal_rate": "Over-Refusal",
    "mean_harm_score": "Mean harm grade (0-10)",
}
RETURN_METRICS = ["attacker_return", "defender_return"]

# A colour-blind-safe qualitative palette (Wong 2011), reused across figures.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_report(path: str) -> dict:
    with open(path) as f:
        report = json.load(f)
    if "results" not in report:
        raise ValueError(f"{path}: not a Markov-game report (no 'results' key)")
    return report


def _std_of(attack_block: dict, metric: str) -> float:
    """Std for a metric if the run used --repeats, else 0."""
    return float(attack_block.get("stats", {}).get(metric, {}).get("std", 0.0) or 0.0)


def _label_for(report: dict, fallback: str) -> str:
    """Short human label for a run, from its config (models / defender)."""
    cfg = report.get("config", {})
    atk = cfg.get("attacker_model", "?")
    judge = cfg.get("judge_model", "?")
    dfn = cfg.get("defender", "?")
    return f"{fallback}\n{atk}|{judge}|{dfn}"


#: Budget knobs that must match for two runs to be comparable. Two runs over a
#: different number of goals are not the same experiment, however identical their
#: models are.
BUDGET_KEYS = ("num_goals", "max_iterations", "max_queries", "breach_threshold")


def diff_budgets(a: dict, b: dict) -> List[str]:
    """Budget//scope differences between two run configs."""
    ca, cb = a.get("config", {}), b.get("config", {})
    return [
        f"budget[{key}]: {ca.get(key)} -> {cb.get(key)}"
        for key in BUDGET_KEYS
        if ca.get(key) != cb.get(key)
    ]


def missing_attacks(baseline: dict, other: dict) -> List[str]:
    """Attacks present in the baseline but absent from ``other``.

    An absent attack is not a zero -- it is a hole. Without this, a missing
    ``signature`` block renders as "PII leaks fell from 100% to 0%", a result the
    attack is structurally incapable of producing.
    """
    return sorted(set(baseline.get("results", {})) - set(other.get("results", {})))


def confound_report(reports: List[dict], paths: List[str]) -> List[str]:
    """Every way run *k* differs from the baseline, as one flat list of lines."""
    lines: List[str] = []
    for report, path in zip(reports[1:], paths[1:]):
        tag = os.path.basename(path)
        diffs = diff_configs(reports[0], report) + diff_budgets(reports[0], report)
        gone = missing_attacks(reports[0], report)
        if gone:
            diffs.append(f"MISSING attacks (not measured, NOT zero): {', '.join(gone)}")
        if diffs:
            lines.extend(f"{tag}: {d}" for d in diffs)
        else:
            lines.append(
                f"{tag}: NO model/prompt change vs baseline "
                "(identical config -- comparison measures only run-to-run noise)"
            )
    return lines


def _confound_dimensions(a: dict, b: dict) -> List[str]:
    """Independent variables that differ between two runs.

    More than one means the comparison cannot attribute an effect to anything.
    """
    return diff_configs(a, b) + diff_budgets(a, b)


def diff_configs(a: dict, b: dict) -> List[str]:
    """Human-readable list of what changed between two run configs.

    Separates **model** changes (target/attacker/judge/victim/defender) from
    **prompt** changes (any guardrail config file whose hash differs). This is the
    mechanism that ensures a model-or-prompt change is caught: if two runs differ,
    exactly what differs is named, so a comparison is never mistaken for a like-for
    -like baseline.
    """
    ca, cb = a.get("config", {}), b.get("config", {})
    changes: List[str] = []
    for key in ("attacker_model", "judge_model", "victim_model", "defender", "backend"):
        if ca.get(key) != cb.get(key):
            changes.append(f"model[{key}]: {ca.get(key)} -> {cb.get(key)}")
    ga, gb = ca.get("guardrail_config", {}) or {}, cb.get("guardrail_config", {}) or {}
    for fname in sorted(set(ga) | set(gb)):
        if ga.get(fname) != gb.get(fname):
            changes.append(f"prompt[{fname}]: changed")
    return changes


# --------------------------------------------------------------------------- #
# Single-run figures
# --------------------------------------------------------------------------- #
def plot_rates(report: dict, out: str) -> str:
    """Grouped ASR/DSR/over-refusal bars per attack, with std error bars."""
    results = report["results"]
    attacks = list(results)
    x = range(len(attacks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(attacks) + 2), 4.5))
    for j, metric in enumerate(RATE_METRICS):
        means = [results[a].get(metric, 0.0) for a in attacks]
        errs = [_std_of(results[a], metric) for a in attacks]
        offs = [i + (j - 1) * width for i in x]
        ax.bar(offs, means, width, yerr=errs, capsize=3,
               label=RATE_LABELS[metric], color=PALETTE[j])

    ax.set_xticks(list(x))
    ax.set_xticklabels(attacks, rotation=0)
    # Headroom above 1.0 so the legend sits in an empty band, never on a full-
    # height (ASR=1.0) bar.
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("rate")
    ax.set_title(_title("Attack vs Defense rates", report))
    ax.legend(loc="upper center", ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def plot_returns(report: dict, out: str) -> str:
    """Attacker vs defender cumulative return per attack (zero-sum view)."""
    results = report["results"]
    attacks = list(results)
    x = range(len(attacks))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(attacks) + 2), 4.5))
    for j, metric in enumerate(RETURN_METRICS):
        means = [results[a].get(metric, 0.0) for a in attacks]
        errs = [_std_of(results[a], metric) for a in attacks]
        offs = [i + (j - 0.5) * width for i in x]
        ax.bar(offs, means, width, yerr=errs, capsize=3,
               label=metric.replace("_", " "), color=PALETTE[j])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(attacks)
    ax.set_ylabel("cumulative reward")
    ax.set_title(_title("Agent returns", report))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def plot_pii(report: dict, out: str) -> Optional[str]:
    """The signature attack's PII campaign metrics (mean +/- std over repeats)."""
    sig = report["results"].get("signature")
    if not sig or "pii_metrics" not in sig:
        return None
    m = sig["pii_metrics"]
    keys = ["attack_success_rate", "refusal_rate", "target_accuracy"]
    labels = ["PII leak ASR", "refusal rate", "target accuracy"]
    means = [m.get(k, {}).get("mean", 0.0) for k in keys]
    errs = [m.get(k, {}).get("std", 0.0) for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, means, yerr=errs, capsize=4, color=PALETTE[:3])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    reps = m.get("attack_success_rate", {}).get("repeats", 1)
    ax.set_title(_title(f"Signature PII campaign (n={reps})", report))
    for i, v in enumerate(means):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def plot_harm_grades(report: dict, out: str) -> str:
    """Mean judge harm grade (0-10) per attack, with std error bars.

    The graded companion to the binary ASR bar: 0 = fully refused, 10 = clearly
    harmful. Shows *how bad* leaks were, not just whether one happened.
    """
    results = report["results"]
    attacks = list(results)
    means = [results[a].get("mean_harm_score", 0.0) for a in attacks]
    errs = [_std_of(results[a], "mean_harm_score") for a in attacks]

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(attacks) + 2), 4.5))
    ax.bar(attacks, means, yerr=errs, capsize=4, color=PALETTE[1])
    ax.set_ylim(0, 10.5)
    ax.set_ylabel("mean harm grade (0-10)")
    ax.set_title(_title("Judge harm grade per attack", report))
    for i, v in enumerate(means):
        ax.text(i, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def plot_harm_heatmap(report: dict, out: str) -> Optional[str]:
    """Attack x goal heatmap of per-goal harm grades (0-10)."""
    results = report["results"]
    attacks = [a for a in results if results[a].get("harm_grades")]
    if not attacks:
        return None
    ngoals = max(len(results[a]["harm_grades"]) for a in attacks)
    mat = []
    for a in attacks:
        row = list(results[a]["harm_grades"])
        row += [float("nan")] * (ngoals - len(row))
        mat.append(row)

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * ngoals + 2), 0.7 * len(attacks) + 2))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=10, aspect="auto")
    ax.set_xticks(range(ngoals))
    ax.set_xticklabels([f"g{i+1}" for i in range(ngoals)])
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels(attacks)
    for r in range(len(attacks)):
        for c in range(ngoals):
            v = mat[r][c]
            if v == v:  # not NaN
                ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                        color="black" if v < 6 else "white", fontsize=8)
    fig.colorbar(im, ax=ax, label="harm grade (0-10)")
    ax.set_title(_title("Per-goal harm grade (attack x goal)", report))
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# Cross-run benchmark comparison
# --------------------------------------------------------------------------- #
def plot_benchmark(reports: List[dict], labels: List[str], metric: str, out: str,
                   subtitle: str = "") -> str:
    """One grouped bar chart: metric per attack, one series per run.

    This is the "did my model/prompt change help?" figure -- put the runs you
    want to compare side by side and read off the delta per attack. ``subtitle``
    (from :func:`diff_configs`) names what actually changed between runs.
    """
    attacks = sorted({a for r in reports for a in r["results"]})
    x = range(len(attacks))
    n = len(reports)
    width = 0.8 / max(n, 1)

    # Fix the ceiling up front: a "not measured" placeholder is drawn at full
    # height, so it must be known before any bar is placed.
    if metric in RATE_METRICS:
        top = 1.05
    elif metric == "mean_harm_score":
        top = 10.5
    else:
        vals = [r["results"].get(a, {}).get(metric, 0.0) for r in reports for a in attacks]
        top = max([*vals, 0.0]) * 1.15 or 1.0

    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(attacks) + 2), 4.8))
    for k, (report, label) in enumerate(zip(reports, labels)):
        res = report["results"]
        offs = [i + (k - (n - 1) / 2) * width for i in x]
        # An attack a run never measured must NOT become a zero bar -- that
        # reads as "the metric fell to zero" for something never run.
        present = [i for i, a in enumerate(attacks) if metric in res.get(a, {})]
        absent = [i for i in range(len(attacks)) if i not in present]
        ax.bar([offs[i] for i in present],
               [res[attacks[i]][metric] for i in present],
               width,
               yerr=[_std_of(res[attacks[i]], metric) for i in present],
               capsize=3, label=label, color=PALETTE[k % len(PALETTE)])
        # A hatched, unfilled full-height slot reads as a hole in the data --
        # visually the opposite of a bar at zero.
        for i in absent:
            ax.bar([offs[i]], [top], width, facecolor="none", edgecolor="0.65",
                   hatch="///", linewidth=0.8, zorder=0)
            ax.text(offs[i], top * 0.5, "not measured", rotation=90, ha="center",
                    va="center", fontsize=7, color="0.3", fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(attacks)
    ax.set_ylim(0, top)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel(metric.replace("_", " "))
    title = f"Benchmark comparison -- {RATE_LABELS.get(metric, metric)}"
    if subtitle:
        title += "\nchanged: " + _wrap(subtitle)
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _wrap(text: str, width: int = 95, max_lines: int = 3) -> str:
    """Wrap a long "what changed" subtitle so it stays inside the figure."""
    import textwrap

    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += f"  (+{len(textwrap.wrap(text, width=width)) - max_lines} more, see changes.txt)"
    return "\n".join(lines)


def _title(base: str, report: dict) -> str:
    cfg = report.get("config", {})
    reps = cfg.get("repeats", 1)
    tag = f"{cfg.get('attacker_model', '?')} vs {cfg.get('defender', '?')}"
    if reps and reps > 1:
        tag += f"  (n={reps})"
    return f"{base}\n{tag}"


def _save(fig, out: str) -> str:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def generate_all(paths: List[str], outdir: str, labels: Optional[List[str]] = None,
                 allow_confounded: bool = False) -> List[str]:
    """Produce every applicable figure for the given report(s)."""
    reports = [load_report(p) for p in paths]
    written: List[str] = []

    if len(reports) == 1:
        r = reports[0]
        written.append(plot_rates(r, os.path.join(outdir, "rates.png")))
        written.append(plot_harm_grades(r, os.path.join(outdir, "harm_grades.png")))
        heat = plot_harm_heatmap(r, os.path.join(outdir, "harm_heatmap.png"))
        if heat:
            written.append(heat)
        written.append(plot_returns(r, os.path.join(outdir, "returns.png")))
        pii = plot_pii(r, os.path.join(outdir, "signature_pii.png"))
        if pii:
            written.append(pii)
    else:
        # Detect and report what changed between the first run (baseline) and each
        # later run, so a model/prompt change is explicit, not implicit.
        change_lines = confound_report(reports, paths)
        print("[plots] config changes vs baseline:")
        for line in change_lines:
            print(f"  - {line}")

        # A caller-supplied label can otherwise overwrite the recorded config --
        # e.g. labelling a run that changed attacker, judge AND defender as
        # "Guarded-MedGemma", which reads as a clean guardrail ablation.
        labels = [
            _label_for(r, lab)
            for r, lab in zip(
                reports,
                labels or [os.path.splitext(os.path.basename(p))[0] for p in paths],
            )
        ]

        confounded = {
            os.path.basename(p): dims
            for r, p in zip(reports[1:], paths[1:])
            if len(dims := _confound_dimensions(reports[0], r)) > 1
        }
        if confounded and not allow_confounded:
            sys.stdout.flush()   # keep the change list above the refusal
            detail = "\n".join(
                f"  {tag}: {len(dims)} variables changed at once\n"
                + "\n".join(f"    - {d}" for d in dims)
                for tag, dims in confounded.items()
            )
            raise SystemExit(
                "[plots] REFUSING to draw a confounded comparison.\n"
                f"{detail}\n"
                "  With more than one variable changed, no difference in these\n"
                "  figures can be attributed to any of them. Re-run holding all\n"
                "  but one fixed, or pass --allow-confounded if you understand\n"
                "  the figure is descriptive only."
            )

        subtitle = "; ".join(_confound_dimensions(reports[0], reports[1])) if len(reports) == 2 else ""
        for metric in RATE_METRICS + ["mean_harm_score"]:
            written.append(
                plot_benchmark(reports, labels, metric,
                               os.path.join(outdir, f"cmp_{metric}.png"),
                               subtitle=subtitle)
            )
        # Persist the change summary next to the figures for the record.
        summary_path = os.path.join(outdir, "changes.txt")
        os.makedirs(outdir, exist_ok=True)
        with open(summary_path, "w") as f:
            f.write("Config changes vs baseline (%s):\n" % os.path.basename(paths[0]))
            f.write("\n".join(change_lines) + "\n")
        written.append(summary_path)
    return written


def parse_args():
    p = argparse.ArgumentParser(description="Plot Markov-game evaluation reports")
    p.add_argument("reports", nargs="+", help="one or more results/*.json reports")
    p.add_argument("--outdir", default="results/figures", help="where to write PNGs")
    p.add_argument("--labels", nargs="*", default=None,
                   help="series labels for benchmark comparison (>=2 reports). "
                        "Always annotated with the recorded config -- a label "
                        "cannot override what the run actually was.")
    p.add_argument("--allow-confounded", action="store_true",
                   help="draw a comparison even when more than one variable "
                        "changed between runs. The figure is then descriptive "
                        "only: no effect can be attributed to any one change.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    written = generate_all(args.reports, args.outdir, args.labels,
                           allow_confounded=args.allow_confounded)
    print(f"[plots] wrote {len(written)} figure(s):")
    for w in written:
        print(f"  - {w}")


if __name__ == "__main__":
    main()
