"""Post-hoc analysis of Markov-game evaluation runs.

The orchestrator (``simulation.py --report``) writes one JSON per run into
``results/``; :mod:`analysis.plots` turns those JSONs into statistical figures
(ASR/DSR, agent returns, PII leakage, and cross-run benchmark comparisons).

Kept strictly read-only w.r.t. ``results/`` and importing nothing from the game
loop, so plotting can never perturb an experiment.
"""
