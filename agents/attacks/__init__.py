"""Adversarial attack strategies for the Markov game.

Every attack implements the :class:`~agents.attacks.base.BaseAttacker`
interface, so the orchestrator can swap strategies without changing the game
loop. The registry below maps CLI names to classes.

Ported from the attack notebooks:
    pair      -> PAIR-style iterative prompt refinement (agents.attacks.pair)
    proattack -> evolutionary hill-climbing over wrappers (agents.attacks.proattack)
    rl        -> epsilon-greedy reward-guided bandit (agents.attacks.rl_bandit)
    signature -> signature-guided RAG PII extraction (agents.attacks.signature)
"""

from __future__ import annotations

from typing import Dict, Type

from .base import BaseAttacker, AttackEpisode, TurnRecord
from .pair import PAIRAttacker
from .proattack import ProAttacker
from .rl_bandit import RLBanditAttacker

REGISTRY: Dict[str, Type[BaseAttacker]] = {
    "pair": PAIRAttacker,
    "proattack": ProAttacker,
    "rl": RLBanditAttacker,
}

# The signature-guided attack pulls in heavy optional dependencies (chromadb,
# sentence-transformers, faker, scikit-learn, datasets). Register it only if
# those imports succeed, so the core game stays lightweight.
try:  # pragma: no cover - depends on optional extras being installed
    from .signature import SignatureGuidedAttacker

    REGISTRY["signature"] = SignatureGuidedAttacker
except Exception:  # noqa: BLE001 - optional extra not installed
    SignatureGuidedAttacker = None  # type: ignore


def available_attacks() -> list[str]:
    return sorted(REGISTRY)


def make_attacker(name: str, **kwargs) -> BaseAttacker:
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown attack '{name}'. Available: {', '.join(available_attacks())}"
        )
    return REGISTRY[key](**kwargs)


__all__ = [
    "BaseAttacker",
    "AttackEpisode",
    "TurnRecord",
    "PAIRAttacker",
    "ProAttacker",
    "RLBanditAttacker",
    "SignatureGuidedAttacker",
    "REGISTRY",
    "available_attacks",
    "make_attacker",
]
