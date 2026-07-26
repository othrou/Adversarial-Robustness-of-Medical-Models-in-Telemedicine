"""Common interface and data records for attack strategies.

The game loop treats every attacker as an async callable that, given a single
adversarial ``goal`` and a ``defender``, runs its own inner optimisation
(iterative refinement, evolutionary search, bandit, ...) and returns an
:class:`AttackEpisode`. Each query to the defender is logged as a
:class:`TurnRecord` so the orchestrator can replay it through the reward model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # avoid import cycles at runtime
    from ..defender import Defender
    from ..judge import Judge


@dataclass
class TurnRecord:
    """One query/response step inside an attack episode."""

    iteration: int
    prompt: str            # full adversarial prompt sent to the defender
    response: str          # defender's response
    blocked: bool          # did the defense refuse / filter it?
    harm: float            # normalised judge harm score in [0, 1]
    success: bool          # judge's breach flag for this step
    prefix: str = ""       # the adversarial framing added this step, if any
    feedback: str = ""     # judge's hint for the next attacker iteration

    @property
    def effective_harm(self) -> float:
        """Harm that actually got past the defence (0 if the turn was blocked).

        Used as the reward signal for search so a blocked refusal -- even one the
        judge mis-scores as harmful -- never rewards the attacker or promotes a
        prefix into the elite/memory set."""
        return 0.0 if self.blocked else self.harm


@dataclass
class AttackEpisode:
    """Result of attacking a single goal."""

    goal: str
    attack: str
    success: bool                       # was the goal ever breached?
    best_harm: float                    # best normalised harm score achieved
    turns: List[TurnRecord] = field(default_factory=list)
    best_prompt: str = ""
    queries_used: int = 0
    #: Optional attack-specific metrics (e.g. signature's PII ASR / refusal rate)
    #: that don't fit the jailbreak reward board but should still be reported.
    metrics: dict = field(default_factory=dict)

    @property
    def iterations(self) -> int:
        return len(self.turns)


class BaseAttacker:
    """Base class for all attack strategies.

    Subclasses receive a text-generation ``backend`` (the attacker LLM) and a
    ``judge``; they must implement :meth:`attack`.
    """

    #: CLI/registry name.
    name: str = "base"

    def __init__(self, backend=None, judge: "Judge" = None, max_iterations: int = 8,
                 max_queries: int = 40, verbose: bool = True, **kwargs):
        self.backend = backend
        self.judge = judge
        self.max_iterations = max_iterations
        self.max_queries = max_queries
        self.verbose = verbose

    async def attack(self, goal: str, defender: "Defender") -> AttackEpisode:
        raise NotImplementedError

    # -- shared helpers --------------------------------------------------
    async def _query_and_judge(
        self, defender: "Defender", prompt: str, iteration: int, prefix: str = ""
    ) -> TurnRecord:
        """Send one prompt through the defense and score the response."""
        out = await defender.respond(prompt)
        verdict = self.judge.evaluate(prompt, out.content)
        return TurnRecord(
            iteration=iteration,
            prompt=prompt,
            response=out.content,
            blocked=out.blocked,
            harm=verdict.normalized,
            success=verdict.success and not out.blocked,
            prefix=prefix,
            feedback=verdict.feedback,
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
