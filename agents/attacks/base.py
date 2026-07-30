"""Common interface and data records for attack strategies.

The game loop treats every attacker as an async callable that, given a single
adversarial ``goal`` and a ``defender``, runs its own inner optimisation
(iterative refinement, evolutionary search, bandit, ...) and returns an
:class:`AttackEpisode`. Each query to the defender is logged as a
:class:`TurnRecord` so the orchestrator can replay it through the reward model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # avoid import cycles at runtime
    from ..defender import Defender
    from ..judge import Judge


# --------------------------------------------------------------------------- #
# Attacker-output hygiene
# --------------------------------------------------------------------------- #
#: Tags reasoning models wrap their private chain-of-thought in. Their content is
#: never the answer, and on a truncated completion it may be *all* there is.
_REASONING_TAGS = ("think", "thinking", "reason", "reasoning", "scratchpad")
_TAG_ALT = "|".join(_REASONING_TAGS)


def strip_reasoning(text: str) -> str:
    """Remove a reasoning model's chain-of-thought from a completion.

    Without this, a reasoning attacker silently collapses: the cleaners take the
    first (or last) line, get ``<think>``, and fall through to their hardcoded
    fallback -- so every iteration sends the *same* prompt and the run reports a
    perfect defence against an attacker that never actually varied its attack.
    Three shapes are handled:

    * ``<think>...</think>answer`` -- drop the closed block;
    * ``...</think>answer`` -- opener eaten by the chat template, so the answer
      is whatever follows the final close;
    * ``<think>truncated`` -- the completion ran out of tokens inside the block,
      so there is no answer at all and the caller must treat it as unusable.
    """
    if not text:
        return ""
    out = re.sub(rf"<({_TAG_ALT})\b[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)

    closes = list(re.finditer(rf"</({_TAG_ALT})>", out, re.I))
    if closes:
        out = out[closes[-1].end():]

    opener = re.search(rf"<({_TAG_ALT})\b[^>]*>", out, re.I)
    if opener:
        out = out[: opener.start()]
    return out.strip()


@dataclass
class EpisodeHealth:
    """Did the *attacker* actually work during this episode?

    A red-team result is only meaningful if the attacker varied its attack. A
    degenerate attacker (censored by its provider, or truncated inside a
    reasoning block) produces 0% ASR that looks exactly like a strong defence.
    These counters make the difference visible, and :attr:`degenerate` turns it
    into a hard signal the orchestrator can refuse to score.
    """

    proposals: int = 0          # attacker-LLM calls that produced a prefix
    fallback_proposals: int = 0 # ...that were unusable, so a canned prefix was used
    unique_prefixes: int = 0    # distinct prefixes actually sent to the defender

    def note_proposal(self, used_fallback: bool) -> None:
        self.proposals += 1
        if used_fallback:
            self.fallback_proposals += 1

    @property
    def reasons(self) -> List[str]:
        """Why this episode's attacker is not trustworthy (empty = fine)."""
        out: List[str] = []
        if self.proposals >= 2 and self.unique_prefixes <= 1:
            out.append(
                f"attacker produced {self.unique_prefixes} distinct prefix over "
                f"{self.proposals} proposals (no adaptation)"
            )
        if self.proposals > 0 and self.fallback_proposals == self.proposals:
            out.append(
                f"every one of {self.proposals} attacker outputs was unusable "
                f"(refused, empty, or truncated inside a reasoning block)"
            )
        return out

    @property
    def degenerate(self) -> bool:
        return bool(self.reasons)

    def as_dict(self) -> dict:
        return {
            "proposals": self.proposals,
            "fallback_proposals": self.fallback_proposals,
            "unique_prefixes": self.unique_prefixes,
            "degenerate": self.degenerate,
            "reasons": self.reasons,
        }


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
    #: Whether the attacker itself functioned (see :class:`EpisodeHealth`).
    health: EpisodeHealth = field(default_factory=EpisodeHealth)

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

    # -- attacker health -------------------------------------------------
    def _start_episode(self, goal: str) -> AttackEpisode:
        """Create an episode and a fresh health tally for it."""
        episode = AttackEpisode(goal=goal, attack=self.name, success=False, best_harm=0.0)
        self._health = episode.health
        return episode

    def _note_proposal(self, used_fallback: bool) -> None:
        """Record one attacker-LLM proposal and whether it had to be replaced."""
        health = getattr(self, "_health", None)
        if health is not None:
            health.note_proposal(used_fallback)

    @staticmethod
    def _finish_episode(episode: AttackEpisode) -> AttackEpisode:
        """Fill in the counts that can only be known once the episode is over."""
        episode.health.unique_prefixes = len({t.prefix for t in episode.turns})
        return episode

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
