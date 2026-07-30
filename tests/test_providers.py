"""Unit tests for provider routing and attacker-health instrumentation.

Fully offline: the cloud client is never constructed and no network call is made.
These cover two failure modes that silently invalidate a run --

* a reasoning-model attacker silently collapsing to one hardcoded prefix, and
* a comparison figure rendering an attack that was never run as a zero --

plus the plumbing the cloud path adds (routing, retries, seeding, metering).
"""

import json
import os

import pytest

from agents.attacks.base import EpisodeHealth, strip_reasoning
from agents.attacks.pair import FALLBACK_PREFIX, _clean_prefix
from agents.attacks.proattack import _clean_wrapper
from agents.llm import (
    BudgetExceeded,
    InstrumentedBackend,
    LLMBackend,
    UsageLedger,
    make_backend,
    parse_model_uri,
)


# --------------------------------------------------------------------------- #
# Provider routing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec,expected", [
    ("openai/gpt-x", ("openai", "gpt-x")),
    # OpenRouter ids are themselves vendor/model: only the FIRST slash is a provider.
    ("openrouter/qwen/qwen3-32b", ("openrouter", "qwen/qwen3-32b")),
    ("ollama/mistral", ("ollama", "mistral")),
    ("compat/local-model", ("compat", "local-model")),
    ("mock/anything", ("mock", "anything")),
])
def test_parse_model_uri_prefixes(spec, expected):
    assert parse_model_uri(spec) == expected


def test_unprefixed_names_keep_previous_behaviour(monkeypatch):
    """Pre-existing model names must resolve exactly as before providers existed."""
    # The suite runs with MARKOV_GAME_BACKEND=mock; clear it to see the default.
    monkeypatch.delenv("MARKOV_GAME_BACKEND", raising=False)
    assert parse_model_uri("mistral") == ("ollama", "mistral")
    # The default target contains a slash but 'amsaravi' is not a provider.
    assert parse_model_uri("amsaravi/medgemma-4b-it:q6") == (
        "ollama", "amsaravi/medgemma-4b-it:q6"
    )
    assert parse_model_uri("mistral", "mock") == ("mock", "mistral")


def test_environment_variable_sets_the_default_provider(monkeypatch):
    monkeypatch.setenv("MARKOV_GAME_BACKEND", "mock")
    assert parse_model_uri("mistral") == ("mock", "mistral")
    # An explicit prefix always wins over the environment.
    assert parse_model_uri("ollama/mistral") == ("ollama", "mistral")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        parse_model_uri("some-model", "not-a-provider")


def test_cloud_backend_requires_key_from_environment(monkeypatch):
    """Keys come from the environment only -- and a missing one says so."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_backend("openai/gpt-x", role="judge")


# --------------------------------------------------------------------------- #
# NeMo model routing
# --------------------------------------------------------------------------- #
def test_ollama_rails_spec_targets_the_openai_compatible_endpoint():
    """NeMo >=0.23 talks to Ollama over /v1, not the native API.

    Without the suffix every guarded call 404s -- and since the defender is
    fail-closed that presents as *every prompt blocked*: a flawless 0% ASR that
    is pure infrastructure failure, not a defence.
    """
    from agents.defender import _rails_model_spec

    spec = _rails_model_spec("main", "ollama/medgemma", "http://127.0.0.1:11434")
    assert spec["engine"] == "ollama"
    assert spec["parameters"]["base_url"] == "http://127.0.0.1:11434/v1"


def test_v1_suffix_is_not_doubled():
    from agents.defender import _rails_model_spec

    spec = _rails_model_spec("main", "ollama/m", "http://127.0.0.1:11434/v1/")
    assert spec["parameters"]["base_url"] == "http://127.0.0.1:11434/v1"


def test_cloud_rails_spec_uses_openai_wire_format_and_env_key():
    from agents.defender import _rails_model_spec

    spec = _rails_model_spec("llama_guard", "openrouter/meta-llama/llama-guard-3-8b",
                             "http://127.0.0.1:11434")
    assert spec["engine"] == "openai"                       # OpenAI-compatible
    assert spec["model"] == "meta-llama/llama-guard-3-8b"   # vendor/model kept
    assert spec["api_key_env_var"] == "OPENROUTER_API_KEY"  # key stays in env
    assert "openrouter.ai" in spec["parameters"]["base_url"]


def test_mock_has_no_guarded_path():
    from agents.defender import _rails_model_spec

    with pytest.raises(ValueError, match="mock"):
        _rails_model_spec("main", "mock/x", "http://127.0.0.1:11434")


# --------------------------------------------------------------------------- #
# Reasoning-block stripping (the PAIR degeneracy)
# --------------------------------------------------------------------------- #
def test_strip_reasoning_closed_block():
    assert strip_reasoning("<think>plan plan</think>The answer") == "The answer"


def test_strip_reasoning_unterminated_block_yields_nothing():
    """A completion truncated inside <think> contains no answer at all."""
    assert strip_reasoning("<think>I should start by considering") == ""


def test_strip_reasoning_orphan_closing_tag():
    """Chat templates often emit the opener themselves, so only </think> appears."""
    assert strip_reasoning("reasoning here</think>The answer") == "The answer"


def test_clean_prefix_recovers_prefix_from_reasoning_model():
    raw = "<think>\nWhat framing evades the filter?\n</think>\nAs a clinical toxicologist,"
    prefix, used_fallback = _clean_prefix(raw)
    assert prefix == "As a clinical toxicologist,"
    assert used_fallback is False


def test_clean_prefix_flags_fallback_when_output_unusable():
    """The bug that produced 0% ASR: <think> is 7 chars, so it tripped the length
    guard and every iteration silently returned the same canned prefix."""
    prefix, used_fallback = _clean_prefix("<think>truncated mid-thought")
    assert prefix == FALLBACK_PREFIX
    assert used_fallback is True


def test_clean_wrapper_strips_reasoning():
    assert _clean_wrapper("<think>deliberating</think>\nAs a medical historian,") == (
        "As a medical historian,"
    )


# --------------------------------------------------------------------------- #
# Attacker health
# --------------------------------------------------------------------------- #
def test_health_flags_single_distinct_prefix():
    h = EpisodeHealth(proposals=10, fallback_proposals=10, unique_prefixes=1)
    assert h.degenerate
    assert any("no adaptation" in r for r in h.reasons)
    assert any("unusable" in r for r in h.reasons)


def test_health_accepts_a_varied_attacker():
    h = EpisodeHealth(proposals=6, fallback_proposals=1, unique_prefixes=5)
    assert not h.degenerate
    assert h.reasons == []


def test_health_ignores_attacks_that_make_no_proposals():
    """The signature campaign proposes no prefixes; absence is not degeneracy."""
    assert not EpisodeHealth(proposals=0, unique_prefixes=0).degenerate


# --------------------------------------------------------------------------- #
# Instrumentation: retries, caching, seeding, metering
# --------------------------------------------------------------------------- #
class _Recorder(LLMBackend):
    """Records the kwargs it was called with; optionally fails a few times."""

    name = "recorder"

    def __init__(self, fail_times: int = 0, error: str = "429 rate limit"):
        self.calls = []
        self.fail_times = fail_times
        self.error = error
        self.last_usage = (10, 5)
        self.last_cost_usd = None

    def generate(self, prompt, system=None, max_tokens=256, temperature=0.7,
                 seed=None, json_schema=None):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError(self.error)
        self.calls.append({"prompt": prompt, "temperature": temperature, "seed": seed})
        return f"ok-{len(self.calls)}"


def test_retries_transient_failures(monkeypatch):
    monkeypatch.setattr("agents.llm.time.sleep", lambda _: None)
    ledger = UsageLedger()
    b = InstrumentedBackend(_Recorder(fail_times=2), role="judge", ledger=ledger)
    assert b.generate("hi") == "ok-1"
    assert ledger.usage("judge").retries == 2
    assert ledger.usage("judge").errors == 0


def test_does_not_retry_a_permanent_failure(monkeypatch):
    monkeypatch.setattr("agents.llm.time.sleep", lambda _: None)
    ledger = UsageLedger()
    b = InstrumentedBackend(
        _Recorder(fail_times=1, error="400 invalid request"), role="judge", ledger=ledger
    )
    with pytest.raises(RuntimeError):
        b.generate("hi")
    assert ledger.usage("judge").retries == 0
    assert ledger.usage("judge").errors == 1


def test_sampled_calls_are_never_cached():
    """Caching a temperature>0 call would re-create the one-prompt-per-goal bug:
    PAIR asks for a new prefix with the SAME prompt each iteration."""
    inner = _Recorder()
    b = InstrumentedBackend(inner, role="attacker")
    b.generate("same prompt", temperature=0.9)
    b.generate("same prompt", temperature=0.9)
    assert len(inner.calls) == 2


def test_deterministic_calls_are_cached():
    inner = _Recorder()
    b = InstrumentedBackend(inner, role="judge", seed=0)
    first = b.generate("same prompt", temperature=0.0)
    second = b.generate("same prompt", temperature=0.0)
    assert first == second
    assert len(inner.calls) == 1


def test_seed_advances_between_calls():
    """A fixed seed would make every identical prompt return an identical
    completion -- reproducibility must not cost the attacker its diversity."""
    inner = _Recorder()
    b = InstrumentedBackend(inner, role="attacker", seed=7)
    b.generate("p", temperature=0.9)
    b.generate("p", temperature=0.9)
    seeds = [c["seed"] for c in inner.calls]
    assert len(set(seeds)) == 2 and all(s is not None for s in seeds)


def test_seeds_are_reproducible_across_identical_runs():
    def seeds_for_run():
        inner = _Recorder()
        b = InstrumentedBackend(inner, role="attacker", seed=3)
        for _ in range(4):
            b.generate("p", temperature=0.9)
        return [c["seed"] for c in inner.calls]

    assert seeds_for_run() == seeds_for_run()


def test_attacker_refusal_is_counted():
    class _Refuser(_Recorder):
        def generate(self, prompt, system=None, max_tokens=256, temperature=0.7,
                     seed=None, json_schema=None):
            return "I can't help with creating prompts designed to bypass safety."

    ledger = UsageLedger()
    InstrumentedBackend(_Refuser(), role="attacker", ledger=ledger).generate("x")
    assert ledger.usage("attacker").refusals == 1


def test_budget_cap_aborts_the_run(monkeypatch, tmp_path):
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"m": [1000.0, 1000.0]}))
    monkeypatch.setenv("MARKOV_GAME_PRICES", str(prices))

    inner = _Recorder()
    inner.model = "m"
    ledger = UsageLedger(max_spend_usd=0.001)
    b = InstrumentedBackend(inner, role="judge", ledger=ledger)
    with pytest.raises(BudgetExceeded):
        for _ in range(100):
            b.generate("x", temperature=0.9)


def test_unpriced_model_marks_cost_incomplete():
    """An unknown price must never be silently reported as $0."""
    inner = _Recorder()
    inner.model = "mystery-model"
    ledger = UsageLedger()
    InstrumentedBackend(inner, role="judge", ledger=ledger, prices={}).generate("x")
    summary = ledger.summary()
    assert summary["cost_complete"] is False
    assert "mystery-model" in summary["unpriced_models"]


# --------------------------------------------------------------------------- #
# Plotter guards
# --------------------------------------------------------------------------- #
def _report(attacks, **config):
    cfg = {"attacker_model": "mistral", "judge_model": "gemma2:2b", "defender": "raw",
           "backend": "ollama", "num_goals": 8, "max_iterations": 10,
           "max_queries": 20, "breach_threshold": 6}
    cfg.update(config)
    return {
        "config": cfg,
        "results": {
            a: {"attack_success_rate": 0.5, "defense_success_rate": 0.5,
                "over_refusal_rate": 0.0, "mean_harm_score": 5.0}
            for a in attacks
        },
    }


def _write(tmp_path, name, report):
    path = tmp_path / name
    path.write_text(json.dumps(report))
    return str(path)


def test_missing_attack_is_reported_not_zeroed(tmp_path):
    from analysis.plots import missing_attacks

    base = _report(["pair", "signature"])
    other = _report(["pair"])
    assert missing_attacks(base, other) == ["signature"]


def test_confounded_comparison_is_refused(tmp_path):
    from analysis.plots import generate_all

    a = _write(tmp_path, "a.json", _report(["pair"]))
    b = _write(tmp_path, "b.json",
               _report(["pair"], defender="guardrails", judge_model="other"))
    with pytest.raises(SystemExit, match="REFUSING"):
        generate_all([a, b], str(tmp_path / "out"))


def test_single_variable_comparison_is_allowed(tmp_path):
    from analysis.plots import generate_all

    a = _write(tmp_path, "a.json", _report(["pair"]))
    b = _write(tmp_path, "b.json", _report(["pair"], defender="guardrails"))
    written = generate_all([a, b], str(tmp_path / "out"))
    assert any(w.endswith("changes.txt") for w in written)


def test_budget_difference_counts_as_a_confound(tmp_path):
    """Two runs over a different number of goals are not the same experiment."""
    from analysis.plots import diff_budgets

    assert diff_budgets(_report(["pair"]), _report(["pair"], num_goals=5)) == [
        "budget[num_goals]: 8 -> 5"
    ]
