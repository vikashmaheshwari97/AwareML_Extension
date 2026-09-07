from awareml.explanation_integrity.benchmark import run_llm_phase15
from awareml.explanation_integrity.controlled import build_controlled_cases
from awareml.explanation_integrity.faithfulness import ControlledExplanationGenerator


class _AlwaysTimeout:
    source = "test-timeout"
    model = "fake-model"

    def generate(self, case):
        raise TimeoutError("synthetic timeout")


class _FailOnce:
    source = "test-flaky"
    model = "fake-model"

    def __init__(self):
        self.calls = 0
        self.base = ControlledExplanationGenerator()

    def generate(self, case):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("first call only")
        return self.base.generate(case)


def test_llm_runner_preserves_partial_failure_instead_of_crashing():
    cases = build_controlled_cases()[:2]
    result = run_llm_phase15(
        cases=cases,
        generator=_AlwaysTimeout(),
        retries=0,
    )

    assert result["complete"] is False
    assert result["failure_count"] == 2
    assert result["completed_correctness_count"] == 0
    assert len(result["failures"]) == 2


def test_llm_runner_retries_transient_generation_failure():
    case = build_controlled_cases()[:1]
    result = run_llm_phase15(
        cases=case,
        generator=_FailOnce(),
        retries=1,
    )

    assert result["complete"] is True
    assert result["failure_count"] == 0
    assert result["completed_correctness_count"] == 1
    assert result["completed_faithfulness_count"] == 1
