from awareml.explanation_integrity.faithfulness import (
    OllamaEvidenceExplanationGenerator,
)
from awareml.explanation_integrity.live_guided import (
    LiveGuidedOllamaGenerator,
)


def test_empirical_prompt_remains_v4_and_live_v3():
    assert (
        OllamaEvidenceExplanationGenerator.prompt_version
        == "phase15_explanation_prompt_v4"
    )
    assert (
        LiveGuidedOllamaGenerator.prompt_version
        == "phase15_live_guided_prompt_v3"
    )
