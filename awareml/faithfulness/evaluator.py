from __future__ import annotations

from copy import deepcopy
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from awareml.llm.evidence import (
    EvidenceBundle,
    build_before_evidence,
)
from awareml.llm.goal_parser import (
    deterministic_goal_parse,
)
from awareml.recommender.v2_ranking import (
    rank_candidates,
)
from awareml.recommender.v2_service import (
    V2Recommender,
)

from .attribution import (
    objective_influence,
)
from .counterfactuals import (
    build_objective_counterfactual,
    build_sustainability_counterfactual,
)
from .metrics import (
    evidence_fidelity_score,
    set_distance,
    text_similarity,
)
from .rationale import (
    FaithfulRationaleGenerator,
)
from .schemas import (
    CounterfactualRecord,
    DatasetFaithfulnessResult,
)


class FaithfulnessEvaluator:
    """FaithLM/Serum-inspired evidence-intervention evaluator.

    The evaluator is deliberately model-agnostic. It tests whether a textual
    rationale changes when the *actual recommendation evidence* is changed.

    It does not claim access to Ollama attention weights or internal PE-LRP
    signals. Internal attribution-guided interventions remain a future
    transformer/HPC experiment.
    """

    def __init__(
        self,
        recommender: Optional[
            V2Recommender
        ] = None,
        rationale_generator: Optional[
            FaithfulRationaleGenerator
        ] = None,
    ):
        self.recommender = (
            recommender
            or V2Recommender()
        )
        self.generator = (
            rationale_generator
            or FaithfulRationaleGenerator()
        )

    def evaluate_profile(
        self,
        dataset_id: str,
        profile: Mapping[str, object],
        goal: str,
        weights: Optional[
            Mapping[str, float]
        ] = None,
        use_llm: bool = False,
        coverage: float = 0.90,
    ) -> DatasetFaithfulnessResult:
        interpretation = (
            deterministic_goal_parse(
                goal
            )
        )
        resolved_weights = (
            dict(weights)
            if weights is not None
            else (
                interpretation
                .primary_weights
                .normalized_dict()
            )
        )

        ranked, meta = (
            self.recommender
            .recommend_profile(
                dict(profile),
                weights=resolved_weights,
                ranking_mode="point",
                coverage=coverage,
            )
        )

        influence = objective_influence(
            ranked,
            weights=resolved_weights,
            objective_correlations=(
                self.recommender
                .manifest
                .get(
                    "objective_correlations"
                )
                or {}
            ),
        )

        evidence = build_before_evidence(
            interpretation,
            dict(profile),
            ranked,
            meta,
        )
        original = (
            self.generator.generate(
                evidence,
                ranked,
                influence,
                use_llm=use_llm,
            )
        )

        counterfactual_records = []

        scenarios = []

        for objective in [
            "accuracy",
            "runtime",
            "energy",
            "co2",
        ]:
            scenarios.append(
                build_objective_counterfactual(
                    ranked,
                    objective,
                    resolved_weights,
                    objective_correlations=(
                        self.recommender
                        .manifest
                        .get(
                            "objective_correlations"
                        )
                        or {}
                    ),
                )
            )

        scenarios.append(
            build_sustainability_counterfactual(
                ranked,
                resolved_weights,
                objective_correlations=(
                    self.recommender
                    .manifest
                    .get(
                        "objective_correlations"
                    )
                    or {}
                ),
            )
        )

        original_top = str(
            ranked.iloc[0][
                "framework"
            ]
        )

        for (
            cf_ranked,
            cf_meta,
            scenario_meta,
        ) in scenarios:
            cf_influence = objective_influence(
                cf_ranked,
                weights=resolved_weights,
                objective_correlations=(
                    self.recommender
                    .manifest
                    .get(
                        "objective_correlations"
                    )
                    or {}
                ),
            )

            cf_evidence = (
                build_before_evidence(
                    interpretation,
                    dict(profile),
                    cf_ranked,
                    cf_meta,
                )
            )

            cf_rationale = (
                self.generator.generate(
                    cf_evidence,
                    cf_ranked,
                    cf_influence,
                    use_llm=use_llm,
                )
            )

            cf_top = str(
                cf_ranked.iloc[0][
                    "framework"
                ]
            )
            flipped = (
                cf_top != original_top
            )

            changed = set(
                scenario_meta[
                    "changed_objectives"
                ]
            )
            cited_changed = set(
                cf_rationale
                .cited_objectives
            ) & changed
            changed_objective_cited = (
                float(
                    len(cited_changed)
                    / len(changed)
                )
                if changed
                else 1.0
            )

            new_winner_ack = (
                1.0
                if (
                    not flipped
                    or (
                        cf_rationale
                        .top_framework_mentioned
                        and cf_top.lower()
                        in (
                            cf_rationale
                            .text
                            .lower()
                        )
                    )
                )
                else 0.0
            )

            citation_change = (
                set_distance(
                    original.evidence_keys,
                    cf_rationale.evidence_keys,
                )
            )

            if flipped:
                counterfactual_sensitivity = float(
                    0.45
                    * new_winner_ack
                    + 0.35
                    * changed_objective_cited
                    + 0.20
                    * citation_change
                )
            else:
                # If the intervention is not sufficient to change the winning
                # decision, a faithful explanation should still acknowledge
                # the changed evidence when that evidence remains relevant.
                counterfactual_sensitivity = float(
                    0.70
                    * changed_objective_cited
                    + 0.30
                    * (
                        cf_rationale
                        .citation_validity
                    )
                )

            counterfactual_records.append(
                CounterfactualRecord(
                    scenario=(
                        scenario_meta[
                            "scenario"
                        ]
                    ),
                    changed_objectives=list(
                        scenario_meta[
                            "changed_objectives"
                        ]
                    ),
                    original_top_framework=(
                        original_top
                    ),
                    counterfactual_top_framework=(
                        cf_top
                    ),
                    decision_flipped=flipped,
                    original_rationale=original,
                    counterfactual_rationale=(
                        cf_rationale
                    ),
                    changed_objective_cited=(
                        changed_objective_cited
                    ),
                    new_winner_acknowledged=(
                        new_winner_ack
                    ),
                    citation_change=(
                        citation_change
                    ),
                    counterfactual_sensitivity=(
                        counterfactual_sensitivity
                    ),
                )
            )

        # Irrelevant-control perturbation: add a metadata note that is not used
        # by the recommender and must not change the decision.
        control_facts = deepcopy(
            evidence.facts
        )
        control_facts[
            "faithfulness_control"
        ] = {
            "irrelevant_note": (
                "control_metadata_only"
            )
        }
        control_evidence = EvidenceBundle(
            "before",
            control_facts,
        )
        control = (
            self.generator.generate(
                control_evidence,
                ranked,
                influence,
                use_llm=use_llm,
            )
        )

        control_same_top = (
            1.0
            if (
                control
                .top_framework_mentioned
                and original_top.lower()
                in control.text.lower()
            )
            else 0.0
        )
        control_similarity = (
            text_similarity(
                original.text,
                control.text,
            )
        )
        irrelevant_invariance = float(
            0.70 * control_same_top
            + 0.30 * control_similarity
        )

        grounding_validity = float(
            np.mean(
                [
                    original.citation_validity
                ]
                + [
                    row.counterfactual_rationale
                    .citation_validity
                    for row in (
                        counterfactual_records
                    )
                ]
                + [
                    control.citation_validity
                ]
            )
        )

        decision_alignment = (
            1.0
            if (
                original
                .top_framework_mentioned
                and original_top.lower()
                in original.text.lower()
            )
            else 0.0
        )

        attribution_alignment = (
            1.0
            if original.top_influence_cited
            else 0.0
        )

        counterfactual_sensitivity = float(
            np.mean(
                [
                    row
                    .counterfactual_sensitivity
                    for row in (
                        counterfactual_records
                    )
                ]
            )
        )

        fidelity = (
            evidence_fidelity_score(
                grounding_validity,
                decision_alignment,
                attribution_alignment,
                counterfactual_sensitivity,
                irrelevant_invariance,
            )
        )

        return DatasetFaithfulnessResult(
            dataset_id=str(
                dataset_id
            ),
            explanation_source=(
                original.source
            ),
            model=original.model,
            original_top_framework=(
                original_top
            ),
            original_top_influence_objective=(
                original
                .top_influence_objective
            ),
            grounding_validity=(
                grounding_validity
            ),
            decision_alignment=(
                decision_alignment
            ),
            attribution_alignment=(
                attribution_alignment
            ),
            counterfactual_sensitivity=(
                counterfactual_sensitivity
            ),
            irrelevant_invariance=(
                irrelevant_invariance
            ),
            evidence_fidelity_score=(
                fidelity
            ),
            counterfactuals=(
                counterfactual_records
            ),
        )
