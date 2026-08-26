from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config_diff import (
    deep_merge,
    diff_configs,
)
from .schemas import (
    CopilotConfiguration,
    CopilotProposal,
    ReviewDecision,
)


def review_proposal(
    proposal: CopilotProposal,
    decision: str,
    edits: Optional[
        Dict[str, Any]
    ] = None,
    note: Optional[str] = None,
) -> ReviewDecision:
    if decision not in {
        "approved",
        "approved_with_edits",
        "rejected",
    }:
        raise ValueError(
            "Unsupported review decision: {}".format(
                decision
            )
        )

    if decision == "rejected":
        return ReviewDecision(
            proposal_id=proposal.proposal_id,
            decision="rejected",
            note=note,
            final_config=None,
            config_diff=[],
        )

    base = (
        proposal
        .proposed_config
        .model_dump()
    )

    if decision == "approved":
        if edits:
            raise ValueError(
                "Use approved_with_edits when edits are supplied."
            )
        final = (
            proposal.proposed_config
        )
    else:
        if not edits:
            raise ValueError(
                "approved_with_edits requires at least one edit."
            )
        merged = deep_merge(
            base,
            edits,
        )
        final = (
            CopilotConfiguration
            .model_validate(merged)
        )

    return ReviewDecision(
        proposal_id=proposal.proposal_id,
        decision=decision,
        note=note,
        final_config=final,
        config_diff=diff_configs(
            proposal.proposed_config,
            final,
        ),
    )


class ReviewStore:
    """Append-only human-review audit log."""

    def __init__(
        self,
        path: Optional[Path] = None,
    ):
        self.path = (
            Path(path)
            if path is not None
            else Path(
                "artifacts/copilot/reviews.jsonl"
            )
        )

    def append(
        self,
        proposal: CopilotProposal,
        review: ReviewDecision,
    ) -> Path:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        record = {
            "record_type": (
                "copilot_human_review"
            ),
            "timestamp_utc": (
                datetime.now(
                    timezone.utc
                )
                .replace(
                    microsecond=0
                )
                .isoformat()
            ),
            "proposal": (
                proposal.model_dump()
            ),
            "review": (
                review.model_dump()
            ),
        }

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

        return self.path
