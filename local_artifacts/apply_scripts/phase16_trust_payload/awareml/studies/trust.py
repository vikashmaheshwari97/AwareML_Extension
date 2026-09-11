from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PHASE16_SCHEMA_VERSION = "phase16_trust_calibration_v1"
RANDOMIZATION_VERSION = "phase16_balanced_within_subject_v1"
DEFAULT_STIMULUS_ROOT = Path("data/journal/trust_stimulus_bank_v1/frozen")
DEFAULT_PROTOCOL_PATH = Path("data/journal/trust_calibration_phase16_v1/design/protocol.json")
DEFAULT_DB_PATH = Path("artifacts/phase16/trust_calibration.sqlite")
FINAL_DESIGN_MANIFEST = Path("data/journal/trust_calibration_phase16_v1/frozen_design/manifest.json")
ALLOWED_COLLECTION_MODES = ("pilot", "final")
ALLOWED_EXPERTISE_GROUPS = ("Novice / student", "Practitioner", "ML / AutoML expert")
ALLOWED_DECISION_ACTIONS = ("Accept", "Override", "Reject")


class Phase16Error(RuntimeError):
    """Base exception for Phase-16 trust-calibration failures."""


class StimulusIntegrityError(Phase16Error):
    """Raised when the frozen Phase-15 stimulus bank is incomplete or changed."""


class ProtocolGateError(Phase16Error):
    """Raised when a draft protocol is used for final collection or freezing."""


class EligibilityError(Phase16Error):
    """Raised when a participant is not eligible for Track 2."""


@dataclass(frozen=True)
class TrustStimulus:
    stimulus_id: str
    pair_id: str
    dataset_id: str
    source_stage: str
    source_name: str
    prompt: str
    explanation: str
    correctness_condition: str
    researcher_label: str
    error_type: Optional[str]
    explanation_sha256: Optional[str]

    def participant_dict(self, opaque_item_id: str, order_index: int) -> Dict[str, Any]:
        # Deliberately omits condition, researcher label, pair id and error type.
        return {
            "item_id": opaque_item_id,
            "order": int(order_index),
            "source": self.source_name,
            "prompt": self.prompt,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class AssignmentItem:
    order_index: int
    opaque_item_id: str
    stimulus_id: str
    pair_id: str
    source_stage: str
    correctness_condition: str
    error_type: Optional[str]
    randomization_version: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParticipantRegistration:
    participant_hash: str
    collection_mode: str
    expertise_group: str
    expertise_self_rating: int
    n_items: int
    consented: bool
    track1_author_attested_separate: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _stable_int(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def load_phase16_protocol(path: Optional[Path] = None) -> Dict[str, Any]:
    protocol_path = Path(path or os.environ.get("AWAREML_PHASE16_PROTOCOL", DEFAULT_PROTOCOL_PATH))
    if not protocol_path.exists():
        raise ProtocolGateError("Phase-16 protocol file is missing: {}".format(protocol_path))
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if int(protocol.get("phase", -1)) != 16:
        raise ProtocolGateError("Protocol does not declare phase=16.")
    if int(protocol.get("track", -1)) != 2:
        raise ProtocolGateError("Protocol does not declare track=2.")
    if protocol.get("schema_version") != PHASE16_SCHEMA_VERSION:
        raise ProtocolGateError(
            "Unexpected protocol schema_version: {}".format(protocol.get("schema_version"))
        )
    return protocol


def validate_protocol_for_design_freeze(protocol: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    trust_measure = protocol.get("trust_measure") or {}
    power = protocol.get("power_calculation") or {}
    ethics = protocol.get("ethics") or {}
    materials = protocol.get("participant_materials") or {}
    design = protocol.get("design") or {}

    if str(protocol.get("status", "")).lower() != "final":
        errors.append("protocol.status must be 'final'")
    if trust_measure.get("status") != "final":
        errors.append("trust_measure.status must be 'final' (Morten gate)")
    items = trust_measure.get("items") or []
    if not items:
        errors.append("trust_measure.items must contain the finalized validated trust item(s)")
    if any(bool(item.get("pilot_only")) for item in items):
        errors.append("final trust_measure.items cannot contain pilot_only items")
    if not trust_measure.get("instrument_name"):
        errors.append("trust_measure.instrument_name is required")
    if not trust_measure.get("scoring_rule"):
        errors.append("trust_measure.scoring_rule is required")

    if power.get("status") != "final":
        errors.append("power_calculation.status must be 'final' (Morten gate)")
    required_n = power.get("required_completed_participants")
    if not isinstance(required_n, int) or required_n <= 0:
        errors.append("power_calculation.required_completed_participants must be a positive integer")

    if ethics.get("status") not in {"approved", "exempt", "not_required"}:
        errors.append("ethics.status must be approved, exempt, or not_required before final collection")
    if ethics.get("status") in {"approved", "exempt"} and not ethics.get("reference"):
        errors.append("ethics.reference is required for approved/exempt status")

    if materials.get("status") != "final":
        errors.append("participant_materials.status must be 'final'")
    if not materials.get("consent_text"):
        errors.append("participant_materials.consent_text is required")
    if not materials.get("instructions"):
        errors.append("participant_materials.instructions is required")

    n_items = design.get("items_per_participant")
    if not isinstance(n_items, int) or n_items < 16 or n_items > 24 or n_items % 2:
        errors.append("design.items_per_participant must be an even integer from 16 to 24")
    if design.get("within_subject") is not True:
        errors.append("design.within_subject must be true")
    if design.get("randomized") is not True:
        errors.append("design.randomized must be true")
    if design.get("blinded") is not True:
        errors.append("design.blinded must be true")
    if design.get("track1_authors_excluded") is not True:
        errors.append("design.track1_authors_excluded must be true")

    return errors


def trust_measure_items(protocol: Dict[str, Any], collection_mode: str) -> List[Dict[str, Any]]:
    measure = protocol.get("trust_measure") or {}
    items = list(measure.get("items") or [])
    if collection_mode == "final":
        if measure.get("status") != "final":
            raise ProtocolGateError("Final collection requires Morten's finalized trust measure.")
        items = [item for item in items if not item.get("pilot_only")]
    if not items:
        raise ProtocolGateError("No trust-rating items are available for {} mode.".format(collection_mode))
    return items


def score_trust_items(protocol: Dict[str, Any], ratings: Dict[str, int], collection_mode: str) -> float:
    measure = protocol.get("trust_measure") or {}
    low = int(measure.get("scale_min", 1))
    high = int(measure.get("scale_max", 7))
    values: List[float] = []
    for item in trust_measure_items(protocol, collection_mode):
        item_id = str(item["id"])
        if item_id not in ratings:
            raise ValueError("Missing trust response for item '{}'".format(item_id))
        value = int(ratings[item_id])
        if value < low or value > high:
            raise ValueError("Trust rating {} must be between {} and {}".format(item_id, low, high))
        if bool(item.get("reverse")):
            value = (low + high) - value
        values.append(float(value))
    if not values:
        raise ValueError("No trust responses were supplied.")
    return float(sum(values) / len(values))


class Phase15StimulusBank:
    """Verified reader for the frozen Phase-15 Track-2 correct/incorrect bank.

    The participant-facing JSON never contains correctness labels. Researcher labels are
    read from the separate researcher file and are used only server-side for assignment,
    logging and analysis.
    """

    def __init__(self, root: Optional[Path] = None, verify_hashes: bool = True):
        configured = os.environ.get("AWAREML_PHASE16_STIMULUS_ROOT")
        self.root = Path(root or configured or DEFAULT_STIMULUS_ROOT)
        self.verify_hashes = bool(verify_hashes)
        self.manifest_path = self.root / "manifest.json"
        self.participant_path = self.root / "stimuli_participant.json"
        self.researcher_path = self.root / "stimuli_researcher.json"
        self.summary_path = self.root / "summary.json"
        self._manifest: Dict[str, Any] = {}
        self._researcher_rows: List[Dict[str, Any]] = []
        self._participant_rows: List[Dict[str, Any]] = []
        self._stimuli: Dict[str, TrustStimulus] = {}
        self._pairs: Dict[str, Dict[str, TrustStimulus]] = {}
        self._load()

    @property
    def manifest(self) -> Dict[str, Any]:
        return dict(self._manifest)

    @property
    def stimuli(self) -> Dict[str, TrustStimulus]:
        return dict(self._stimuli)

    @property
    def pairs(self) -> Dict[str, Dict[str, TrustStimulus]]:
        return {key: dict(value) for key, value in self._pairs.items()}

    def _load_json(self, path: Path) -> Any:
        if not path.exists():
            raise StimulusIntegrityError("Missing frozen stimulus artifact: {}".format(path))
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load(self) -> None:
        self._manifest = self._load_json(self.manifest_path)
        if self._manifest.get("status") != "frozen":
            raise StimulusIntegrityError("Phase-15 trust stimulus bank must have status='frozen'.")
        if self._manifest.get("schema_version") != "trust_stimulus_bank_v1":
            raise StimulusIntegrityError("Unexpected Phase-15 stimulus bank schema.")

        if self.verify_hashes:
            for name, meta in (self._manifest.get("files") or {}).items():
                path = self.root / name
                if not path.exists():
                    raise StimulusIntegrityError("Manifest-listed file is missing: {}".format(path))
                expected = str((meta or {}).get("sha256") or "").strip().lower()
                if expected and sha256_file(path).lower() != expected:
                    raise StimulusIntegrityError("Frozen stimulus hash mismatch for {}".format(path))

        self._participant_rows = list(self._load_json(self.participant_path))
        self._researcher_rows = list(self._load_json(self.researcher_path))
        participant_by_id = {
            str(row.get("stimulus_id")): row for row in self._participant_rows
        }
        if len(participant_by_id) != len(self._participant_rows):
            raise StimulusIntegrityError("Duplicate stimulus_id in participant-facing bank.")

        for row in self._researcher_rows:
            stimulus_id = str(row.get("stimulus_id") or "")
            pair_id = str(row.get("pair_id") or "")
            researcher_label = str(row.get("researcher_label") or "")
            if researcher_label == "known_correct":
                condition = "correct"
            elif researcher_label == "known_incorrect":
                condition = "incorrect"
            else:
                raise StimulusIntegrityError(
                    "Unknown researcher_label '{}' for {}".format(researcher_label, stimulus_id)
                )
            p = participant_by_id.get(stimulus_id)
            if p is None:
                raise StimulusIntegrityError(
                    "Researcher stimulus {} has no participant-facing version.".format(stimulus_id)
                )
            participant_version = row.get("participant_version") or {}
            if participant_version:
                for field in ("stimulus_id", "prompt", "explanation"):
                    if str(participant_version.get(field)) != str(p.get(field)):
                        raise StimulusIntegrityError(
                            "Participant/researcher mismatch for {} field {}".format(stimulus_id, field)
                        )
            provenance = row.get("provenance") or {}
            explanation_hash = provenance.get("explanation_sha256")
            if explanation_hash:
                actual = _sha256_bytes(str(p.get("explanation") or "").encode("utf-8"))
                if actual != explanation_hash:
                    raise StimulusIntegrityError(
                        "Explanation provenance hash mismatch for {}".format(stimulus_id)
                    )

            stimulus = TrustStimulus(
                stimulus_id=stimulus_id,
                pair_id=pair_id,
                dataset_id=str(row.get("dataset_id") or p.get("scenario") or ""),
                source_stage=str(row.get("source_stage") or ""),
                source_name=str(row.get("source_name") or p.get("source") or ""),
                prompt=str(p.get("prompt") or ""),
                explanation=str(p.get("explanation") or ""),
                correctness_condition=condition,
                researcher_label=researcher_label,
                error_type=(None if row.get("error_type") is None else str(row.get("error_type"))),
                explanation_sha256=(None if explanation_hash is None else str(explanation_hash)),
            )
            if not stimulus_id or not pair_id or not stimulus.source_stage:
                raise StimulusIntegrityError("Incomplete stimulus metadata for {}".format(stimulus_id))
            if stimulus_id in self._stimuli:
                raise StimulusIntegrityError("Duplicate researcher stimulus_id {}".format(stimulus_id))
            self._stimuli[stimulus_id] = stimulus
            self._pairs.setdefault(pair_id, {})[condition] = stimulus

        self.validate_design_pool()

    def validate_design_pool(self, min_per_condition: int = 20) -> Dict[str, Any]:
        correct = sum(1 for item in self._stimuli.values() if item.correctness_condition == "correct")
        incorrect = sum(1 for item in self._stimuli.values() if item.correctness_condition == "incorrect")
        complete_pairs = [
            pair_id for pair_id, variants in self._pairs.items()
            if set(variants) == {"correct", "incorrect"}
        ]
        if correct < min_per_condition or incorrect < min_per_condition:
            raise StimulusIntegrityError(
                "Phase-16 requires at least {} correct and {} incorrect stimuli; found {} and {}.".format(
                    min_per_condition, min_per_condition, correct, incorrect
                )
            )
        if len(complete_pairs) < min_per_condition:
            raise StimulusIntegrityError(
                "Phase-16 requires at least {} complete correct/incorrect pairs; found {}.".format(
                    min_per_condition, len(complete_pairs)
                )
            )
        if len(complete_pairs) != len(self._pairs):
            missing = sorted(set(self._pairs) - set(complete_pairs))
            raise StimulusIntegrityError("Incomplete correct/incorrect stimulus pairs: {}".format(missing))
        stages: Dict[str, int] = {}
        for pair_id in complete_pairs:
            stage = self._pairs[pair_id]["correct"].source_stage
            stages[stage] = stages.get(stage, 0) + 1
        return {
            "correct": correct,
            "incorrect": incorrect,
            "pairs": len(complete_pairs),
            "stages": stages,
            "manifest_sha256": sha256_file(self.manifest_path),
        }

    def get(self, stimulus_id: str) -> TrustStimulus:
        try:
            return self._stimuli[str(stimulus_id)]
        except KeyError as exc:
            raise KeyError("Unknown Phase-16 stimulus_id: {}".format(stimulus_id)) from exc


class BalancedWithinSubjectRandomizer:
    """Deterministic, participant-specific, blinded correct/incorrect randomizer."""

    def __init__(self, bank: Phase15StimulusBank, max_same_condition_run: int = 3):
        self.bank = bank
        self.max_same_condition_run = int(max_same_condition_run)
        if self.max_same_condition_run < 1:
            raise ValueError("max_same_condition_run must be >=1")

    def _select_pairs(self, participant_hash: str, n_items: int) -> List[str]:
        stages: Dict[str, List[str]] = {}
        for pair_id, variants in self.bank.pairs.items():
            stage = variants["correct"].source_stage
            stages.setdefault(stage, []).append(pair_id)
        stage_names = sorted(stages)
        if not stage_names:
            raise StimulusIntegrityError("No source stages are available.")
        if n_items > len(self.bank.pairs):
            raise ValueError("Requested {} items from only {} pairs.".format(n_items, len(self.bank.pairs)))

        base = n_items // len(stage_names)
        remainder = n_items % len(stage_names)
        rotation = _stable_int(participant_hash, RANDOMIZATION_VERSION, "stage-rotation") % len(stage_names)
        rotated = stage_names[rotation:] + stage_names[:rotation]
        quotas = {stage: base + (1 if stage in rotated[:remainder] else 0) for stage in stage_names}

        selected: List[str] = []
        for stage in stage_names:
            pairs = stages[stage]
            quota = quotas[stage]
            if quota > len(pairs):
                raise StimulusIntegrityError(
                    "Stage {} has {} pairs but randomization requires {}.".format(stage, len(pairs), quota)
                )
            ranked = sorted(
                pairs,
                key=lambda pair_id: _stable_int(participant_hash, RANDOMIZATION_VERSION, "pair", stage, pair_id),
            )
            selected.extend(ranked[:quota])
        return selected

    def _assign_conditions(self, participant_hash: str, pair_ids: Sequence[str]) -> Dict[str, str]:
        n_items = len(pair_ids)
        correct_total = n_items // 2
        by_stage: Dict[str, List[str]] = {}
        for pair_id in pair_ids:
            stage = self.bank.pairs[pair_id]["correct"].source_stage
            by_stage.setdefault(stage, []).append(pair_id)

        floor_correct = {stage: len(ids) // 2 for stage, ids in by_stage.items()}
        extras_needed = correct_total - sum(floor_correct.values())
        odd_stages = [stage for stage, ids in by_stage.items() if len(ids) % 2 == 1]
        odd_stages = sorted(
            odd_stages,
            key=lambda stage: _stable_int(participant_hash, RANDOMIZATION_VERSION, "condition-extra", stage),
        )
        correct_quota = dict(floor_correct)
        for stage in odd_stages[:extras_needed]:
            correct_quota[stage] += 1

        assignments: Dict[str, str] = {}
        for stage, ids in sorted(by_stage.items()):
            ranked = sorted(
                ids,
                key=lambda pair_id: _stable_int(
                    participant_hash, RANDOMIZATION_VERSION, "condition", stage, pair_id
                ),
            )
            correct_ids = set(ranked[: correct_quota[stage]])
            for pair_id in ids:
                assignments[pair_id] = "correct" if pair_id in correct_ids else "incorrect"
        return assignments

    def _condition_run_ok(self, ordered: Sequence[Tuple[str, str]]) -> bool:
        run = 0
        last: Optional[str] = None
        for _, condition in ordered:
            if condition == last:
                run += 1
            else:
                last = condition
                run = 1
            if run > self.max_same_condition_run:
                return False
        return True

    def _order_items(
        self,
        participant_hash: str,
        assignments: Dict[str, str],
    ) -> List[Tuple[str, str]]:
        pair_ids = list(assignments)
        # Try deterministic salts until the max-run constraint is satisfied.
        for attempt in range(512):
            ordered_pairs = sorted(
                pair_ids,
                key=lambda pair_id: _stable_int(
                    participant_hash,
                    RANDOMIZATION_VERSION,
                    "order",
                    str(attempt),
                    pair_id,
                ),
            )
            ordered = [(pair_id, assignments[pair_id]) for pair_id in ordered_pairs]
            if self._condition_run_ok(ordered):
                return ordered
        raise Phase16Error("Unable to construct a randomized order satisfying max condition run.")

    def build(self, participant_hash: str, n_items: int) -> List[AssignmentItem]:
        if n_items < 16 or n_items > 24 or n_items % 2:
            raise ValueError("Phase-16 items_per_participant must be an even integer from 16 to 24.")
        pair_ids = self._select_pairs(participant_hash, n_items)
        assignments = self._assign_conditions(participant_hash, pair_ids)
        ordered = self._order_items(participant_hash, assignments)
        output: List[AssignmentItem] = []
        for index, (pair_id, condition) in enumerate(ordered, start=1):
            stimulus = self.bank.pairs[pair_id][condition]
            opaque = "T16-{}".format(
                hmac.new(
                    participant_hash.encode("utf-8"),
                    "{}|{}|{}".format(RANDOMIZATION_VERSION, index, stimulus.stimulus_id).encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()[:10].upper()
            )
            output.append(
                AssignmentItem(
                    order_index=index,
                    opaque_item_id=opaque,
                    stimulus_id=stimulus.stimulus_id,
                    pair_id=pair_id,
                    source_stage=stimulus.source_stage,
                    correctness_condition=condition,
                    error_type=stimulus.error_type,
                    randomization_version=RANDOMIZATION_VERSION,
                )
            )
        self._validate_assignment(output, n_items)
        return output

    def _validate_assignment(self, items: Sequence[AssignmentItem], n_items: int) -> None:
        if len(items) != n_items:
            raise Phase16Error("Assignment length mismatch.")
        if len({item.pair_id for item in items}) != n_items:
            raise Phase16Error("A participant cannot see both variants of the same pair.")
        conditions = [item.correctness_condition for item in items]
        if conditions.count("correct") != n_items // 2:
            raise Phase16Error("Assignment is not 50/50 correct/incorrect.")
        if conditions.count("incorrect") != n_items // 2:
            raise Phase16Error("Assignment is not 50/50 correct/incorrect.")
        if not self._condition_run_ok([(item.pair_id, item.correctness_condition) for item in items]):
            raise Phase16Error("Assignment violates the maximum same-condition run constraint.")


class Phase16Store:
    """Append-only SQLite store for participant registration, assignment and responses."""

    def __init__(self, path: Optional[Path] = None):
        configured = os.environ.get("AWAREML_PHASE16_DB")
        self.path = Path(path or configured or DEFAULT_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path), timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=FULL")
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS participants (
                    collection_mode TEXT NOT NULL,
                    participant_hash TEXT NOT NULL,
                    expertise_group TEXT NOT NULL,
                    expertise_self_rating INTEGER NOT NULL,
                    n_items INTEGER NOT NULL,
                    consented INTEGER NOT NULL,
                    track1_author_attested_separate INTEGER NOT NULL,
                    randomization_version TEXT NOT NULL,
                    protocol_sha256 TEXT NOT NULL,
                    stimulus_manifest_sha256 TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    completed_at TEXT,
                    PRIMARY KEY(collection_mode, participant_hash)
                );

                CREATE TABLE IF NOT EXISTS assignments (
                    collection_mode TEXT NOT NULL,
                    participant_hash TEXT NOT NULL,
                    order_index INTEGER NOT NULL,
                    opaque_item_id TEXT NOT NULL,
                    stimulus_id TEXT NOT NULL,
                    pair_id TEXT NOT NULL,
                    source_stage TEXT NOT NULL,
                    correctness_condition TEXT NOT NULL,
                    error_type TEXT,
                    randomization_version TEXT NOT NULL,
                    assigned_at TEXT NOT NULL,
                    PRIMARY KEY(collection_mode, participant_hash, order_index),
                    UNIQUE(collection_mode, participant_hash, opaque_item_id),
                    UNIQUE(collection_mode, participant_hash, pair_id),
                    FOREIGN KEY(collection_mode, participant_hash)
                        REFERENCES participants(collection_mode, participant_hash)
                );

                CREATE TABLE IF NOT EXISTS responses (
                    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_mode TEXT NOT NULL,
                    participant_hash TEXT NOT NULL,
                    expertise_group TEXT NOT NULL,
                    order_index INTEGER NOT NULL,
                    opaque_item_id TEXT NOT NULL,
                    stimulus_id TEXT NOT NULL,
                    source_stage TEXT NOT NULL,
                    correctness_condition TEXT NOT NULL,
                    trust_rating REAL NOT NULL,
                    trust_items_json TEXT NOT NULL,
                    perceived_correctness INTEGER NOT NULL,
                    fluency_rating INTEGER NOT NULL,
                    perceived_confidence INTEGER NOT NULL,
                    decision_action TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    response_time_sec REAL NOT NULL,
                    submitted_at TEXT NOT NULL,
                    UNIQUE(collection_mode, participant_hash, order_index),
                    UNIQUE(collection_mode, participant_hash, opaque_item_id),
                    FOREIGN KEY(collection_mode, participant_hash, order_index)
                        REFERENCES assignments(collection_mode, participant_hash, order_index)
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_mode TEXT NOT NULL,
                    participant_hash TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def log_event(
        self,
        collection_mode: str,
        event_type: str,
        payload: Dict[str, Any],
        participant_hash: Optional[str] = None,
    ) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO audit_events(collection_mode, participant_hash, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    collection_mode,
                    participant_hash,
                    event_type,
                    _canonical_json(payload),
                    _utc_now(),
                ),
            )

    def get_participant(self, collection_mode: str, participant_hash: str) -> Optional[Dict[str, Any]]:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM participants WHERE collection_mode=? AND participant_hash=?",
                (collection_mode, participant_hash),
            ).fetchone()
        return None if row is None else dict(row)

    def register_participant(
        self,
        registration: ParticipantRegistration,
        assignment: Sequence[AssignmentItem],
        protocol_sha256: str,
        stimulus_manifest_sha256: str,
    ) -> None:
        if not registration.consented:
            raise EligibilityError("Consent is required before participation.")
        if not registration.track1_author_attested_separate:
            raise EligibilityError(
                "Track-2 participants must be separate from people who authored Track-1 scenarios."
            )
        if registration.expertise_group not in ALLOWED_EXPERTISE_GROUPS:
            raise ValueError("Unknown expertise group: {}".format(registration.expertise_group))
        if registration.expertise_self_rating < 1 or registration.expertise_self_rating > 5:
            raise ValueError("expertise_self_rating must be 1..5")
        if registration.collection_mode not in ALLOWED_COLLECTION_MODES:
            raise ValueError("collection_mode must be pilot or final")

        existing = self.get_participant(registration.collection_mode, registration.participant_hash)
        if existing is not None:
            # Resume is allowed, but study-defining attributes may not silently change.
            immutable = {
                "expertise_group": registration.expertise_group,
                "expertise_self_rating": registration.expertise_self_rating,
                "n_items": registration.n_items,
                "randomization_version": RANDOMIZATION_VERSION,
                "protocol_sha256": protocol_sha256,
                "stimulus_manifest_sha256": stimulus_manifest_sha256,
            }
            for key, expected in immutable.items():
                if existing.get(key) != expected:
                    raise Phase16Error(
                        "Participant already registered under a different Phase-16 design attribute {} "
                        "(stored={!r}, requested={!r}). Do not resume this participant across a protocol, "
                        "stimulus-bank, or randomization change.".format(key, existing.get(key), expected)
                    )
            return

        now = _utc_now()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                """
                INSERT INTO participants(
                    collection_mode, participant_hash, expertise_group, expertise_self_rating,
                    n_items, consented, track1_author_attested_separate, randomization_version,
                    protocol_sha256, stimulus_manifest_sha256, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    registration.collection_mode,
                    registration.participant_hash,
                    registration.expertise_group,
                    registration.expertise_self_rating,
                    registration.n_items,
                    1,
                    1,
                    RANDOMIZATION_VERSION,
                    protocol_sha256,
                    stimulus_manifest_sha256,
                    now,
                ),
            )
            for item in assignment:
                con.execute(
                    """
                    INSERT INTO assignments(
                        collection_mode, participant_hash, order_index, opaque_item_id,
                        stimulus_id, pair_id, source_stage, correctness_condition, error_type,
                        randomization_version, assigned_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        registration.collection_mode,
                        registration.participant_hash,
                        item.order_index,
                        item.opaque_item_id,
                        item.stimulus_id,
                        item.pair_id,
                        item.source_stage,
                        item.correctness_condition,
                        item.error_type,
                        item.randomization_version,
                        now,
                    ),
                )

    def assignment_rows(self, collection_mode: str, participant_hash: str) -> List[Dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM assignments WHERE collection_mode=? AND participant_hash=? ORDER BY order_index",
                (collection_mode, participant_hash),
            ).fetchall()
        return [dict(row) for row in rows]

    def response_rows(self, collection_mode: str, participant_hash: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM responses WHERE collection_mode=?"
        params: List[Any] = [collection_mode]
        if participant_hash is not None:
            query += " AND participant_hash=?"
            params.append(participant_hash)
        query += " ORDER BY participant_hash, order_index"
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def participant_rows(self, collection_mode: str) -> List[Dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM participants WHERE collection_mode=? ORDER BY registered_at, participant_hash",
                (collection_mode,),
            ).fetchall()
        return [dict(row) for row in rows]

    def all_assignment_rows(self, collection_mode: str) -> List[Dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM assignments WHERE collection_mode=? ORDER BY participant_hash, order_index",
                (collection_mode,),
            ).fetchall()
        return [dict(row) for row in rows]

    def next_order_index(self, collection_mode: str, participant_hash: str) -> Optional[int]:
        participant = self.get_participant(collection_mode, participant_hash)
        if participant is None:
            return None
        with self._connect() as con:
            completed = {
                int(row[0])
                for row in con.execute(
                    "SELECT order_index FROM responses WHERE collection_mode=? AND participant_hash=?",
                    (collection_mode, participant_hash),
                ).fetchall()
            }
        for order_index in range(1, int(participant["n_items"]) + 1):
            if order_index not in completed:
                return order_index
        return None

    def save_response(
        self,
        collection_mode: str,
        participant_hash: str,
        order_index: int,
        trust_rating: float,
        trust_items: Dict[str, int],
        perceived_correctness: int,
        fluency_rating: int,
        perceived_confidence: int,
        decision_action: str,
        response_time_sec: float,
    ) -> None:
        if decision_action not in ALLOWED_DECISION_ACTIONS:
            raise ValueError("decision_action must be one of {}".format(ALLOWED_DECISION_ACTIONS))
        for name, value in (
            ("perceived_correctness", perceived_correctness),
            ("fluency_rating", fluency_rating),
            ("perceived_confidence", perceived_confidence),
        ):
            if int(value) < 1 or int(value) > 7:
                raise ValueError("{} must be between 1 and 7".format(name))
        if float(response_time_sec) <= 0.0:
            raise ValueError("response_time_sec must be positive")

        participant = self.get_participant(collection_mode, participant_hash)
        if participant is None:
            raise Phase16Error("Participant is not registered.")
        with self._connect() as con:
            assignment = con.execute(
                """
                SELECT * FROM assignments
                WHERE collection_mode=? AND participant_hash=? AND order_index=?
                """,
                (collection_mode, participant_hash, int(order_index)),
            ).fetchone()
            if assignment is None:
                raise Phase16Error("No assignment exists for order {}.".format(order_index))
            accepted = 1 if decision_action == "Accept" else 0
            try:
                con.execute(
                    """
                    INSERT INTO responses(
                        collection_mode, participant_hash, expertise_group, order_index,
                        opaque_item_id, stimulus_id, source_stage, correctness_condition,
                        trust_rating, trust_items_json, perceived_correctness, fluency_rating,
                        perceived_confidence, decision_action, accepted, response_time_sec, submitted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        collection_mode,
                        participant_hash,
                        participant["expertise_group"],
                        int(order_index),
                        assignment["opaque_item_id"],
                        assignment["stimulus_id"],
                        assignment["source_stage"],
                        assignment["correctness_condition"],
                        float(trust_rating),
                        _canonical_json(trust_items),
                        int(perceived_correctness),
                        int(fluency_rating),
                        int(perceived_confidence),
                        decision_action,
                        accepted,
                        float(response_time_sec),
                        _utc_now(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise Phase16Error("This trial response has already been stored; responses are append-only.") from exc

        if self.next_order_index(collection_mode, participant_hash) is None:
            with self._connect() as con:
                con.execute(
                    """
                    UPDATE participants SET completed_at=?
                    WHERE collection_mode=? AND participant_hash=? AND completed_at IS NULL
                    """,
                    (_utc_now(), collection_mode, participant_hash),
                )

    def summary(self, collection_mode: str) -> Dict[str, Any]:
        participants = self.participant_rows(collection_mode)
        responses = self.response_rows(collection_mode)
        completed = [row for row in participants if row.get("completed_at")]
        by_expertise: Dict[str, int] = {}
        for row in completed:
            key = str(row.get("expertise_group"))
            by_expertise[key] = by_expertise.get(key, 0) + 1
        return {
            "collection_mode": collection_mode,
            "registered_participants": len(participants),
            "completed_participants": len(completed),
            "responses": len(responses),
            "completed_by_expertise": by_expertise,
            "db_path": str(self.path),
        }


class TrustCalibrationStudy:
    """Phase-16 Track-2 human trust-calibration study engine.

    This replaces the former top/second/worst utility scaffold. Trial stimuli now come
    exclusively from the frozen Phase-15 correct-vs-incorrect explanation bank.
    """

    def __init__(
        self,
        protocol_path: Optional[Path] = None,
        stimulus_root: Optional[Path] = None,
        db_path: Optional[Path] = None,
        id_salt: Optional[str] = None,
        verify_stimulus_hashes: bool = True,
        seed: Optional[int] = None,
    ):
        # seed is accepted only for source compatibility with the removed scaffold.
        # Phase-16 randomization is deterministic from the pseudonymous participant hash.
        self.legacy_seed_ignored = seed
        self.protocol_path = Path(protocol_path or os.environ.get("AWAREML_PHASE16_PROTOCOL", DEFAULT_PROTOCOL_PATH))
        self.protocol = load_phase16_protocol(self.protocol_path)
        self.bank = Phase15StimulusBank(stimulus_root, verify_hashes=verify_stimulus_hashes)
        randomization = self.protocol.get("randomization") or {}
        self.randomizer = BalancedWithinSubjectRandomizer(
            self.bank,
            max_same_condition_run=int(randomization.get("max_same_condition_run", 3)),
        )
        self.store = Phase16Store(db_path)
        self.id_salt = str(id_salt or os.environ.get("AWAREML_PHASE16_ID_SALT") or "phase16-pilot-local-salt")

    @property
    def protocol_sha256(self) -> str:
        return sha256_file(self.protocol_path)

    @property
    def stimulus_manifest_sha256(self) -> str:
        return sha256_file(self.bank.manifest_path)

    @staticmethod
    def collection_mode() -> str:
        mode = str(os.environ.get("AWAREML_PHASE16_COLLECTION_MODE", "pilot")).strip().lower()
        if mode not in ALLOWED_COLLECTION_MODES:
            raise ProtocolGateError("AWAREML_PHASE16_COLLECTION_MODE must be pilot or final.")
        return mode

    def participant_hash(self, participant_code: str, collection_mode: str) -> str:
        code = str(participant_code or "").strip()
        if len(code) < 3:
            raise ValueError("Participant/session code must contain at least 3 characters.")
        if collection_mode == "final" and (
            self.id_salt == "phase16-pilot-local-salt" or len(self.id_salt) < 16
        ):
            raise ProtocolGateError(
                "Final collection requires a private AWAREML_PHASE16_ID_SALT of at least 16 characters."
            )
        payload = "{}|{}|{}".format(PHASE16_SCHEMA_VERSION, collection_mode, code).encode("utf-8")
        return hmac.new(self.id_salt.encode("utf-8"), payload, hashlib.sha256).hexdigest()[:24]

    def assert_final_collection_ready(self) -> None:
        errors = validate_protocol_for_design_freeze(self.protocol)
        if errors:
            raise ProtocolGateError("Final collection protocol gate failed: " + "; ".join(errors))
        if not FINAL_DESIGN_MANIFEST.exists():
            raise ProtocolGateError(
                "Final collection requires a frozen design manifest. Run scripts/freeze_phase16_design.py first."
            )
        with FINAL_DESIGN_MANIFEST.open("r", encoding="utf-8") as handle:
            frozen_design = json.load(handle)
        frozen_protocol_sha = str(
            (((frozen_design.get("files") or {}).get("protocol.json") or {}).get("sha256") or "")
        )
        frozen_stimulus_sha = str(
            ((frozen_design.get("source_stimulus_bank") or {}).get("manifest_sha256") or "")
        )
        if frozen_protocol_sha and self.protocol_sha256 != frozen_protocol_sha:
            raise ProtocolGateError(
                "Current Phase-16 protocol differs from the frozen design. Final collection is locked."
            )
        if frozen_stimulus_sha and self.stimulus_manifest_sha256 != frozen_stimulus_sha:
            raise ProtocolGateError(
                "Current Phase-15 stimulus bank differs from the Phase-16 frozen design. Final collection is locked."
            )
        armed = str(os.environ.get("AWAREML_PHASE16_FINAL_ARMED", "")).strip().upper()
        if armed != "YES":
            raise ProtocolGateError(
                "Set AWAREML_PHASE16_FINAL_ARMED=YES explicitly before collecting FINAL responses."
            )
        if self.id_salt == "phase16-pilot-local-salt" or len(self.id_salt) < 16:
            raise ProtocolGateError("Final collection requires a private participant-ID salt.")

    def _n_items(self) -> int:
        n_items = int((self.protocol.get("design") or {}).get("items_per_participant", 20))
        if n_items < 16 or n_items > 24 or n_items % 2:
            raise ProtocolGateError("Protocol items_per_participant must be an even integer from 16 to 24.")
        return n_items

    def register(
        self,
        participant_code: str,
        expertise_group: str,
        expertise_self_rating: int,
        consented: bool,
        track1_author_attested_separate: bool,
        collection_mode: Optional[str] = None,
    ) -> ParticipantRegistration:
        mode = collection_mode or self.collection_mode()
        if mode == "final":
            self.assert_final_collection_ready()
        if not consented:
            raise EligibilityError("The participant must provide consent before continuing.")
        if not track1_author_attested_separate:
            raise EligibilityError(
                "This participant is not eligible for Track 2 because Track-2 participants must be separate "
                "from people who authored Track-1 scenarios."
            )
        participant_hash = self.participant_hash(participant_code, mode)
        n_items = self._n_items()
        assignment = self.randomizer.build(participant_hash, n_items)
        registration = ParticipantRegistration(
            participant_hash=participant_hash,
            collection_mode=mode,
            expertise_group=expertise_group,
            expertise_self_rating=int(expertise_self_rating),
            n_items=n_items,
            consented=bool(consented),
            track1_author_attested_separate=bool(track1_author_attested_separate),
        )
        self.store.register_participant(
            registration,
            assignment,
            protocol_sha256=self.protocol_sha256,
            stimulus_manifest_sha256=self.stimulus_manifest_sha256,
        )
        self.store.log_event(
            mode,
            "participant_registered_or_resumed",
            {
                "expertise_group": expertise_group,
                "expertise_self_rating": int(expertise_self_rating),
                "n_items": n_items,
                "randomization_version": RANDOMIZATION_VERSION,
            },
            participant_hash=participant_hash,
        )
        return registration

    def current_trial(
        self,
        participant_hash: str,
        collection_mode: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        mode = collection_mode or self.collection_mode()
        order_index = self.store.next_order_index(mode, participant_hash)
        if order_index is None:
            return None
        rows = self.store.assignment_rows(mode, participant_hash)
        row = next((candidate for candidate in rows if int(candidate["order_index"]) == int(order_index)), None)
        if row is None:
            raise Phase16Error("Current assignment row is missing.")
        stimulus = self.bank.get(str(row["stimulus_id"]))
        participant = stimulus.participant_dict(str(row["opaque_item_id"]), int(row["order_index"]))
        participant["total_items"] = len(rows)
        return participant

    def submit_trial(
        self,
        participant_hash: str,
        order_index: int,
        trust_items: Dict[str, int],
        perceived_correctness: int,
        fluency_rating: int,
        perceived_confidence: int,
        decision_action: str,
        response_time_sec: float,
        collection_mode: Optional[str] = None,
    ) -> float:
        mode = collection_mode or self.collection_mode()
        trust_rating = score_trust_items(self.protocol, trust_items, mode)
        self.store.save_response(
            collection_mode=mode,
            participant_hash=participant_hash,
            order_index=int(order_index),
            trust_rating=trust_rating,
            trust_items=trust_items,
            perceived_correctness=int(perceived_correctness),
            fluency_rating=int(fluency_rating),
            perceived_confidence=int(perceived_confidence),
            decision_action=decision_action,
            response_time_sec=float(response_time_sec),
        )
        return trust_rating

    def participant_progress(
        self,
        participant_hash: str,
        collection_mode: Optional[str] = None,
    ) -> Dict[str, int]:
        mode = collection_mode or self.collection_mode()
        participant = self.store.get_participant(mode, participant_hash)
        if participant is None:
            return {"completed": 0, "total": 0}
        completed = len(self.store.response_rows(mode, participant_hash))
        return {"completed": completed, "total": int(participant["n_items"])}

    def build_preview_assignment(self, preview_key: str = "researcher-preview") -> List[Dict[str, Any]]:
        # No database write; useful for the Advanced Labs participant-facing preview.
        participant_hash = hashlib.sha256(preview_key.encode("utf-8")).hexdigest()[:24]
        items = self.randomizer.build(participant_hash, self._n_items())
        output: List[Dict[str, Any]] = []
        for item in items:
            stimulus = self.bank.get(item.stimulus_id)
            output.append(stimulus.participant_dict(item.opaque_item_id, item.order_index))
        return output

    def researcher_assignment(self, participant_hash: str, collection_mode: str) -> List[Dict[str, Any]]:
        return self.store.assignment_rows(collection_mode, participant_hash)

    # Deliberate failure for the removed utility-based scaffold API. Keeping an explicit
    # method gives callers a clear migration message rather than silently recreating the
    # scientifically obsolete top/second/worst manipulation.
    def build_case(self, *args: Any, **kwargs: Any) -> Any:
        raise Phase16Error(
            "Phase 16 removed the utility-based correct/weak/wrong TrustCalibrationStudy scaffold. "
            "Use register()/current_trial()/submit_trial() with the frozen Phase-15 correct-vs-incorrect bank."
        )


def calibration_metrics(responses: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Backward-compatible lightweight summary using Phase-16 correct/incorrect labels."""
    rows = list(responses)
    if not rows:
        return {}
    trust_by: Dict[str, List[float]] = {"correct": [], "incorrect": []}
    accepted_incorrect: List[float] = []
    for row in rows:
        condition = str(row.get("correctness_condition") or row.get("condition") or "")
        trust = row.get("trust_rating", row.get("trust"))
        if condition in trust_by and trust is not None:
            trust_by[condition].append(float(trust))
        if condition == "incorrect" and ("accepted" in row):
            accepted_incorrect.append(float(bool(row.get("accepted"))))
    mean_trust = {
        condition: (sum(values) / len(values) if values else None)
        for condition, values in trust_by.items()
    }
    return {
        "mean_trust_by_condition": mean_trust,
        "incorrect_acceptance_rate": (
            sum(accepted_incorrect) / len(accepted_incorrect) if accepted_incorrect else None
        ),
    }
