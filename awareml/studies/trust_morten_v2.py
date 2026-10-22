from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

SCHEMA_VERSION = "phase16_trust_calibration_v2_1"
RANDOMIZATION_VERSION = "phase16_counterbalanced_v3_morten"
DEFAULT_PROTOCOL_PATH = Path("data/journal/trust_calibration_phase16_v2/frozen_design/protocol.json")
DEFAULT_MANIFEST_PATH = Path("data/journal/trust_calibration_phase16_v2/frozen_design/manifest.json")
DEFAULT_STIMULUS_ROOT = Path("data/journal/trust_stimulus_bank_v2/frozen")
DEFAULT_DB_PATH = Path("artifacts/phase16/trust_calibration.sqlite")
ALLOWED_EXPERTISE_GROUPS = ("University / CS student", "Practitioner", "ML / AutoML expert")
ALLOWED_DECISION_ACTIONS = ("Accept", "Override", "Reject")


class StudyV2Error(RuntimeError):
    pass


def _stable_int(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def load_protocol(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or DEFAULT_PROTOCOL_PATH)
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise StudyV2Error("Unexpected Phase-16 v2 protocol schema.")
    return data


@dataclass(frozen=True)
class Stimulus:
    stimulus_id: str
    pair_id: str
    scenario_id: str
    scenario_text: str
    domain: str
    framework_context: str
    source_stage: str
    source_name: str
    prompt: str
    explanation: str
    reference_evidence: Dict[str, Any]
    condition: str
    error_type: Optional[str]

    def participant_dict(self, opaque_item_id: str, order_index: int) -> Dict[str, Any]:
        return {
            "item_id": opaque_item_id,
            "order": int(order_index),
            "scenario_id": self.scenario_id,
            "scenario_text": self.scenario_text,
            "domain": self.domain,
            "framework_context": self.framework_context,
            "source_stage": self.source_stage,
            "source": self.source_name,
            "prompt": self.prompt,
            "explanation": self.explanation,
            "reference_evidence": self.reference_evidence,
        }


@dataclass(frozen=True)
class Assignment:
    order_index: int
    opaque_item_id: str
    stimulus_id: str
    pair_id: str
    scenario_id: str
    framework_context: str
    source_stage: str
    correctness_condition: str
    error_type: Optional[str]
    counterbalance_arm: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StimulusBankV2:
    def __init__(self, root: Optional[Path] = None, verify: bool = True):
        self.root = Path(root or os.getenv("AWAREML_PHASE16_V2_STIMULUS_ROOT") or DEFAULT_STIMULUS_ROOT)
        self.manifest_path = self.root / "manifest.json"
        self.rows = json.loads((self.root / "stimuli_researcher.json").read_text(encoding="utf-8"))
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen" or manifest.get("schema_version") != "trust_stimulus_bank_v2_1":
            raise StudyV2Error("Trust stimulus bank v2 is not frozen/valid.")
        if verify:
            for name, meta in (manifest.get("files") or {}).items():
                p = self.root / name
                if not p.exists() or _sha(p) != str(meta.get("sha256")):
                    raise StudyV2Error("Stimulus-bank integrity check failed for {}".format(name))
        self.by_id: Dict[str, Stimulus] = {}
        self.pairs: Dict[str, Dict[str, Stimulus]] = {}
        self.pair_order: List[str] = []
        for row in self.rows:
            cond = "correct" if row.get("researcher_label") == "known_correct" else "incorrect"
            stim = Stimulus(
                stimulus_id=str(row["stimulus_id"]), pair_id=str(row["pair_id"]),
                scenario_id=str(row.get("scenario_id") or ""), scenario_text=str(row.get("scenario_text") or ""),
                domain=str(row.get("domain") or ""), framework_context=str(row.get("framework_context") or ""),
                source_stage=str(row.get("source_stage") or ""), source_name=str(row.get("source_name") or ""),
                prompt=str(row.get("prompt") or ""), explanation=str(row.get("explanation") or ""),
                reference_evidence=dict(row.get("evidence_summary") or {}), condition=cond,
                error_type=row.get("error_type"),
            )
            self.by_id[stim.stimulus_id] = stim
            if stim.pair_id not in self.pairs:
                self.pair_order.append(stim.pair_id)
            self.pairs.setdefault(stim.pair_id, {})[cond] = stim
        self._validate()

    def _validate(self):
        if len(self.pairs) != 20 or any(set(v) != {"correct", "incorrect"} for v in self.pairs.values()):
            raise StudyV2Error("Phase-16 v2 requires exactly 20 complete matched pair families.")
        correct = [self.pairs[p]["correct"] for p in self.pair_order]
        if len({x.scenario_id for x in correct}) != 20:
            raise StudyV2Error("Phase-16 v2 requires 20 unique Track-1 scenarios.")
        fw = [x.framework_context for x in correct]
        expected_fw = {"AutoStreamML": 4, "AutoClass": 4, "EvoAutoML": 4, "OAML": 4, "ChaCha": 4}
        if {k: fw.count(k) for k in set(fw)} != expected_fw:
            raise StudyV2Error("Framework coverage must be four trial families per framework.")
        stages = [x.source_stage for x in correct]
        if {k: stages.count(k) for k in set(stages)} != {"OBJ_FIT": 5, "COMPARE": 5, "TRADEOFF": 5, "F_CHAT": 5}:
            raise StudyV2Error("Source-stage coverage must be five trial families per stage.")

    def get(self, stimulus_id: str) -> Stimulus:
        return self.by_id[str(stimulus_id)]


class Phase16V2Store:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or os.getenv("AWAREML_PHASE16_DB") or DEFAULT_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        con = sqlite3.connect(str(self.path), timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=30000")
        return con

    @staticmethod
    def _columns(con, table: str) -> set:
        return {str(r[1]) for r in con.execute("PRAGMA table_info({})".format(table)).fetchall()}

    @staticmethod
    def _add_column(con, table: str, definition: str):
        name = definition.split()[0]
        if name not in Phase16V2Store._columns(con, table):
            con.execute("ALTER TABLE {} ADD COLUMN {}".format(table, definition))

    def _init_db(self):
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript("""
            CREATE TABLE IF NOT EXISTS participants(
              collection_mode TEXT NOT NULL, participant_hash TEXT NOT NULL,
              expertise_group TEXT NOT NULL, expertise_self_rating INTEGER NOT NULL,
              n_items INTEGER NOT NULL, consented INTEGER NOT NULL,
              track1_author_attested_separate INTEGER NOT NULL,
              randomization_version TEXT NOT NULL, protocol_sha256 TEXT NOT NULL,
              stimulus_manifest_sha256 TEXT NOT NULL, registered_at TEXT NOT NULL,
              completed_at TEXT, PRIMARY KEY(collection_mode, participant_hash));
            CREATE TABLE IF NOT EXISTS assignments(
              collection_mode TEXT NOT NULL, participant_hash TEXT NOT NULL,
              order_index INTEGER NOT NULL, opaque_item_id TEXT NOT NULL,
              stimulus_id TEXT NOT NULL, pair_id TEXT NOT NULL,
              source_stage TEXT NOT NULL, correctness_condition TEXT NOT NULL,
              error_type TEXT, randomization_version TEXT NOT NULL,
              assigned_at TEXT NOT NULL,
              PRIMARY KEY(collection_mode, participant_hash, order_index));
            CREATE TABLE IF NOT EXISTS responses(
              response_id INTEGER PRIMARY KEY AUTOINCREMENT,
              collection_mode TEXT NOT NULL, participant_hash TEXT NOT NULL,
              expertise_group TEXT NOT NULL, order_index INTEGER NOT NULL,
              opaque_item_id TEXT NOT NULL, stimulus_id TEXT NOT NULL,
              source_stage TEXT NOT NULL, correctness_condition TEXT NOT NULL,
              trust_rating REAL NOT NULL, trust_items_json TEXT NOT NULL,
              perceived_correctness INTEGER NOT NULL, fluency_rating INTEGER NOT NULL,
              perceived_confidence INTEGER NOT NULL, decision_action TEXT NOT NULL,
              accepted INTEGER NOT NULL, response_time_sec REAL NOT NULL,
              submitted_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS phase16_initial_ratings_v2(
              collection_mode TEXT NOT NULL, participant_hash TEXT NOT NULL,
              order_index INTEGER NOT NULL, trust_rating REAL NOT NULL,
              trust_items_json TEXT NOT NULL, perceived_correctness INTEGER NOT NULL,
              fluency_rating INTEGER NOT NULL, perceived_confidence INTEGER NOT NULL,
              locked_at TEXT NOT NULL,
              PRIMARY KEY(collection_mode, participant_hash, order_index));
            CREATE TABLE IF NOT EXISTS phase16_trial_events_v2(
              id INTEGER PRIMARY KEY AUTOINCREMENT, collection_mode TEXT NOT NULL,
              participant_hash TEXT NOT NULL, order_index INTEGER NOT NULL,
              event_type TEXT NOT NULL, payload_json TEXT NOT NULL, elapsed_sec REAL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS phase16_secondary_questionnaire_v2(
              collection_mode TEXT NOT NULL, participant_hash TEXT NOT NULL,
              ratings_json TEXT NOT NULL, open_comment TEXT, submitted_at TEXT NOT NULL,
              PRIMARY KEY(collection_mode, participant_hash));
            """)
            for definition in ["counterbalance_arm TEXT", "protocol_version TEXT"]:
                self._add_column(con, "participants", definition)
            for definition in ["scenario_id TEXT", "framework_context TEXT", "counterbalance_arm TEXT"]:
                self._add_column(con, "assignments", definition)
            for definition in ["pair_id TEXT", "scenario_id TEXT", "framework_context TEXT",
                               "override_targets_json TEXT", "override_correction_text TEXT",
                               "evidence_question_count INTEGER DEFAULT 0", "initial_locked_at TEXT"]:
                self._add_column(con, "responses", definition)

    def participant(self, mode: str, ph: str) -> Optional[dict]:
        with self._connect() as con:
            row = con.execute("SELECT * FROM participants WHERE collection_mode=? AND participant_hash=?", (mode, ph)).fetchone()
        return None if row is None else dict(row)

    def registration_count(self, mode: str) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM participants WHERE collection_mode=?", (mode,)).fetchone()[0])

    def register(self, mode: str, ph: str, group: str, expertise: int, assignments: Sequence[Assignment], protocol_sha: str, stimulus_sha: str):
        arm = assignments[0].counterbalance_arm
        with self._connect() as con:
            con.execute("""INSERT INTO participants(collection_mode,participant_hash,expertise_group,expertise_self_rating,n_items,
            consented,track1_author_attested_separate,randomization_version,protocol_sha256,stimulus_manifest_sha256,
            registered_at,counterbalance_arm,protocol_version) VALUES(?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP,?,?)""",
            (mode, ph, group, int(expertise), len(assignments), 1, 1, RANDOMIZATION_VERSION, protocol_sha, stimulus_sha, arm, SCHEMA_VERSION))
            for a in assignments:
                con.execute("""INSERT INTO assignments(collection_mode,participant_hash,order_index,opaque_item_id,stimulus_id,pair_id,
                source_stage,correctness_condition,error_type,randomization_version,assigned_at,scenario_id,framework_context,counterbalance_arm)
                VALUES(?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP,?,?,?)""",
                (mode, ph, a.order_index, a.opaque_item_id, a.stimulus_id, a.pair_id, a.source_stage,
                 a.correctness_condition, a.error_type, RANDOMIZATION_VERSION, a.scenario_id, a.framework_context, arm))

    def assignment_rows(self, mode: str, ph: Optional[str] = None) -> List[dict]:
        q, params = "SELECT * FROM assignments WHERE collection_mode=?", [mode]
        if ph is not None:
            q += " AND participant_hash=?"; params.append(ph)
        q += " ORDER BY participant_hash,order_index"
        with self._connect() as con:
            return [dict(r) for r in con.execute(q, params).fetchall()]

    def response_rows(self, mode: str, ph: Optional[str] = None) -> List[dict]:
        q, params = "SELECT * FROM responses WHERE collection_mode=?", [mode]
        if ph is not None:
            q += " AND participant_hash=?"; params.append(ph)
        q += " ORDER BY participant_hash,order_index"
        with self._connect() as con:
            return [dict(r) for r in con.execute(q, params).fetchall()]

    def participant_rows(self, mode: str) -> List[dict]:
        with self._connect() as con:
            return [dict(r) for r in con.execute("SELECT * FROM participants WHERE collection_mode=? ORDER BY registered_at", (mode,)).fetchall()]

    def current_order(self, mode: str, ph: str) -> Optional[int]:
        p = self.participant(mode, ph)
        if not p: return None
        with self._connect() as con:
            done = {int(r[0]) for r in con.execute("SELECT order_index FROM responses WHERE collection_mode=? AND participant_hash=?", (mode, ph)).fetchall()}
        for i in range(1, int(p["n_items"]) + 1):
            if i not in done: return i
        return None

    def save_initial(self, mode: str, ph: str, order: int, trust_items: Dict[str, int], perceived: int, fluency: int, confidence: int, trust_rating: float):
        with self._connect() as con:
            con.execute("""INSERT OR REPLACE INTO phase16_initial_ratings_v2(collection_mode,participant_hash,order_index,trust_rating,
            trust_items_json,perceived_correctness,fluency_rating,perceived_confidence,locked_at)
            VALUES(?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""", (mode, ph, order, trust_rating, _json(trust_items), perceived, fluency, confidence))

    def initial(self, mode: str, ph: str, order: int) -> Optional[dict]:
        with self._connect() as con:
            row = con.execute("SELECT * FROM phase16_initial_ratings_v2 WHERE collection_mode=? AND participant_hash=? AND order_index=?", (mode, ph, order)).fetchone()
        if not row: return None
        out = dict(row); out["trust_items"] = json.loads(out.pop("trust_items_json")); return out

    def log_event(self, mode: str, ph: str, order: int, event_type: str, payload: dict, elapsed: Optional[float] = None):
        with self._connect() as con:
            con.execute("INSERT INTO phase16_trial_events_v2(collection_mode,participant_hash,order_index,event_type,payload_json,elapsed_sec) VALUES(?,?,?,?,?,?)",
                        (mode, ph, order, event_type, _json(payload), elapsed))

    def event_rows(self, mode: str, ph: Optional[str] = None) -> List[dict]:
        q, params = "SELECT * FROM phase16_trial_events_v2 WHERE collection_mode=?", [mode]
        if ph is not None:
            q += " AND participant_hash=?"; params.append(ph)
        q += " ORDER BY id"
        with self._connect() as con:
            return [dict(r) for r in con.execute(q, params).fetchall()]

    def save_response(self, mode: str, ph: str, assignment: dict, decision: str, response_time: float, override_targets: List[str], correction: str):
        init = self.initial(mode, ph, int(assignment["order_index"]))
        if init is None: raise StudyV2Error("Initial evaluation must be locked before the final decision.")
        with self._connect() as con:
            qcount = int(con.execute("SELECT COUNT(*) FROM phase16_trial_events_v2 WHERE collection_mode=? AND participant_hash=? AND order_index=? AND event_type='evidence_question'",
                                     (mode, ph, assignment["order_index"])).fetchone()[0])
            group = con.execute("SELECT expertise_group FROM participants WHERE collection_mode=? AND participant_hash=?", (mode, ph)).fetchone()[0]
            con.execute("""INSERT INTO responses(collection_mode,participant_hash,expertise_group,order_index,opaque_item_id,stimulus_id,
            source_stage,correctness_condition,trust_rating,trust_items_json,perceived_correctness,fluency_rating,perceived_confidence,
            decision_action,accepted,response_time_sec,submitted_at,pair_id,scenario_id,framework_context,override_targets_json,
            override_correction_text,evidence_question_count,initial_locked_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP,?,?,?,?,?,?,?)""",
            (mode, ph, group, assignment["order_index"], assignment["opaque_item_id"], assignment["stimulus_id"], assignment["source_stage"],
             assignment["correctness_condition"], init["trust_rating"], _json(init["trust_items"]), init["perceived_correctness"], init["fluency_rating"],
             init["perceived_confidence"], decision, 1 if decision == "Accept" else 0, response_time, assignment["pair_id"], assignment.get("scenario_id"),
             assignment.get("framework_context"), _json(override_targets), str(correction or "").strip(), qcount, init["locked_at"]))
        if self.current_order(mode, ph) is None:
            with self._connect() as con:
                con.execute("UPDATE participants SET completed_at=CURRENT_TIMESTAMP WHERE collection_mode=? AND participant_hash=? AND completed_at IS NULL", (mode, ph))

    def secondary_exists(self, mode: str, ph: str) -> bool:
        with self._connect() as con:
            return con.execute("SELECT 1 FROM phase16_secondary_questionnaire_v2 WHERE collection_mode=? AND participant_hash=?", (mode, ph)).fetchone() is not None

    def save_secondary(self, mode: str, ph: str, ratings: dict, comment: str):
        with self._connect() as con:
            con.execute("INSERT INTO phase16_secondary_questionnaire_v2(collection_mode,participant_hash,ratings_json,open_comment,submitted_at) VALUES(?,?,?,?,CURRENT_TIMESTAMP)",
                        (mode, ph, _json(ratings), str(comment or "")[:4000]))

    def secondary_rows(self, mode: str) -> List[dict]:
        with self._connect() as con:
            return [dict(r) for r in con.execute(
                "SELECT * FROM phase16_secondary_questionnaire_v2 WHERE collection_mode=? ORDER BY submitted_at",
                (mode,),
            ).fetchall()]


class TrustCalibrationStudyV2:
    def __init__(self, protocol_path: Optional[Path] = None, bank_root: Optional[Path] = None, db_path: Optional[Path] = None, id_salt: Optional[str] = None):
        self.protocol_path = Path(protocol_path or DEFAULT_PROTOCOL_PATH)
        self.protocol = load_protocol(self.protocol_path)
        self.bank = StimulusBankV2(bank_root)
        self.store = Phase16V2Store(db_path)
        self.id_salt = str(id_salt or os.getenv("AWAREML_PHASE16_ID_SALT") or "phase16-v2-pilot-local-salt")

    @staticmethod
    def collection_mode() -> str:
        return "final" if str(os.getenv("AWAREML_PHASE16_COLLECTION_MODE", "pilot")).strip().lower() == "final" else "pilot"

    @property
    def protocol_sha(self): return _sha(self.protocol_path)
    @property
    def stimulus_sha(self): return _sha(self.bank.manifest_path)

    def assert_final_ready(self):
        manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen": raise StudyV2Error("Phase-16 v2 design is not frozen.")
        expected = str(((manifest.get("files") or {}).get("protocol.json") or {}).get("sha256") or "")
        source = str((manifest.get("source_stimulus_bank") or {}).get("manifest_sha256") or "")
        if expected != self.protocol_sha or source != self.stimulus_sha: raise StudyV2Error("Phase-16 v2 frozen design integrity check failed.")
        if str(os.getenv("AWAREML_PHASE16_FINAL_ARMED", "")).strip().upper() != "YES": raise StudyV2Error("Set AWAREML_PHASE16_FINAL_ARMED=YES before collecting revised Main Study responses.")
        if self.id_salt == "phase16-v2-pilot-local-salt" or len(self.id_salt) < 16: raise StudyV2Error("Revised Main Study requires a private AWAREML_PHASE16_ID_SALT of at least 16 characters.")

    def participant_hash(self, code: str, mode: str) -> str:
        code = str(code or "").strip()
        if len(code) < 3: raise StudyV2Error("Participant/session code must contain at least 3 characters.")
        if mode == "final" and (self.id_salt == "phase16-v2-pilot-local-salt" or len(self.id_salt) < 16): raise StudyV2Error("Revised Main Study requires a private participant-ID salt.")
        return hmac.new(self.id_salt.encode(), f"{SCHEMA_VERSION}|{mode}|{code}".encode(), hashlib.sha256).hexdigest()[:24]

    def _assignment(self, ph: str, arm: str) -> List[Assignment]:
        pair_ids = list(self.bank.pair_order)
        base = {pair_id: ("correct" if idx % 2 == 0 else "incorrect") for idx, pair_id in enumerate(pair_ids)}
        if arm == "B": base = {p: ("incorrect" if c == "correct" else "correct") for p, c in base.items()}
        ok, ordered = False, pair_ids
        for attempt in range(1000):
            ordered = sorted(pair_ids, key=lambda p: _stable_int(ph, RANDOMIZATION_VERSION, "order", str(attempt), p))
            conds = [base[p] for p in ordered]
            run, ok = 1, True
            for j in range(1, len(conds)):
                run = run + 1 if conds[j] == conds[j-1] else 1
                if run > 3: ok = False; break
            if ok: break
        if not ok: raise StudyV2Error("Unable to create balanced randomized order.")
        out = []
        for order, pair_id in enumerate(ordered, 1):
            cond, stim = base[pair_id], self.bank.pairs[pair_id][base[pair_id]]
            opaque = "T16V2-" + hmac.new(ph.encode(), f"{order}|{stim.stimulus_id}".encode(), hashlib.sha256).hexdigest()[:10].upper()
            out.append(Assignment(order, opaque, stim.stimulus_id, pair_id, stim.scenario_id, stim.framework_context, stim.source_stage, cond, stim.error_type, arm))
        if sum(x.correctness_condition == "correct" for x in out) != 10: raise StudyV2Error("Assignment is not 10/10 correct/incorrect.")
        for fw in ALLOWED_FRAMEWORKS:
            items = [x for x in out if x.framework_context == fw]
            if len(items) != 4 or sum(x.correctness_condition == "correct" for x in items) != 2: raise StudyV2Error("Framework-level balance failed for {}".format(fw))
        return out

    def register(self, code: str, group: str, expertise: int, consented: bool, track1_separate: bool, mode: Optional[str] = None) -> dict:
        mode = mode or self.collection_mode()
        if mode == "final": self.assert_final_ready()
        if not consented: raise StudyV2Error("Participation and data-processing consent are required.")
        if not track1_separate: raise StudyV2Error("People who authored the Track-1 scenario bank are not eligible.")
        if group not in ALLOWED_EXPERTISE_GROUPS: raise StudyV2Error("Unknown experience group.")
        if not 1 <= int(expertise) <= 7: raise StudyV2Error("ML/AutoML knowledge/experience must be 1–7.")
        ph = self.participant_hash(code, mode)
        existing = self.store.participant(mode, ph)
        if existing:
            if existing.get("protocol_version") != SCHEMA_VERSION: raise StudyV2Error("This code belongs to a pre-revision session. Use a new participant code for v2.1.")
            return existing
        arm = "A" if self.store.registration_count(mode) % 2 == 0 else "B"
        assignments = self._assignment(ph, arm)
        self.store.register(mode, ph, group, expertise, assignments, self.protocol_sha, self.stimulus_sha)
        return self.store.participant(mode, ph) or {}

    def current_trial(self, ph: str, mode: Optional[str] = None) -> Optional[dict]:
        mode, order = mode or self.collection_mode(), self.store.current_order(mode or self.collection_mode(), ph)
        if order is None: return None
        row = next(r for r in self.store.assignment_rows(mode, ph) if int(r["order_index"]) == order)
        out = self.bank.get(row["stimulus_id"]).participant_dict(row["opaque_item_id"], order)
        out["pair_id"], out["counterbalance_arm"] = row["pair_id"], row.get("counterbalance_arm")
        return out

    def lock_initial(self, ph: str, order: int, trust_items: Dict[str, int], perceived: int, fluency: int, confidence: int, mode: Optional[str] = None) -> float:
        mode = mode or self.collection_mode(); vals = []
        for item_id in ["stias_confident", "stias_reliable", "stias_trust"]:
            value = trust_items.get(item_id)
            if value is None or not 1 <= int(value) <= 7: raise StudyV2Error("Complete all three S-TIAS statements.")
            vals.append(int(value))
        for name, value in [("perceived correctness", perceived), ("fluency", fluency), ("confident-sounding language", confidence)]:
            if value is None or not 1 <= int(value) <= 7: raise StudyV2Error("Complete the {} rating.".format(name))
        score = sum(vals) / 3.0
        self.store.save_initial(mode, ph, order, trust_items, perceived, fluency, confidence, score)
        self.store.log_event(mode, ph, order, "initial_evaluation_locked", {"trust_rating": score, "perceived_correctness": perceived, "fluency_rating": fluency, "perceived_confidence": confidence})
        return score

    def final_submit(self, ph: str, order: int, decision: str, response_time: float, override_targets: List[str], correction: str, mode: Optional[str] = None):
        mode = mode or self.collection_mode()
        if decision not in ALLOWED_DECISION_ACTIONS: raise StudyV2Error("Choose Accept, Override or Reject.")
        if decision == "Override" and (not override_targets or not str(correction or "").strip()): raise StudyV2Error("For Override, select what you would correct and briefly state the correction.")
        a = next(r for r in self.store.assignment_rows(mode, ph) if int(r["order_index"]) == order)
        self.store.log_event(mode, ph, order, "final_decision", {"decision": decision, "override_targets": override_targets, "correction": str(correction or "").strip()})
        self.store.save_response(mode, ph, a, decision, max(float(response_time), 0.001), override_targets, correction)

    def progress(self, ph: str, mode: Optional[str] = None) -> Tuple[int, int]:
        mode = mode or self.collection_mode(); p = self.store.participant(mode, ph)
        return (0, 20) if not p else (len(self.store.response_rows(mode, ph)), int(p["n_items"]))


ALLOWED_FRAMEWORKS = ("AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha")


def mean_sd(values: Sequence[Any]) -> Tuple[Optional[float], Optional[float]]:
    s = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return (None, None) if s.empty else (float(s.mean()), None if len(s) < 2 else float(s.std(ddof=1)))


def analyze_v2(store: Phase16V2Store, mode: str = "final") -> Dict[str, Any]:
    participants, responses = pd.DataFrame(store.participant_rows(mode)), pd.DataFrame(store.response_rows(mode))
    if participants.empty: return {"status": "insufficient_data", "registered": 0, "completed": 0}
    completed = participants[participants["completed_at"].notna()].copy()
    valid = set(completed["participant_hash"].astype(str))
    frame = responses[responses["participant_hash"].astype(str).isin(valid)].copy() if not responses.empty else responses
    out = {"status": "ok" if len(frame) else "insufficient_data", "registered": len(participants), "completed": len(completed), "trials": len(frame)}
    if frame.empty: return out
    for c in ["trust_rating", "response_time_sec", "order_index", "evidence_question_count"]: frame[c] = pd.to_numeric(frame[c], errors="coerce")
    out["condition_summary"] = {}
    for cond in ["correct", "incorrect"]:
        g = frame[frame["correctness_condition"] == cond]; mean, sd = mean_sd(g["trust_rating"])
        out["condition_summary"][cond] = {"n": len(g), "mean_trust": mean, "sd_trust": sd, "acceptance_rate": None if len(g) == 0 else float((g["decision_action"] == "Accept").mean())}
    pivot = frame.pivot_table(index="participant_hash", columns="correctness_condition", values="trust_rating", aggfunc="mean")
    if "correct" in pivot and "incorrect" in pivot:
        mean, sd = mean_sd(pivot["correct"] - pivot["incorrect"]); out["trust_gap"] = {"mean": mean, "sd": sd, "n": len(pivot)}
    out["pair_coverage"] = [{"pair_id": p, "scenario_id": s, "framework": fw, "source_stage": st, "evaluations": len(g), "correct": int((g["correctness_condition"] == "correct").sum()), "incorrect": int((g["correctness_condition"] == "incorrect").sum())}
                            for (p, s, fw, st), g in frame.groupby(["pair_id", "scenario_id", "framework_context", "source_stage"], dropna=False)]
    out["decision_counts"] = frame["decision_action"].value_counts().to_dict()
    overrides = frame[frame["decision_action"] == "Override"]
    out["override_correction_capture_rate"] = None if len(overrides) == 0 else float(overrides["override_correction_text"].fillna("").astype(str).str.strip().ne("").mean())
    mean, sd = mean_sd(frame["evidence_question_count"]); out["evidence_questions"] = {"mean": mean, "sd": sd}
    out["expertise_descriptive"] = []
    for exp, g in frame.groupby("expertise_group"):
        mean, sd = mean_sd(g["trust_rating"]); out["expertise_descriptive"].append({"expertise_group": exp, "participants": g["participant_hash"].nunique(), "mean_trust": mean, "sd_trust": sd})
    out["trust_vs_trial_order_spearman"] = float(frame["trust_rating"].rank().corr(frame["order_index"].rank()))
    out["response_time_vs_trial_order_spearman"] = float(frame["response_time_sec"].rank().corr(frame["order_index"].rank()))
    out["completed_counterbalance_arms"] = completed["counterbalance_arm"].value_counts().to_dict() if "counterbalance_arm" in completed else {}
    return out
