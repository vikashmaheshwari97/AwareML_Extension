"""Optional white-box collector. Run in a separate Python 3.11+ environment.

Independent grouped permutation-Shapley adaptation, NOT an exact CC-SHAP port.
The default does not execute remote model code. Model downloads require explicit
--allow-download. Input records contain the actual fixed decision/explanation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awareml.explanation_integrity.sequence_attribution import paired_shapley


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Local causal-LM directory or Hugging Face model ID")
    parser.add_argument("--revision", default="main", help="Prefer an immutable model commit")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--permutations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 2 <= args.groups <= 64 or args.permutations < 2:
        parser.error("groups must be 2–64 and permutations >= 2")
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                              local_files_only=not args.allow_download, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                local_files_only=not args.allow_download, trust_remote_code=False)
    model = model.to(args.device).eval()
    rows = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input must be a nonempty list of fixed generation records.")
    out = []
    for row in rows:
        for k in ("case_id", "input", "decision", "explanation"):
            if not isinstance(row.get(k), str) or not row[k].strip():
                raise ValueError("Missing nonempty generation field: " + k)
        tokens = tokenizer.encode(row["input"], add_special_tokens=False)
        n = min(args.groups, len(tokens))
        if n < 2:
            raise ValueError("Input needs at least two tokens.")
        groups = [g.tolist() for g in np.array_split(np.arange(len(tokens)), n)]
        suffixes = [row.get("decision_suffix", "\nAnswer:"),
                    row.get("explanation_suffix", "\nAnswer: " + row["decision"] + "\nBrief explanation:")]
        targets = [row["decision"], row["explanation"]]
        suffix_ids = [tokenizer.encode(s, add_special_tokens=False) for s in suffixes]
        target_ids = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
        bos = tokenizer.bos_token_id
        prefix = [bos] if bos is not None else tokenizer.encode("Context:\n", add_special_tokens=False)
        if not prefix or any(not target for target in target_ids):
            raise ValueError("Tokenizer must yield a prefix and nonempty targets.")
        max_context = getattr(model.config, "max_position_embeddings", 2048)
        if max(len(prefix) + len(tokens) + len(suffix_ids[j]) + len(target_ids[j]) for j in range(2)) > max_context:
            raise ValueError("Input and target exceed model context; truncate explicitly, never silently.")

        def score(selected):
            kept = [tokens[i] for gi, group in enumerate(groups) if gi in selected for i in group]
            values = []
            for suffix, target in zip(suffix_ids, target_ids):
                context = prefix + kept + suffix
                ids = torch.tensor([context + target], device=args.device)
                with torch.no_grad():
                    logits = model(input_ids=ids).logits[0]
                    positions = logits[len(context) - 1:len(context) + len(target) - 1]
                    scores = torch.log_softmax(positions.float(), dim=-1)
                    value = scores.gather(1, torch.tensor(target, device=args.device)[:, None]).sum()
                values.append(float(value.cpu()))
            return values

        measured = paired_shapley(score, n, args.permutations, args.seed)
        record = {"case_id": row["case_id"], "input_sha256": hashlib.sha256(row["input"].encode()).hexdigest(),
                  "model": args.model, "model_revision": getattr(model.config, "_commit_hash", None) or args.revision,
                  "attribution_method": "grouped_permutation_shapley_sequence_log_probability",
                  "target_definition": "sum teacher-forced log P(fixed output tokens | coalition input, fixed suffix)",
                  "feature_ids": ["token_group_%d" % i for i in range(n)],
                  "feature_labels": [tokenizer.decode([tokens[i] for i in group]) for group in groups],
                  "token_indices": groups, "decision": row["decision"], "explanation": row["explanation"],
                  "split": row.get("split", "test"), "suffixes": suffixes,
                  "masking": "delete contiguous input token groups, retain original order and fixed suffixes",
                  "limitations": "Grouped, deletion-baseline Shapley approximation; not original CC-SHAP normalization. Masked inputs may be out of distribution.",
                  **measured}
        out.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Checkpoint after every expensive case, with an atomic replacement.
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temp.replace(args.output)
        print("Collected", row["case_id"], "coalitions:", measured["coalitions_evaluated"])


if __name__ == "__main__":
    main()
