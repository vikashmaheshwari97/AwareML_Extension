
from __future__ import annotations



import argparse

import json

from pathlib import Path





def fail(msg: str):

    raise SystemExit(f"QC FAILED: {msg}")





p = argparse.ArgumentParser()

p.add_argument("--summary", required=True)

p.add_argument("--framework", required=True)

p.add_argument("--sha256", required=True)

p.add_argument("--success", required=True)

args = p.parse_args()



summary_path = Path(args.summary)

if not summary_path.exists():

    fail(f"summary missing: {summary_path}")



with summary_path.open() as f:

    data = json.load(f)



if not isinstance(data, list) or len(data) != 1:

    fail(f"expected exactly one result, got {type(data).__name__}/{len(data) if isinstance(data, list) else 'NA'}")



r = data[0]



if r.get("status") != "ok":

    fail(f"framework status={r.get('status')} error={r.get('error')}")



if r.get("framework") != args.framework:

    fail(f"framework mismatch: expected={args.framework}, got={r.get('framework')}")



prov = r.get("dataset_provenance") or {}

if prov.get("source_sha256") != args.sha256:

    fail(

        f"dataset SHA mismatch: expected={args.sha256}, "

        f"got={prov.get('source_sha256')}"

    )



if int(r.get("samples") or 0) <= 0:

    fail("zero processed samples")



s = r.get("sustainability") or {}

if s.get("status") != "measured":

    fail(f"sustainability status={s.get('status')}")



energy = s.get("energy_kwh")

co2 = s.get("co2_kg")



if energy is None or float(energy) <= 0:

    fail(f"invalid energy_kwh={energy}")



if co2 is None or float(co2) <= 0:

    fail(f"invalid co2_kg={co2}")



params = r.get("parameters") or {}

backend = str(r.get("backend") or "")



if args.framework == "ChaCha":

    if params.get("native_autovw_active") is not True:

        fail(f"ChaCha native AutoVW inactive: {params}")

    if "AutoVW" not in backend:

        fail(f"unexpected ChaCha backend={backend}")



elif args.framework == "OAML":

    if params.get("mode") != "online":

        fail(f"OAML mode is not online: {params.get('mode')}")

    if params.get("river_version") != "0.8.0":

        fail(f"OAML River mismatch: {params.get('river_version')}")

    if params.get("isolated_environment") is not True:

        fail("OAML isolated environment flag missing")



elif args.framework == "EvoAutoML":

    if params.get("evo_version") != "0.0.14":

        fail(f"EvoAutoML version mismatch: {params.get('evo_version')}")

    if params.get("isolated_environment") is not True:

        fail("EvoAutoML isolated environment flag missing")

    if "native" not in backend.lower():

        fail(f"unexpected EvoAutoML backend={backend}")



success = {

    "status": "success",

    "framework": args.framework,

    "backend": backend,

    "samples": r.get("samples"),

    "accuracy": r.get("accuracy"),

    "f1_macro": r.get("f1_macro"),

    "runtime_sec": r.get("runtime_sec"),

    "energy_kwh": energy,

    "co2_kg": co2,

    "experiment_id": r.get("experiment_id"),

}



success_path = Path(args.success)

success_path.write_text(json.dumps(success, indent=2) + "\n")



print("QC OK")

print(json.dumps(success, indent=2))

