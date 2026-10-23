from __future__ import annotations

import collections
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from .common import (ROOT, LABELS, environment, load_campaign, read_json, safe_path,
                     sha256, utc_now, write_json)


def backend_check(framework, params):
    if params.get("research_error_count", 0):
        raise RuntimeError("Suppressed backend errors occurred.")
    if params.get("fallback_used") or "fallback" in str(getattr(framework, "backend", "")).lower():
        raise RuntimeError("Fallback-backed runs are not valid framework evidence.")
    if framework.name == "ChaCha" and not params.get("native_autovw_active"):
        raise RuntimeError("ChaCha-OVR requires active FLAML AutoVW.")
    if framework.name == "OAML" and params.get("mode") != "online":
        raise RuntimeError("The frozen V3 comparison uses OAML online mode.")
    if framework.name == "OAML" and params.get("river_version") != "0.8.0":
        raise RuntimeError("OAML must run in its pinned River 0.8.0 environment.")
    if framework.name == "EvoAutoML" and params.get("evo_version") != "0.0.14":
        raise RuntimeError("EvoAutoML must run in its pinned 0.0.14 environment.")


class Confusion:
    def __init__(self):
        self.total = self.correct = self.missing = 0
        self.tp, self.fp, self.fn = collections.Counter(), collections.Counter(), collections.Counter()

    def update(self, y, pred):
        self.total += 1
        if pred is None:
            self.missing += 1
        if pred is not None and pred == y:
            self.correct += 1
            self.tp[y] += 1
        else:
            self.fn[y] += 1
            if pred is not None:
                self.fp[pred] += 1

    @property
    def accuracy(self):
        return self.correct / self.total if self.total else None

    @property
    def f1(self):
        labels = set(self.tp) | set(self.fp) | set(self.fn)
        return float(np.mean([2 * self.tp[k] / (2 * self.tp[k] + self.fp[k] + self.fn[k])
                              if 2 * self.tp[k] + self.fp[k] + self.fn[k] else 0.0 for k in labels])) if labels else None


def prequential(framework, frame, target, profile_size, window_size, checkpoint=None):
    # No wall-clock termination condition. The parent process owns the safeguard.
    from awareml.data import StreamingEncoder
    encoder = StreamingEncoder()
    cumulative, suffix = Confusion(), Confusion()
    labels = {}
    correct_prefix, windows = 0, []
    rolling = collections.deque(maxlen=window_size)
    for index, row in enumerate(frame.itertuples(index=False, name=None)):
        record = dict(zip(frame.columns, row))
        raw = record.pop(target)
        if isinstance(raw, np.generic):
            raw = raw.item()
        token = (type(raw).__name__, str(raw))
        if token not in labels:
            labels[token] = len(labels)
        y = labels[token]
        x = encoder.transform_row(record)
        pred = framework.predict_one(x)
        if isinstance(pred, np.generic):
            pred = pred.item()
        cumulative.update(y, pred)
        if index >= profile_size:
            suffix.update(y, pred)
        else:
            correct_prefix = cumulative.correct
        rolling.append(int(pred is not None and pred == y))
        framework.learn_one(x, y)
        # A transition into fallback is rejected at the very instance it occurs.
        if getattr(framework, "_fallback_active", False) or getattr(framework, "research_error_count", 0):
            raise RuntimeError("Backend entered fallback or suppressed an error: {}".format(getattr(framework, "native_error", None)))
        done = index + 1
        if done % window_size == 0 or done == len(frame):
            params = framework.get_params()
            backend_check(framework, params)
            snapshot = {"samples_processed": done, "accuracy": cumulative.accuracy,
                        "correct": cumulative.correct, "rolling_accuracy": float(np.mean(rolling)),
                        "classes_seen": len(labels), "missing_predictions": cumulative.missing}
            windows.append(snapshot)
            if checkpoint:
                checkpoint(snapshot)
    return {"samples_processed": cumulative.total, "accuracy": cumulative.accuracy,
            "f1_macro": cumulative.f1, "correct_count": cumulative.correct,
            "missing_predictions": cumulative.missing, "correct_in_profile_prefix": correct_prefix,
            "post_profile_samples": suffix.total, "post_profile_accuracy": suffix.accuracy,
            "post_profile_f1_macro": suffix.f1, "classes_seen": len(labels), "windows": windows}


def worker(campaign_dir, task_id, attempt_dir):
    campaign_dir, attempt_dir = Path(campaign_dir), Path(attempt_dir)
    manifest = load_campaign(campaign_dir)
    cfg = manifest["frozen"]["protocol"]
    tasks = pd.read_csv(campaign_dir / "tasks.csv")
    task = tasks.loc[tasks.task_id.eq(int(task_id))].iloc[0].to_dict()
    datasets = {d["dataset_id"]: d for d in read_json(campaign_dir / "datasets.json")}
    dataset = datasets[task["dataset_id"]]
    stream_path = safe_path(campaign_dir, dataset["stream_path"])
    if sha256(stream_path) != dataset["stream_sha256"]:
        raise ValueError("The task stream differs from the frozen data.")
    frame = pd.read_csv(stream_path, low_memory=False)
    env = environment()
    production = manifest["frozen"]["mode"] == "production"
    if production and env["packages"]["river"] != cfg["required_main_river"]:
        raise RuntimeError("Main River version differs from the frozen protocol: {}".format(env["packages"]["river"]))
    tracker, framework = None, None
    result = {"task": task, "campaign_sha256": manifest["campaign_sha256"], "environment": env,
              "framework_label": LABELS[task["framework"]], "status": "failed", "eligible": False,
              "energy_kwh": None, "co2_kg": None, "started_utc": utc_now()}
    # Import modules before the declared timing interval, then time construction
    # and streaming together. CSV loading and post-hoc XAI are not in that interval.
    from awareml.frameworks import create_frameworks
    if cfg["track_energy"]:
        import codecarbon
        from codecarbon import OfflineEmissionsTracker
        if production and codecarbon.__version__ != cfg["codecarbon_version"]:
            raise RuntimeError("Pin CodeCarbon to {} for this campaign.".format(cfg["codecarbon_version"]))
        tracker = OfflineEmissionsTracker(country_iso_code=cfg["country_iso"],
                                         tracking_mode="machine", save_to_file=False, log_level="error",
                                         allow_multiple_runs=False, measure_power_secs=1)
        tracker.start()
    start = time.perf_counter()
    write_json(attempt_dir / "progress.json", {"phase": "initializing", "samples_processed": 0, "started_utc": utc_now()})
    try:
        framework = create_frameworks([task["framework"]], seed=int(task["seed"]))[0]
        backend_check(framework, framework.get_params())
        init_elapsed = time.perf_counter() - start
        def checkpoint(snapshot):
            write_json(attempt_dir / "progress.json", {"phase": "streaming", **snapshot})
        metrics = prequential(framework, frame, dataset["target"], int(cfg["profile_size"]), int(cfg["window_size"]), checkpoint)
        result.update(metrics)
        result["runtime_sec"] = time.perf_counter() - start
        result["initialization_sec"] = init_elapsed
        result["streaming_sec"] = result["runtime_sec"] - init_elapsed
        params = framework.get_params()
        backend_check(framework, params)
        result.update({"backend": framework.backend, "parameters": params})
        if result["samples_processed"] != int(task["expected_samples"]):
            raise RuntimeError("Incomplete stream execution.")
        result.update(status="completed", stop_reason="common_sample_cap_or_eof")
    except Exception as exc:
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
        if framework is not None:
            with contextlib.suppress(Exception):
                result.update(backend=framework.backend, parameters=framework.get_params())
        result["stop_reason"] = "backend_failure"
        result["runtime_sec"] = time.perf_counter() - start
    finally:
        if tracker:
            try:
                co2 = tracker.stop()
                final = tracker.final_emissions_data
                energy = getattr(final, "energy_consumed", None)
                if co2 is None or energy is None or not np.isfinite([float(energy), float(co2)]).all() or energy <= 0 or co2 < 0:
                    raise RuntimeError("CodeCarbon did not return valid measured energy and emissions.")
                result.update(energy_kwh=float(energy), co2_kg=float(co2), energy_status="measured",
                              tracking_mode="machine", carbon_intensity_g_kwh=float(co2) * 1000 / float(energy),
                              carbon_intensity_source="CodeCarbon Offline country factor " + cfg["country_iso"])
                result["measurement_provenance"] = {
                    "codecarbon_version": env["packages"]["codecarbon"],
                    "cpu_power_mode": str(getattr(getattr(tracker, "_cpu", None), "_mode", "unknown")),
                    "cpu_energy_kwh": getattr(final, "cpu_energy", None),
                    "ram_energy_kwh": getattr(final, "ram_energy", None),
                    "gpu_energy_kwh": getattr(final, "gpu_energy", None),
                    "power_sampling_secs": 1, "scope": "framework initialization plus streaming evaluation",
                    "region": cfg["country_iso"], "host_ram_gib": psutil.virtual_memory().total / 1073741824.0,
                }
            except Exception as exc:
                result.update(energy_status="measurement_failed", measurement_error=str(exc), status="failed")
        else:
            result["energy_status"] = "not_measured_smoke"
        if framework:
            framework.close()
    result["ended_utc"] = utc_now()
    result["eligible"] = bool(production and result["status"] == "completed" and result.get("energy_status") == "measured")
    write_json(attempt_dir / "worker_result.json", result)
    return result


@contextlib.contextmanager
def host_lock():
    path = Path(tempfile.gettempdir()) / "awareml-v3-measured-workload.lock"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("Another V3 workload holds {}. If a previous process died, inspect that lock before removing it.".format(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "started_utc": utc_now()}, f)
        yield
    finally:
        path.unlink()


def isolation_evidence(dedicated_host=False):
    job = os.getenv("SLURM_JOB_ID")
    if job:
        details = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True, timeout=15)
        if "OverSubscribe=NO" not in details and "Shared=0" not in details:
            raise RuntimeError("Slurm allocation is not recorded as exclusive. Submit with --exclusive.")
        return {"kind": "slurm_exclusive", "job_id": job, "allocation": details.strip()}
    if dedicated_host:
        return {"kind": "user_declared_dedicated_host", "declaration": "one measured workload and no other compute users"}
    raise RuntimeError("Measured execution needs an exclusive Slurm allocation or --dedicated-host on a dedicated workstation.")


def kill_tree(proc):
    # Capture children before killing the root; terminate the entire Unix group
    # as well so a worker created between enumeration and shutdown is not missed.
    try:
        children = psutil.Process(proc.pid).children(recursive=True)
    except psutil.Error:
        children = []
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in children:
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        proc.kill()
        proc.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


def run_task(campaign_dir, task_id, dedicated_host=False, retry=False):
    campaign_dir = Path(campaign_dir).resolve()
    manifest = load_campaign(campaign_dir)
    cfg = manifest["frozen"]["protocol"]
    tasks = pd.read_csv(campaign_dir / "tasks.csv")
    matches = tasks.loc[tasks.task_id.eq(int(task_id))]
    if len(matches) != 1:
        raise ValueError("Task ID not found in frozen manifest.")
    task = matches.iloc[0].to_dict()
    base = campaign_dir / "runs" / ("task_{:05d}".format(int(task_id)))
    attempts = sorted(base.glob("attempt_*")) if base.exists() else []
    if attempts and not retry:
        raise RuntimeError("This task has an attempt. Inspect it, then use --retry to preserve a new attempt.")
    isolation = isolation_evidence(dedicated_host) if cfg["track_energy"] else {"kind": "unmeasured_smoke"}
    with host_lock():
        attempt = base / ("attempt_{:03d}".format(len(attempts) + 1))
        attempt.mkdir(parents=True)
        env = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(int(cfg["cpu_threads"]))
        env.update(AWAREML_V3_STRICT="1", AWAREML_OAML_MODE="online", PYTHONHASHSEED=str(task["seed"]))
        command = [sys.executable, str(ROOT / "scripts/awareml_v3.py"), "worker", "--campaign", str(campaign_dir),
                   "--task-id", str(task_id), "--attempt", str(attempt)]
        started, peak, baseline, timed_out = time.monotonic(), 0, None, False
        known_children = set()
        with (attempt / "stdout.log").open("w") as out, (attempt / "stderr.log").open("w") as err:
            proc = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=out, stderr=err,
                                    start_new_session=(os.name != "nt"))
            try:
                while proc.poll() is None:
                    if time.monotonic() - started >= float(cfg["safety_timeout_sec"]):
                        timed_out = True
                        kill_tree(proc)
                        break
                    try:
                        root = psutil.Process(proc.pid)
                        children = root.children(recursive=True)
                        known_children.update(children)
                        rss = sum(p.memory_info().rss for p in [root] + children if p.is_running())
                        if baseline is None:
                            baseline = rss
                        peak = max(peak, rss)
                    except psutil.Error:
                        pass
                    time.sleep(float(cfg["memory_poll_sec"]))
            except BaseException:
                kill_tree(proc)
                raise
        # Release orphaned isolated workers even when the main worker exited.
        for child in known_children:
            try:
                if child.is_running():
                    child.kill()
            except psutil.Error:
                pass
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elapsed = time.monotonic() - started
        result_path = attempt / "worker_result.json"
        result = read_json(result_path) if result_path.exists() else {"task": task, "eligible": False}
        if timed_out:
            result.update(status="timeout", stop_reason="external_safety_timeout", eligible=False,
                          accuracy=None, f1_macro=None, runtime_sec=None, energy_kwh=None, co2_kg=None)
        elif proc.returncode != 0 or not result_path.exists():
            result.update(status="failed", stop_reason="worker_process_failure", eligible=False)
        if result.get("samples_processed") != int(task["expected_samples"]):
            result["eligible"] = False
        if not result.get("eligible") and (attempt / "progress.json").exists():
            result["last_checkpoint_diagnostic_only"] = read_json(attempt / "progress.json")
        result.update(task=task, campaign_sha256=manifest["campaign_sha256"], isolation=isolation,
                      peak_process_tree_rss_mib=peak / 1048576.0,
                      baseline_process_tree_rss_mib=(baseline / 1048576.0) if baseline is not None else None,
                      memory_sampling_interval_sec=float(cfg["memory_poll_sec"]),
                      memory_scope="sampled full worker lifetime including input data and isolated children",
                      watchdog_elapsed_sec=elapsed, attempt_number=len(attempts) + 1, parent_environment=environment())
        if peak <= 0:
            result["eligible"] = False
        write_json(attempt / "result.json", result)
        return result
