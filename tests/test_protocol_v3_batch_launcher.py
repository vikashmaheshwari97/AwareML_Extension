"""Standard-library checks; no frameworks, energy measurement or Slurm jobs run."""
import importlib.util
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "hpc/protocol_v3/batch_launcher.py"
SPEC = importlib.util.spec_from_file_location("batch_launcher", SCRIPT)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sha = "campaign-sha"
        self.rows = [{"task_id": str(i), "dataset_id": "d%d" % i,
                      "framework": "AutoClass", "seed": "42", "expected_samples": "100"}
                     for i in range(8)]
        import csv
        with (self.root / "tasks.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.manifest = {"campaign_sha256": self.sha, "frozen": {"protocol": {
            "cpu_threads": 5, "safety_timeout_sec": 7200}}}
        batch.STOP = False
        self.save(0, self.good(0))

    def good(self, i):
        return {"task": self.rows[i], "campaign_sha256": self.sha,
                "status": "completed", "eligible": True, "samples_processed": 100,
                "accuracy": .8, "runtime_sec": 10, "energy_kwh": .001,
                "co2_kg": .0004, "peak_process_tree_rss_mib": 100,
                "energy_status": "measured", "tracking_mode": "machine",
                "isolation": {"kind": "slurm_exclusive"},
                "measurement_provenance": {"gpu_energy_kwh": 0, "host_ram_gib": 1000},
                "environment": {"cpu_model": "AMD EPYC 7763 64-Core Processor",
                                "thread_limits": {k: "5" for k in
                                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}}

    def save(self, i, result, attempt=1):
        path = self.root / "runs" / ("task_%05d" % i) / ("attempt_%03d" % attempt) / "result.json"
        batch.write(path, result)

    def plan(self, retries=()):
        return batch.build_plan(self.root, self.manifest, 4, 12, ["123"], retries)

    def test_read_only_inventory_preserves_frozen_bytes(self):
        before = (self.root / "tasks.csv").read_bytes()
        inv = batch.inventory(self.root, self.manifest)
        self.assertEqual(inv["completed"], [0])
        self.assertEqual(inv["fresh"], list(range(1, 8)))
        self.assertEqual(before, (self.root / "tasks.csv").read_bytes())

    def test_shards_are_disjoint_and_preserve_completed(self):
        plan = self.plan()
        ids = [int(item["task"]["task_id"]) for shard in plan["shards"] for item in shard]
        self.assertEqual(sorted(ids), list(range(1, 8)))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(plan["preserved_completed"], [0])

    def test_incomplete_attempt_requires_explicit_retry(self):
        (self.root / "runs/task_00001/attempt_001").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "Review existing"):
            self.plan()
        plan = self.plan([1])
        item = next(x for s in plan["shards"] for x in s if int(x["task"]["task_id"]) == 1)
        self.assertTrue(item["retry_authorized"])

    def test_latest_failed_attempt_is_not_hidden(self):
        self.save(0, {"status": "failed"}, attempt=2)
        with self.assertRaisesRegex(ValueError, "Review existing"):
            self.plan()

    def test_max_nodes_and_watchdog_headroom(self):
        for nodes, hours in ((8, 12), (0, 12), (4, 2), (4, 200)):
            with self.assertRaises(ValueError):
                batch.build_plan(self.root, self.manifest, nodes, hours, ["123"], [])

    def test_active_jobs_block_handover(self):
        with patch.object(batch.subprocess, "check_output", return_value="123|oldjob\n456|awareml-v3-cpu2\n"):
            with self.assertRaisesRegex(RuntimeError, "123, 456"):
                batch.assert_drained(["123"])

    def test_scheduler_error_fails_closed(self):
        with patch.object(batch.subprocess, "check_output", side_effect=subprocess.CalledProcessError(1, "squeue")):
            with self.assertRaises(subprocess.CalledProcessError):
                batch.assert_drained(["123"])

    def test_sbatch_retains_scope_budget_and_single_node(self):
        command = batch.submit_command(self.plan(), self.root / "plan")
        for flag in ("--exclusive", "--mem=0", "--cpus-per-task=5", "--nodes=1",
                     "--time=12:00:00", "--array=0-3%4", "--partition=main"):
            self.assertIn(flag, command)
        self.assertTrue(command[-1].startswith("--wrap=exec "))

    def test_slurm_time_parse_and_exclusive_requirement(self):
        self.assertEqual(batch.parse_slurm_duration("1-02:03:04"), 93784)
        self.assertEqual(batch.parse_slurm_duration("02:10:00"), 7800)
        self.assertEqual(batch.parse_slurm_duration("03:04"), 184)
        with patch.dict(batch.os.environ, {"SLURM_JOB_ID": "123"}):
            with patch.object(batch.subprocess, "check_output", return_value="OverSubscribe=NO TimeLimit=12:00:00 RunTime=00:10:00"):
                self.assertEqual(batch.allocation_seconds_left(), 42600)
            with patch.object(batch.subprocess, "check_output", return_value="OverSubscribe=YES TimeLimit=12:00:00 RunTime=00:10:00"):
                with self.assertRaisesRegex(RuntimeError, "not exclusive"):
                    batch.allocation_seconds_left()

    def test_no_new_task_near_allocation_end(self):
        plan = self.plan()
        with patch.object(batch.time, "monotonic", return_value=100):
            result = batch.run_shard(plan, plan["shards"][0], self.root, 100 + 7799,
                                     runner=lambda *a, **k: self.fail("Task should not start"))
        self.assertEqual(result["state"], "allocation_boundary")

    def test_launch_unchanged_executor_and_measure_each_task(self):
        plan = self.plan()
        calls = []
        def runner(command, **kwargs):
            i = int(command[command.index("--task-id") + 1])
            calls.append(i)
            self.assertNotIn("--dedicated-host", command)
            self.save(i, self.good(i))
            return 0
        result = batch.run_shard(plan, plan["shards"][0], self.root, time.monotonic() + 50000,
                                 runner=runner, node_check=lambda p: None)
        self.assertEqual(result["state"], "shard_complete")
        self.assertEqual(calls, [1, 5])

    def test_stops_on_measurement_failure_without_retrying(self):
        plan = self.plan()
        calls = []
        def runner(command, **kwargs):
            i = int(command[command.index("--task-id") + 1])
            calls.append(i)
            self.save(i, {"status": "failed", "energy_status": "measurement_failed"})
            return 1
        result = batch.run_shard(plan, plan["shards"][0], self.root, time.monotonic() + 50000,
                                 runner=runner, node_check=lambda p: None)
        self.assertEqual(result["state"], "stopped_on_failure")
        self.assertEqual(calls, [1])

    def test_history_change_blocks_duplicate_or_silent_retry(self):
        plan = self.plan()
        (self.root / "runs/task_00001/attempt_001").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "history changed"):
            batch.run_shard(plan, plan["shards"][0], self.root, time.monotonic() + 50000,
                            node_check=lambda p: None)

    def test_newly_completed_task_is_skipped(self):
        plan = self.plan()
        self.save(1, self.good(1))
        result = batch.run_shard(plan, [plan["shards"][0][0]], self.root, time.monotonic() + 50000,
                                 runner=lambda *a, **k: self.fail("Must skip completed task"))
        self.assertEqual(result["outcomes"][0]["state"], "skipped_completed")

    def test_missing_metric_or_wrong_campaign_rejected(self):
        for key, value in (("energy_kwh", None), ("energy_kwh", float("nan")),
                           ("co2_kg", -1), ("campaign_sha256", "different"),
                           ("samples_processed", 99), ("eligible", False)):
            result = self.good(0)
            result[key] = value
            self.assertFalse(batch.eligible(result, self.rows[0], self.sha))

    def test_stale_node_lock_is_never_removed(self):
        plan = self.plan()
        lock = self.root / ".codecarbon.lock"
        lock.write_text("preserve")
        def node_text(path, *args, **kwargs):
            if str(path) == "/proc/cpuinfo":
                return "model name : AMD EPYC 7763 64-Core Processor"
            return "MemTotal: 1048576000 kB"
        with patch.object(batch.socket, "gethostname", return_value="artemis2"), \
             patch.object(batch.Path, "read_text", autospec=True, side_effect=node_text), \
             patch.dict(batch.os.environ, {"SLURM_CPUS_PER_TASK": "5"}), \
             patch.object(batch.tempfile, "gettempdir", return_value=str(self.root)):
            with self.assertRaisesRegex(RuntimeError, "node-local lock"):
                batch.check_node(plan)
        self.assertTrue(lock.exists())

    def test_preserved_hardware_and_budget_cannot_be_mixed(self):
        for edit in ("cpu", "threads", "gpu"):
            result = self.good(1)
            if edit == "cpu":
                result["environment"]["cpu_model"] = "Different CPU"
            elif edit == "threads":
                result["environment"]["thread_limits"]["OMP_NUM_THREADS"] = "10"
            else:
                result["measurement_provenance"]["gpu_energy_kwh"] = .01
            self.save(1, result)
            with self.assertRaises(ValueError):
                self.plan()

    def test_finish_current_task_on_stop_signal_then_launch_no_more(self):
        plan = self.plan()
        calls = []
        def runner(command, **kwargs):
            i = int(command[command.index("--task-id") + 1])
            calls.append(i)
            self.save(i, self.good(i))
            batch.stop_signal(None, None)
            return 0
        result = batch.run_shard(plan, plan["shards"][0], self.root, time.monotonic() + 50000,
                                 runner=runner, node_check=lambda p: None)
        self.assertEqual(result["state"], "allocation_boundary")
        self.assertEqual(calls, [1])

    def test_plan_tampering_rejected(self):
        plan = self.plan()
        payload = {"sha256": batch.digest(plan), "frozen": plan}
        payload["frozen"]["cpu_threads"] = 10
        batch.write(self.root / "plan/plan.json", payload)
        with self.assertRaisesRegex(ValueError, "changed"):
            batch.load_plan(self.root / "plan")

    def test_submission_is_once_only_and_dry_run_is_nonmutating(self):
        plan = self.plan()
        directory = self.root / "plan"
        directory.mkdir()
        (directory / "logs").mkdir()
        with patch.object(batch, "load_plan", return_value=plan), \
             patch.object(batch, "assert_drained"), \
             patch.object(batch.subprocess, "check_output", return_value="12345\n") as submit:
            batch.main(["submit", "--plan", str(directory), "--dry-run"])
            submit.assert_not_called()
            self.assertFalse((directory / "submission.json").exists())
            batch.main(["submit", "--plan", str(directory)])
            self.assertEqual(submit.call_count, 1)
            with self.assertRaises(FileExistsError):
                batch.main(["submit", "--plan", str(directory)])
            self.assertEqual(submit.call_count, 1)


if __name__ == "__main__":
    unittest.main()
