"""Synthetic scheduling checks. No Slurm jobs or energy trackers are started."""
import importlib.util
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "hpc/protocol_v3/resume_ended_workers.py"
SPEC = importlib.util.spec_from_file_location("resume_ended_workers", SCRIPT)
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "old_plan"
        self.source.mkdir()
        self.tasks = [{"task_id": str(i), "dataset_id": "d%d" % i, "framework": "AutoClass",
                       "seed": "42", "expected_samples": "100"} for i in range(8)]
        self.old = {"schema": "awareml_v3_exclusive_batch_v1", "campaign": str(self.root),
                    "campaign_sha256": "campaign-sha", "nodes": 4, "cpu_threads": 5,
                    "allocation_seconds": 43200, "safety_timeout_sec": 7200, "cleanup_seconds": 600,
                    "shards": [[{"task": self.tasks[i], "history": [], "retry_authorized": False}
                                for i in range(k, 8, 4)] for k in range(4)]}
        resume.batch.write(self.source / "plan.json", {"frozen": self.old, "sha256": resume.batch.digest(self.old)})
        for i in (0, 1, 3):
            resume.batch.write(self.source / ("shard_%d.summary.json" % i),
                               {"state": "allocation_boundary", "outcomes": []})
        self.patch_load = patch.object(resume.batch, "load_plan", return_value=self.old)
        self.patch_load.start()
        self.addCleanup(self.patch_load.stop)
        self.live = "123_2|awareml-v3-batch|RUNNING\n"
        self.states = "123_0|COMPLETED|0:0\n123_1|COMPLETED|0:0\n123_3|COMPLETED|0:0\n"
        self.calls = []
        def scheduler(args, **kwargs):
            self.calls.append((args, kwargs))
            if args[0] == "sacct":
                return self.states
            if args[0] == "squeue":
                return self.live
            if args[0] == "sbatch":
                return "456\n"
            raise AssertionError(args)
        self.patch_scheduler = patch.object(resume.subprocess, "check_output", side_effect=scheduler)
        self.patch_scheduler.start()
        self.addCleanup(self.patch_scheduler.stop)
        resume.batch.STOP = False
        self.save(0, self.good(0))

    def good(self, i):
        return {"task": self.tasks[i], "campaign_sha256": "campaign-sha", "status": "completed",
                "eligible": True, "samples_processed": 100, "accuracy": .8, "runtime_sec": 10,
                "energy_kwh": .001, "co2_kg": .0004, "peak_process_tree_rss_mib": 100,
                "energy_status": "measured", "tracking_mode": "machine",
                "isolation": {"kind": "slurm_exclusive"}}

    def save(self, task, result, attempt=1):
        resume.batch.write(self.root / "runs" / ("task_%05d" % task) /
                           ("attempt_%03d" % attempt) / "result.json", result)

    def plan(self, **kwargs):
        return resume.make_plan(self.source, "123", [0, 1, 3], **kwargs)

    def prepared(self, name="new_plan"):
        directory = self.root / name
        plan = resume.prepare(self.source, "123", [0, 1, 3], 3, 12, 4, directory)
        return directory, plan

    def test_disjoint_subset_preserves_completed_and_reserves_live_worker_future(self):
        plan = self.plan()
        ids = [int(x["task"]["task_id"]) for shard in plan["shards"] for x in shard]
        self.assertEqual(sorted(ids), [1, 3, 4, 5, 7])
        self.assertEqual(plan["reserved_task_ids"], [2, 6])
        self.assertEqual(plan["source_completed_skipped"], [0])
        self.assertEqual(len(ids), len(set(ids)))

    def test_selected_source_worker_still_running_is_rejected(self):
        self.states = self.states.replace("123_0|COMPLETED", "123_0|RUNNING")
        with self.assertRaisesRegex(RuntimeError, "COMPLETED"):
            self.plan()

    def test_selected_source_requeued_after_accounting_is_rejected(self):
        self.live += "123_0|awareml-v3-batch|PENDING\n"
        with self.assertRaisesRegex(RuntimeError, "still live"):
            self.plan()

    def test_foreign_v3_submission_blocks_possible_duplicates(self):
        self.live += "999_0|awareml-v3-cpu|PENDING\n"
        with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
            self.plan()

    def test_worker_cap_includes_live_source_workers(self):
        with self.assertRaisesRegex(RuntimeError, "at most four"):
            self.plan(nodes=4)

    def test_short_minimum_allocation_cannot_remove_watchdog_headroom(self):
        for kw in ({"min_hours": 2}, {"hours": 3, "min_hours": 4}, {"nodes": 5}):
            with self.assertRaises(ValueError):
                self.plan(**kw)

    def test_incomplete_ended_attempt_is_not_silently_retried(self):
        (self.root / "runs/task_00001/attempt_001").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.plan()

    def test_newer_incomplete_attempt_does_not_hide_behind_old_completed(self):
        (self.root / "runs/task_00000/attempt_002").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.plan()

    def test_failed_final_result_requires_review(self):
        self.save(1, {"status": "failed"})
        with self.assertRaisesRegex(RuntimeError, "unsuccessful"):
            self.plan()

    def test_failed_source_summary_requires_review(self):
        resume.batch.write(self.source / "shard_3.summary.json", {"state": "stopped_on_failure", "outcomes": []})
        with self.assertRaisesRegex(RuntimeError, "failure"):
            self.plan()

    def test_scheduler_failure_is_not_treated_as_no_live_jobs(self):
        with patch.object(resume.subprocess, "check_output", side_effect=subprocess.CalledProcessError(1, "squeue")):
            with self.assertRaises(subprocess.CalledProcessError):
                self.plan()

    def test_accounting_missing_or_ambiguous_is_rejected(self):
        for states in ("", self.states + "123_0|COMPLETED|0:0\n"):
            self.states = states
            with self.assertRaises(RuntimeError):
                self.plan()

    def test_prepare_claim_prevents_two_plans_for_same_ended_shards(self):
        directory, plan = self.prepared()
        with self.assertRaises(FileExistsError):
            self.prepared("duplicate_plan")
        self.assertEqual(resume.load(directory), plan)

    def test_failed_multi_claim_prepare_releases_only_own_new_claims(self):
        claimed = self.source / "resume_claims/shard_1.json"
        resume.batch.write(claimed, {"other_plan": True})
        with self.assertRaises(FileExistsError):
            self.prepared()
        self.assertTrue(claimed.exists())
        self.assertFalse((claimed.parent / "shard_0.json").exists())

    def test_plan_tampering_is_rejected(self):
        directory, _ = self.prepared()
        payload = resume.batch.read(directory / "plan.json")
        payload["frozen"]["cpu_threads"] = 10
        resume.batch.write(directory / "plan.json", payload)
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            resume.load(directory)

    def test_source_summary_changes_are_rejected(self):
        directory, _ = self.prepared()
        resume.batch.write(self.source / "shard_0.summary.json", {"state": "shard_complete", "outcomes": []})
        with self.assertRaisesRegex(RuntimeError, "summary changed"):
            resume.load(directory)

    def test_prepare_leaves_original_plan_and_completed_results_unchanged(self):
        path = self.root / "runs/task_00000/attempt_001/result.json"
        before = (self.source / "plan.json").read_bytes(), path.read_bytes()
        self.prepared()
        self.assertEqual(before, ((self.source / "plan.json").read_bytes(), path.read_bytes()))

    def test_exact_array_element_lookup_including_placeholder(self):
        record = "JobId=123 ArrayJobId=123 ArrayTaskId=3 OverSubscribe=NO TimeLimit=04:00:00 RunTime=00:01:00"
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_ARRAY_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "3"}), \
             patch.object(resume.subprocess, "check_output", return_value=record) as scheduler:
            self.assertEqual(resume.seconds_left(), 14340)
            self.assertEqual(scheduler.call_args[0][0], ["scontrol", "show", "job", "-o", "123_3"])

    def test_multiple_or_wrong_array_records_fail_closed(self):
        record = "ArrayJobId=123 ArrayTaskId=3 OverSubscribe=NO TimeLimit=12:00:00 RunTime=00:00:00"
        with patch.dict(os.environ, {"SLURM_ARRAY_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "3"}):
            for output in (record + "\n" + record, record.replace("ArrayTaskId=3", "ArrayTaskId=2")):
                with patch.object(resume.subprocess, "check_output", return_value=output):
                    with self.assertRaises(RuntimeError):
                        resume.seconds_left()

    def test_original_multi_record_parser_can_produce_empty_allocation_boundary(self):
        records = "TimeLimit=12:00:00 RunTime=00:00:00 OverSubscribe=NO\nTimeLimit=12:00:00 RunTime=09:51:40 OverSubscribe=NO"
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), \
             patch.object(resume.subprocess, "check_output", return_value=records):
            self.assertLess(resume.batch.allocation_seconds_left(), 7800)

    def test_submission_retains_measurement_scope_and_allocation_bounds(self):
        directory, plan = self.prepared()
        args = resume.command(plan, directory)
        for flag in ("--exclusive", "--mem=0", "--nodes=1", "--ntasks=1", "--cpus-per-task=5",
                     "--time=12:00:00", "--time-min=4:00:00", "--array=0-2%3", "--partition=main"):
            self.assertIn(flag, args)
        self.assertIn("resume_ended_workers.py", args[-1])

    def test_dry_run_does_not_submit_and_submission_is_once_only(self):
        directory, _ = self.prepared()
        resume.submit(directory, True)
        self.assertFalse((directory / "submission.json").exists())
        with patch.dict(os.environ, {"SBATCH_NODELIST": "ares3", "AWAREML_EVO_PYTHON": "evo-python"}):
            resume.submit(directory)
        submissions = [(args, kw) for args, kw in self.calls if args[0] == "sbatch"]
        self.assertEqual(len(submissions), 1)
        self.assertNotIn("SBATCH_NODELIST", submissions[0][1]["env"])
        self.assertEqual(submissions[0][1]["env"]["AWAREML_EVO_PYTHON"], "evo-python")
        with self.assertRaises(FileExistsError):
            resume.submit(directory)
        self.assertEqual(len([a for a, k in self.calls if a[0] == "sbatch"]), 1)

    def test_worker_uses_original_per_task_executor_and_skips_newly_completed(self):
        plan = self.plan()
        shard = plan["shards"][0]
        self.save(int(shard[0]["task"]["task_id"]), self.good(int(shard[0]["task"]["task_id"])))
        calls = []
        def runner(args, **kwargs):
            i = int(args[args.index("--task-id") + 1])
            calls.append(i)
            self.assertTrue(args[1].endswith("scripts/awareml_v3.py"))
            self.assertNotIn("--retry", args)
            self.save(i, self.good(i))
            return 0
        result = resume.batch.run_shard(plan, shard, self.root / "new_plan", time.monotonic() + 20000,
                                        runner=runner, node_check=lambda p: None)
        self.assertEqual(result["state"], "shard_complete")
        self.assertEqual(result["outcomes"][0]["state"], "skipped_completed")
        self.assertEqual(len(calls), len(shard) - 1)


if __name__ == "__main__":
    unittest.main()
