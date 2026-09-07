# Phase-15 HPC production runner

Main commands:

```bash
sbatch hpc/production/phase15/setup_phase15_ollama.sbatch

bash hpc/production/phase15/submit_phase15_campaign.sh \
  --campaign-id phase15_llama3_8b_final_v1 \
  --concurrency 4 \
  --gpu-gres gpu:a100-40g:1
```

Resume:

```bash
bash hpc/production/phase15/resume_phase15_campaign.sh \
  artifacts/phase15/hpc_runs/phase15_llama3_8b_final_v1 4
```

Reduce:

```bash
python hpc/reduce/phase15/collect_phase15_results.py \
  --campaign-root artifacts/phase15/hpc_runs/phase15_llama3_8b_final_v1
```

See `docs/PHASE15_HPC_EMPIRICAL_RUN.md` for the full protocol.
