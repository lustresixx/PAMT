# Benchmark Notes (AMEM vs AMEM+PAMU)

This folder contains the LoCoMo benchmark script used to compare AMEM vs AMEM+PAMU.
These notes capture the current run setup, results, and how to quickly restart
in a fresh window.

## Quick Start

1) Ensure Ollama is running and the model exists:
   - Model: `qwen2.5:3b`
   - Check: `ollama list`

2) (Optional) Precompute cache (faster runs later):
   - Single-hop:
     `python example\benchmark\benchmark_amem_pamu.py --data data\locomo10.json --task single-hop --prepare-cache --pref-max-turns 24 --pref-stride 3 --pref-min-turns 6 --notes-session-summary --notes-observation --notes-event-summary --cache-dir data\benchmark_cache\sh_all_mt24_s3`
   - Multi-hop:
     `python example\benchmark\benchmark_amem_pamu.py --data data\locomo10.json --task multi-hop --prepare-cache --pref-max-turns 24 --pref-stride 3 --pref-min-turns 6 --notes-session-summary --notes-observation --notes-event-summary --cache-dir data\benchmark_cache\mh_all_mt24_s3`
   - Temporal:
     `python example\benchmark\benchmark_amem_pamu.py --data data\locomo10.json --task temporal --prepare-cache --pref-max-turns 24 --pref-stride 3 --pref-min-turns 6 --notes-session-summary --notes-observation --notes-event-summary --cache-dir data\benchmark_cache\tr_all_mt24_s3`

3) Run evaluation (results are printed and logged):
   - Single-hop:
     `python example\benchmark\benchmark_amem_pamu.py --data data\locomo10.json --task single-hop --pref-max-turns 24 --pref-stride 3 --pref-min-turns 6 --notes-session-summary --notes-observation --notes-event-summary --cache-dir data\benchmark_cache\sh_all_mt24_s3`
   - Multi-hop:
     `python example\benchmark\benchmark_amem_pamu.py --data data\locomo10.json --task multi-hop --pref-max-turns 24 --pref-stride 3 --pref-min-turns 6 --notes-session-summary --notes-observation --notes-event-summary --cache-dir data\benchmark_cache\mh_all_mt24_s3`
   - Temporal:
     `python example\benchmark\benchmark_amem_pamu.py --data data\locomo10.json --task temporal --pref-max-turns 24 --pref-stride 3 --pref-min-turns 6 --notes-session-summary --notes-observation --notes-event-summary --cache-dir data\benchmark_cache\tr_all_mt24_s3`

## Where Results Are Logged

Each run appends a JSONL record to:
`runs/benchmark_results.jsonl`

Quick view:
`Get-Content runs\benchmark_results.jsonl | Select-Object -Last 5`

## Notes / Fixes Applied

- Preference prompt placement was moved into the instruction section (before
  "Memory notes") to avoid contaminating the question text. This helped PAMU
  outperform baseline on locomo10.

## Current Results (locomo10, Qwen2.5-3B, full subset)

All results below are on `data/locomo10.json` (10-dialogue subset).

- Single-hop (841 samples):
  - baseline F1 32.90 / BLEU-1 28.83
  - PAMU     F1 34.53 / BLEU-1 30.89

- Multi-hop (282 samples):
  - baseline F1 20.55 / BLEU-1 14.66
  - PAMU     F1 21.31 / BLEU-1 15.26

- Temporal (321 samples):
  - baseline F1 11.98 / BLEU-1 8.83
  - PAMU     F1 13.62 / BLEU-1 10.34

## Caveats vs Paper (Table 1)

- These results use the 10-dialogue LoCoMo subset (not the full 50).
- AMEM here uses summary/observation/event_summary notes (not the full AMEM
  graph memory implementation).
- Prompt and evaluation details differ from the paper.
- The paper averages multiple seeds.
