# Experiments: Evaluation Protocol & Ablation Matrix

## Overview
This document defines the experimental methodology for evaluating the self-evolving reasoning system across milestones M0-M4, including datasets, metrics, ablations, and reproducibility requirements.

---

## Datasets

### Primary: GSM8K (Grade School Math)
- **Source**: https://github.com/openai/grade-school-math
- **Split Strategy**:
  - M0: 100 problems (dev subset, fixed seed 42)
  - M1: 500 problems (expanded dev)
  - M2+: Full test set (1319 problems)
- **Metric**: Exact match (normalized numeric/string comparison)
- **Setup**: 
  ```bash
  # Download and prepare
  git clone https://github.com/openai/grade-school-math data/gsm8k
  python scripts/prepare_gsm8k.py --seed 42 --subset 100 --output data/gsm8k_100.json
  ```

### Secondary: MATH (Competition Math)
- **Source**: https://github.com/hendrycks/math
- **Split**: 500 problems from Level 1-3 (easier subset)
- **Metric**: Exact match with LaTeX normalization
- **Timeline**: M2+

### Future Tracks

#### Code Reasoning: HumanEval / MBPP
- **HumanEval**: 164 Python function synthesis problems
- **MBPP**: 500 basic Python problems
- **Metric**: pass@1, pass@10 (unit test execution)
- **Timeline**: M2+

#### Multi-hop QA: HotpotQA
- **Source**: https://hotpotqa.github.io/
- **Split**: Distractor setting (10k dev)
- **Metric**: Exact match (answer span) + F1
- **Timeline**: M3+

---

## Metrics

### Primary Metrics
| Metric | Definition | Target |
|--------|------------|--------|
| **pass@1** | Fraction of problems solved correctly on first attempt | >0.60 (GSM8K) |
| **retrieval_hit@k** | Fraction where top-k retrievals include ≥1 correct similar episode | >0.70 (k=5) |
| **solver_depth** | Average reasoning steps (thought-action pairs) per problem | <10 steps |
| **token_cost** | Average tokens (input + output) per problem | <2000 tokens |
| **latency_ms** | End-to-end time per problem (retrieval + reasoning + execution) | <5000ms |

### Secondary Metrics
- **memory_efficiency**: Episodes stored per unique problem solved
- **self_correction_rate**: Fraction of problems where reflection led to correct answer after initial failure
- **diversity@k**: Average pairwise dissimilarity in top-k retrievals (cosine distance)

---

## Ablation Matrix

### M0: Baseline Reproduction
| Variant | Memory | Planner | Beam | Expected pass@1 |
|---------|--------|---------|------|-----------------|
| reasoning_bank | None | None | 1 | ~0.50 (baseline) |
| M0 + episodic | Episodic (k=5) | None | 1 | 0.52-0.55 |

**Goal**: Reproduce reasoning_bank baseline; establish episodic memory delta.

---

### M1: Memory Variants
| Variant | Memory Type | Retrieval k | Dataset | pass@1 | hit@k | latency_ms |
|---------|-------------|-------------|---------|--------|-------|------------|
| M0 (episodic) | Episodic | 5 | GSM8K-500 | - | - | - |
| M1-A (dual) | Episodic + Semantic | 3+2 | GSM8K-500 | - | - | - |
| M1-B (tool) | SQL + Vector | 5 | GSM8K-500 | - | - | - |
| M1-C (graph) | Graph-walk | 2-hop | GSM8K-500 | - | - | - |
| M1 (no-memory) | None | 0 | GSM8K-500 | - | - | - |

**Hypothesis**: Dual memory (M1-A) will show highest pass@1 due to concept reuse.

---

### M2: Planner Integration
| Variant | Memory | Planner | Beam | Depth | Dataset | pass@1 | solver_depth |
|---------|--------|---------|------|-------|---------|--------|--------------|
| M1-A | Dual | None | 1 | - | GSM8K-500 | - | - |
| M2 | Dual | BFS | 3 | 2 | GSM8K-500 | - | - |
| M2 (no-planner) | Dual | None | 1 | - | GSM8K-500 | - | - |

**Hypothesis**: Shallow planning (beam=3, depth=2) will improve pass@1 by 3-5% over M1-A.

---

### M3: Self-Evolution Loop
| Variant | Memory | Self-Evo | Reflection | Dataset | pass@1 | self_correct_rate |
|---------|--------|----------|------------|---------|--------|-------------------|
| M2 | Dual | Off | Off | GSM8K-full | - | - |
| M3 | Dual | On | Every 10 episodes | GSM8K-full | - | - |
| M3 (no-reflect) | Dual | On | Never | GSM8K-full | - | - |

**Hypothesis**: Self-evolution with reflection will show incremental improvement over time (learning curve).

---

## Run Logging & Reproducibility

### Run ID Convention
```
{YYYYMMDD}-{milestone}-{commit_hash}
Example: 20251102-m0-abc1234
```

### Config File (YAML)
Every run must log a config file:
```yaml
run_id: 20251102-m0-abc1234
milestone: M0
dataset:
  name: gsm8k
  split: dev-100
  seed: 42
model:
  name: gpt-4
  temperature: 0.7
  max_tokens: 1500
memory:
  type: episodic
  retrieval_k: 5
  embedding_model: all-MiniLM-L6-v2
planner:
  enabled: false
environment:
  python_version: 3.10
  torch_version: 2.0.1
  cuda_version: 11.8
  seed: 42
```

### Output Structure
```
results/
  20251102-m0-abc1234/
    config.yaml              # Run configuration
    episodes.jsonl           # All episodic traces
    metrics.json             # Aggregated metrics
    predictions.jsonl        # Per-problem predictions
    retrieval_logs.jsonl     # Retrieved episodes per problem
    timings.json             # Latency breakdown
```

### Reproducibility Checklist
- [ ] Fixed random seeds (Python, NumPy, PyTorch)
- [ ] Exact package versions logged (`pip freeze > requirements_run.txt`)
- [ ] Dataset split and order deterministic
- [ ] Model API calls logged (if external)
- [ ] Retrieval index saved (FAISS snapshot)

---

## Evaluation Scripts

### Run Evaluation
```bash
# M0 baseline
python scripts/evaluate.py \
  --milestone m0 \
  --dataset data/gsm8k_100.json \
  --config configs/m0_baseline.yaml \
  --output results/$(date +%Y%m%d)-m0-$(git rev-parse --short HEAD)

# M1 ablations (parallel)
for memory_type in episodic dual tool graph; do
  python scripts/evaluate.py \
    --milestone m1 \
    --memory_type $memory_type \
    --dataset data/gsm8k_500.json \
    --config configs/m1_${memory_type}.yaml \
    --output results/$(date +%Y%m%d)-m1-${memory_type}-$(git rev-parse --short HEAD)
done
```

### Compute Metrics
```bash
python scripts/compute_metrics.py \
  --run_dir results/20251102-m0-abc1234 \
  --output results/20251102-m0-abc1234/metrics.json
```

### Compare Runs
```bash
python scripts/compare_runs.py \
  --runs results/20251102-m0-abc1234 results/20251102-m1-dual-def5678 \
  --metrics pass@1 hit@5 latency_ms \
  --output plots/m0_vs_m1.png
```

---

## Statistical Significance

- Use bootstrap resampling (n=1000) for confidence intervals on pass@1
- Report 95% CI alongside point estimates
- Require non-overlapping CIs for claims of improvement
- Minimum sample size: 100 problems per condition

---

## Safety & Guardrails

### Per-Problem Budgets
- Max tokens: 3000 (input + output)
- Max latency: 30s (timeout and skip)
- Max retrieval: 10 episodes (cap memory lookup)

### Run-Level Budgets
- Max episodes stored: 10,000
- Max disk usage: 1GB for memory
- Refuse problems with PII or unsafe content (keyword filter)

---

## Future Enhancements (M4+)

- **Online Evaluation**: Continual learning on streaming problems
- **Human-in-the-Loop**: Periodic annotation of failure cases
- **Multi-Model Ensemble**: Compare GPT-4 vs Claude vs open-source models
- **Cross-Domain Transfer**: Train on math, evaluate on code (zero-shot)

---

## References
- GSM8K: Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems.
- MATH: Hendrycks et al. (2021). Measuring Mathematical Problem Solving.
- HumanEval: Chen et al. (2021). Evaluating Large Language Models Trained on Code.
- HotpotQA: Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.

