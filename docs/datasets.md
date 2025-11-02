# Datasets: Setup, Splits, and Reproducibility

## Overview
This document provides detailed setup instructions, split strategies, and reproducibility checklists for all datasets used in the self-evolving reasoning project.

---

## 1. GSM8K (Grade School Math)

### Description
- **Full Name**: Grade School Math 8K
- **Domain**: Math word problems (arithmetic, basic algebra)
- **Size**: 8,500 problems (7,473 train + 1,319 test)
- **Answer Format**: Numeric (integers, decimals)
- **Source**: https://github.com/openai/grade-school-math

### Download & Setup
```bash
# Clone repository
cd /dengwenhao2/projects/20251102_selfevo_llm
git clone https://github.com/openai/grade-school-math data/gsm8k

# Verify structure
ls data/gsm8k/grade_school_math/data/
# Expected: train.jsonl, test.jsonl
```

### Dataset Format
Each line in JSONL:
```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

### Split Strategy
| Milestone | Source | Size | Seed | Purpose |
|-----------|--------|------|------|---------|
| M0 | test | 100 | 42 | Quick baseline verification |
| M1 | test | 500 | 42 | Memory ablations |
| M2 | test | 1319 | - | Full evaluation (planner) |
| M3 | test | 1319 | - | Self-evolution learning curve |

### Preprocessing Script
```python
# scripts/prepare_gsm8k.py
import json
import random
import re

def extract_numeric_answer(answer_str):
    """Extract final numeric answer from GSM8K format."""
    match = re.search(r'#### (.+)$', answer_str)
    if match:
        return match.group(1).strip()
    return None

def prepare_subset(input_path, output_path, n_samples, seed):
    """Create a reproducible subset."""
    random.seed(seed)
    
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Sample without replacement
    subset = random.sample(data, n_samples)
    
    # Extract clean answers
    for item in subset:
        item['answer_numeric'] = extract_numeric_answer(item['answer'])
    
    with open(output_path, 'w') as f:
        for item in subset:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created subset: {n_samples} problems -> {output_path}")

if __name__ == '__main__':
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    prepare_subset('data/gsm8k/grade_school_math/data/test.jsonl',
                   'data/gsm8k_100.json', 100, seed)
    prepare_subset('data/gsm8k/grade_school_math/data/test.jsonl',
                   'data/gsm8k_500.json', 500, seed)
```

### Evaluation Metric
```python
def evaluate_gsm8k(prediction: str, ground_truth: str) -> bool:
    """Exact match with numeric normalization."""
    pred = normalize_numeric(prediction)
    gt = normalize_numeric(ground_truth)
    return pred == gt

def normalize_numeric(s: str) -> str:
    """Normalize numeric strings for comparison."""
    # Remove commas, whitespace
    s = s.replace(',', '').strip()
    # Convert to float and back to handle '72' vs '72.0'
    try:
        return str(float(s))
    except ValueError:
        return s.lower()
```

### Reproducibility Checklist
- [ ] Use test split only (never train split)
- [ ] Fix random seed for subset sampling (seed=42)
- [ ] Verify subset hash (MD5: `scripts/verify_gsm8k_hash.sh`)
- [ ] Log exact OpenAI commit/version if using official repo
- [ ] Use numeric normalization for evaluation

---

## 2. MATH (Competition Math)

### Description
- **Full Name**: Mathematics Aptitude Test of Heuristics
- **Domain**: Competition-level math (algebra, geometry, number theory, etc.)
- **Size**: 12,500 problems
- **Difficulty**: 5 levels (1=easiest, 5=hardest)
- **Answer Format**: Numeric, symbolic (LaTeX)
- **Source**: https://github.com/hendrycks/math

### Download & Setup
```bash
# Clone repository
git clone https://github.com/hendrycks/math data/math

# Verify structure
ls data/math/
# Expected: train/ test/ (subdirs for each subject)
```

### Dataset Format
Each problem is a JSON file:
```json
{
  "problem": "Simplify $\\frac{1}{1+\\sqrt{2}}$.",
  "level": "Level 2",
  "type": "Algebra",
  "solution": "Multiply numerator and denominator by $\\sqrt{2}-1$...",
  "answer": "\\sqrt{2}-1"
}
```

### Split Strategy
| Milestone | Levels | Subjects | Size | Purpose |
|-----------|--------|----------|------|---------|
| M2 | 1-2 | Algebra, Prealgebra | 500 | Easier subset for initial testing |
| M3 | 1-3 | All | 2000 | Expanded evaluation |
| M4 | 1-5 | All | Full | Challenge benchmark |

### Preprocessing Script
```python
# scripts/prepare_math.py
import json
import os
import random
from pathlib import Path

def load_math_dataset(base_path, levels=[1, 2], max_per_subject=100, seed=42):
    """Load MATH dataset filtered by level."""
    random.seed(seed)
    problems = []
    
    for subject_dir in Path(base_path).iterdir():
        if not subject_dir.is_dir():
            continue
        
        subject_problems = []
        for problem_file in subject_dir.glob('*.json'):
            with open(problem_file) as f:
                data = json.load(f)
            
            # Extract level number
            level_num = int(data['level'].split()[-1])
            if level_num in levels:
                data['subject'] = subject_dir.name
                subject_problems.append(data)
        
        # Sample up to max_per_subject
        sampled = random.sample(subject_problems, 
                                min(len(subject_problems), max_per_subject))
        problems.extend(sampled)
    
    return problems
```

### Evaluation Metric
```python
import re

def normalize_latex(s: str) -> str:
    """Normalize LaTeX expressions for comparison."""
    # Remove whitespace
    s = re.sub(r'\s+', '', s)
    # Normalize common patterns
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('\\,', '').replace('\\:', '')
    return s.lower()

def evaluate_math(prediction: str, ground_truth: str) -> bool:
    """LaTeX-aware exact match."""
    pred = normalize_latex(prediction)
    gt = normalize_latex(ground_truth)
    return pred == gt
```

### Reproducibility Checklist
- [ ] Use test split only
- [ ] Fix random seed for subset sampling (seed=42)
- [ ] Log exact level filters and subject selection
- [ ] Use LaTeX normalization for evaluation
- [ ] Verify subset hash

---

## 3. HumanEval (Code Synthesis)

### Description
- **Full Name**: HumanEval Python Programming Problems
- **Domain**: Python function synthesis
- **Size**: 164 problems
- **Answer Format**: Python function (evaluated via unit tests)
- **Source**: https://github.com/openai/human-eval

### Download & Setup
```bash
# Install evaluation package
pip install human-eval

# Download dataset (auto-downloaded via package)
python -c "from human_eval.data import read_problems; read_problems()"
```

### Dataset Format
```python
{
  "task_id": "HumanEval/0",
  "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    ...",
  "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False",
  "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    ...",
  "entry_point": "has_close_elements"
}
```

### Split Strategy
- No train/test split (all 164 problems used for evaluation)
- Subsets for quick testing: first 20 problems (sorted by task_id)

### Evaluation Metric
```python
from human_eval.evaluation import evaluate_functional_correctness

def evaluate_humaneval(predictions_path: str, k: list[int] = [1, 10]):
    """Compute pass@k using official evaluator."""
    results = evaluate_functional_correctness(
        sample_file=predictions_path,
        k=k,
        n_workers=4,
        timeout=3.0
    )
    return results
```

### Reproducibility Checklist
- [ ] Use official `human-eval` package (version 0.1.0)
- [ ] Set execution timeout (3s per test)
- [ ] Log number of workers for parallel execution
- [ ] Use deterministic task_id ordering for subsets

---

## 4. MBPP (Mostly Basic Python Problems)

### Description
- **Full Name**: Mostly Basic Python Problems
- **Domain**: Python function synthesis (simpler than HumanEval)
- **Size**: 974 problems (374 train + 90 dev + 500 test)
- **Answer Format**: Python function
- **Source**: https://github.com/google-research/google-research/tree/master/mbpp

### Download & Setup
```bash
# Download dataset
wget https://github.com/google-research/google-research/raw/master/mbpp/mbpp.jsonl -O data/mbpp.jsonl
```

### Split Strategy
- Use test split (500 problems) for final evaluation
- Use dev split (90 problems) for quick iterations

### Evaluation Metric
Similar to HumanEval: execute generated code against unit tests.

---

## 5. HotpotQA (Multi-hop Question Answering)

### Description
- **Full Name**: HotpotQA
- **Domain**: Multi-hop reasoning over Wikipedia
- **Size**: 113k problems (90k train + 7.4k dev)
- **Answer Format**: Text span
- **Source**: https://hotpotqa.github.io/

### Download & Setup
```bash
# Download distractor setting
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O data/hotpotqa_dev.json
```

### Dataset Format
```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "answer": "yes",
  "supporting_facts": [
    ["Scott Derrickson", 0],
    ["Ed Wood", 0]
  ],
  "context": [
    ["Scott Derrickson", ["Scott Derrickson (born July 16, 1966) is an American director..."]],
    ...
  ]
}
```

### Split Strategy
- Dev set (7.4k) for evaluation
- Subset (500) for quick testing

### Evaluation Metric
```python
def evaluate_hotpotqa(prediction: str, ground_truth: str):
    """Exact match + F1 (token overlap)."""
    em = normalize_text(prediction) == normalize_text(ground_truth)
    f1 = compute_f1(prediction, ground_truth)
    return {'exact_match': em, 'f1': f1}
```

---

## Cross-Dataset Reproducibility

### Universal Setup Script
```bash
# scripts/setup_datasets.sh
#!/bin/bash
set -e

echo "Setting up datasets..."

# GSM8K
if [ ! -d "data/gsm8k" ]; then
    git clone https://github.com/openai/grade-school-math data/gsm8k
    python scripts/prepare_gsm8k.py 42
fi

# MATH
if [ ! -d "data/math" ]; then
    git clone https://github.com/hendrycks/math data/math
fi

# HumanEval
pip install human-eval==0.1.0

# MBPP
if [ ! -f "data/mbpp.jsonl" ]; then
    wget https://github.com/google-research/google-research/raw/master/mbpp/mbpp.jsonl -O data/mbpp.jsonl
fi

# HotpotQA
if [ ! -f "data/hotpotqa_dev.json" ]; then
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O data/hotpotqa_dev.json
fi

echo "All datasets ready!"
```

### Verification Script
```bash
# scripts/verify_datasets.sh
#!/bin/bash

# Check file counts and hashes
echo "GSM8K test: $(wc -l < data/gsm8k/grade_school_math/data/test.jsonl) lines"
echo "GSM8K-100 hash: $(md5sum data/gsm8k_100.json | cut -d' ' -f1)"
echo "MATH test dirs: $(ls data/math/test/ | wc -l)"
echo "HumanEval: 164 problems (via package)"
echo "MBPP: $(wc -l < data/mbpp.jsonl) lines"
echo "HotpotQA dev: $(jq '. | length' data/hotpotqa_dev.json)"
```

---

## References
- GSM8K: Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems. https://arxiv.org/abs/2110.14168
- MATH: Hendrycks et al. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. https://arxiv.org/abs/2103.03874
- HumanEval: Chen et al. (2021). Evaluating Large Language Models Trained on Code. https://arxiv.org/abs/2107.03374
- MBPP: Austin et al. (2021). Program Synthesis with Large Language Models. https://arxiv.org/abs/2108.07732
- HotpotQA: Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. https://arxiv.org/abs/1809.09600

