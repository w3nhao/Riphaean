# Memory Design for Self-Evolving Reasoning System

## Overview
This document describes the memory architecture for the self-evolving reasoning system, starting with a simple episodic baseline (M0) and expanding to more sophisticated variants in M1.

## M0: Episodic Memory with RAG

### Schema
Each episode is a trace of a single problem-solving attempt:

```python
{
    "episode_id": str,           # Unique identifier
    "timestamp": datetime,       # When the attempt occurred
    "input": {
        "problem": str,          # Original problem text
        "problem_id": str,       # Dataset problem identifier
        "domain": str            # "math", "code", "qa"
    },
    "trace": [
        {
            "step": int,         # Sequential step number
            "thought": str,      # Reasoning thought
            "action": str,       # Action taken (if any)
            "observation": str   # Result of action
        }
    ],
    "output": {
        "answer": str,           # Final answer
        "correct": bool,         # Whether answer was correct
        "confidence": float      # Model's confidence (0-1)
    },
    "metadata": {
        "tokens_used": int,      # Total tokens in this episode
        "latency_ms": float,     # End-to-end latency
        "model": str,            # Model used
        "temperature": float     # Sampling parameters
    }
}
```

### Storage
- Format: JSONL file per run (e.g., `memory/episodic_20251102_m0_abc123.jsonl`)
- Embedding: Store vector embeddings of `input.problem` using a frozen encoder (e.g., sentence-transformers)
- Index: FAISS or simple numpy-based kNN for retrieval

### Retrieval Policy
When solving a new problem:
1. Embed the new problem text
2. Retrieve top-k (k=3-5) similar past episodes by cosine similarity
3. Filter by domain if available
4. Include in context: problem, trace (first 3 steps), outcome, and key insight (if correct)

### API
```python
class EpisodicMemory:
    def add_episode(self, episode: dict) -> None
    def retrieve(self, query: str, k: int = 5, domain: str = None) -> list[dict]
    def size(self) -> int
    def clear(self) -> None
```

---

## M1 Variants: Extended Memory Architectures

### Variant A: Dual Memory (Episodic + Semantic)

**Motivation**: Separate short-term problem traces from long-term reusable knowledge.

**Semantic Memory Schema**:
```python
{
    "concept_id": str,           # Unique identifier
    "domain": str,               # "math", "code", "qa"
    "concept": str,              # e.g., "quadratic formula", "dynamic programming"
    "definition": str,           # Short description
    "examples": list[str],       # Problem IDs where this was used
    "success_rate": float,       # How often this concept led to correct answers
    "created_from": list[str]    # Episode IDs that generated this concept
}
```

**Retrieval Policy**:
1. Query episodic memory for similar problem traces (as in M0)
2. Query semantic memory for relevant concepts (keyword or embedding match)
3. Merge results: episodic for procedural context, semantic for declarative knowledge

**Extraction Heuristic**:
- After every N successful episodes (N=10), extract common patterns
- Use LLM to summarize: "What general principle or concept was used?"
- Deduplicate and add to semantic memory

---

### Variant B: Tool-Augmented Memory (Vector Store + SQLite)

**Motivation**: Enable complex queries over metadata (e.g., "Show me failed attempts on algebra problems from last week").

**Architecture**:
- **Vector Store**: FAISS/ChromaDB for semantic similarity (same as M0)
- **Metadata DB**: SQLite with schema:
  ```sql
  CREATE TABLE episodes (
      episode_id TEXT PRIMARY KEY,
      timestamp DATETIME,
      problem_id TEXT,
      domain TEXT,
      correct BOOLEAN,
      tokens_used INTEGER,
      latency_ms REAL
  );
  
  CREATE TABLE steps (
      step_id INTEGER PRIMARY KEY,
      episode_id TEXT,
      step_num INTEGER,
      thought TEXT,
      action TEXT,
      FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
  );
  ```

**Retrieval Policy**:
1. SQL query for structured filters (e.g., `WHERE correct=1 AND domain='math'`)
2. Vector search over filtered set for semantic similarity
3. Return episodes with both metadata and trace content

**Tooling**:
- Support natural language to SQL conversion for flexible queries
- Add analytics: "What's my success rate on geometry problems?"

---

### Variant C: Graph-Based Memory

**Motivation**: Capture relationships between problems, concepts, and strategies.

**Schema**:
- Nodes: Problems, Concepts, Strategies, Episodes
- Edges: 
  - `Problem --similar_to--> Problem`
  - `Problem --requires--> Concept`
  - `Episode --solved--> Problem`
  - `Episode --used--> Strategy`

**Retrieval Policy**:
1. Find similar problems via graph traversal (1-2 hops)
2. Retrieve successful episodes connected to those problems
3. Optionally traverse to concepts/strategies for additional context

**Implementation**: NetworkX for lightweight graph operations; defer to graph DB (Neo4j) if scale requires.

---

## M1 Evaluation Plan

### Ablation Matrix
| Memory Type | Retrieval | Dataset | Metric |
|-------------|-----------|---------|--------|
| M0 (episodic) | top-5 | GSM8K-100 | pass@1 |
| M1-A (dual) | episodic + semantic | GSM8K-100 | pass@1, hit@k |
| M1-B (tool-augmented) | SQL + vector | GSM8K-100 | pass@1, query_time |
| M1-C (graph) | graph-walk | GSM8K-100 | pass@1, hop_count |

### Metrics
- **pass@1**: Fraction of problems solved correctly on first attempt
- **hit@k**: Fraction of retrievals where at least one similar correct episode is in top-k
- **latency**: Retrieval + reasoning time per problem
- **memory_size**: Number of episodes/concepts stored
- **token_cost**: Average tokens used per problem

### Success Criteria
- M1 variants should improve pass@1 by ≥5% over M0 baseline
- Retrieval latency must remain <500ms for 1000 episodes
- Memory growth should be sublinear (compression via semantic/graph)

---

## Safety and Guardrails

1. **PII Filtering**: Strip any personal information from problem text before storage
2. **Budget Caps**: Limit memory size (max 10k episodes, max 1GB disk) to prevent runaway growth
3. **Retrieval Diversity**: Ensure top-k includes both successes and instructive failures
4. **Garbage Collection**: Periodically prune low-value episodes (duplicate or low-confidence)

---

## References
- Episodic Memory: [1] Tulving, E. (1972). Episodic and semantic memory.
- RAG: [2] Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Memory for Reasoning: [3] Self-RAG, Reflexion, related works in reasoning_bank paper.

