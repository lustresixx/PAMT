**English** | [中文](README_zh.md)
# PAMT: Preference-Aware Memory Tree
Project name: **PAMT (Preference-Aware Memory Tree)**. The Python package name is `pamt`.

A runnable preference-aware personalization system with a memory tree. It extracts preference signals, smooths them via SW/EMA fusion, and injects control prompts for personalized generation.

---

## Highlights

- Paper-aligned preference extraction (RoBERTa + SKEP + OpenNRE + formality)
- SW/EMA fusion + change detection
- 3-layer memory tree (root -> category -> leaf)
- Memory plugin for external agents
- JSON persistence (auto load/create, auto save)

---

## Contents

- [Quickstart](#quickstart)
- [Memory Plugin (External Agents)](#memory-plugin-external-agents)
- [System Design](#system-design)
- [End-to-End Flow](#end-to-end-flow)
- [Configuration](#configuration)
- [Project Layout](#project-layout)

---

## Quickstart

### CLI Chatbot (DeepSeek + HF embeddings)

```bash
python example/chatbot/chatbot.py \
  --llm-model deepseek-chat \
  --deepseek-key YOUR_API_KEY \
  --embed-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --memory-file data/pamt_memory.json
```

---

### Environment\n\n- PAMT_HF_CACHE_DIR (optional): where HF models are cached/downloaded locally.\n- `PAMT_HF_TOKEN` (optional): Hugging Face access token if your environment requires auth.
- `PAMT_DEEPSEEK_API_KEY` or `DEEPSEEK_API_KEY`: DeepSeek API key for memory-tree labeling.
## Memory Plugin (External Agents)

The plugin exposes a minimal interface:
- `augment(query)` -> returns **memory-augmented prompt**
- `update(query, response)` -> writes back to memory

```python
from pamt import create_memory_plugin, ModelConfig, EmbeddingConfig

plugin = create_memory_plugin(
    model_config=ModelConfig(model_name="deepseek-chat", api_key="YOUR_API_KEY"),
    embedding_config=EmbeddingConfig(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    storage_path="data/pamt_memory.json",
)

aug = plugin.augment("Write an email to my manager")
prompt = aug.prompt

# send prompt to any agent / LLM
response = "..."
plugin.update("Write an email to my manager", response)
```

---

## System Design

### 1) Preference Extraction
Signals:
- Tone
- Emotion
- Response length
- Information density
- Formality

Paper-aligned models:
- RoBERTa encoder + multi-class head (Tone)
- SKEP (Emotional tone)
- OpenNRE (Information density via triples)
- Formality classifier

Length uses model token count when available; otherwise word count.

### 2) SW/EMA Fusion + Change Detection

Continuous dimensions:
- `SW_t(d) = mean(p_{t-W+1:t}(d))`
- `EMA_t(d) = alpha * EMA_{t-1}(d) + (1-alpha) * p_t(d)`
- `w_t(d) = lambda * SW_t(d) + (1-lambda) * EMA_t(d)`

Categorical dimensions apply SW/EMA to probability vectors.

Change detection:
- `Delta_t(d) = |SW_t(d) - EMA_t(d)|`
- `C_t(d) = Delta_t(d) / (epsilon + Var(SW) + Var(EMA))`
- Trigger if `C_t(d) > threshold`

### 3) Memory Tree

Structure:
- root -> category -> leaf

Routing:
- Strict match -> descend
- Loose match -> merge candidates
- Otherwise fallback

Leaf management:
- Labels are inferred from **query + response**
- Leaf reuse uses **content embeddings**
- If leaf count exceeds limit, merge to best leaf
- LLM summary generates a **coarser leaf label** when needed

Update weights:
- Leaf: 1.0
- Category: `1 / category_leaf_count`
- Root: `1 / total_leaf_count`

### 4) Prompt Injection
If preferences exist, a control prefix is added before the user prompt. Otherwise, the raw prompt is used.

### 5) Persistence
The tree + preference state serialize to JSON. Plugin auto load-or-create and auto-save on update.

---

## End-to-End Flow

1) User query arrives
2) Memory retrieval (short-query fallback uses context)
3) Build prompt with preference controls
4) LLM generates response
5) Extract preference signals
6) Route + update memory tree
7) Persist to disk

---

## Configuration

Key knobs:
- `pamt/config.py`:
  - window size, EMA decay, fuse weight
  - change detection thresholds
  - prompt quantization bins
- `RetrievalConfig`:
  - strict/loose similarity
  - max leaf count
  - merge / reuse thresholds

---

## Project Layout

```
pamt/
  core/                 # prompting, memory tree, update logic
  extractors/           # preference extraction
  embeddings/           # HF embeddings
  llms/                 # DeepSeek LLM client
  memory_plugin.py      # external plugin wrapper
example/
  chatbot/              # CLI example
prompts/
  category_leaf.txt     # label inference
  leaf_only.txt
  leaf_merge_summary.txt
```




