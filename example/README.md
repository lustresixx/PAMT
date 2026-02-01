# PAMT Memory Tree Chatbot Example

This example wires the hierarchical memory tree into a simple chat loop
using real embeddings and an LLM backend.

## Quick start (Ollama)
1) Make sure Ollama is running and the models are pulled:
```
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

2) Run the chatbot:
```
python example/chatbot.py
```

## Visual UI (tree + chat)
```
python example/visual_chatbot.py
```
Open `http://127.0.0.1:7860` in your browser.

## DeepSeek (optional)
```
python example/chatbot.py \
  --llm-provider deepseek \
  --embed-provider deepseek \
  --llm-model deepseek-chat \
  --embed-model deepseek-embedding \
  --deepseek-key YOUR_API_KEY
```

## Notes
- Category/leaf labeling prompts live in `prompts/category_leaf.txt` and `prompts/leaf_only.txt`.
- `--use-model-extractor` switches preference extraction to the HuggingFace-backed models
  (requires extra dependencies as noted in the main README).
- For local embeddings, set `embed_provider` to `hf` and `embed_model_name` to a
  HuggingFace model id (for example `sentence-transformers/all-MiniLM-L6-v2`).
  You will also need `transformers` and `torch` installed.

