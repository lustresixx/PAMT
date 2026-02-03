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

## Langfuse Tracing (Optional)

Langfuse provides observability and tracing for LLM applications.

### Setup Langfuse with Docker

1) Start Langfuse locally using Docker Compose:
```bash
cd example
docker-compose -f docker-compose.langfuse.yml up -d
```

2) Open http://localhost:3000 and create an account

3) Create a new project and get your API keys from Settings > API Keys

4) Configure your `config/api_config.json`:
```json
{
  "langfuse_enabled": true,
  "langfuse_host": "http://localhost:3000",
  "langfuse_public_key": "pk-lf-xxx",
  "langfuse_secret_key": "sk-lf-xxx"
}
```

### Running with Langfuse

Using config file (recommended):
```bash
python example/run_visual_chatbot.py
```

Auto-start Langfuse Docker and run:
```bash
python example/run_visual_chatbot.py --start-langfuse
```

Using command line arguments:
```bash
# Set environment variables
export LANGFUSE_PUBLIC_KEY=pk-lf-xxx
export LANGFUSE_SECRET_KEY=sk-lf-xxx

# Run with --langfuse flag
python example/visual_chatbot.py --langfuse --langfuse-host http://localhost:3000
```

### What gets traced

- **chat_response**: Overall conversation turn trace
- **preference_retrieval**: Memory tree preference lookup
- **llm_generate**: LLM API calls with token usage
- **preference_extraction**: Extracting preferences from responses
- **route_query**: Routing to memory tree nodes
- **memory_update**: Updating memory tree with new information

### Stop Langfuse
```bash
cd example
docker-compose -f docker-compose.langfuse.yml down
```

