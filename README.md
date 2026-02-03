
# smashingLearning

smashingLearning: local LLM pipeline — dataset → fine-tune → serve → minimal agents.

Setup (fast)

```bash
git clone https://github.com/ConardLi/easy-dataset.git
git clone https://github.com/hiyouga/LLaMA-Factory.git

# easy-dataset (dev)
cd easy-dataset && npm install && npm run start   # http://localhost:1717

# llma-factory (fine-tune)
cd ../LLaMA-Factory && pip install -r requirements.txt
```

Quick operations

- Prepare training data with `easy-dataset` and export JSONL.
- Fine-tune via `LLaMA-Factory` (see repo for params).
- Serve model locally (Ollama or any http-compatible server).

Notes

- See `STATE_OF_ART_RECAP.md` and `STATE_OF_ART_RECAP_IT.md` for current recommendations (RAG, agent-loop, swarm).
- Replace model names with your served model identifiers.

Ollama & Testing

See `ollamaDocs/` for official Ollama documentation (core API, tool calling, streaming, web search).

Test local models with `ollama_test_llms.py`:

```bash
cd ollamaExperiments
python ollama_test_llms.py > ollama_test_llms.log 2>&1
```

Tests: reasoning (system design), code generation, tool calling, finance analysis, streaming. Toggle flags in `config`.

Resources

See `resources_sorted.md` and `resources_extendex.md` for 2026 tech stack: AI frameworks, data tools, trading bots, e-commerce platforms, automation, and more.

---

