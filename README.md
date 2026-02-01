
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

Minimal examples

1) send prompt (curl)

```bash
curl -s -X POST "http://localhost:8000/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"Summarize the repo in one sentence."}] }'
```

2) tiny worker/validator (Python)

```python
import requests
BASE='http://localhost:8000/v1'
def ask(m,p): return requests.post(f"{BASE}/chat",json={"model":m,"messages":[{"role":"user","content":p}]}).json()
q='What is the capital of France?'
a=ask('worker-model',q); text=a['choices'][0]['message']['content']
v=ask('validator-model',f"Answer: {text}\nIs this correct? Reply VALID or INVALID.")
print(text, '=>', v['choices'][0]['message']['content'].strip())
```

Notes

- Old large skeletons removed; keep examples tiny and functional. See `STATE_OF_ART_RECAP.md` and `STATE_OF_ART_RECAP_IT.md` for current recommendations (RAG, agent-loop, swarm).
- Replace model names with your served model identifiers.

---

