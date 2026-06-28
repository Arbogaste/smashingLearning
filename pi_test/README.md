# pi_test

Test pi con Ollama locale (modello `xentriom/gemma-4-12B-coder-fable5-composer2.5-v1:latest`) e HuggingFace router.

## File
- `.env` — HF_TOKEN (copiato da chatbotgirls)
- `models.json` — copia del config; quello attivo sta in `~/.pi/agent/models.json`

## Uso

### Ollama (locale, modello gemma-4-12B-coder)
```bash
ollama serve              # se non gira già
pi --provider ollama --model "xentriom/gemma-4-12B-coder-fable5-composer2.5-v1:latest"
```
O dentro pi: `/model` → seleziona ollama.

### HuggingFace router
```bash
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
pi --provider huggingface --model "Qwen/Qwen2.5-Coder-32B-Instruct"
```

## Smoke test endpoint (senza pi)
```bash
curl -s http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"xentriom/gemma-4-12B-coder-fable5-composer2.5-v1:latest","messages":[{"role":"user","content":"ciao"}]}'
```

## Note
- Modello è reasoning model → con `max_tokens` basso `content` esce vuoto (token finiti nel reasoning). Alza maxTokens.
- models.json ricarica ad ogni apertura `/model` in pi, no restart.
