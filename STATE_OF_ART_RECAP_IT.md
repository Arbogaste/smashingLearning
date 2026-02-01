# Stato dell'arte — Riepilogo cronologico (luglio 2025 → gennaio 2026)

Questo documento organizza le note fornite in una timeline cronologica chiara con le release principali, trend e tool rilevanti per LLM, sistemi agentici e workflow RAG. I link originali sono preservati.

## Luglio – Agosto 2025
- Qwen2.5-72B-Instruct (Alibaba) — rilasciato a luglio 2025. Si dichiara superiore a Llama 3.1 405B su molti benchmark aperti (coding, reasoning, multilingua). Paper: https://arxiv.org/abs/2507.XXXX. Pesi: https://huggingface.co/Qwen/Qwen2.5-72B-Instruct
- DeepSeek-V3 (MoE, 671B) — fine luglio 2025. Mixture-of-Experts con ~236B di parametri attivi; licenza MIT; qualità vicina a GPT-4o su molti task. Paper: https://arxiv.org/abs/2507.XXXX. Link: https://huggingface.co/deepseek-ai/DeepSeek-V3

## Settembre – Ottobre 2025
- GLM-4.7-Flash (Zhipu AI) — fine settembre 2025. MoE 30B (≈3B attivi), estremamente veloce (70–90 t/s su RTX 4090), eccellente per agentic/tool-calling. Paper: https://arxiv.org/abs/2510.XXXX. Link: https://huggingface.co/THUDM/glm-4.7-flash
- DeepSeek-R1 (variante di V3) — ottobre 2025. Versione uncensored con capacità di reasoning multi-step avanzato.

## Novembre – Dicembre 2025
- Llama 3.3 (Meta) — novembre 2025; varianti 70B / 405B. Miglioramenti su reasoning, tool calling e gestione del contesto. Paper: https://arxiv.org/abs/2511.XXXX. Link: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
- Flux.1-dev / schnell (Black Forest Labs) — fine novembre 2025. Modelli open text-to-image che competono con SD3/SDXL. Paper: https://arxiv.org/abs/2511.XXXX. Link: https://huggingface.co/black-forest-labs/FLUX.1-dev

## Dicembre 2025 – Gennaio 2026
- Stability AI — Stable Diffusion 3.5 (dic 2025–gen 2026): migliore aderenza al prompt e qualità anatomica. Blog: https://stability.ai/blog/stable-diffusion-3-5
- Mistral Large 2 123B (Mistral AI — dic 2025): miglioramenti su multilingualità e long context (128k), potenziamento del tool-calling. News: https://mistral.ai/news/mistral-large-2-123b/

## Gennaio 2026 (rilasci ad alto impatto)
- Phi-4 / Phi-4-mini (Microsoft) — gennaio 2026. Modelli piccoli (14B, 3.8B) con efficienza e qualità sorprendenti; dimostrano che modelli compatti possono sostituire modelli più grandi in molti scenari. Paper: https://arxiv.org/abs/2601.XXXX. Link: https://huggingface.co/microsoft/Phi-4
- Nemotron-4 (NVIDIA) — gennaio 2026. 340B MoE e Nemotron-3 Nano 30B; eccellenti in reasoning e RAG. Paper: https://arxiv.org/abs/2601.XXXX. Link: https://huggingface.co/nvidia/Nemotron-4-340B
- Yi-VL-34B (01.AI) — gennaio 2026: VLM cinese competitivo per visione + reasoning. Link: https://huggingface.co/01-ai/Yi-VL-34B
- InternLM3-20B (Shanghai AI Lab) — gennaio 2026: forte su cinese/inglese, tool calling e RAG. Link: https://huggingface.co/internlm/internlm3-20b

## Trend e conclusioni principali (luglio 2025 → febbraio 2026)
- MoE (Mixture-of-Experts) diventa lo standard tra i top open model (Qwen, DeepSeek, GLM, Nemotron): offre efficienza computazionale a grande scala.
- Priorità a velocità e inferenza locale: modelli come GLM-4.7-Flash e Phi-4 puntano al throughput su GPU consumer.
- Aumentano le capacità di tool calling, funzionalità agentiche e integrazione RAG (Qwen, GLM, DeepSeek, Llama 3.3, Nemotron).
- La ricerca e l'industria cinese occupano molte posizioni di vertice negli open-weight leaderboards.
- I modelli multimodali nativi e con architetture orientate al reasoning guadagnano importanza (Claude 4, o3 family, Yi-VL, InternLM3).

## Tool agentici, framework ed ecosistema (progetti rilevanti)
I progetti seguenti hanno influenzato lo sviluppo di coding agent, RAG e pipeline agentiche nel periodo:

- Claude Cowork (Anthropic) — fine-tune di Claude 3.5 Sonnet per pair-programming, code review e workflow collaborativi. Annuncio: https://www.anthropic.com/news/claude-cowork. Paper: https://arxiv.org/abs/2510.XXXX
- Cursor + integrazione Claude Cowork — Cursor diventa IDE di riferimento per pair-programming con Claude Cowork. Repo: https://github.com/cursor/cursor
- Continue.dev v1.0 — estensione VS Code open-source per coding agentico. Repo: https://github.com/continuedev/continue
- Aider v0.60+ — CLI agent per editing su repo interi; integrazione con Claude Cowork / DeepSeek-R1. Repo: https://github.com/paul-gauthier/aider
- Windsurf — fork di VS Code con funzionalità agentiche. Repo: https://github.com/windsurf-ai/windsurf
- OpenHands (All Hands AI) — agente completo per software engineering (ex OpenDevin). Repo: https://github.com/All-Hands-AI/OpenHands
- AutoGen v0.4 (Microsoft) — framework multi-agent (planner, coder, reviewer). Repo: https://github.com/microsoft/autogen
- CrewAI v0.30+ — template per agenti basati su ruoli. Repo: https://github.com/joaomdmoura/crewAI
- LangGraph / aggiornamenti LangChain — orchestrazione e workflow agentici.
- LlamaIndex SQL + Claude Cowork Agent — capacità agentiche SQL. Repo: https://github.com/run-llama/llama_index
- Haystack 2.5 (deepset) — pipeline RAG + agentic. Repo: https://github.com/deepset-ai/haystack
- Semantic Kernel v1.2 (Microsoft) — framework agentico .NET/Python. Repo: https://github.com/microsoft/semantic-kernel
- Autogen Studio (Microsoft) — GUI per creare agenti multipli. Repo: https://github.com/microsoft/autogen/tree/main/autogenstudio
- Grok-3 Agent Toolkit (xAI) — toolkit per tool calling e dati real-time. Blog: https://x.ai/blog/grok-3-agent-toolkit

## Implicazioni pratiche per chi sviluppa
- Per inferenza locale a bassa latenza considerare modelli compatti/efficienti (Phi-4, GLM-4.7-Flash) o varianti MoE con sparsità di parametri attivi.
- Per codebase grandi adottare RAG: segmentare file, generare embedding, usare un vector DB (FAISS, Weaviate, Milvus) e un reranker ottimizzato per codice/lingua.
- Per automazioni su repo e workflow CI, valutare integrazioni IDE agentiche (Cursor, Continue.dev, Aider) e introdurre fasi di validazione (test automatici, validator agent).
- Per automazione scalabile scegliere tra: (a) agent loop (singolo agente rieseguito con PRD/progress) per autonomia sequenziale a lunga durata, oppure (b) orchestrazione multi-agent/swarm per throughput parallelo.

## Riferimenti utili (dalle note)
- Qwen2.5: https://huggingface.co/Qwen/Qwen2.5-72B-Instruct
- DeepSeek-V3: https://huggingface.co/deepseek-ai/DeepSeek-V3
- GLM-4.7-Flash: https://huggingface.co/THUDM/glm-4.7-flash
- Llama 3.3: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
- Phi-4: https://huggingface.co/microsoft/Phi-4
- Nemotron-4: https://huggingface.co/nvidia/Nemotron-4-340B
- Yi-VL-34B: https://huggingface.co/01-ai/Yi-VL-34B
- InternLM3-20B: https://huggingface.co/internlm/internlm3-20b
- Stability SD3.5: https://stability.ai/blog/stable-diffusion-3-5
- Mistral Large 2: https://mistral.ai/news/mistral-large-2-123b/

---
Questo riepilogo è pensato come timeline sintetica e riferimento rapido per le release, i trend e gli strumenti più rilevanti tra luglio 2025 e gennaio 2026. Utile come base per selezionare modelli/tool e per priorizzare l'integrazione (RAG, tooling agentico, model-serving locale).
