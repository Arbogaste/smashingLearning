# 30 progetti GitHub non-banali da usare davvero

Non li ho scelti perché “famosi”.
Li ho scelti perché aprono **delta di capacità**.

## A. Computer-use, browser-use, attacco operativo

| # | Progetto | Perché non è banale |
|---|---|---|
| 1 | [microsoft/playwright-mcp](https://github.com/microsoft/playwright-mcp) | browser automation esposto come MCP, cioè plug diretto per agenti |
| 2 | [browser-use/browser-use](https://github.com/browser-use/browser-use) | layer agentico sopra browser reali, utile per flussi sporchi |
| 3 | [browser-use/macOS-use](https://github.com/browser-use/macOS-use) | controllo delle app locali Mac, non solo del browser |
| 4 | [OpenHands/software-agent-sdk](https://github.com/OpenHands/software-agent-sdk/) | SDK focalizzato su agenti che lavorano col codice, non solo chat |
| 5 | [All-Hands-AI/openhands-aci](https://github.com/All-Hands-AI/openhands-aci) | agent computer interface: pezzo raro, non da brochure |
| 6 | [screenpipe/screenpipe](https://github.com/screenpipe/screenpipe) | memoria continua di schermo/audio locale: potentissimo per feedback loop |

**Perché contano:**  
questi non sono “altri agent framework”. Sono pezzi che danno agli agenti **mani, occhi e memoria operativa del desktop**, cosa che quasi tutti i setup banali non hanno. [playwright-mcp](https://github.com/microsoft/playwright-mcp) [browser-use](https://github.com/browser-use/browser-use) [screenpipe](https://github.com/screenpipe/screenpipe)

## B. Raccolta web offensiva e ingest ad alto leverage

| # | Progetto | Perché non è banale |
|---|---|---|
| 7 | [firecrawl/firecrawl](https://github.com/firecrawl/firecrawl) | scrape/crawl/extract fatto per AI, non per spider anni 2010 |
| 8 | [firecrawl/firecrawl-mcp-server](https://github.com/firecrawl/firecrawl-mcp-server) | Firecrawl come capacità tool-native per agenti |
| 9 | [unclecode/crawl4ai](https://github.com/unclecode/crawl4ai) | crawler LLM-friendly, ottimo per pipeline massive |
| 10 | [firecrawl/open-lovable](https://github.com/firecrawl/open-lovable) | clone/recreate siti velocemente: utile per reverse engineering di landing |
| 11 | [microsoft/graphrag](https://github.com/microsoft/graphrag) | retrieval relazionale, utile quando ti serve struttura e non solo chunk |
| 12 | [qdrant/mcp-server-qdrant](https://github.com/qdrant/mcp-server-qdrant) | memoria semantica esposta già come MCP |

**Perché contano:**  
qui stai costruendo il tuo **motore di intelligence privata**. Non cercare “risposte”; cerca **pattern, mutazioni, offerte concorrenti, nuove promesse, nicchie emerse, pagine che cambiano**. [Firecrawl](https://github.com/firecrawl/firecrawl) [Crawl4AI](https://github.com/unclecode/crawl4ai) [GraphRAG](https://github.com/microsoft/graphrag)

## C. Parsing brutale di documenti, PDF, policy, listini, T&C

| # | Progetto | Perché non è banale |
|---|---|---|
| 13 | [docling-project/docling](https://github.com/docling-project/docling) | parsing serio di documenti complessi |
| 14 | [docling-project/docling-mcp](https://github.com/docling-project/docling-mcp) | parsing documentale tool-native per agenti |
| 15 | [datalab-to/marker](https://github.com/datalab-to/marker) | PDF → markdown/JSON molto veloce |
| 16 | [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) | ETL documentale ampio, utile per corpus sporchi |
| 17 | [gotenberg/gotenberg](https://github.com/gotenberg/gotenberg) | output PDF puliti per proposal, audit, offerte, contratti |
| 18 | [documenso/documenso](https://github.com/documenso/documenso) | firma embedded self-hosted: chiude il loop soldi |

**Perché contano:**  
gran parte dei soldi B2B non sono nel browser, ma in **PDF, deck, brochure, listini, contratti, formulari e documenti firmabili**. Se non li domini, il tuo swarm resta un ragazzino brillante ma inutile. [Docling](https://github.com/docling-project/docling) [Marker](https://github.com/datalab-to/marker) [Documenso](https://github.com/documenso/documenso)

## D. Swarm, handoff, memoria, inferenza non-da-asilo

| # | Progetto | Perché non è banale |
|---|---|---|
| 19 | [langchain-ai/langgraph-swarm-py](https://github.com/langchain-ai/langgraph-swarm-py) | swarm con handoff dinamici, più interessante del solito agent loop |
| 20 | [mem0ai/mem0](https://github.com/mem0ai/mem0) | memory layer pensato per agenti, non semplice vettore |
| 21 | [mem0ai/mem0-mcp](https://github.com/mem0ai/mem0-mcp) | memoria condivisibile via MCP |
| 22 | [langfuse/langfuse](https://github.com/langfuse/langfuse) | observability/evals per agenti: utile se vuoi migliorare davvero |
| 23 | [SigNoz/signoz-mcp-server](https://github.com/SigNoz/signoz-mcp-server) | telemetria/observability accessibile dagli agenti |
| 24 | [superset-sh/superset](https://github.com/superset-sh/superset) | questo è interessante: orchestrazione di coding agents su worktree isolati |

**Perché contano:**  
qui smetti di giocare al “chatbot con tools”. Cominci a costruire una colonia con **memoria, telemetria, handoff e controllo di qualità**. [langgraph-swarm-py](https://github.com/langchain-ai/langgraph-swarm-py) [mem0](https://github.com/mem0ai/mem0) [Langfuse](https://github.com/langfuse/langfuse) [superset-sh/superset](https://github.com/superset-sh/superset)

## E. Growth, attribution, inbox, experiment layer

| # | Progetto | Perché non è banale |
|---|---|---|
| 25 | [dubinc/dub](https://github.com/dubinc/dub) | attribution link platform open-source, utilissima per GTM agentico |
| 26 | [growthbook/growthbook](https://github.com/growthbook/growthbook) | feature flags + experimentation seria |
| 27 | [chatwoot/chatwoot](https://github.com/chatwoot/chatwoot) | inbox unificata self-hosted per contatto/lead |
| 28 | [knadh/listmonk](https://github.com/knadh/listmonk) | email engine rapido e semplice, zero pachidermi |
| 29 | [calcom/cal.com](https://github.com/calcom/cal.com) | scheduling infrastructure integrabile nei funnel |
| 30 | [apache/answer](https://github.com/apache/answer) | Q&A/knowledge layer con nuove capacità AI/MCP, utile per internal market memory |

**Perché contano:**  
fare soldi non è “avere agenti intelligenti”; è avere **attribuzione, test, contatto, risposta, chiusura, ricontatto**. Dub e GrowthBook in particolare sono molto più importanti di tanta roba “AI wow”. [Dub](https://github.com/dubinc/dub) [GrowthBook](https://github.com/growthbook/growthbook) [Chatwoot](https://github.com/chatwoot/chatwoot) [listmonk](https://github.com/knadh/listmonk)

---

# I 10 progetti più “nuovi / non scontati” del lotto
Se vuoi il sottoinsieme meno-banale, io terrei d’occhio soprattutto questi:

- [microsoft/playwright-mcp](https://github.com/microsoft/playwright-mcp)
- [browser-use/macOS-use](https://github.com/browser-use/macOS-use)
- [screenpipe/screenpipe](https://github.com/screenpipe/screenpipe)
- [firecrawl/firecrawl-mcp-server](https://github.com/firecrawl/firecrawl-mcp-server)
- [docling-project/docling-mcp](https://github.com/docling-project/docling-mcp)
- [qdrant/mcp-server-qdrant](https://github.com/qdrant/mcp-server-qdrant)
- [mem0ai/mem0-mcp](https://github.com/mem0ai/mem0-mcp)
- [SigNoz/signoz-mcp-server](https://github.com/SigNoz/signoz-mcp-server)
- [superset-sh/superset](https://github.com/superset-sh/superset)
- [firecrawl/open-lovable](https://github.com/firecrawl/open-lovable)

Questi sono i pezzi dove davvero senti aria 2026 invece che 2023.

---

# Le 20 cose da scrivere tu in locale
Queste sono il vero moat.
Non sono “wrapper”.
Sono coltelli.

## 1) **Public Exhaust Miner**
Legge changelog, pricing page, docs, subdomain, robots, sitemap, job post, status page, JS bundles e costruisce un profilo business di un’azienda.  
**Output:** stack, mercato, pricing motion, buyer type, geografie, dolore probabile.

## 2) **Offer Delta Synthesizer**
Prende 20 competitor di una nicchia e ti genera le **5 promesse di vendita non presidiate**.  
Non “riassunto”.  
**Gap di mercato convertibile in headline**.

## 3) **Sales Trigger Radar**
Monitora eventi che rendono un prospect improvvisamente comprabile:
- nuova assunzione
- aumento prezzi
- nuova compliance
- nuovo piano enterprise
- apertura nuovo mercato
- redesign checkout  
**Quando scatta, l’agente attacca.**

## 4) **ICP Mutation Engine**
Non fissa un ICP.
Lo evolve.  
Prende tutti i closed-won e closed-lost e riscrive automaticamente il profilo ideale cliente.

## 5) **Pain-to-Product Compiler**
Da recensioni 1-3 stelle, thread Reddit, ticket pubblici, FAQ e community post, compila:
- problema ricorrente
- soluzione attuale schifosa
- angolo d’offerta
- MVP minimo vendibile

## 6) **Checkout Leak Detector**
Esegue signup/checkout flow concorrenti e misura:
- passaggi
- drop friction
- copy tossico
- trust failure
- timing
- deposit friction  
Produce un report utilizzabile per vendere ottimizzazione.

## 7) **Price Mutation Diff**
Tiene versione storica di:
- pricing page
- piani
- limitazioni
- add-on
- CTA  
Ti dice **quando un mercato si muove** prima che lo capiscano gli altri.

## 8) **Landing Parasite Cloner**
Clona struttura, non testo, delle landing che convertono:
- hero geometry
- social proof ordering
- CTA rhythm
- form depth
- FAQ pressure relief  
Serve per generare landing nuove velocemente senza copiare alla cieca.

## 9) **Testimonial Harvester**
Estrae prove sociali reali da:
- G2/Capterra
- community
- X/Reddit
- case study
- commenti YouTube  
Le trasforma in blocchi di prova riusabili per nicchia.

## 10) **Invisible CRM**
CRM non pensato per umani ma per agenti:
- lead come oggetti mutabili
- thread memory
- objections memory
- readiness score
- next best action  
Niente UI barocca; solo stato pulito e API.

## 11) **Lead Skeletonizer**
Prende un prospect e lo riduce a uno “scheletro decisionale”:
- chi compra
- chi blocca
- quale rischio teme
- quale KPI vuole
- quale leva emotiva funziona

## 12) **Dead Lead Necromancer**
Riprende lead morti e li riattiva quando compaiono nuovi trigger:
- funding
- compliance shock
- pricing shock competitor
- hiring spike
- calo rating pubblico

## 13) **Revenue Leak Cartographer**
Dato un business pubblico, stima dove perde soldi:
- no-show
- triage lento
- supporto lento
- payment friction
- dispute
- reactivation assente
- onboarding incompleto  
Non deve essere perfetto. Deve essere **vendibile**.

## 14) **Funnel Genome Extractor**
Fa reverse engineering dei funnel vincenti:
- ad angle
- landing angle
- form angle
- sales angle
- upsell angle
- retention angle  
Produce una “sequenza genetica” confrontabile tra nicchie.

## 15) **Offer Stress Tester**
Prima di uscire, fa passare ogni offerta su 10 filtri:
- comprabile in 1 frase?
- problema già sentito?
- ROI visibile?
- richiede educazione?
- rischio platform?
- può essere productized?
- può essere automatizzata?
- si può chiudere entro 14 giorni?
- ha buyers con budget?
- crea referrals?

## 16) **Cold Email Forge**
Non un mail merge.
Un sistema che fonde:
- trigger event
- stack del prospect
- competitor delta
- hypothesized leak
- CTA corta  
per produrre email che sembrano quasi scritte da uno che ha studiato l’azienda.

## 17) **Micro-SaaS Seeder**
Genera automaticamente skeleton di prodotto per nuove idee:
- auth
- billing
- admin
- logs
- webhooks
- feature flag
- eval hooks  
Serve a seminare 10 idee in parallelo, non a fare artigianato lento.

## 18) **Autonomous Offer Shop**
Un catalogo locale di offerte machine-readable:
- servizio
- prodotto
- audit
- trial
- setup
- revshare
- guarantee  
Gli agenti pescano da qui e montano proposte, landing e outreach.

## 19) **Objection Library Distiller**
Ascolta call, inbox, DMs, chat e costruisce:
- obiezione
- sottotesto
- risposta efficace
- prova necessaria
- momento giusto per usarla

## 20) **Capital Reallocator**
Il pezzo più importante.  
Ogni settimana decide dove spostare:
- compute
- scraping budget
- outreach volume
- tempo umano
- priorità agenti  
in base a **soldi attesi**, non a entusiasmo.

---

# I 7 tools locali più “antirez style”
Se vuoi la parte veramente non-banalotta, questi sono i miei favoriti:

1. **Public Exhaust Miner**  
2. **Sales Trigger Radar**  
3. **Price Mutation Diff**  
4. **Revenue Leak Cartographer**  
5. **Lead Skeletonizer**  
6. **Dead Lead Necromancer**  
7. **Capital Reallocator**

Perché?  
Perché non sono “AI assistant features”.  
Sono **macchine di compressione dell’informazione verso denaro**.

---

# Come li combinerei davvero

## Layer 1: sensing
- Playwright MCP
- browser-use
- Firecrawl
- Crawl4AI
- Screenpipe

## Layer 2: memory
- Qdrant MCP
- Mem0
- GraphRAG
- Invisible CRM
- Offer Shop

## Layer 3: strike
- Cold Email Forge
- Landing Parasite Cloner
- Checkout Leak Detector
- Lead Skeletonizer

## Layer 4: closing loop
- Dub
- GrowthBook
- Chatwoot
- listmonk
- Capital Reallocator

Questo è molto più vicino a una macchina viva che a un “AI stack”.

---

# Ordine di priorità spietato

## Prima clona/studia questi 10
1. [playwright-mcp](https://github.com/microsoft/playwright-mcp)  
2. [browser-use](https://github.com/browser-use/browser-use)  
3. [screenpipe](https://github.com/screenpipe/screenpipe)  
4. [firecrawl](https://github.com/firecrawl/firecrawl)  
5. [crawl4ai](https://github.com/unclecode/crawl4ai)  
6. [mem0](https://github.com/mem0ai/mem0)  
7. [qdrant mcp](https://github.com/qdrant/mcp-server-qdrant)  
8. [langgraph-swarm-py](https://github.com/langchain-ai/langgraph-swarm-py)  
9. [dub](https://github.com/dubinc/dub)  
10. [growthbook](https://github.com/growthbook/growthbook)

## Poi scrivi questi 8
1. Public Exhaust Miner  
2. Sales Trigger Radar  
3. Offer Delta Synthesizer  
4. Revenue Leak Cartographer  
5. Cold Email Forge  
6. Invisible CRM  
7. Dead Lead Necromancer  
8. Capital Reallocator  

Se fai bene solo questi 18 pezzi, sei già molto oltre il 90% delle installazioni “AI agent” che esistono online.

---

# Quello che NON farei
Non perderei mesi a costruire:
- UI belle
- dashboard decorative
- agent memories “magiche”
- mega orchestratori enterprise
- plugin factory infinita
- knowledge base generiche

Sono quasi tutte fughe dalla parte dura: **trovare soldi e chiuderli**.

---

# Sintesi finale
La roba non-banale non è “più agenti”.
È:

- **più sensing**
- **più diff**
- **più trigger**
- **più decisione allocativa**
- **più tooling che converte rumore pubblico in offerte private**

Claude ti dà già molta intelligenza di base.
Quello che ti manca non è il cervello.
Sono i **coltelli specializzati**.

Se vuoi, nel messaggio dopo ti preparo una cosa molto più utile ancora:

## **“war-machine-v1”**
con:
- **monorepo layout**
- **30 repo mappati in cartelle**
- **20 tools locali con nomi file**
- **ordine di implementazione 14 giorni**
- **quali 5 far scrivere subito a Claude Code**
- **quali 3 far girare 24/7 come daemon**