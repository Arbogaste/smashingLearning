# DwarfStar (ds4) — inferenza locale di un modello quasi-frontier

Sintesi pratica di **DwarfStar / ds4**, il motore di inferenza scritto da
antirez (Salvatore Sanfilippo) per far girare **DeepSeek V4 Flash/PRO** — e, sul
branch dedicato, **GLM 5.2** — interamente in locale, senza API remote, senza
costo per token, senza dipendenza da un fornitore.

Fonti:
- `README.md` del progetto (branch main) — https://github.com/antirez/ds4
- `README_glm.md` (branch GLM 5.2)
- `STRIXHALO.md`, `MODEL_CARD.md`

> Nota di scope: ds4 **non** è un runner GGUF generico. Carica solo i GGUF
> pubblicati per il progetto (`huggingface.co/antirez/deepseek-v4-gguf`), tarati
> tensore per tensore sul motore. È una scommessa volutamente stretta: un modello
> alla volta, validato contro i logit ufficiali, con agente e API già integrati.

---

## 1. Cos'è, in una frase

Un singolo eseguibile nativo (C + kernel Metal/CUDA/ROCm) che prende un modello
open weight di poche centinaia di miliardi di parametri e lo rende **usabile
davvero** su una macchina personale di fascia alta: chat CLI, server
OpenAI/Anthropic-compatibile, agente di coding, gestione della KV cache su RAM e
su disco. Non è un wrapper: fa il loading, il rendering dei prompt, il tool
calling e il caching da solo.

Il modello di riferimento, **DeepSeek V4 Flash**, è un MoE da 284B parametri
totali con ~13B attivi per token, contesto fino a **1M di token**, licenza MIT.
antirez lo definisce *quasi-frontier*: sui benchmark duri (GPQA Diamond,
LiveCodeBench, SWE-bench, AIME) sta vicino ai modelli chiusi, soprattutto in
modalità reasoning. Il PRO è ancora migliore ma richiede macchine enormi.

## 2. L'idea che cambia le regole: la KV cache è cittadina del disco

Due intuizioni reggono tutto il progetto:

1. **Quantizzazione 2-bit asimmetrica.** Solo gli esperti MoE *routed* vengono
   quantizzati (up/gate a `IQ2_XXS`, down a `Q2_K`). Tutto il resto — esperti
   condivisi, proiezioni, routing, attenzione, output — resta a piena precisione.
   Gli esperti routed sono la maggior parte dello spazio del modello, quindi si
   guadagna moltissima memoria **senza** distruggere la qualità: il quant 2-bit
   chiama i tool in modo affidabile e regge gli agenti di coding.

2. **La KV cache non deve stare per forza in RAM.** DeepSeek V4 comprime molto la
   KV cache, e gli SSD dei Mac moderni sono velocissimi. Questo trasforma il
   vincolo "il modello ci sta in RAM oppure no?" da soglia netta a **spettro
   continuo di velocità**: con lo *SSD streaming* i pesi non-routed restano
   residenti, gli esperti routed stanno in una cache in RAM e vengono letti dal
   GGUF su disco quando mancano. Più RAM = meno miss = più veloce; meno RAM =
   ancora eseguibile, solo più lento.

È questa seconda idea che rende la domanda "64GB o 32GB?" sensata invece che un
muro.

---

## 3. Per l'utente con computer base: cosa ci fai con 64GB (e la verità sui 32GB)

Il GGUF 2-bit di Flash pesa **~81GB**. Da qui derivano tutte le fasce.

| RAM macchina | Come gira | Esperienza |
| --- | --- | --- |
| **≥96–128GB** (Mac) | modello **residente** in memoria | fascia consigliata: veloce, la normale |
| **64GB** (Mac) | **SSD streaming** obbligatorio, cache esperti ~32GB | usabile: prefill buono, generazione più lenta per i cache miss |
| **32GB** | sotto il floor pratico | **non è lo strumento giusto** — vedi sotto |
| **≥256GB / 512GB** | quant Q4 / modello PRO | qualità massima, macchine da workstation |

### 64GB — configurazione reale

Questa è la fascia "entry" credibile. Il modello non ci sta tutto in RAM, quindi
si usa lo streaming da SSD tenendo residenti i pesi non-routed e mettendo gli
esperti in una cache limitata:

```sh
./download_model.sh q2-imatrix

./ds4 \
  -m ./ds4flash.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 32GB \
  --ctx 32768 \
  --nothink
```

Cosa aspettarsi, in concreto:

- **Prefill** (lettura del prompt) resta veloce: gli esperti servono in blocco.
- **Generazione** è la parte che soffre: ogni token nuovo passa di nuovo dagli
  esperti, quindi ogni cache miss è una lettura da disco. Più grande la cache
  esperti, meno miss.
- Parti con `--nothink` per risposte dirette; attiva il reasoning solo quando
  serve, con un budget di token conservativo.
- Se scotta o fa rumore, `--power 50` dimezza l'uso GPU senza cambiare l'output.

Come riferimento di velocità *sulla fascia sopra* (Mac 128GB, Metal, quant 2-bit,
generazione greedy): un M3 Max fa ~**27 token/s** su prompt corto e ~**21
token/s** a 12k di contesto; un M5 Max sale a ~**34 token/s**. Su 64GB in
streaming aspettati numeri di generazione più bassi di questi, con il prefill che
regge meglio.

### 32GB — la risposta onesta

**Con 32GB non far girare ds4.** Il floor reale del progetto è 64GB (in
streaming) / 96GB (residente): sotto, dopo il sistema operativo restano ~24GB,
che non bastano a tenere i pesi non-routed + la KV cache + una cache esperti
utile. Il risultato sarebbe miss continui su disco e una generazione troppo lenta
per essere utile.

Su una macchina da 32GB la mossa giusta **non** è spremere un modello da 284B, ma
usare gli strumenti giusti per quella taglia — ed è esattamente il tema di questo
repo:

- **Ollama** o **llama.cpp** con modelli 7B–14B–32B quantizzati (Qwen, Llama,
  Gemma, DeepSeek-Distill): girano bene in 32GB e coprono la stragrande
  maggioranza dei task quotidiani.
- **OpenRouter** quando serve un modello frontier vero senza avere l'hardware:
  paghi a consumo e non gestisci nulla.

Regola pratica: ds4 serve quando vuoi *quel* modello quasi-frontier **in locale**
e hai la RAM per reggerlo. Se il vincolo è la RAM, scendi di modello, non di
motore.

---

## 4. Per l'azienda: cosa ci fai davvero

Qui ds4 smette di essere una curiosità da smanettoni e diventa infrastruttura.
Il server è OpenAI-, Anthropic- e Responses-compatibile, quindi si aggancia agli
agenti che già usi.

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Espone `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/completions`.
Ci puoi puntare **Claude Code**, **Codex CLI**, **opencode**, **Pi** con poche
righe di config (esempi completi nel README del progetto).

### Casi d'uso concreti

1. **Agente di coding self-hosted a costo per-token zero.**
   Il collo di bottiglia economico degli agenti è il costo delle API su usi
   intensi. Con ds4 il costo marginale è l'elettricità: capex sull'hardware (un
   Mac 128GB) contro opex sulle API. Per un team che macina agenti tutto il
   giorno il break-even arriva in fretta.

2. **Sovranità del dato e compliance.**
   Il codice e i documenti **non lasciano la macchina**. Niente prompt inviati a
   un fornitore US, niente training di terzi sui tuoi dati, niente kill switch
   remoto. Rilevante per settori regolati, GDPR, e deployment air-gapped.

3. **Contesto da 1M di token su documenti privati.**
   Contratti, codebase intere, corpora legali processati in locale. Con 128GB si
   usano finestre da 100–300k token (il contesto pieno da 1M costa ~26GB di
   memoria da solo). La **disk KV cache** fa sì che un prompt di sistema grande e
   ripetuto (gli agenti ne mandano spesso ~25k token) venga riusato invece che
   ri-processato a ogni richiesta.

4. **Cluster distribuito per qualità superiore.**
   ds4 sa spezzare i layer del modello su più macchine collegate (Thunderbolt 5
   ideale, ma va anche su Ethernet/WiFi). Due MacBook da 128GB fanno girare il
   Flash Q4; due Mac Studio da 512GB fanno girare il **PRO Q4** completo. Il
   prefill lungo accelera fino a ~1.85×; la generazione resta poco più lenta del
   singolo nodo. Serve a *far entrare* modelli più grandi, non a velocizzare il
   decode.

5. **Chatbot vincolato con lo steering.**
   Con una singola direzione di attivazione (`dir-steering`) rendi il modello più
   o meno verboso, o gli impedisci di rispondere fuori tema (es. il bot del sito
   noleggio auto che non deve fare da assistente di programmazione) — molto più
   rapido di un fine-tune. Utile anche per ridurre risposte dual-use.

### Limiti da mettere in conto (onestà tecnica)

- **Beta.** Codice e GGUF sono di qualità beta dichiarata; serviranno mesi per
  stabilizzarsi. Per i bug si logga con `--trace`.
- **Una richiesta alla volta.** Il server serializza l'inferenza su un solo
  graph worker: **non** fa batching di richieste concorrenti. Va benissimo per un
  team piccolo o un agente per sviluppatore, non come endpoint multi-tenant ad
  alto QPS.
- **Rete distribuita fidata.** Il protocollo distribuito non ha cifratura né
  autenticazione: solo macchine e reti fidate, tutti i nodi buildati dallo stesso
  commit.
- **Mac-first.** Metal è il target primario; CUDA (incluso DGX Spark) e ROCm
  (Strix Halo / Framework Desktop 128GB) sono supportati ma meno maturi. Il path
  CPU è solo per diagnostica.

---

## 5. In sintesi

- **ds4 = un modello quasi-frontier, in locale, fatto funzionare bene end-to-end.**
- **Utente 64GB:** sì, con SSD streaming e cache esperti; prefill ok, generazione
  più lenta. È la fascia entry onesta.
- **Utente 32GB:** no ds4 — scendi di modello con Ollama/llama.cpp, o vai su
  OpenRouter. Il vincolo è la RAM, non il motore.
- **Azienda:** agente di coding self-hosted a costo per-token zero, dato che non
  esce dalla macchina, contesto da 1M su documenti privati, cluster distribuito
  per il PRO. Con i limiti chiari: beta, una richiesta alla volta, rete fidata.
