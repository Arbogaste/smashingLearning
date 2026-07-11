# Google Chrome Built-in AI — Documentazione

Sintesi delle API AI **gratuite e locali** integrate in Chrome. Modelli girano
on-device (nessun costo server, funziona offline, dati non lasciano il device).

Fonti:
- https://developer.chrome.com/docs/ai/built-in/overview
- https://developer.chrome.com/docs/ai/built-in-apis
- https://developer.chrome.com/docs/ai/webmcp
- https://developer.chrome.com/docs/ai/get-started

---

## 1. Cos'è

Web platform API che eseguono task AI **nel browser**, senza deploy/gestione di
modelli remoti. Chrome usa **Gemini Nano** (LLM generico) + **expert models**
(modelli specializzati, più performanti, meno hardware). Il browser gestisce
download, update e ottimizzazione modelli.

Benefici: privacy (dati client-side), offline, bassa latenza, costo inferenza
zero. Possibile approccio ibrido (client + fallback server).

Hardware: CPU / GPU / NPU locali, auto-detection capacità + degrado graceful.

---

## 2. Requisiti

| Voce | Requisito |
|------|-----------|
| OS | Windows 10/11, macOS 13+, Linux, ChromeOS (Chromebook Plus) |
| Storage | **≥ 22 GB liberi** |
| GPU | > 4 GB VRAM **oppure** |
| CPU | 16 GB RAM + 4+ core |
| Rete | Non a consumo (solo download iniziale modello) |

Modelli text-to-text. Prompt API supporta **multimodale** (immagini/audio input).

---

## 3. API disponibili (stato)

| API | Stato | Scopo |
|-----|-------|-------|
| **Translator** | Chrome 138 stable (Web+Ext) | Traduce contenuto dinamico/user |
| **Language Detector** | Chrome 138 stable (Web+Ext) | Rileva lingua del testo |
| **Summarizer** | Chrome 138 stable (Web+Ext) | Condensa long-form |
| **Prompt** | Chrome 148 stable (solo Ext); Origin trial (Web) | Richieste NL a Gemini Nano |
| **Writer** | Developer trial (Web+Ext) | Genera nuovo contenuto |
| **Rewriter** | Developer trial (Web+Ext) | Riscrive/ristruttura testo |
| **Proofreader** | Origin trial (Web+Ext) | Correzione grammaticale interattiva |

Stabili e usabili subito in produzione web: **Translator, Language Detector,
Summarizer**. Prompt API stabile solo in Extensions.

---

## 4. Setup dev locale

Flags in `chrome://flags`:

1. `#optimization-guide-on-device-model` → **Enabled**
2. Per Gemini Nano/Prompt: `#prompt-api-for-gemini-nano` → **Enabled**
3. Riavvia Chrome

WebMCP testing: `chrome://flags/#enable-webmcp-testing`.

---

## 5. Pattern base (feature detection + availability)

Ogni API espone `availability()` → 4 stati:
`"unavailable"` | `"downloadable"` | `"downloading"` | `"available"`.

```javascript
// feature detection
if (!('Summarizer' in self)) { /* non supportato */ }

// check disponibilità (con lingue opzionali)
const status = await LanguageModel.availability({ languages: ["en", "ja"] });

// download modello richiede user activation
if (navigator.userActivation.isActive) {
  const model = await LanguageModel.create({
    monitor(m) {
      m.addEventListener('downloadprogress', e => {
        console.log(`scaricato ${Math.round(e.loaded * 100)}%`);
      });
    }
  });
}
```

Modello scarica automaticamente al primo uso; usi successivi non richiedono rete.

---

## 6. Esempi per API

### Prompt API (Gemini Nano — general purpose)
```javascript
const session = await LanguageModel.create({
  initialPrompts: [{ role: 'system', content: 'Sei assistente conciso.' }],
  temperature: 0.7,
  topK: 3,
});

// singola risposta
const res = await session.prompt('Riassumi la fotosintesi.');

// streaming
const stream = session.promptStreaming('Scrivi una poesia.');
for await (const chunk of stream) console.log(chunk);

// structured output via JSON schema
const json = await session.prompt('Estrai nome e età', {
  responseConstraint: schemaObject,
});

session.destroy();
```
Supporta session management, streaming, structured output, sampling params
(`temperature`, `topK`), multimodale (immagini/audio come input).

### Summarizer API
```javascript
const summarizer = await Summarizer.create({
  type: 'key-points',      // 'tl;dr' | 'teaser' | 'headline' | 'key-points'
  format: 'markdown',      // 'plain-text' | 'markdown'
  length: 'short',         // 'short' | 'medium' | 'long'
});
const summary = await summarizer.summarize(longText, {
  context: 'Articolo tecnico',
});
```
Uso: transcript riunioni, review prodotti, punti chiave articoli, thread forum.

### Translator API
```javascript
const translator = await Translator.create({
  sourceLanguage: 'en',
  targetLanguage: 'it',
});
const out = await translator.translate('Hello world');
// streaming disponibile: translator.translateStreaming(text)
```

### Language Detector API
```javascript
const detector = await LanguageDetector.create();
const results = await detector.detect('Bonjour le monde');
// [{ detectedLanguage: 'fr', confidence: 0.99 }, ...]
```

### Writer API
```javascript
const writer = await Writer.create({
  tone: 'formal',          // 'formal' | 'neutral' | 'casual'
  format: 'plain-text',
  length: 'medium',
});
const email = await writer.write('Email di sollecito pagamento', {
  context: 'Cliente in ritardo 30gg',
});
```

### Rewriter API
```javascript
const rewriter = await Rewriter.create({
  tone: 'more-casual',     // 'as-is' | 'more-formal' | 'more-casual'
  length: 'shorter',       // 'as-is' | 'shorter' | 'longer'
});
const revised = await rewriter.rewrite(text, { context: '...' });
```

### Proofreader API
```javascript
const proofreader = await Proofreader.create();
const result = await proofreader.proofread('I has a apple');
// correzioni con posizioni + suggerimenti
```

---

## 7. WebMCP (Chrome 149+ origin trial)

Standard proposto: espone **tool strutturati per agenti AI** dentro la pagina
web. Permette agli agenti di simulare azioni utente (click, input) e automatizzare
task (es. acquisti, prenotazioni multi-tappa, form filling).

Componenti: discovery/registrazione tool, JSON Schema per input/output, shared
state per contesto real-time.

Due approcci:
1. **Imperative** — JS per definire tool (input form, navigazione, custom fn).
2. **Declarative** — annotazioni su form HTML standard.

```javascript
// esempio imperative (concettuale)
navigator.modelContext.registerTool({
  name: 'add_to_cart',
  description: 'Aggiunge prodotto al carrello',
  inputSchema: { type: 'object', properties: { sku: { type: 'string' } } },
  async execute({ sku }) { /* ... */ return { success: true }; },
});
```

Sicurezza: origin isolation, Permissions Policy `tools` (default `self`).
Cross-origin iframe richiede `allow="tools"`. Supporto sperimentale in Angular.

Use case: customer support flow, booking viaggi, form filling, UI complesse,
tool di debug.

---

## 8. Note pratiche

- Distinguere **feature detection** (`'API' in self`) da **availability** (modello scaricato).
- `create()` che scarica modello richiede **user activation** (click/tap/keystroke).
- Sempre `.destroy()` sessioni per liberare memoria.
- API in *trial* (Writer/Rewriter/Proofreader): serve token origin trial o EPP
  (Early Preview Program) — non garantite in stable.
- Tutto **gratis**: nessun costo API, nessuna key, gira in locale.
