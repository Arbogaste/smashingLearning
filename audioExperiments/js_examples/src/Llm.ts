const OLLAMA_HOST = 'http://localhost:11434';
const MODEL = 'nexusriot/Qwen3.5-Uncensored-HauhauCS-Aggressive:4b';
const PROMPT = 'write the first 4 sentences of the Red Book by Mao.';

async function streamOllama() {
  const response = await fetch(`${OLLAMA_HOST}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: 'user', content: PROMPT }],
      stream: true,
    }),
  });

  if (!response.ok) {
    throw new Error(`Ollama error ${response.status}: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No stream reader available');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const chunk = JSON.parse(line);
        const text = chunk.message?.content || chunk.response || '';
        if (text) {
          process.stdout.write(text);
        }
      } catch (err) {
        // ignore partial JSON chunks or unexpected lines
      }
    }
  }

  console.log();
}

streamOllama().catch((err) => {
  console.error('Streaming failed:', err);
  process.exit(1);
});
