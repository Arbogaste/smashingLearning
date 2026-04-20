/**
 * Ollama streaming client - vanilla JavaScript
 * Calls /api/generate endpoint directly with streaming support
 */

const OLLAMA_HOST = 'http://localhost:11434';
const MODEL = 'nexusriot/Qwen3.5-Uncensored-HauhauCS-Aggressive:4b';
const PROMPT = 'write the first 4 sentences of the Red Book by Mao.';

/**
 * Stream text generation from Ollama /api/generate endpoint
 * @param {string} prompt - The prompt to send
 * @param {Function} onChunk - Callback for each text chunk received
 * @returns {Promise<string>} - Full generated text
 */
async function generateStreaming(prompt, onChunk) {
  console.log(`[Ollama] Connecting to ${OLLAMA_HOST}`);
  console.log(`[Ollama] Model: ${MODEL}`);
  console.log(`[Ollama] Prompt: ${prompt}\n`);

  const response = await fetch(`${OLLAMA_HOST}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      prompt: prompt,
      stream: true,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = '';
  let buffer = '';
  let lineCount = 0;

  console.log('[Stream] Starting iteration...\n');

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        console.log(`\n[Stream] Done. Received ${lineCount} chunks`);
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      // Process complete lines, keep incomplete line in buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;

        lineCount++;

        try {
          const chunk = JSON.parse(line);
          const text = chunk.response || '';

          if (text) {
            fullText += text;
            process.stdout.write(text); // Stream to stdout
            if (onChunk) onChunk(text);
          }

          // Log final stats when done
          if (chunk.done) {
            console.log(`\n\n[Stats] Model: ${chunk.model}`);
            console.log(`[Stats] Total generated: ${fullText.length} chars`);
            if (chunk.eval_count) {
              console.log(
                `[Stats] Tokens: ${chunk.eval_count} eval, ${chunk.prompt_eval_count} prompt`
              );
            }
          }
        } catch (e) {
          console.error(`[Error] JSON parse failed on line ${lineCount}:`, e.message);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  return fullText;
}

/**
 * Main entry point
 */
async function main() {
  console.log('============================================================');
  console.log('Ollama LLM Streaming Test (JavaScript)');
  console.log('============================================================\n');

  try {
    const generatedText = await generateStreaming(PROMPT, (chunk) => {
      // Optional callback for each chunk
      // Could be used to update UI, save to file, etc.
    });

    console.log('\n============================================================');
    if (generatedText.length > 0) {
      console.log('✅ Success!');
      console.log(`Generated text (${generatedText.length} chars):`);
      console.log(generatedText);
    } else {
      console.log('❌ No text generated');
    }
    console.log('============================================================\n');
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}

module.exports = { generateStreaming };
