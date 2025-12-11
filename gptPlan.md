Ok — buonissimo. Ecco 5 esempi di codice (ossia “scheletri / skeleton”) che mostrano come potresti orchestrare due LLM locali (nel tuo caso: Gemma3 e superSL) in sistemi multi-agente. Ogni esempio usa una tecnologia diversa: da framework “pronto” a vanilla Python, così puoi confrontare approcci e complessità e scegliere quello più leggero e adatto a te.

🔧 Esempi: 5 setup con 2 LLM locali (Gemma3 + superSL)
1) Con PicoAgents (Python multi-agent framework)
# file: agent_pico.py
import asyncio
from picoagents import Agent, OllamaChatCompletionClient

# supponiamo che Ollama server gestisca Gemma3 e superSL con due endpoint distinti:
gemma_client = OllamaChatCompletionClient(model="gemma3", base_url="http://localhost:8000/v1")
supersl_client = OllamaChatCompletionClient(model="superSL", base_url="http://localhost:8000/v1")

agent_worker = Agent(
    name="Worker",
    instructions="You answer questions. Provide a factual answer.",
    model_client=supersl_client,
    tools=[]
)

agent_validator = Agent(
    name="Validator",
    instructions="You review the answer from Worker. If answer is correct, say VALID; otherwise INVALID.",
    model_client=gemma_client,
    tools=[]
)

async def main():
    question = "What is the capital of France?"
    result = await agent_worker.run(question)
    answer = result.messages[-1].content

    validation = await agent_validator.run(f"Answer: {answer}\n Question: {question}\nIs this correct? Reply VALID or INVALID.")
    verdict = validation.messages[-1].content.strip()

    print("Answer:", answer)
    print("Verdict:", verdict)
    if verdict.upper() == "VALID":
        print("→ Action A")
    else:
        print("→ Action B")

if __name__ == "__main__":
    asyncio.run(main())


Cosa fa:

“Worker” usa superSL per generare una risposta.

“Validator” usa Gemma3 per verificare la risposta.

In base al giudizio, si decide un’azione A o B.

2) Con MultiAgent Framework (più semplice, CLI-based)
# file: simple_multiagent.py
from multiagent_framework import Conversation, AgentConfig, ChatAgent

config_worker = AgentConfig(name="worker", model="superSL", base_url="http://localhost:8000/v1")
config_val   = AgentConfig(name="validator", model="gemma3", base_url="http://localhost:8000/v1")

agent_worker = ChatAgent(config_worker)
agent_validator = ChatAgent(config_val)

conv = Conversation(agents=[agent_worker, agent_validator])

question = "List the first three prime numbers greater than 10."
response_worker = conv.send(agent_worker, question)
answer = response_worker.message

response_val = conv.send(agent_validator, f"Is this answer correct? Answer: {answer}\nQuestion: {question}")
verdict = response_val.message.strip()

print("Answer:", answer)
print("Validator says:", verdict)
if verdict.lower().startswith("yes") or "valid" in verdict.lower():
    print("→ Action A")
else:
    print("→ Action B")


Cosa fa:
Più minimale, senza tools extra — definisci due agenti, fai girare la conversazione, vali­dazione, branching sull’azione.

3) Con AutoGen (multi-agent via chat + workflow)
# file: autogen_agents.py
from autogen import GroupChat, GroupChatManager

models = [
    {"model": "superSL", "base_url": "http://localhost:8000/v1"},
    {"model": "gemma3", "base_url": "http://localhost:8000/v1"}
]

def build_agents():
    return [
        {"name": "worker", "model": models[0], "system_prompt": "Answer the question."},
        {"name": "validator", "model": models[1], "system_prompt": "Check whether answer is correct; reply VALID or INVALID."}
    ]

def run_task(question: str):
    agents = build_agents()
    chat = GroupChat(agents=agents, messages=[])
    manager = GroupChatManager(groupchat=chat, llm_config={})
    manager.agents[0].send_message(question)    # worker gets question
    manager.run(max_round=2)
    answer = manager.get_messages("worker")[-1].content
    manager.agents[1].send_message(f"Question: {question}\nAnswer: {answer}\nIs this correct?")
    manager.run(max_round=1)
    verdict = manager.get_messages("validator")[-1].content.strip()
    return answer, verdict

if __name__ == "__main__":
    q = "What is 2+2?"
    ans, ver = run_task(q)
    print("Answer:", ans)
    print("Validator:", ver)
    print("=>", "Action A" if ver.lower().startswith("valid") else "Action B")


Cosa fa:
Usa AutoGen per orchestrare un gruppo chat tra worker e validator. Approccio dichiarativo, “chat-based”.

4) “Vanilla” Python senza framework — pipe manuali
import subprocess
import json

def call_llm(model: str, prompt: str) -> str:
    # esempio: chiamata HTTP a Ollama, o subprocess a binario locale
    # ritorna la risposta come stringa
    # qui pseudo-codice
    data = {"model": model, "prompt": prompt}
    result = subprocess.run(["curl", "-X", "POST", "http://localhost:8000/v1/chat", "-d", json.dumps(data)],
                             capture_output=True, text=True)
    return result.stdout.strip()

def main():
    question = "Who discovered penicillin?"
    answer = call_llm("superSL", question)
    verdict = call_llm("gemma3", f"Answer: {answer}\nQuestion: {question}\nIs this answer correct? Reply VALID or INVALID.")
    print("Answer:", answer)
    print("Verdict:", verdict)
    if "VALID" in verdict.upper():
        print("→ Action A")
    else:
        print("→ Action B")

if __name__ == "__main__":
    main()


Cosa fa:
Zero librerie speciali — solo chiamate HTTP/subprocess al server LLM. Massimo controllo, minima dipendenza.

5) Con Simpliflow — lightweight workflow + LLM orchestration
# file: simpliflow_workflow.py
from simpliflow import Workflow, Step

def worker_step(ctx):
    ctx['answer'] = ctx.llm("superSL", ctx['question'])
    return ctx

def validator_step(ctx):
    ans = ctx['answer']
    verdict = ctx.llm("gemma3", f"Answer: {ans}\nQuestion: {ctx['question']}\nIs this correct? Reply VALID or INVALID.")
    ctx['verdict'] = verdict.strip()
    return ctx

wf = Workflow([
    Step(func=worker_step),
    Step(func=validator_step),
])

if __name__ == "__main__":
    ctx = {'question': "What is the speed of light in vacuum?"}
    result = wf.run(ctx)
    print("Answer:", result['answer'])
    print("Verdict:", result['verdict'])
    if result['verdict'].upper().startswith("VALID"):
        print("→ Action A")
    else:
        print("→ Action B")


Nota: simpliflow è un framework leggero e recente pensato per orchestrare agenti/LLM in workflow più gestibili. 
arXiv

🎯 Quale scegliere in che scenario
Esigenza / obiettivo	Setup consigliato
Vuoi qualcosa che “funziona subito”, con minima configurazione	PicoAgents (es. 1) o AutoGen (es. 3)
Vuoi massima semplicità e dipendenze minime	Vanilla Python (es. 4) oppure simpliflow (es. 5)
Vuoi struttura per espandere e aggiungere tool, memory, workflow	MultiAgent Framework (es. 2) o PicoAgents
Vuoi controllo totale, senza librerie “grandi”	Vanilla Python