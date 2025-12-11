
# smashingLearning

## Project Overview

smashingLearning is a practical framework for building, fine-tuning, and deploying local Large Language Models (LLMs) with a focus on simplicity, reproducibility, and smart architecture. The project is designed to:

1. **Run and infer with a local LLM** using the most straightforward implementation (Ollama recommended for local inference, with alternatives like datapizza).
2. **Generate useful training data** from easy datasets and custom sources, leveraging [easy-dataset](https://github.com/ConardLi/easy-dataset) for streamlined data preparation.
3. **Fine-tune LLMs** using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (Ollama Factory) for targeted model adaptation.
4. **Demonstrate multi-agent inference**: process a query through multiple agents, each providing output for a simple input/output example.

## Step-by-Step Functional Workflow


### 1. Clone and Set Up Dependencies

#### Clone easy-dataset and LLaMA-Factory

```bash
git clone https://github.com/ConardLi/easy-dataset.git
git clone https://github.com/hiyouga/LLaMA-Factory.git
```

#### Install easy-dataset (choose one method)

- **NPM (recommended for development):**
	```bash
	cd easy-dataset
	npm install
	npm run build
	npm run start
	# Visit http://localhost:1717 in your browser
	```
- **Docker:**
	```bash
	cd easy-dataset
	docker-compose up -d
	# Or build manually:
	docker build -t easy-dataset .
	docker run -d -p 1717:1717 -v ./local-db:/app/local-db -v ./prisma:/app/prisma --name easy-dataset easy-dataset
	# Visit http://localhost:1717
	```

#### Install LLaMA-Factory

- **Python (recommended):**
	```bash
	cd LLaMA-Factory
	pip install -r requirements.txt
	# Or use Docker, see official repo for details
	```

### 2. Local LLM Inference
Set up a local LLM environment using Ollama (or datapizza). The system should allow easy import and inference with a chosen model, providing a simple API for text generation.

### 3. Data Preparation
Use easy-dataset to create and curate datasets. Upload domain-specific files, split and label text, generate questions and answers, and export datasets in OpenAI-compatible formats (JSON/JSONL).

### 4. Fine-Tuning Pipeline
Utilize LLaMA-Factory to fine-tune the local LLM. Import your dataset, configure training parameters, and run the fine-tuning process. Export the model for local inference or deployment.

### 5. Multi-Agent Inference Example
Implement a simple multi-agent system where a query is processed by several agents (models or model instances), each returning an output. Provide a clear example script demonstrating input, agent processing, and output aggregation.

## Pseudoarchitecture

```
smashingLearning/
│
├── llm_inference/         # Local LLM import and inference (Ollama integration)
├── data_preparation/      # Scripts for dataset creation and source integration
├── fine_tuning/           # Fine-tuning pipeline (LLaMA-Factory integration)
├── multi_agent_demo/      # Multi-agent inference example
├── README.md              # Project documentation
└── LICENSE                # License information
```

### Design Principles
- Minimal, purposeful files only
- Modular scripts for each step
- Clear separation of concerns
- Easy to extend and adapt

## Getting Started

### Minimal Setup Workflow
1. Clone easy-dataset and LLaMA-Factory
2. Install dependencies (NPM/Docker for easy-dataset, Python for LLaMA-Factory)
3. Prepare your dataset with easy-dataset
4. Fine-tune your model with LLaMA-Factory
5. Integrate and run local inference and multi-agent demo

---

## Multi-Agent Orchestration: Example Skeletons

Once you have two local LLMs available (e.g., one fine-tuned with LLaMA-Factory and another like Gemma3 or superSL, both served via Ollama or compatible API), you can orchestrate them in a multi-agent system. Below are practical code skeletons for different levels of complexity and dependency:

### 1. PicoAgents (Python multi-agent framework)
```python
import asyncio
from picoagents import Agent, OllamaChatCompletionClient

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
```

### 2. MultiAgent Framework (CLI-based)
```python
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
```

### 3. AutoGen (multi-agent chat + workflow)
```python
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
```

### 4. Vanilla Python (no framework)
```python
import subprocess
import json

def call_llm(model: str, prompt: str) -> str:
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
```

### 5. Simpliflow (lightweight workflow)
```python
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
```

#### Which approach to choose?

- For minimal setup and quick results: PicoAgents or AutoGen
- For maximum simplicity and minimal dependencies: Vanilla Python or Simpliflow
- For extensibility (tools, memory, workflow): MultiAgent Framework or PicoAgents
- For total control, no big libraries: Vanilla Python

---
These skeletons can be adapted to your models and workflow. After fine-tuning with LLaMA-Factory, serve your models locally (e.g., with Ollama) and use one of these approaches to build your multi-agent system.
