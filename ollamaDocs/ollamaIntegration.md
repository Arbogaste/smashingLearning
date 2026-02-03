> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Claude Code

Claude Code is Anthropic's agentic coding tool that can read, modify, and execute code in your working directory.

Open models can be used with Claude Code through Ollama's Anthropic-compatible API, enabling you to use models such as `glm-4.7`, `qwen3-coder`, `gpt-oss`.

![Claude Code with Ollama](https://files.ollama.com/claude-code.png)

## Install

Install [Claude Code](https://code.claude.com/docs/en/overview):

<CodeGroup>
  ```shell macOS / Linux theme={"system"}
  curl -fsSL https://claude.ai/install.sh | bash
  ```

  ```powershell Windows theme={"system"}
  irm https://claude.ai/install.ps1 | iex
  ```
</CodeGroup>

## Usage with Ollama

### Quick setup

```shell  theme={"system"}
ollama launch claude
```

To configure without launching:

```shell  theme={"system"}
ollama launch claude --config
```

### Manual setup

Claude Code connects to Ollama using the Anthropic-compatible API.

1. Set the environment variables:

```shell  theme={"system"}
export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_API_KEY=""
export ANTHROPIC_BASE_URL=http://localhost:11434
```

2. Run Claude Code with an Ollama model:

```shell  theme={"system"}
claude --model gpt-oss:20b
```

Or run with environment variables inline:

```shell  theme={"system"}
ANTHROPIC_AUTH_TOKEN=ollama ANTHROPIC_BASE_URL=http://localhost:11434 ANTHROPIC_API_KEY="" claude --model qwen3-coder 
```

**Note:** Claude Code requires a large context window. We recommend at least 64k tokens. See the [context length documentation](/context-length) for how to adjust context length in Ollama.

## Recommended Models

* `qwen3-coder`
* `glm-4.7`
* `gpt-oss:20b`
* `gpt-oss:120b`

Cloud models are also available at [ollama.com/search?c=cloud](https://ollama.com/search?c=cloud).

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# OpenClaw

OpenClaw is a personal AI assistant that runs on your own devices. It bridges messaging services (WhatsApp, Telegram, Slack, Discord, iMessage, and more) to AI coding agents through a centralized gateway.

## Install

Install [OpenClaw](https://openclaw.ai/)

```bash  theme={"system"}
npm install -g openclaw@latest
```

Then run the onboarding wizard:

```bash  theme={"system"}
openclaw onboard --install-daemon
```

<Note>OpenClaw requires a larger context window. It is recommended to use a context window of at least 64k tokens. See [Context length](/context-length) for more information.</Note>

## Usage with Ollama

### Quick setup

```bash  theme={"system"}
ollama launch openclaw
```

<Note>Previously known as Clawdbot. `ollama launch clawdbot` still works as an alias.</Note>

This configures OpenClaw to use Ollama and starts the gateway.
If the gateway is already running, no changes need to be made as the gateway will auto-reload the changes.

To configure without launching:

```shell  theme={"system"}
ollama launch openclaw --config
```

## Recommended Models

* `qwen3-coder`
* `glm-4.7`
* `gpt-oss:20b`
* `gpt-oss:120b`

Cloud models are also available at [ollama.com/search?c=cloud](https://ollama.com/search?c=cloud).

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# JetBrains

<Note>This example uses **IntelliJ**; same steps apply to other JetBrains IDEs (e.g., PyCharm).</Note>

## Install

Install [IntelliJ](https://www.jetbrains.com/idea/).

## Usage with Ollama

<Note>
  To use **Ollama**,  you will need a [JetBrains AI Subscription](https://www.jetbrains.com/ai-ides/buy/?section=personal\&billing=yearly).
</Note>

1. In Intellij, click the **chat icon** located in the right sidebar

<div style={{ display: 'flex', justifyContent: 'center' }}>
  <img src="https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-chat-sidebar.png?fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=3744faa4bdfb6e817ad68d7da792bf18" alt="Intellij Sidebar Chat" width="50%" data-og-width="668" data-og-height="476" data-path="images/intellij-chat-sidebar.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-chat-sidebar.png?w=280&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=b8b775ae96fdf260c9f9ffb001dee5f5 280w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-chat-sidebar.png?w=560&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=7636af78546e0de3a1ba1f0895d512b9 560w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-chat-sidebar.png?w=840&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=98ceeb48201d4c25a70ae3c723f394a4 840w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-chat-sidebar.png?w=1100&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=cfb6f2e863bef8b1cee3aad638a4bf7a 1100w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-chat-sidebar.png?w=1650&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=30a37f1fcebc59171c937ad65f222202 1650w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-chat-sidebar.png?w=2500&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=9a35c626ad3352cfc65846c756633ddd 2500w" />
</div>

2. Select the **current model** in the sidebar, then click **Set up Local Models**

<div style={{ display: 'flex', justifyContent: 'center' }}>
  <img src="https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-current-model.png?fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=cc42c11b23f4a9b57b3e7c69ae42b60b" alt="Intellij model bottom right corner" width="50%" data-og-width="778" data-og-height="546" data-path="images/intellij-current-model.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-current-model.png?w=280&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=48a6ec059b1eb155c1ebcf9059642c91 280w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-current-model.png?w=560&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=7b416430af08c499ca30b92aad33f71a 560w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-current-model.png?w=840&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=5c3d156e7ce892035a7c753893decd9a 840w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-current-model.png?w=1100&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=3f41f2e81f918d6ad424d317f86be381 1100w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-current-model.png?w=1650&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=dcc25dd0a711f5bd3d304c7cdea45617 1650w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-current-model.png?w=2500&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=47d24cfabf9aefb42d44a2438b0e4285 2500w" />
</div>

3. Under **Third Party AI Providers**, choose **Ollama**
4. Confirm the **Host URL** is `http://localhost:11434`, then click **Ok**
5. Once connected, select a model under **Local models by Ollama**

<div style={{ display: 'flex', justifyContent: 'center' }}>
  <img src="https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-local-models.png?fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=0cc866166e1d6af65b3d8a16c3f396f5" alt="Zed star icon in bottom right corner" width="50%" data-og-width="522" data-og-height="602" data-path="images/intellij-local-models.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-local-models.png?w=280&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=cef5847f7b97a11ed6bc785d93181062 280w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-local-models.png?w=560&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=9c64e22cc5cc3db1d5dfd160450aeede 560w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-local-models.png?w=840&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=c2e29b300a2630189df542906012b957 840w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-local-models.png?w=1100&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=10a5578b086d21b104caf3a2d415a181 1100w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-local-models.png?w=1650&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=91a64e8e4e91a53dac2a3ff057c2b212 1650w, https://mintcdn.com/ollama-9269c548/YbLeuKjU_QVFWOuC/images/intellij-local-models.png?w=2500&fit=max&auto=format&n=YbLeuKjU_QVFWOuC&q=85&s=243144901c8988b937ead0dd1579f98a 2500w" />
</div>

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# n8n

## Install

Install [n8n](https://docs.n8n.io/choose-n8n/).

## Using Ollama Locally

1. In the top right corner, click the dropdown and select **Create Credential**

<div style={{ display: 'flex', justifyContent: 'center' }}>
  <img src="https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-credential-creation.png?fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=5b7f955f792e8b9899f3b0b3a1846d06" alt="Create a n8n Credential" width="75%" data-og-width="896" data-og-height="436" data-path="images/n8n-credential-creation.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-credential-creation.png?w=280&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=ca26ff9b910a94d1120a0df887424072 280w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-credential-creation.png?w=560&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=f0209dbb3aa364c2801d8c02bf2c834e 560w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-credential-creation.png?w=840&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=9d291f0dca191861c5eebc5236985bdd 840w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-credential-creation.png?w=1100&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=5bc5d64f6eb7398e83c264382064879f 1100w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-credential-creation.png?w=1650&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=c866783cc5819354b4f69f0e7e8acded 1650w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-credential-creation.png?w=2500&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=a4b7c6e4cf3d38e6898901f64556e16a 2500w" />
</div>

2. Under **Add new credential** select **Ollama**

<div style={{ display: 'flex', justifyContent: 'center' }}>
  <img src="https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-ollama-form.png?fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=46c5ec6678b7323405a1ca98d9f88af0" alt="Select Ollama under Credential" width="75%" data-og-width="1014" data-og-height="580" data-path="images/n8n-ollama-form.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-ollama-form.png?w=280&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=969d64d58c346578b68028e051ecee33 280w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-ollama-form.png?w=560&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=b5f84f26821ad5dbf1b05e01ce03fbcc 560w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-ollama-form.png?w=840&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=8f8b5b1301e0cc3e81ab32c839c52a89 840w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-ollama-form.png?w=1100&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=163156b985eca789a4cf2f441cf2b461 1100w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-ollama-form.png?w=1650&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=812d4591c0a3ebe160b6c78884ad487c 1650w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-ollama-form.png?w=2500&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=1ec83dd378b05cd17ac60bd9a4522e6c 2500w" />
</div>

3. Confirm Base URL is set to `http://localhost:11434` if running locally or `http://host.docker.internal:11434` if running through docker and click **Save**

<Note>
  In environments that don't use Docker Desktop (ie, Linux server installations), `host.docker.internal` is not automatically added.

  Run n8n in docker with `--add-host=host.docker.internal:host-gateway`

  or add the following to a docker compose file:

  ```yaml  theme={"system"}
  extra_hosts:
    - "host.docker.internal:host-gateway"
  ```
</Note>

You should see a `Connection tested successfully` message.

4. When creating a new workflow, select **Add a first step** and select an **Ollama node**

<div style={{ display: 'flex', justifyContent: 'center' }}>
  <img src="https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-chat-node.png?fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=78c15518fa96d509096af63307063ae6" alt="Add a first step with Ollama node" width="75%" data-og-width="822" data-og-height="674" data-path="images/n8n-chat-node.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-chat-node.png?w=280&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=66a5f3201aae6a8ff315efe181fe5a36 280w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-chat-node.png?w=560&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=241b695dc152e97bc9992d2389c4c0cf 560w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-chat-node.png?w=840&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=be8484e52dab096ef1a2178a4d24e934 840w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-chat-node.png?w=1100&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=1be199b1714e4413d1d85fb5d679d6a1 1100w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-chat-node.png?w=1650&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=df82f732f724dd0079d62e39b519713a 1650w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-chat-node.png?w=2500&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=9f235598a4fade17b04c967e478f26a4 2500w" />
</div>

5. Select your model of choice (e.g. `qwen3-coder`)

<div style={{ display: 'flex', justifyContent: 'center' }}>
  <img src="https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-models.png?fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=45e705696dc7c8f40112e1c6dbee0e7e" alt="Set up Ollama credentials" width="75%" data-og-width="1088" data-og-height="1058" data-path="images/n8n-models.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-models.png?w=280&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=b0e7aede75a0098d023db708a37f9966 280w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-models.png?w=560&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=5786bf38961730b03e7415f9255c06d7 560w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-models.png?w=840&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=2182809e057f870dbf8f5b30b4069fe5 840w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-models.png?w=1100&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=6635c63c203627d6def2c398d22efeba 1100w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-models.png?w=1650&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=5715ed9f2bcf4157a29f77b59b0c69ae 1650w, https://mintcdn.com/ollama-9269c548/AZjkMSgDaM1B-WFi/images/n8n-models.png?w=2500&fit=max&auto=format&n=AZjkMSgDaM1B-WFi&q=85&s=cbb012310cc5a74f7e2616423bc4f902 2500w" />
</div>

## Connecting to ollama.com

1. Create an [API key](https://ollama.com/settings/keys) on **ollama.com**.
2. In n8n, click **Create Credential** and select **Ollama**
3. Set the **API URL** to `https://ollama.com`
4. Enter your **API Key** and click **Save**

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# OpenCode

OpenCode is an open-source AI coding assistant that runs in your terminal.

## Install

Install the [OpenCode CLI](https://opencode.ai):

```bash  theme={"system"}
curl -fsSL https://opencode.ai/install | bash
```

<Note>OpenCode requires a larger context window. It is recommended to use a context window of at least 64k tokens. See [Context length](/context-length) for more information.</Note>

## Usage with Ollama

### Quick setup

```bash  theme={"system"}
ollama launch opencode
```

To configure without launching:

```shell  theme={"system"}
ollama launch opencode --config
```

### Manual setup

Add a configuration block to `~/.config/opencode/opencode.json`:

```json  theme={"system"}
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "qwen3-coder": {
          "name": "qwen3-coder"
        }
      }
    }
  }
}
```

## Cloud Models

`glm-4.7:cloud` is the recommended model for use with OpenCode.

Add the cloud configuration to `~/.config/opencode/opencode.json`:

```json  theme={"system"}
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "glm-4.7:cloud": {
          "name": "glm-4.7:cloud"
        }
      }
    }
  }
}
```

## Connecting to ollama.com

1. Create an [API key](https://ollama.com/settings/keys) from ollama.com and export it as `OLLAMA_API_KEY`.
2. Update `~/.config/opencode/opencode.json` to point to ollama.com:

```json  theme={"system"}
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama Cloud",
      "options": {
        "baseURL": "https://ollama.com/v1"
      },
      "models": {
        "glm-4.7:cloud": {
          "name": "glm-4.7:cloud"
        }
      }
    }
  }
}
```

Run `opencode` in a new terminal to load the new settings.

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# VS Code

## Install

Install [VS Code](https://code.visualstudio.com/download).

## Usage with Ollama

1. Open Copilot side bar found in top right window
   <div style={{ display: "flex", justifyContent: "center" }}>
     <img src="https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-sidebar.png?fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=8d841164c3a8c2e6cb502f9dece6079c" alt="VS Code chat Sidebar" width="75%" data-og-width="838" data-og-height="304" data-path="images/vscode-sidebar.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-sidebar.png?w=280&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=8baa6af2c2f307707730aff500625719 280w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-sidebar.png?w=560&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=790f257751bb80213223c2d897988793 560w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-sidebar.png?w=840&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=4c1d2ba0a7e7f4c32fc7818a213eaa85 840w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-sidebar.png?w=1100&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=5470dddd3a1a42d8c599968e8a4613b1 1100w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-sidebar.png?w=1650&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=cf1e7e5ec1aa98136b76e93db93f6116 1650w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-sidebar.png?w=2500&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=6a995df960d939abd4c3ee29f3e58fac 2500w" />
   </div>
2. Select the model dropdown > **Manage models**
   <div style={{ display: "flex", justifyContent: "center" }}>
     <img src="https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-models.png?fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=9a1715817d228c9c103708da3c5ecd37" alt="VS Code model picker" width="75%" data-og-width="1064" data-og-height="462" data-path="images/vscode-models.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-models.png?w=280&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=bcb1de0c96fde6b44d95a816dd81a99a 280w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-models.png?w=560&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=4941a05a32b420adabcd827ebf635097 560w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-models.png?w=840&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=1ef98e50f14c73d9027d38c2f4f28e06 840w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-models.png?w=1100&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=546c696a9857ab6721f7c084836b5921 1100w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-models.png?w=1650&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=98bf91da4065a79774c7b199a99b730f 1650w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-models.png?w=2500&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=6cbd0767ca3440fac141c2e5657f55e7 2500w" />
   </div>
3. Enter **Ollama** under **Provider Dropdown** and select desired models (e.g `qwen3, qwen3-coder:480b-cloud`)
   <div style={{ display: "flex", justifyContent: "center" }}>
     <img src="https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-model-options.png?fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=1b08a9ccc2f275e6eb039de37cceaf31" alt="VS Code model options dropdown" width="75%" data-og-width="1202" data-og-height="552" data-path="images/vscode-model-options.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-model-options.png?w=280&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=bdb15602e971a34695cadc7f6d90d64d 280w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-model-options.png?w=560&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=cb22702c3f6c295ae8822e8ca5f163cf 560w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-model-options.png?w=840&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=bdf5eb8776e9163afe8437f5413c67cc 840w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-model-options.png?w=1100&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=a46fc6100e91907298d223c52f306a5b 1100w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-model-options.png?w=1650&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=c134ec18e18bcdfe471fd5eb329acd9a 1650w, https://mintcdn.com/ollama-9269c548/Q0hzAGiFk9hDuXaH/images/vscode-model-options.png?w=2500&fit=max&auto=format&n=Q0hzAGiFk9hDuXaH&q=85&s=f1be99a8b9069d8d213b6a10debe73a9 2500w" />
   </div>

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Zed

## Install

Install [Zed](https://zed.dev/download).

## Usage with Ollama

1. In Zed, click the **star icon** in the bottom-right corner, then select **Configure**.

<div style={{ display: 'flex', justifyContent: 'center' }}>
  <img src="https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-settings.png?fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=0a63b1359b1472dde2cdf6c37e314e22" alt="Zed star icon in bottom right corner" width="50%" data-og-width="944" data-og-height="224" data-path="images/zed-settings.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-settings.png?w=280&fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=bfc876e9fb924521ab6cdb7ce0f8f8b2 280w, https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-settings.png?w=560&fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=c4b923c2b68da82a39d5834f71e595ec 560w, https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-settings.png?w=840&fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=427b91b58eaff6245edb4d19cacbe13c 840w, https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-settings.png?w=1100&fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=3c135be074ecb974ea19b4e286ec4439 1100w, https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-settings.png?w=1650&fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=974ceda223ced89cf821c3f1899040db 1650w, https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-settings.png?w=2500&fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=d75863b4421cc3f5229b673d673ec7eb 2500w" />
</div>

2. Under **LLM Providers**, choose **Ollama**
3. Confirm the **Host URL** is `http://localhost:11434`, then click **Connect**
4. Once connected, select a model under **Ollama**

<div style={{ display: 'flex', justifyContent: 'center' }}>
  <img src="https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-ollama-dropdown.png?fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=722c315dcb2563079865eccae2e0c56d" alt="Zed star icon in bottom right corner" width="50%" data-og-width="646" data-og-height="370" data-path="images/zed-ollama-dropdown.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-ollama-dropdown.png?w=280&fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=3f1d23fad0d96ddd6eacb1e79f2e4489 280w, https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-ollama-dropdown.png?w=560&fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=a83649ed5beedb5fa741276d3f0de691 560w, https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-ollama-dropdown.png?w=840&fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=5338e29259c73686c55ccc42a1ac1e11 840w, https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-ollama-dropdown.png?w=1100&fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=63cca84dff4a209986daaadd2f1a1fd4 1100w, https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-ollama-dropdown.png?w=1650&fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=51f3d300d365ffc31661b83985fa2008 1650w, https://mintcdn.com/ollama-9269c548/qP80N64oeC4tOgE5/images/zed-ollama-dropdown.png?w=2500&fit=max&auto=format&n=qP80N64oeC4tOgE5&q=85&s=d82841949d806094684e5e28c6c0a064 2500w" />
</div>

## Connecting to ollama.com

1. Create an [API key](https://ollama.com/settings/keys) on **ollama.com**
2. In Zed, open the **star icon** → **Configure**
3. Under **LLM Providers**, select **Ollama**
4. Set the **API URL** to `https://ollama.com`
