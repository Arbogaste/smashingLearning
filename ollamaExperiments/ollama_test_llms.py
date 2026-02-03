import time
from ollama import Client

# Specs:
# - Scan available ollama models
# - If models are defined in config, test only those models
# - For each model, ask complex programming questions and log response time
# - If "tools" is flagged, test the model's ability to use external tools (calculator, web search)
#   Tools are passed to the chat and the model can invoke them via tool_calls
# - If "code" is flagged, test the model's ability to generate correct code
# - If "finance" is flagged, test the model's ability to answer finance-related questions
# - If "stream" is flagged, test streaming responses with real-time token generation

"""redirect outputs to log file if needed 
    python ollama_test_llms.py > ollama_test_llms.log 2>&1
"""
def calculate_expression(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: The mathematical expression to evaluate
        
    Returns:
        The result of the calculation
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


class TestLLMs:
    def __init__(self, config):
        """Initialize with config dict."""
        self.config = config
        self.client = Client(host='http://localhost:11434')
        
        # Get models to test
        if "models" in config and config["models"]:
            self.models = config["models"]
        else:
            response = self.client.list()
            self.models = [m['name'] for m in response['models']]
        
        # Define tools schema for function calling (used when tools=True)
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_expression",
                    "description": "Evaluate a mathematical expression and return the result",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to evaluate (e.g., '12345 * 67890')"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

    def test_model(self, model):
        """Test a single model with programming questions."""
        print(f"\n=== {model} ===")
        
        # Question 1: Complex reasoning task - controlled by "reasoning" flag
        if self.config.get("reasoning", True):  # Enabled by default
            q1 = """You are designing a caching layer for a distributed system that handles millions of requests per second.
Given constraints:
- Limited memory (32GB)
- Cache hit rate is critical for performance
- Data has variable TTLs (1 min to 1 hour)
- Need to support both read-heavy and write-heavy workloads
- Geographic distribution across 5 regions

Explain:
1. What eviction policy would you choose and why?
2. How would you handle cache coherence across regions?
3. What trade-offs exist between consistency and performance?
4. Provide pseudocode for the core cache lookup/update logic."""
            t0 = time.time()
            r1 = self.client.chat(model=model, messages=[{"role": "user", "content": q1}], stream=False)
            t1 = time.time() - t0
            print(f"Q1 Reasoning ({t1:.2f}s):\n{r1['message']['content'][:400]}...")
        else:
            print("Q1 Reasoning: SKIPPED")

        # Question 2: Code generation (if enabled)
        if self.config.get("code"):
            q2 = "Write a Python function for factorial."
            t0 = time.time()
            r2 = self.client.chat(model=model, messages=[{"role": "user", "content": q2}], stream=False)
            t2 = time.time() - t0
            print(f"Q2 Code ({t2:.2f}s): {r2['message']['content'][:100]}...")
        else:
            print("Q2 Code: SKIPPED")

        # Question 3: Tools (if enabled)
        # Uses actual tool calling: model will invoke calculate_expression if needed
        if self.config.get("tools"):
            q3 = "Calculate this complex expression and show your work: (1234552247244724472477242 * 67890242457)/7.5"
            t0 = time.time()
            # Pass tools to the chat - model can invoke them via tool_calls
            r3 = self.client.chat(
                model=model, 
                messages=[{"role": "user", "content": q3}], 
                stream=False,
                tools=self.tools  # Enable tool calling
            )
            t3 = time.time() - t0
            
            # Check if model made tool calls
            content = r3['message']['content']
            if r3['message'].get('tool_calls'):
                print(f"Q3 Tools ({t3:.2f}s): Model invoked tools")
                for tool_call in r3['message']['tool_calls']:
                    tool_name = tool_call['function']['name']
                    args = tool_call['function']['arguments']
                    print(f"  → {tool_name}({args})")
                    # Execute the tool
                    result = calculate_expression(args.get('expression', ''))
                    print(f"  ← Result: {result}")
            else:
                print(f"Q3 Tools ({t3:.2f}s): {content[:100]}...")
        else:
            print("Q3 Tools: SKIPPED")

        # Question 4: Finance decision (if enabled)
        if self.config.get("finance"):
            q4 = """
You are a senior financial analyst. Analyze the following asset and provide a buy/sell/hold recommendation with clear reasoning.

Asset: Apple Inc. (AAPL)
Current Price: $195.23
52-Week High: $252.88
52-Week Low: $168.47
P/E Ratio: 28.5
Forward P/E: 24.2
Dividend Yield: 0.42%
Market Cap: $3.2T
Revenue Growth (YoY): 2.1%
Net Profit Margin: 26.3%
Free Cash Flow: $110.5B
Debt-to-Equity: 1.85
Current Macro Environment: Fed has kept rates stable at 4.5%, inflation at 2.1%, bond yields rising slightly

Given these metrics and the current macroeconomic conditions in February 2026, should an institutional investor with a 2-year horizon buy, sell, or hold AAPL? Justify your decision.
"""
            t0 = time.time()
            r4 = self.client.chat(model=model, messages=[{"role": "user", "content": q4}], stream=False)
            t4 = time.time() - t0
            print(f"Q4 Finance ({t4:.2f}s): {r4['message']['content'][:150]}...")
        else:
            print("Q4 Finance: SKIPPED")

        # Question 5: Streaming response (if enabled)
        # Uses stream=True: model generates tokens progressively, better for long responses
        if self.config.get("stream"):
            q5 = "Describe quantum computing in 3 paragraphs."
            t0 = time.time()
            print(f"Q5 Streaming: ", end="", flush=True)
            full_response = ""
            
            # Stream chunks as they arrive - model generates progressively
            for chunk in self.client.chat(
                model=model, 
                messages=[{"role": "user", "content": q5}], 
                stream=True  # Enable streaming
            ):
                content = chunk['message']['content']
                full_response += content
                print(content, end="", flush=True)
            
            t5 = time.time() - t0
            # Calculate tokens/second if available
            tokens = len(full_response.split())
            tokens_per_sec = tokens / t5 if t5 > 0 else 0
            print(f"\n({t5:.2f}s, ~{tokens} tokens, {tokens_per_sec:.1f} tokens/sec)")
        else:
            print("Q5 Streaming: SKIPPED")

    def run_all(self):
        """Run tests on all configured models."""
        for model in self.models:
            try:
                self.test_model(model)
            except Exception as e:
                print(f"{model}: ERROR - {e}")


if __name__ == "__main__":
    config = {
        "models": ["llama2:latest"], # Specify models to test; empty list means all available models
        "reasoning": True,   # Q1: Explain complex system design
        "code": True,       # Q2: Code generation
        "tools": True,      # Q3: Tool calling
        "finance": True,     # Q4: Finance analysis
        "stream": True       # Q5: Streaming response
    }
    
    # Print enabled tests, redirect outputs to log file if needed 
    # python ollama_test_llms.py > ollama_test_llms.log 2>&1
    enabled = [k for k, v in config.items() if k != "models" and v]
    print(f"[Config] Testing: {', '.join(enabled) if enabled else 'NONE'}")
    
    tester = TestLLMs(config)
    tester.run_all()

