#!/usr/bin/env python3
"""
Project 9: Finance Multi-Agent Advisor
Analyst (GLM-4.7) → Validator (Phi-4) → final recommendation.
Antirez-style: minimal, direct, no bloat.

Usage:
    python project_9_finance_advisor.py
    python project_9_finance_advisor.py --ticker AAPL --price 213.5 --pe 34.2 --mcap 3.2T

Setup:
    pip install ollama
"""
import argparse
import json
import time
from ollama import Client

ANALYST_MODEL = "glm-4.7-flash:latest"
VALIDATOR_MODEL = "phi4:latest"

ANALYST_PROMPT = """You are a senior equity analyst. Analyze this asset and provide:
1. Key metrics interpretation (2-3 lines)
2. Main risk and main opportunity (1 line each)
3. Signal: BUY / SELL / HOLD with confidence score 1-10

Asset data:
{data}

Be concise. End your response with a line: SIGNAL: <BUY|SELL|HOLD> CONFIDENCE: <1-10>"""

VALIDATOR_PROMPT = """You are a risk compliance officer reviewing an analyst recommendation.

Analyst report:
{analysis}

Provide:
1. Any missed risk or bias in the analysis (1-2 lines)
2. Alternative interpretation of the data (1 line)
3. Your confidence adjustment: ADJUSTED_CONFIDENCE: <1-10>

Be concise and critical."""


class FinanceAdvisor:
    def __init__(self):
        self.client = Client(host="http://localhost:11434")

    def analyze(self, asset_data: dict) -> str:
        prompt = ANALYST_PROMPT.format(data=json.dumps(asset_data, indent=2))
        t0 = time.time()
        r = self.client.chat(
            model=ANALYST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        elapsed = time.time() - t0
        content = r["message"]["content"]
        print(f"  [Analyst/{ANALYST_MODEL}] {elapsed:.1f}s")
        return content

    def validate(self, analysis: str) -> str:
        prompt = VALIDATOR_PROMPT.format(analysis=analysis[:800])
        t0 = time.time()
        r = self.client.chat(
            model=VALIDATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        elapsed = time.time() - t0
        content = r["message"]["content"]
        print(f"  [Validator/{VALIDATOR_MODEL}] {elapsed:.1f}s")
        return content

    def _extract_signal(self, text: str) -> tuple[str, int]:
        """Parse 'SIGNAL: BUY CONFIDENCE: 7' from analyst output."""
        signal, confidence = "HOLD", 5
        for line in text.splitlines():
            line = line.strip().upper()
            if "SIGNAL:" in line:
                for s in ("BUY", "SELL", "HOLD"):
                    if s in line:
                        signal = s
                        break
            if "CONFIDENCE:" in line and "ADJUSTED" not in line:
                for token in line.split():
                    if token.isdigit():
                        confidence = max(1, min(10, int(token)))
                        break
        return signal, confidence

    def _extract_adjusted_confidence(self, text: str) -> int | None:
        for line in text.splitlines():
            if "ADJUSTED_CONFIDENCE:" in line.upper():
                for token in line.split():
                    if token.isdigit():
                        return max(1, min(10, int(token)))
        return None

    def recommend(self, asset_data: dict) -> dict:
        print(f"\nAnalyzing {asset_data.get('ticker', 'unknown')}...")

        analysis = self.analyze(asset_data)
        validation = self.validate(analysis)

        signal, analyst_conf = self._extract_signal(analysis)
        adjusted_conf = self._extract_adjusted_confidence(validation)
        final_conf = adjusted_conf if adjusted_conf is not None else analyst_conf

        return {
            "ticker": asset_data.get("ticker"),
            "signal": signal,
            "analyst_confidence": analyst_conf,
            "final_confidence": final_conf,
            "analyst_view": analysis,
            "validator_critique": validation,
        }


def print_report(result: dict):
    signal_color = {"BUY": "↑", "SELL": "↓", "HOLD": "→"}.get(result["signal"], "?")
    print(f"\n{'='*60}")
    print(f"  {result['ticker']}  {signal_color} {result['signal']}  "
          f"confidence {result['final_confidence']}/10")
    print(f"{'='*60}")
    print("\n[Analyst view]")
    print(result["analyst_view"])
    print("\n[Validator critique]")
    print(result["validator_critique"])
    print(f"\n[Final] {result['signal']} — confidence {result['final_confidence']}/10"
          f" (analyst: {result['analyst_confidence']}/10)")


DEMO_ASSETS = [
    {
        "ticker": "NVDA",
        "price": 118.5,
        "pe_ratio": 35.2,
        "forward_pe": 29.1,
        "market_cap": "2.9T",
        "revenue_growth_yoy": "122%",
        "gross_margin": "74%",
        "debt_to_equity": 0.42,
        "sector": "Semiconductors",
        "notes": "AI infrastructure demand driving data center GPU sales. Competition from AMD and custom silicon growing.",
    },
    {
        "ticker": "BTCUSD",
        "price": 95400,
        "market_cap": "1.89T",
        "dominance": "53%",
        "30d_change": "+18%",
        "spot_etf_inflows_7d": "+2.1B",
        "funding_rate": "0.012% (elevated)",
        "notes": "Post-halving cycle. Institutional inflows via ETFs. Macro uncertainty from Fed policy.",
    },
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", help="Run a custom asset (requires --price)")
    p.add_argument("--price", type=float)
    p.add_argument("--pe", type=float, help="P/E ratio")
    p.add_argument("--mcap", help="Market cap string (e.g. 3.2T)")
    p.add_argument("--notes", default="")
    p.add_argument("--demo", type=int, choices=[0, 1], default=None,
                   help="Run only demo asset 0 or 1 (default: run both)")
    args = p.parse_args()

    advisor = FinanceAdvisor()

    if args.ticker:
        if not args.price:
            print("--price required with --ticker")
            return
        asset = {"ticker": args.ticker, "price": args.price}
        if args.pe: asset["pe_ratio"] = args.pe
        if args.mcap: asset["market_cap"] = args.mcap
        if args.notes: asset["notes"] = args.notes
        assets = [asset]
    elif args.demo is not None:
        assets = [DEMO_ASSETS[args.demo]]
    else:
        assets = DEMO_ASSETS

    for asset in assets:
        result = advisor.recommend(asset)
        print_report(result)


if __name__ == "__main__":
    main()
