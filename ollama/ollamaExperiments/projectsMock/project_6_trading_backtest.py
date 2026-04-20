#!/usr/bin/env python3
"""
Project 6: Trading Strategy Backtest + Multi-Agent Advisor
GLM-4.7 (Analyst) generates signals → Phi-4 (Validator) reviews them →
manual vectorized backtest (no external library, just pandas + stdlib).

Note: backtrader is abandoned (last release 2022). We implement a simple
equity-curve backtest manually: iterate signals, track cash/position, compute
return, Sharpe ratio, max drawdown. Fully reproducible.

Usage:
    python project_6_trading_backtest.py                  # uses synthetic OHLCV
    python project_6_trading_backtest.py --csv prices.csv # real OHLCV file
    python project_6_trading_backtest.py --ticker SPY     # requires yfinance

Setup:
    pip install ollama pandas
    pip install yfinance   # only if using --ticker
"""
import argparse
import json
import math
import time
from ollama import Client

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARN] pandas not installed. Using built-in synthetic data only.")

ANALYST_MODEL = "glm-4.7-flash:latest"
VALIDATOR_MODEL = "phi4:latest"

ANALYST_PROMPT = """You are a quantitative analyst. Review this OHLCV price data (last 30 bars).

{data}

Generate exactly one trading signal per row in this JSON format — no other text:
{{
  "signal": "BUY" | "SELL" | "HOLD",
  "reason": "one line rationale",
  "entry_price": <float>,
  "stop_loss": <float>,
  "take_profit": <float>
}}

Base your analysis on: trend direction, momentum, volume confirmation.
Respond with JSON only."""

VALIDATOR_PROMPT = """You are a risk manager reviewing a trading signal.

Signal: {signal}

Check for:
1. Is the signal consistent with the price data trend?
2. Is the stop_loss reasonable (not too tight, not too wide)?
3. Any obvious bias (e.g., chasing momentum without confirmation)?

Respond with JSON only:
{{
  "approved": true | false,
  "risk_level": "low" | "medium" | "high",
  "comment": "one line"
}}"""


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def _synthetic_ohlcv(n=60) -> list[dict]:
    """Generate synthetic trending + noisy OHLCV for demo purposes."""
    import random
    rng = random.Random(42)
    price = 100.0
    rows = []
    for i in range(n):
        change = rng.gauss(0.001, 0.015)  # slight upward drift
        open_ = price
        close = round(price * (1 + change), 2)
        high = round(max(open_, close) * (1 + abs(rng.gauss(0, 0.005))), 2)
        low = round(min(open_, close) * (1 - abs(rng.gauss(0, 0.005))), 2)
        vol = int(rng.gauss(1_000_000, 200_000))
        rows.append({"date": f"2024-{(i//30)+1:02d}-{(i%30)+1:02d}",
                     "open": open_, "high": high, "low": low,
                     "close": close, "volume": max(100_000, vol)})
        price = close
    return rows


def load_csv(path: str) -> list[dict]:
    if not HAS_PANDAS:
        raise RuntimeError("pandas required for CSV loading")
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[["date", "open", "high", "low", "close", "volume"]].to_dict("records")


def fetch_yfinance(ticker: str, period="6mo") -> list[dict]:
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance not installed: pip install yfinance")
    df = yf.download(ticker, period=period, progress=False)
    df = df.reset_index()
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    df = df.rename(columns={"adj close": "close"})
    df["date"] = df["date"].astype(str)
    return df[["date", "open", "high", "low", "close", "volume"]].to_dict("records")


# ---------------------------------------------------------------------------
# Agent layer
# ---------------------------------------------------------------------------

class TradingAdvisor:
    def __init__(self):
        self.client = Client(host="http://localhost:11434")

    def _parse_json(self, text: str) -> dict | None:
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(
                l for l in text.splitlines() if not l.strip().startswith("```")
            ).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def generate_signal(self, rows: list[dict]) -> dict | None:
        last30 = rows[-30:]
        data_str = "\n".join(
            f"{r['date']}  O:{r['open']:.2f} H:{r['high']:.2f} "
            f"L:{r['low']:.2f} C:{r['close']:.2f} V:{int(r['volume'])}"
            for r in last30
        )
        prompt = ANALYST_PROMPT.format(data=data_str)
        t0 = time.time()
        r = self.client.chat(
            model=ANALYST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        elapsed = time.time() - t0
        result = self._parse_json(r["message"]["content"])
        status = result.get("signal", "?") if result else "PARSE_ERROR"
        print(f"  [Analyst/{ANALYST_MODEL}] {elapsed:.1f}s → {status}")
        return result

    def validate_signal(self, signal: dict) -> dict | None:
        prompt = VALIDATOR_PROMPT.format(signal=json.dumps(signal))
        t0 = time.time()
        r = self.client.chat(
            model=VALIDATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        elapsed = time.time() - t0
        result = self._parse_json(r["message"]["content"])
        approved = result.get("approved", False) if result else False
        risk = result.get("risk_level", "?") if result else "?"
        print(f"  [Validator/{VALIDATOR_MODEL}] {elapsed:.1f}s → approved={approved} risk={risk}")
        return result


# ---------------------------------------------------------------------------
# Backtest engine (no external library)
# ---------------------------------------------------------------------------

def backtest(rows: list[dict], signal: dict, validation: dict | None,
             initial_cash: float = 10_000.0) -> dict:
    """
    Simple single-trade backtest driven by LLM signal.

    Logic:
    - BUY:  enter at entry_price on day signal is generated,
            exit at take_profit or stop_loss (whichever is hit first).
            If validation is rejected, skip the trade.
    - SELL: short entry at entry_price, cover at stop_loss (stop) or take_profit.
    - HOLD: no trade, return equals 0.
    """
    approved = True
    if validation:
        approved = validation.get("approved", True)
        if not approved:
            print("  [Backtest] signal rejected by validator — no trade executed")
            return {
                "trade_executed": False,
                "reason": "validator rejected",
                "initial_cash": initial_cash,
                "final_cash": initial_cash,
                "total_return_pct": 0.0,
                "sharpe": None,
                "max_drawdown_pct": 0.0,
            }

    direction = signal.get("signal", "HOLD")
    if direction == "HOLD":
        return {
            "trade_executed": False,
            "reason": "HOLD signal",
            "initial_cash": initial_cash,
            "final_cash": initial_cash,
            "total_return_pct": 0.0,
            "sharpe": None,
            "max_drawdown_pct": 0.0,
        }

    entry = float(signal.get("entry_price", rows[-1]["close"]))
    stop = float(signal.get("stop_loss", entry * 0.97))
    target = float(signal.get("take_profit", entry * 1.05))
    shares = math.floor(initial_cash / entry)
    cost = shares * entry
    cash = initial_cash - cost

    # Walk forward from last bar in data
    equity_curve = [initial_cash]
    exit_price = None
    exit_reason = None

    for row in rows:
        price = row["close"]
        current_equity = cash + shares * price
        equity_curve.append(current_equity)

        if direction == "BUY":
            if price <= stop:
                exit_price, exit_reason = stop, "stop_loss"
                break
            if price >= target:
                exit_price, exit_reason = target, "take_profit"
                break
        else:  # SELL / short
            if price >= stop:
                exit_price, exit_reason = stop, "stop_loss"
                break
            if price <= target:
                exit_price, exit_reason = target, "take_profit"
                break
    else:
        # Reached end of data without exit — use last close
        exit_price = rows[-1]["close"]
        exit_reason = "end_of_data"

    if direction == "BUY":
        final_cash = cash + shares * exit_price
    else:
        # Short: profit if exit < entry
        pnl = shares * (entry - exit_price)
        final_cash = initial_cash + pnl

    total_return = (final_cash - initial_cash) / initial_cash * 100

    # Max drawdown on equity curve
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Simplified Sharpe: mean daily return / std daily return * sqrt(252)
    sharpe = None
    if len(equity_curve) > 2:
        daily_returns = [
            (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            for i in range(1, len(equity_curve))
        ]
        mean_r = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_r) ** 2 for r in daily_returns) / len(daily_returns)
        std_r = math.sqrt(variance) if variance > 0 else 0
        sharpe = round((mean_r / std_r) * math.sqrt(252), 2) if std_r > 0 else None

    return {
        "trade_executed": True,
        "direction": direction,
        "entry_price": entry,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "shares": shares,
        "initial_cash": initial_cash,
        "final_cash": round(final_cash, 2),
        "total_return_pct": round(total_return, 2),
        "sharpe": sharpe,
        "max_drawdown_pct": round(max_dd, 2),
        "equity_curve_len": len(equity_curve),
    }


def print_report(signal, validation, bt):
    print(f"\n{'='*60}")
    print("SIGNAL")
    print(f"  {signal.get('signal', '?')}  "
          f"entry={signal.get('entry_price')}  "
          f"stop={signal.get('stop_loss')}  "
          f"target={signal.get('take_profit')}")
    print(f"  Reason: {signal.get('reason', '')}")

    if validation:
        approved = validation.get("approved", False)
        print(f"\nVALIDATION")
        print(f"  approved={approved}  risk={validation.get('risk_level', '?')}")
        print(f"  {validation.get('comment', '')}")

    print(f"\nBACKTEST")
    if not bt["trade_executed"]:
        print(f"  No trade — {bt.get('reason', '')}")
    else:
        r = bt["total_return_pct"]
        sign = "+" if r >= 0 else ""
        print(f"  {bt['direction']}  {bt['shares']} shares @ {bt['entry_price']}"
              f" → exit @ {bt['exit_price']} ({bt['exit_reason']})")
        print(f"  Return:       {sign}{r}%")
        print(f"  Max drawdown: {bt['max_drawdown_pct']}%")
        if bt["sharpe"] is not None:
            print(f"  Sharpe ratio: {bt['sharpe']}")
        print(f"  Final cash:   ${bt['final_cash']:,.2f}  "
              f"(started: ${bt['initial_cash']:,.2f})")
    print("="*60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", help="CSV file with OHLCV columns")
    p.add_argument("--ticker", help="Fetch data via yfinance (e.g. SPY)")
    p.add_argument("--cash", type=float, default=10_000, help="Starting capital")
    args = p.parse_args()

    if args.csv:
        print(f"Loading CSV: {args.csv}")
        rows = load_csv(args.csv)
    elif args.ticker:
        print(f"Fetching {args.ticker} via yfinance...")
        rows = fetch_yfinance(args.ticker)
    else:
        print("No data source provided — using synthetic OHLCV (60 bars)")
        rows = _synthetic_ohlcv(60)

    print(f"Loaded {len(rows)} bars. Last close: {rows[-1]['close']:.2f}")

    advisor = TradingAdvisor()

    print("\nGenerating signal...")
    signal = advisor.generate_signal(rows)
    if not signal:
        print("[ERROR] Could not parse analyst signal")
        return

    print("Validating signal...")
    validation = advisor.validate_signal(signal)

    print("Running backtest...")
    bt = backtest(rows, signal, validation, initial_cash=args.cash)

    print_report(signal, validation, bt)


if __name__ == "__main__":
    main()
