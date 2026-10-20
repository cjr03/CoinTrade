# CoinTrade

Crypto price-prediction research and an automated trader for BTC, ETH, SOL and XRP on Kraken Pro
(or Coinbase Advanced Trade), with a retro terminal and browser trading desk.

Research-backed pipeline: data → causal features → purged walk-forward training → fee-aware
backtest with a go-live gate → hourly trader (paper or live). The Python package is `btcbot`.

> **Risk notice.** This trades real money when you switch it to live mode. Every test in this
> project says short-term prediction does not beat exchange fees; the deployed rule is a simple
> trend filter that cut drawdowns in backtests but did not significantly beat holding. Nothing
> here is financial advice. Start in paper mode.

## Results (out-of-sample, 31 quarterly test folds, 2019-01 → 2026-09, $100 start)

| Strategy (Kraken fees 0.40% maker / 0.80% taker) | CAGR | Sharpe | Max DD | Trades/yr |
|---|---|---|---|---|
| Buy & hold | +49.5% | 0.96 | −77% | – |
| **SMA-50 trend filter** (long when daily close > 50d SMA) | **+53.7%** | **1.20** | −61% | 9.7 |
| SMA-100 trend filter | +50.1% | 1.13 | −44% | 6.4 |
| Best ML: LightGBM / Ridge, 12h horizon | +2.0% | 0.23 | −16% | 7 |
| Validation-selected ML: LightGBM, 72h (deployed ML policy) | −1.1% | −0.06 | −19% | 1 |

* **No ML configuration passed the go-live gate** at any fee level (Coinbase 0.60/1.20,
  0.40/0.60, Kraken 0.40/0.80). Out-of-sample IC is ≈ 0.01–0.03; the model is right
  slightly more often than chance but the edge is far below the ~0.8–1.2% round-trip cost.
* Zero-shot foundation models (Chronos-Bolt, Chronos-2) scored IC −0.03 / −0.08 on 2025+.
* Frequent short-term up/down bets (1h/4h/12h/24h horizons, long and short, generous all-maker
  fills) were tested separately: betting every hour turns $100 into $0.00; the best of 56
  configurations picked in hindsight ends at $139 vs $2,804 for the trend filter. The models are
  right 52–57% of the time, but capture 1–12 bp per bet against an 80 bp round trip.
* Confidence-gated bets ("only bet when the probability is high enough"), three designs with
  validation-only thresholds and an independent audit: calibrated up/down probability $100 → $100.59
  (flat 28 of 31 quarters), cost-aware "profitable move" classifiers → $60.68, take-profit/stop bets
  → $92.26. The models' 70%+ calls came true only ~53% of the time.
* The trend filter beats buy & hold on Sharpe and drawdown, but **not significantly**
  (bootstrap p = 0.20) and only for 50–100 day windows (20 and 200 lose) — treat it as a
  risk-reduction rule, not alpha. 2019–2026 was a strong BTC bull period.

## Coins

BTC, ETH, SOL and XRP each run as a separate **sleeve** with its own cash ledger, position, kill
switch and Kraken pair (`BTCBOT_COINS=BTC,ETH,SOL,XRP`). New sleeves start with $0; fund them with
`move 20 btc eth` (moves bot cash between sleeves) or `deposit 20 eth`. Cash booked to one coin is
reserved from the others, so no sleeve can spend another's money.

Trend rule vs holding per coin at Kraken fees (`python -m btcbot.coin_eval`):

| coin (test span) | trend CAGR / max DD | hold CAGR / max DD | auto trading by default |
|---|---|---|---|
| BTC (2019–2026) | +55.3% / −58% | +48.9% / −77% | on |
| ETH (2019–2026) | +48.8% / −59% | +45.3% / −81% | on |
| SOL (2021–2026) | +44.2% / −73% | +8.8% / −97% | on |
| XRP (2023–2026, Coinbase delisted it 2021–23) | +15.3% / −72% | +39.5% / −73% | **off** (manual only; `bot on xrp` to enable) |

## Cross-selling and the short-term bet pool

* `convert 20 btc eth` / `convert all sol xrp` sell one coin and buy another with the proceeds;
  `rebalance btc 50 eth 30 sol 20` redistributes the sleeves' combined value to those percentages.
  Both use market orders (about 0.8% per trade at Kraken tier 1); converted coins become your manual
  holdings (the trend bot guards them only with the kill switch).
* The **bet pool** (`BTCBOT_BETS=1`) places short-term up/down bets on every coin when a calibrated
  direction model is confident enough (`python -m btcbot.bets --train`). It trades only its own cash
  (`move 15 btc bets`), one ~$6–10 bet per coin at a time, each with a 2-sigma stop, and halts itself
  at −50% of what was put in. Bets are **long-only** in live trading; margin shorts exist but are
  experimental (paper only unless `BTCBOT_BET_SHORTS=1`), because margin eligibility, reduce-only
  closes and Kraken's real rollover charges are not reconciled. Expectation, measured on the 6
  months before training (thresholds forced to stay active):

| coin | horizon / threshold | bets per year | $100 after 6 months |
|---|---|---|---|
| BTC | 24h, P(up) ≥ 60% | 293 | $32 |
| ETH | 24h, P(up) ≥ 55% | 343 | $26 |
| SOL | 24h, P(up) ≥ 55% | 297 | $44 |
| XRP | 12h, P(up) ≥ 55% | 30 | $95 |

  Treat the pool as a cost of curiosity, not a profit engine: every test in this project says these
  bets lose to fees.

## Layout
| module | role |
|---|---|
| `btcbot/data.py` | Coinbase public candle downloader (1h, 5m, ETH) with parquet cache |
| `btcbot/features.py` | strictly causal hourly features (returns, trend, vol, 5m flow, ETH, seasonality) |
| `btcbot/train.py` | purged walk-forward: LightGBM (huber), Ridge, LightGBM triple-barrier; `--final` |
| `btcbot/backtest.py` | event-driven long/flat sim: post-only fills, fees, stops, kill switch, bootstrap |
| `btcbot/evaluate.py` | threshold selection on validation only, baselines, gate, policy export |
| `btcbot/chronos_eval.py` | zero-shot Chronos-Bolt / Chronos-2 challenger |
| `btcbot/broker.py` | Coinbase live broker + paper broker |
| `btcbot/kraken.py` | Kraken live broker (REST, HMAC signed) + Kraken paper broker |
| `btcbot/trader.py` | hourly loop: refresh data → signal → policy → broker; bot ledger, state + journal in `state/` |
| `btcbot/app.py`, `btcbot/web/` | Phosphor Desk: local server, browser UI, command line, model notes |
| `btcbot/coins.py`, `btcbot/coin_eval.py` | coin identifiers per venue; per-coin trend-vs-hold evidence |
| `btcbot/bets.py` | short-term bet pool: per-coin calibrated models, fenced ledger, margin shorts |
| `btcbot/tui.py` | Phosphor Desk in the terminal (Textual), a client of the local server |

## Usage
Market data, model files and your trade ledgers are not in the repository (see `.gitignore`); the
commands below rebuild the first two, and the bot creates `state/` on its first run. Policy files
and evaluation reports are in `artifacts/`.
```sh
git clone https://github.com/cjr03/CoinTrade.git && cd CoinTrade
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env                                  # live trading only: add your exchange keys
.venv/bin/python -m btcbot.data --products BTC-USD,ETH-USD,SOL-USD,XRP-USD --granularity ONE_HOUR --days 4200
.venv/bin/python -m btcbot.data --products BTC-USD --granularity FIVE_MINUTE --days 3580
.venv/bin/python -m btcbot.train                      # walk-forward OOS predictions
.venv/bin/python -m btcbot.evaluate --fees kraken_0.40_0.80
.venv/bin/python -m btcbot.train --final --model lgbm --horizon 72
.venv/bin/python -m btcbot.evaluate --horizons 72 --models lgbm --fees kraken_0.40_0.80 --export 72,lgbm,kraken_0.40_0.80
.venv/bin/python -m btcbot.evaluate --export-momentum kraken_0.40_0.80   # writes policy.json (copy aside)
.venv/bin/python -m btcbot.bets --train                # short-term bet pool models (artifacts/bets_*.pkl)
.venv/bin/python -m btcbot.coin_eval                   # per-coin trend vs hold
.venv/bin/python -m pytest -q tests
```

### Phosphor Desk (terminal and browser UI)
```sh
./run_live.sh          # live bot on Kraken (trend policy) + local server on 127.0.0.1:8787
./tui.sh               # the desk in your terminal — attaches to the running bot
./tui.sh --start       # no bot running? host one inside this terminal session instead
# or open http://127.0.0.1:8787 in a browser
```
The terminal desk is a client of the running bot, so the browser and any number of terminals can
watch at once without a second trader ever touching the account. Click a chart range (10m/1h/1d/1w),
a coin, a market row in the portfolio panel, or a command chip above the prompt (buy, sell, deposit,
withdraw and move fill the prompt for you to finish; status, help and bets run at once; clicking the
bot lamp fills `bot off`/`bot on`). `↑`/`↓` command history, `ctrl+l` clear console, `ctrl+q` quit.
Best at ≥ 110×36; narrower terminals hide the side panels but keep the money lamps.

Both UIs run the hourly bot's local server and show: live BTC price (exchange
WebSocket), chart with 10m / 1h / 1d / 1w views (click to switch) and the model's projection cone, a
crawl of model notes (training philosophy, research results, current projection, recent
trades), account/model/fills panels, and a command line:

| command | paper | live |
|---|---|---|
| `buy 25`, `buy all` | market buy from bot cash | same, after `confirm CODE` |
| `sell 25` ($ worth), `sell 50%`, `sell all`, `sell 0.0002 btc` | market sell | same, after `confirm CODE` |
| `deposit 50` | adds paper cash | hands $50 of exchange USD to the bot (bank deposits must be started in the exchange app) |
| `withdraw 20` | removes paper cash | Kraken: sends to your pre-verified bank account, only with `BTCBOT_ALLOW_WITHDRAW=1`, `KRAKEN_WITHDRAW_KEY=<bank name in Kraken>` and a key with Withdraw permission |
| `bot off` / `bot on`, `resume`, `status`, `view 1d`, `clear`, `help` | | |

The bot trades only its own ledger (deposits − withdrawals ± P&L), never the rest of the
account. The server binds to 127.0.0.1 and every command needs the per-session page token.

Run the trader headless instead (paper by default; state and journal go to `state/`):
```sh
BTCBOT_EXCHANGE=kraken BTCBOT_POLICY=policy_momentum.json .venv/bin/python -m btcbot.trader
BTCBOT_EXCHANGE=kraken BTCBOT_POLICY=policy_ml_H72.json   .venv/bin/python -m btcbot.trader
```
Live trading additionally needs `BTCBOT_MODE=live BTCBOT_CONFIRM_LIVE=I_UNDERSTAND` and API keys in
`.env` (`KRAKEN_API_KEY`/`KRAKEN_API_SECRET` or `COINBASE_API_KEY`/`COINBASE_API_SECRET`; see
`.env.example`). Keys need trade permissions only — never withdrawal.

## Safety rails
post-only maker orders priced at the decision price, never chasing it · market orders only for
stops/kill switch · $110 per-order cap · fee guard (no entries if live fee > backtested) ·
stale-data guard (BTC 1h, ETH 1h, 5m must be current) · open orders cancelled on startup ·
drawdown kill switch halts until a human removes `halted` from the state file.

## License
MIT, see [LICENSE](LICENSE).
