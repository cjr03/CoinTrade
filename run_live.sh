#!/bin/sh
# Live trading on Kraken: one 50-day-trend sleeve per coin + Phosphor Desk at http://127.0.0.1:8787
# New coins start with $0; fund them with 'move 20 btc eth' or 'deposit 20 eth' in the UI.
cd "$(dirname "$0")"
export BTCBOT_MODE=live BTCBOT_CONFIRM_LIVE=I_UNDERSTAND BTCBOT_EXCHANGE=kraken \
       BTCBOT_POLICY=policy_momentum.json BTCBOT_CAPITAL=100 BTCBOT_MAX_NOTIONAL=110 \
       BTCBOT_COINS=BTC,ETH,SOL,XRP BTCBOT_BETS=1
exec .venv/bin/python -W ignore -m btcbot.app
