#!/bin/sh
# Phosphor Desk in the terminal. Attaches to the running bot (start it first with ./run_live.sh).
cd "$(dirname "$0")"
exec .venv/bin/python -W ignore -m btcbot.tui "$@"
