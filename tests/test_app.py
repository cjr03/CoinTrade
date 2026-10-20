import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from btcbot import app


class FakeTrader:
    def __init__(self, live, coin, usd=100.0, qty=0.001):
        self.live, self.coin = live, coin
        self.calls = []
        self.s = {"ledger": {"usd": usd, "qty": qty, "deposited": usd, "withdrawn": 0.0, "fees": 0.0}, "auto": True}
        self.busy = None

    @property
    def ledger(self):
        return self.s["ledger"]

    def status(self):
        return {"mode": "live" if self.live else "paper", "exchange": "kraken", "busy": None,
                "policy": {"model": "momentum", "round_trip": 0.008}, "state": self.s}

    def manual_buy(self, usd):
        self.calls.append(("buy", usd)); return {"size": 0.0003, "price": 80000.0, "fee": 0.1}

    def manual_sell(self, qty):
        self.calls.append(("sell", qty)); return {"size": qty, "price": 80000.0, "fee": 0.1}

    def deposit(self, usd):
        self.calls.append(("deposit", usd)); return "paper deposit"

    def withdraw(self, usd):
        self.calls.append(("withdraw", usd)); return {"refid": "X"}

    def set_auto(self, on):
        self.s["auto"] = on


class FakePortfolio:
    exchange = "kraken"

    def __init__(self, live):
        self.live = live
        self.traders = {"BTC": FakeTrader(live, "BTC"), "ETH": FakeTrader(live, "ETH", usd=10.0, qty=0.0)}
        self.events = []
        self.moves = []

    def __getitem__(self, c):
        return self.traders[c]

    @property
    def coins(self):
        return list(self.traders)

    pool = None

    def move(self, usd, src, dst):
        self.moves.append((usd, src, dst))

    def convert(self, src, dst, usd=None, fraction=None):
        self.moves.append(("convert", src, dst, usd if fraction is None else f"{fraction:.0%}"))
        return {"size": 0.001, "price": 80000.0, "fee": 0.6}, {"size": 0.03, "price": 2400.0, "fee": 0.6}

    def rebalance(self, weights):
        self.moves.append(("rebalance", weights))
        return []


@pytest.fixture(autouse=True)
def fixed_price(monkeypatch):
    monkeypatch.setattr(app, "ticker", lambda ex, coin="BTC": {"bid": 80000.0, "ask": 80000.0, "last": 80000.0, "t": 0})
    app._cache.clear(); app.PENDING.clear()


def test_paper_commands_execute_immediately():
    pf = FakePortfolio(live=False)
    t = pf["BTC"]
    assert app.run_command(pf, "buy 25")["ok"]
    assert app.run_command(pf, "sell 50%")["ok"]
    assert app.run_command(pf, "deposit 50")["ok"]
    assert t.calls == [("buy", 25.0), ("sell", 0.0005), ("deposit", 50.0)]


def test_commands_route_to_the_named_or_selected_coin():
    pf = FakePortfolio(live=False)
    assert app.run_command(pf, "buy 5 eth")["ok"]
    assert app.run_command(pf, "deposit 20", coin="ETH")["ok"]
    assert pf["ETH"].calls == [("buy", 5.0), ("deposit", 20.0)] and pf["BTC"].calls == []
    assert app.run_command(pf, "move 20 btc eth")["ok"] and pf.moves == [(20.0, "BTC", "ETH")]
    assert app.run_command(pf, "move 20 btc btc")["error"]
    assert app.run_command(pf, "move 3 to btc from eth")["ok"] and pf.moves[-1] == (3.0, "ETH", "BTC")
    assert app.run_command(pf, "move 4 from eth to btc")["ok"] and pf.moves[-1] == (4.0, "ETH", "BTC")
    assert app.run_command(pf, "buy 5", coin="SOL")["error"]  # selected coin not enabled: never falls back to BTC
    assert app.run_command(pf, "buy 5 btc eth")["error"]
    assert app.run_command(pf, "buy 5 sol")["error"]  # not enabled in this portfolio


def test_live_money_moves_need_matching_confirmation_code():
    pf = FakePortfolio(live=True)
    t = pf["BTC"]
    r = app.run_command(pf, "withdraw 20")
    assert r["confirm"] and t.calls == []
    assert app.run_command(pf, "confirm WRONG").get("error")
    assert t.calls == []
    r = app.run_command(pf, "withdraw 20")
    assert app.run_command(pf, f"confirm {r['confirm']}")["ok"]
    assert t.calls == [("withdraw", 20.0)]
    r = app.run_command(pf, "move 5 btc eth")
    assert r["confirm"] and pf.moves == []


def test_rejects_oversized_and_malformed_orders():
    pf = FakePortfolio(live=False)
    t = pf["BTC"]
    assert app.run_command(pf, "buy 500")["error"]
    assert app.run_command(pf, "sell 500")["error"]  # $500 worth > the $80 the sleeve holds
    assert app.run_command(pf, "buy lots")["error"]
    assert app.run_command(pf, "withdraw 10%")["error"]
    assert t.calls == []


def test_post_requires_session_token_and_local_host():
    app.Handler.portfolio = FakePortfolio(live=False)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/api/command"
    body = json.dumps({"line": "buy 25"}).encode()

    def post(headers):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers})
        try:
            return urllib.request.urlopen(req).status
        except urllib.error.HTTPError as e:
            return e.code

    assert post({}) == 403
    assert post({"X-Btcbot-Token": app.TOKEN, "Origin": "https://evil.example"}) == 403
    assert post({"X-Btcbot-Token": app.TOKEN, "Host": "evil.example"}) == 403
    assert post({"X-Btcbot-Token": app.TOKEN}) == 200
    srv.shutdown()


def test_convert_and_rebalance_parse_and_stage():
    pf = FakePortfolio(live=False)
    assert app.run_command(pf, "convert 20 btc eth")["ok"] and pf.moves[-1] == ("convert", "BTC", "ETH", 20.0)
    assert app.run_command(pf, "convert all eth btc")["ok"] and pf.moves[-1] == ("convert", "ETH", "BTC", None)
    assert app.run_command(pf, "convert 20 btc")["error"]
    assert app.run_command(pf, "rebalance btc 60 eth 40")["ok"] and pf.moves[-1] == ("rebalance", {"BTC": 60.0, "ETH": 40.0})
    assert app.run_command(pf, "rebalance btc 80 eth 40")["error"]  # over 100%
    assert app.run_command(pf, "rebalance btc eth 50")["error"]     # btc has no number
    assert app.run_command(pf, "rebalance btc 50 doge 50")["error"]
    assert app.run_command(pf, "convert 50% btc eth")["ok"] and pf.moves[-1] == ("convert", "BTC", "ETH", "50%")
    assert app.run_command(pf, "bets")["error"]  # no pool in this portfolio
    live = FakePortfolio(live=True)
    r = app.run_command(live, "rebalance btc 100")
    assert r["confirm"] and live.moves == []


def test_portfolio_value_ignores_internal_moves(tmp_path, monkeypatch):
    import btcbot.trader as trader_mod
    monkeypatch.setattr(trader_mod, "STATE", tmp_path)
    pf = trader_mod.Portfolio("paper", "kraken", "policy_momentum.json", ["BTC", "ETH"], bets=True)
    monkeypatch.setattr(app, "markets", lambda ex, coins: {c: {"last": 100.0, "bid": 100.0, "chg24": 0.01,
                                                               "high24": 101.0, "low24": 99.0, "vol24_usd": 1e6}
                                                           for c in coins})
    app._cache.clear()
    before = app.portfolio_value(pf)
    pf.move(15, "BTC", "BETS")
    pf.move(10, "BTC", "ETH")
    after = app.portfolio_value(pf)
    assert after["equity"] == pytest.approx(before["equity"]) == pytest.approx(100.0)
    assert after["pnl"] == pytest.approx(0.0) and after["basis"] == pytest.approx(100.0)
    assert after["parts"] == {"BTC": 75.0, "ETH": 10.0, "BETS": 15.0}
    assert set(after["coins"]) == {"BTC", "ETH"}
    assert after["coins"]["ETH"]["last"] == 100.0 and after["coins"]["ETH"]["chg24"] == 0.01
    assert after["coins"]["BTC"]["qty"] == 0 and after["coins"]["ETH"]["value"] == pytest.approx(10.0)
