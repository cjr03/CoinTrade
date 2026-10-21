import pandas as pd
import pytest

import btcbot.trader as trader_mod


class FakeBroker:
    min_order_usd = 1.0

    def __init__(self, usd=100.0, btc=0.0, bid=100.0):
        self.usd, self.btc, self.bid = usd, btc, bid
        self.fail_fallback = False
        self.quote_fails = False

    def quote(self):
        if self.quote_fails:
            raise ConnectionError("quote down")
        return self.bid, self.bid + 0.1

    def balances(self):
        return self.usd, self.btc

    def fees(self):
        return 0.004, 0.008

    def product_ok(self):
        return True

    def cancel_open_orders(self):
        return []

    def buy_maker(self, usd, deadline_s=0, ref_px=None):
        size = usd / (self.bid * 1.004)
        self.usd -= usd; self.btc += size
        return {"price": self.bid, "size": size, "fee": usd - size * self.bid, "liquidity": "maker"}

    def market_buy(self, usd):
        return self.buy_maker(usd)

    def sell_maker(self, btc, deadline_s=0, ref_px=None, fallback_taker=True):
        part = btc * 0.6  # 60% fills as maker; then the taker fallback may break
        self.btc -= part; self.usd += part * self.bid
        if self.fail_fallback:
            return {"price": self.bid, "size": part, "fee": 0.0, "liquidity": "maker"}
        self.btc -= btc - part; self.usd += (btc - part) * self.bid
        return {"price": self.bid, "size": btc, "fee": 0.0, "liquidity": "maker"}

    def market_sell(self, btc):
        btc = min(btc, self.btc)
        self.btc -= btc; self.usd += btc * self.bid
        return {"price": self.bid, "size": btc, "fee": 0.0, "liquidity": "taker"}

    def deposit(self, usd):
        self.usd += usd

    def withdraw(self, usd):
        self.usd -= usd
        return {"refid": "R1"}


def make(tmp_path, monkeypatch, live=False, policy="policy_momentum.json"):
    monkeypatch.setattr(trader_mod, "STATE", tmp_path)
    monkeypatch.setattr(trader_mod, "LATE_MINUTES", 61)  # tests may run at any minute of the hour
    t = trader_mod.Trader("paper", "kraken", policy)
    t.live = live
    t.broker = FakeBroker()
    t.s["ledger"] = {"usd": 100.0, "qty": 0.0, "deposited": 100.0, "withdrawn": 0.0, "fees": 0.0}
    t.s.pop("paper", None)
    return t


def signal(t, pred, bar):
    t.predict = lambda: {"bar": bar, "pred": pred, "sigma_h": 0.02, "horizon": 24, "vol_pctile": 0.5,
                         "close": t.broker.bid, "sma": 90.0, "daily_close": 100.0, "sma_days": 50}


def fresh_bar():
    return str(pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1))


def test_entry_then_partial_exit_is_booked(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    signal(t, 1.0, fresh_bar()); t.step()
    assert t.s["position"] and t.ledger["qty"] > 0
    t.broker.fail_fallback = True
    t.s["last_bar"] = None
    t.s["position"]["entry_time"] = "2000-01-01T00:00:00+00:00"
    signal(t, -1.0, fresh_bar()); t.step()
    assert t.ledger["qty"] == pytest.approx(t.broker.btc)  # nothing lost, nothing invented
    assert t.ledger["usd"] == pytest.approx(t.broker.usd, abs=0.5)


def test_btc_missing_from_exchange_halts_instead_of_shrinking_equity(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    t.ledger["qty"] = 0.5; t.ledger["usd"] = 50.0
    t.s["position"] = {"entry_time": trader_mod.now_iso(), "entry_px": 100.0, "qty": 0.5, "frac": 1, "stop_px": 1.0}
    t.broker.btc = 0.1
    signal(t, 1.0, fresh_bar()); t.step()
    assert t.s.get("needs_review") and t.ledger["qty"] == 0.5 and not t.s.get("halted")


def test_manual_position_survives_bot_sell_signal_but_not_kill_switch(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    t.manual_buy(80)
    assert t.s["position"]["manual"]
    t.s["position"]["entry_time"] = "2000-01-01T00:00:00+00:00"
    signal(t, -1.0, fresh_bar()); t.step()
    assert t.s.get("position")  # the bot's sell signal left it alone
    t.s["auto"] = False
    t.s["last_bar"] = None
    t.broker.bid = 50.0  # about -40% equity
    signal(t, -1.0, fresh_bar()); t.step()
    assert t.s.get("halted") and not t.s.get("position")


def test_withdraw_is_booked_even_if_quote_fails_afterwards(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    t.s["hwm"] = 100.0
    t.broker.quote_fails = True
    t.withdraw(30)
    assert t.ledger["usd"] == 70 and t.ledger["withdrawn"] == 30 and t.s["hwm"] == 70
    assert not t.s.get("withdraw_pending")


def test_rejects_non_finite_amounts(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    for bad in (float("inf"), float("nan"), -5, 0):
        with pytest.raises(ValueError):
            t.deposit(bad)


def test_ungated_ml_policy_does_not_enter_live(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, live=True, policy="policy_ml_H72.json")
    signal(t, 0.5, fresh_bar())  # far above the 3.8% entry bar
    t.step()
    assert not t.s.get("position")


def test_old_state_file_is_migrated(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(trader_mod, "STATE", tmp_path)
    (tmp_path / "kraken_paper_policy_momentum.json").write_text(json.dumps({
        "ledger": {"usd": 0.09, "btc": 0.0012039, "deposited": 100.0, "withdrawn": 0.0, "fees": 0.4},
        "position": {"entry_time": "2026-10-20T18:09:53+00:00", "entry_px": 82658.6, "btc": 0.0012039,
                     "frac": 1.0, "manual": False, "stop_px": 0.0}}))
    t = trader_mod.Trader("paper", "kraken", "policy_momentum.json")
    assert t.ledger["qty"] == 0.0012039 and "btc" not in t.ledger
    assert t.s["position"]["qty"] == 0.0012039


def test_portfolio_keeps_sleeves_apart_and_moves_cash(tmp_path, monkeypatch):
    monkeypatch.setattr(trader_mod, "STATE", tmp_path)
    pf = trader_mod.Portfolio("paper", "kraken", "policy_momentum.json", ["BTC", "ETH"])
    assert pf["BTC"].ledger["usd"] == 100 and pf["ETH"].ledger["usd"] == 0  # new coins start unfunded
    assert pf["BTC"].state_path.name == "kraken_paper_policy_momentum.json"
    assert pf["ETH"].state_path.name == "kraken_paper_policy_momentum_ETH.json"
    pf.move(30, "BTC", "ETH")
    assert pf["BTC"].ledger["usd"] == 70 and pf["ETH"].ledger["usd"] == 30
    assert pf["BTC"].broker.balances()[0] == 70 and pf["ETH"].broker.balances()[0] == 30  # paper cash moved too
    assert pf["ETH"].other_sleeves_usd() == 70
    with pytest.raises(ValueError):
        pf.move(500, "BTC", "ETH")


def test_live_room_counts_other_sleeves_resting_buy_once(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, live=True)
    t.ledger["usd"] = 30.0
    t.other_sleeves_usd = lambda: 70.0           # BTC sleeve still books its $70 ...
    t.broker.usd = 30.0                          # ... which is on hold for its resting buy (available)
    t.broker.total_usd = lambda: 100.0           # total includes the hold
    assert t._room(t.broker.usd) == pytest.approx(30.0)  # old code: 30 - 70 = -40, entry skipped


def test_moving_cash_out_does_not_fake_a_drawdown(tmp_path, monkeypatch):
    monkeypatch.setattr(trader_mod, "STATE", tmp_path)
    pf = trader_mod.Portfolio("paper", "kraken", "policy_momentum.json", ["BTC", "ETH"])
    a = pf["BTC"]
    a.ledger.update(usd=50.0, qty=0.625)
    a.s["signal"] = {"close": 80.0}
    a.s["hwm"] = 133.0                            # equity 100 -> 24.8% drawdown
    pf.move(45, "BTC", "ETH")
    eq = a.equity(80.0)
    assert 1 - eq / a.s["hwm"] == pytest.approx(1 - 100 / 133.0)  # same % drawdown, kill switch not tripped


def test_bet_pool_is_fenced_from_sleeves(tmp_path, monkeypatch):
    import btcbot.bets as bets_mod
    monkeypatch.setattr(trader_mod, "STATE", tmp_path)
    pf = trader_mod.Portfolio("paper", "kraken", "policy_momentum.json", ["BTC", "ETH"], bets=True)
    assert pf.pool is not None and set(pf.pool.coins) <= {"BTC", "ETH"}
    pf.move(15, "BTC", "BETS")
    assert pf["BTC"].ledger["usd"] == 85 and pf.pool.ledger["usd"] == 15
    assert pf["ETH"].other_sleeves_usd() == 85 + 15  # sleeves can't spend the pool's cash
    with pytest.raises(ValueError):
        pf.move(20, "BETS", "ETH")                   # more than the pool holds
    pf.move(5, "BETS", "ETH")
    assert pf.pool.ledger["usd"] == 10 and pf["ETH"].ledger["usd"] == 5


@pytest.mark.parametrize("side,exit_px", [(1, 101.0), (1, 99.0), (-1, 99.0), (-1, 101.0)])
def test_bet_pool_books_balance(tmp_path, monkeypatch, side, exit_px):
    """pool cash must always equal what was put in plus realized P&L (all fees included)."""
    import btcbot.bets as bets_mod
    monkeypatch.setattr(trader_mod, "STATE", tmp_path)
    pf = trader_mod.Portfolio("paper", "kraken", "policy_momentum.json", ["BTC"], bets=True)
    pool = pf.pool
    pf.move(20, "BTC", "BETS")
    fills = iter([{"price": 100.0, "size": 0.1, "fee": 0.04, "liquidity": "maker"},
                  {"price": exit_px, "size": 0.1, "fee": 0.04, "liquidity": "maker"}])
    br = pool.brokers["BTC"]
    br.buy_maker = br.short_maker = br.sell_maker = br.cover_maker = lambda *a, **k: next(fills)
    br.quote = lambda: (100.0, 100.0)
    pool._open("BTC", side, {"close": 100.0, "sigma_h": 0.02, "H": 24, "p_up": 0.7}, 100.0, 100.0)
    bet = pool.s["bets"]["BTC"]
    pool._close("BTC", dict(bet), "time", {"close": exit_px})
    L = pool.ledger
    assert "BTC" not in pool.s["bets"]
    assert L["usd"] == pytest.approx(L["deposited"] - L["withdrawn"] + L["realized"])
    gross = side * 0.1 * (exit_px - 100.0)
    assert L["realized"] < gross  # fees (and margin on shorts) always cost something


def _funding_portfolio(tmp_path, monkeypatch, btc_qty=0.8, btc_cash=0.0):
    """Paper portfolio (BTC + ETH sleeves, bet pool) with fixed $100 prices and a sale stub instead of orders."""
    monkeypatch.setattr(trader_mod, "STATE", tmp_path)
    pf = trader_mod.Portfolio("paper", "kraken", "policy_momentum.json", ["BTC", "ETH"], bets=True)
    sales = []
    for t in pf.traders.values():
        t.broker.quote = lambda: (100.0, 100.1)

        def sell_maker(qty, deadline_s=0, ref_px=None, fallback_taker=True, t=t):
            sales.append({"coin": t.coin, "qty": qty, "ref_px": ref_px, "deadline_s": deadline_s, "taker": fallback_taker})
            return {"price": ref_px, "size": qty, "fee": qty * ref_px * 0.004, "liquidity": "maker"}
        t.broker.sell_maker = sell_maker
        t.broker.s["paper"]["usd"] = 1000.0  # paper cash moves need a simulated balance behind them
    for b in pf.pool.brokers.values():
        b.quote = lambda: (100.0, 100.1)
    btc = pf.traders["BTC"]
    btc.ledger.update(usd=btc_cash, qty=btc_qty, deposited=100.0)
    btc.s["position"] = {"entry_time": "2026-10-20T00:00:00+00:00", "entry_px": 100.0, "qty": btc_qty, "frac": 1.0,
                         "manual": False, "stop_px": 0.0}
    return pf, sales


def test_bet_sells_another_coin_for_its_stake(tmp_path, monkeypatch):
    pf, sales = _funding_portfolio(tmp_path, monkeypatch)
    before = sum(t.equity(100.0) for t in pf.traders.values()) + pf.pool.equity({})
    got = pf.fund_bet("ETH", 8.0)
    assert len(sales) == 1 and sales[0]["coin"] == "BTC"
    assert sales[0]["ref_px"] == 100.1 and sales[0]["deadline_s"] == 600 and sales[0]["taker"] is False  # maker only
    assert 7.99 <= got <= 8.1 and pf.pool.ledger["usd"] == pytest.approx(got)
    after = sum(t.equity(100.1) for t in pf.traders.values()) + pf.pool.equity({})
    assert after == pytest.approx(before, abs=0.2)  # money moved, it wasn't created (only the small fee went)
    btc = pf.traders["BTC"]
    assert btc.s["position"]["qty"] == pytest.approx(btc.ledger["qty"])  # the trend position shrank with the sale
    journal = (tmp_path / f"{btc.mode}_journal.jsonl").read_text()
    assert '"sell_for_bet"' in journal and '"bet": "ETH"' in journal


def test_bet_uses_idle_cash_before_selling(tmp_path, monkeypatch):
    pf, sales = _funding_portfolio(tmp_path, monkeypatch, btc_cash=20.0)
    assert pf.fund_bet("ETH", 8.0) == pytest.approx(8.0, abs=0.01)
    assert sales == [] and pf.traders["BTC"].ledger["usd"] == pytest.approx(12.0, abs=0.01)


def test_bet_never_sells_the_coin_it_bets_on(tmp_path, monkeypatch):
    pf, sales = _funding_portfolio(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="only coin to sell is btc"):
        pf.fund_bet("BTC", 8.0)
    assert sales == [] and pf.pool.ledger["usd"] == 0


def test_bet_funding_stops_at_the_lifetime_cap(tmp_path, monkeypatch):
    pf, sales = _funding_portfolio(tmp_path, monkeypatch)
    pf.pool.ledger.update(deposited=20.0, usd=0.0)  # already given $20 of a ~$80 portfolio (cap 25% = ~$20)
    with pytest.raises(ValueError, match="cap"):
        pf.fund_bet("ETH", 8.0)
    assert sales == []


def test_open_asks_the_funder_only_when_allowed(tmp_path, monkeypatch):
    pf, sales = _funding_portfolio(tmp_path, monkeypatch)
    pool, calls = pf.pool, []
    pool.funder = lambda coin, want: calls.append((coin, want)) or 0.0
    sig = {"close": 100.0, "bar": "2026-10-20 20:00:00+00:00", "H": 24, "p_up": 0.7, "sigma_h": 0.02}
    pool.s["autofund"] = False
    pool._open("ETH", 1, sig, 100.0, 100.1)
    assert calls == []
    pool.s["autofund"] = True
    pool._open("ETH", 1, sig, 100.0, 100.1)
    assert calls and calls[0][0] == "ETH" and calls[0][1] > 0


def test_small_shortfall_still_sells_at_least_the_exchange_minimum(tmp_path, monkeypatch):
    pf, sales = _funding_portfolio(tmp_path, monkeypatch)
    pf.traders["BTC"].broker.min_order_usd = 5.5
    pf.pool.ledger.update(usd=4.6, deposited=4.6)
    pf.fund_bet("ETH", 1.0)  # the pool is $1 short, but Kraken won't take a $1 sale
    assert sales and sales[0]["qty"] * sales[0]["ref_px"] >= 5.5 * 1.1
