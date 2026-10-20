import time

import pytest

from btcbot.tui import big_number, bucket_bars, render_chart

NOW = time.time()
STATUS = {"policy": {"model": "momentum", "horizon": 24, "sma_days": 50},
          "state": {"signal": {"pred": 1.0, "sigma_h": 0.02, "horizon": 24},
                    "position": {"entry_px": 100.5, "stop_px": 0.0}}}


def candles(n, secs=60):
    return {"kind": "candles", "secs": secs,
            "bars": [[int(NOW - (n - i) * secs), 100 + i % 3, 101 + i % 3, 99 + i % 3, 100.5 + i % 3, 1] for i in range(n)]}


@pytest.mark.parametrize("w,h", [(120, 30), (60, 12), (25, 7), (8, 3)])
@pytest.mark.parametrize("view,data", [("1h", candles(60)), ("1w", candles(168, 3600)),
                                       ("10m", {"kind": "ticks", "points": [[NOW - 600 + i, 100 + (i % 7) * 0.1] for i in range(600)]}),
                                       ("1h", {"kind": "candles", "secs": 60, "bars": []}), ("1h", None)])
def test_chart_renders_any_size_and_data(w, h, view, data):
    t = render_chart(w, h, view, data, [[NOW, 100.2]], 100.2, STATUS, [{"t": NOW - 30, "price": 100.1, "side": "buy"}])
    lines = t.plain.split("\n")
    assert len(lines) == max(h, 6) and all(len(l) == max(w, 20) for l in lines)


def test_flat_price_does_not_divide_by_zero():
    flat = {"kind": "candles", "secs": 60, "bars": [[int(NOW - 60), 100, 100, 100, 100, 0]]}
    render_chart(80, 20, "1h", flat, [], 100.0, {"policy": None, "state": {}}, [])


def test_bucketing_keeps_extremes():
    bars = candles(300)["bars"]
    b = bucket_bars(bars, 50)
    assert len(b) <= 50 and max(x[2] for x in b) == max(x[2] for x in bars) and min(x[3] for x in b) == min(x[3] for x in bars)


def test_big_number_is_three_rows():
    assert len(big_number("82,551.9", "").plain.split("\n")) == 3


def test_every_bot_event_renders_in_the_console():
    """Regression: a local named `px` once shadowed the px() formatter and crashed on bet events."""
    from types import SimpleNamespace
    from btcbot.tui import Desk
    lines = []
    fake = SimpleNamespace(log_line=lambda text, style=None, t=None: lines.append(text), fills=[],
                           status={"policy": {"model": "momentum"}})
    ts = "2026-10-20T20:00:00+00:00"
    fill = {"price": 2431.77, "size": 0.003, "fee": 0.03}
    events = [
        {"event": "tick", "bar": "2026-10-20T19:00:00+00:00", "close": 2434.8, "pred": 1.0, "coin": "ETH"},
        {"event": "order", "stage": "placing", "side": "buy", "size": 0.003, "price": 2431.77, "coin": "ETH"},
        {"event": "order", "stage": "placing", "side": "market_sell", "size": 0.003, "coin": "ETH"},
        {"event": "order", "stage": "done", "order_id": "X", "status": "closed"},
        {"event": "order", "stage": "UNSETTLED", "order_id": "X"},
        {"event": "startup", "coin": "BTC", "ledger": {"usd": 0.1, "qty": 0.001}},
        {"event": "entry", "coin": "BTC", "fill": fill},
        {"event": "exit", "coin": "BTC", "fill": fill, "pnl_pct": -0.01},
        {"event": "manual_sell", "coin": "BTC", "fill": fill},
        {"event": "bet_open", "coin": "ETH", "side": "long", "fill": fill, "p_up": 0.61, "H": 24},
        {"event": "bet_close", "coin": "ETH", "side": "long", "fill": fill, "pnl": -0.05, "reason": "time"},
        {"event": "pool_startup", "coins": ["BTC", "ETH"], "ledger": {"usd": 14.87}, "shorts": False},
        {"event": "pool_skip", "note": "too small"}, {"event": "skip", "note": "fees"},
        {"event": "error", "note": "boom"}, {"event": "halted", "note": "review"},
    ]
    for e in events:
        Desk.handle_bot_event(fake, {**e, "ts": ts})
    assert len(lines) >= 12 and len(fake.fills) == 5


def test_chart_ranges_coins_and_commands_are_clickable(monkeypatch):
    """Everything the number keys used to do is a mouse click; money commands only fill the prompt."""
    import asyncio
    import btcbot.tui as tui

    monkeypatch.setattr(tui, "BASE", "http://127.0.0.1:9")  # no bot: tests must not touch a running one

    def where(widget, label):  # (x, y) of `label` inside a Static's rendered text, after its 1-cell padding
        for y, line in enumerate(str(widget.render()).split("\n")):
            if label in line:
                return line.index(label) + 2, y
        raise AssertionError(f"{label!r} not shown")

    async def go():
        desk = tui.Desk()
        async with desk.run_test(size=(140, 40)) as pilot:
            desk.coins_cache = ["BTC", "ETH", "SOL", "XRP"]
            desk.paint_coins()
            await pilot.pause()
            keys = desk.query_one("#keys")
            await pilot.click("#keys", offset=where(keys, " 1d "))
            await pilot.pause()
            assert desk.view == "1d"
            await pilot.click("#coins", offset=where(desk.query_one("#coins"), " sol "))
            await pilot.pause()
            assert desk.coin == "SOL"
            await pilot.click("#actions", offset=where(desk.query_one("#actions"), " sell "))
            await pilot.pause()
            assert desk.query_one("#prompt").value == "sell " and desk.history == []  # filled, not sent

    asyncio.run(go())
