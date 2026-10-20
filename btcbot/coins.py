"""Tradable coins and their identifiers on each venue (the one place to add a coin)."""

COINS = {
    "BTC": {"name": "bitcoin", "product": "BTC-USD", "kraken_pair": "XBTUSD", "kraken_key": "XXBTZUSD",
            "kraken_asset": "XXBT", "ws": "BTC/USD"},
    "ETH": {"name": "ethereum", "product": "ETH-USD", "kraken_pair": "ETHUSD", "kraken_key": "XETHZUSD",
            "kraken_asset": "XETH", "ws": "ETH/USD"},
    "SOL": {"name": "solana", "product": "SOL-USD", "kraken_pair": "SOLUSD", "kraken_key": "SOLUSD",
            "kraken_asset": "SOL", "ws": "SOL/USD"},
    "XRP": {"name": "xrp", "product": "XRP-USD", "kraken_pair": "XRPUSD", "kraken_key": "XXRPZUSD",
            "kraken_asset": "XXRP", "ws": "XRP/USD"},
}
ALIASES = {"bitcoin": "BTC", "xbt": "BTC", "ethereum": "ETH", "ether": "ETH", "etherium": "ETH", "solana": "SOL",
           "ripple": "XRP"}


def coin_symbol(word):
    """'eth', 'Ethereum', 'ETH-USD' -> 'ETH'; None if the word is not a coin."""
    w = word.strip().lower().replace("-usd", "")
    w = ALIASES.get(w, w).upper()
    return w if w in COINS else None
