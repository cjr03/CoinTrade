from btcbot.kraken import sign


def test_sign_matches_kraken_documented_example():
    # example vector from docs.kraken.com "Authentication" (public test secret, not a real key)
    secret = "kQH5HW/8p1uGOVjbgWA7FunAmGO8lsSUXNsu3eow76sz84Q18fWxnyRzBHCd3pd5nE9qa99HAZtuZuj6F1huXg=="
    data = {"nonce": 1616492376594, "ordertype": "limit", "pair": "XBTUSD", "price": 37500,
            "type": "buy", "volume": 1.25}
    assert sign("/0/private/AddOrder", data, secret) == \
        "4/dpxb3iT4tp/ZCVEwSnEsLxx0bqyhLpdfOpc6fn7OR8+UClSV5n9E6aSS8MPtnRfp32bAb0nmbRn6H8ndwLUQ=="
