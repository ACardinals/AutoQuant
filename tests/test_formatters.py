from market_monitor.signals.formatters import format_signal_table, signal_with_metadata
from market_monitor.signals.models import StrategySignal


def test_signal_with_metadata_adds_name_and_market():
    signal = StrategySignal(symbol="000001.SZ", action="hold", confidence=0.2, reasons=["未满足条件"])

    row = signal_with_metadata(signal, {"000001.SZ": {"name": "平安银行", "market": "A股"}})

    assert row["name"] == "平安银行"
    assert row["market"] == "A股"
    assert row["signal"] == "hold"


def test_format_signal_table_includes_core_columns():
    table = format_signal_table([
        {
            "symbol": "000001.SZ",
            "name": "平安银行",
            "market": "A股",
            "signal": "buy_candidate",
            "confidence": 0.9,
            "reasons": ["收盘价突破20周期高点"],
            "risk": {"stop_loss": 9.6, "take_profit": 10.8},
        }
    ])

    assert "symbol" in table
    assert "000001.SZ" in table
    assert "平安银行" in table
    assert "buy_candidate" in table
