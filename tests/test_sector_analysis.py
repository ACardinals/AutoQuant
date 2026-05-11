from market_monitor.sector_analysis import format_sector_candidates_table, format_sector_table, score_sectors, top_symbols_by_sector


def test_score_sectors_aggregates_strategy_and_ai_rows():
    strategy_rows = [
        {"market": "银行", "symbol": "A", "score": 10, "total_return_pct": 2, "max_drawdown_pct": 1},
        {"market": "银行", "symbol": "B", "score": 6, "total_return_pct": 1, "max_drawdown_pct": 2},
        {"market": "半导体", "symbol": "C", "score": 3, "total_return_pct": -1, "max_drawdown_pct": 4},
    ]
    ai_rows = [
        {"market": "银行", "symbol": "A", "probability": 0.7},
        {"market": "半导体", "symbol": "C", "probability": 0.45},
    ]

    rows = score_sectors(strategy_rows, ai_rows)

    assert rows[0]["sector"] == "银行"
    assert rows[0]["symbol_count"] == 2
    assert rows[0]["average_probability"] == 0.7


def test_top_symbols_by_sector_returns_ranked_candidates():
    rows = top_symbols_by_sector(
        [
            {"market": "银行", "symbol": "A", "name": "A", "strategy": "x", "score": 1, "total_return_pct": 1},
            {"market": "银行", "symbol": "B", "name": "B", "strategy": "x", "score": 3, "total_return_pct": 2},
        ],
        top_n=1,
    )

    assert rows == [
        {
            "sector": "银行",
            "rank": 1,
            "symbol": "B",
            "name": "B",
            "strategy": "x",
            "score": 3,
            "total_return_pct": 2,
            "max_drawdown_pct": None,
        }
    ]


def test_sector_formatters():
    sector_table = format_sector_table([{"sector": "银行", "sector_score": 1, "symbol_count": 2}])
    candidate_table = format_sector_candidates_table([{"sector": "银行", "rank": 1, "symbol": "A"}])

    assert "sector" in sector_table
    assert "银行" in candidate_table
