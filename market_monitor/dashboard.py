from __future__ import annotations

from datetime import date
from pathlib import Path
import os

from market_monitor.backtest.engine import LongOnlyBacktester
from market_monitor.data.akshare_a_share import download_a_share_watchlist, fetch_a_share_spot_universe
from market_monitor.data.watchlist import WatchlistItem, load_watchlist, load_watchlist_candles, write_watchlist
from market_monitor.evaluation import compare_strategies, compare_watchlist
from market_monitor.ml.models import available_models
from market_monitor.ml.prediction import rank_watchlist_ml
from market_monitor.ml.validation import evaluate_time_series_model
from market_monitor.models import Candle
from market_monitor.signals.formatters import signal_with_metadata
from market_monitor.signals.screener import screen_watchlist
from market_monitor.strategies.registry import available_strategies, create_strategy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_START_DATE = "20240101"
ALL_INDUSTRIES_LABEL = "全部行业"
STRATEGY_LABELS = {
    "bollinger_reversion": "布林均值回归",
    "breakout": "放量突破",
    "ma_trend": "均线趋势（推荐）",
    "macd_trend": "MACD趋势增强",
    "rsi_rebound": "RSI反弹",
    "volume_pullback": "缩量回调",
}
STRATEGY_DESCRIPTIONS = {
    "bollinger_reversion": "适合观察超跌修复：寻找价格从布林下轨附近回升、RSI同步改善的标的。",
    "ma_trend": "适合先从趋势跟随开始：寻找短中长期均线多头排列、价格处于强势区间的标的。",
    "macd_trend": "适合确认趋势动能：关注MACD转强、价格站上趋势均线且成交量健康的标的。",
    "breakout": "适合寻找强势股：关注价格突破近期高点并伴随成交量放大的标的，波动通常更大。",
    "rsi_rebound": "适合观察低位修复：寻找 RSI 从弱势区域回升、价格开始企稳的标的。",
    "volume_pullback": "适合趋势中的低吸观察：寻找上升趋势里缩量回调后重新走强的标的。",
}
STRATEGY_GUIDANCE = "不知道怎么选时，建议先用「均线趋势（推荐）」；想确认趋势动能用「MACD趋势增强」；想找强势股用「放量突破」；想做低位修复观察用「RSI反弹」或「布林均值回归」；想找趋势回调机会用「缩量回调」。"
SIGNAL_LABELS = {
    "全部信号": "all",
    "买入候选": "buy_candidate",
    "观望": "hold",
}
DEFAULT_A_SHARE_SYMBOLS = [
    ("000001.SZ", "平安银行", "银行"),
    ("600519.SH", "贵州茅台", "白酒消费"),
    ("300750.SZ", "宁德时代", "新能源电池"),
    ("600036.SH", "招商银行", "银行"),
    ("601318.SH", "中国平安", "保险"),
    ("600030.SH", "中信证券", "券商"),
    ("601899.SH", "紫金矿业", "有色金属"),
    ("600276.SH", "恒瑞医药", "医药"),
    ("000858.SZ", "五粮液", "白酒消费"),
    ("002594.SZ", "比亚迪", "新能源车"),
    ("601088.SH", "中国神华", "能源"),
    ("688981.SH", "中芯国际", "半导体"),
]


def strategy_options() -> list[str]:
    keys = available_strategies()
    return [STRATEGY_LABELS.get(key, key) for key in keys]


def strategy_key_from_label(label: str) -> str:
    for key, value in STRATEGY_LABELS.items():
        if value == label:
            return key
    return label


def strategy_description(strategy_key: str) -> str:
    return STRATEGY_DESCRIPTIONS.get(strategy_key, "暂无策略说明。")


def _signal_value_from_label(label: str) -> str:
    return SIGNAL_LABELS.get(label, "all")


def industry_options(rows: list[dict]) -> list[str]:
    industries = sorted({row.get("market", "") for row in rows if row.get("market")})
    return [ALL_INDUSTRIES_LABEL] + industries


def industry_value_from_label(label: str) -> str:
    return "all" if label == ALL_INDUSTRIES_LABEL else label


def load_available_watchlist_candles(watchlist_path: str | Path) -> tuple[list[WatchlistItem], dict[str, list[Candle]], list[WatchlistItem]]:
    items = load_watchlist(watchlist_path)
    available_items = []
    missing_items = []
    candles_by_symbol = {}
    from market_monitor.data.csv_loader import load_candles_from_csv

    for item in items:
        if not item.csv_path.exists():
            missing_items.append(item)
            continue
        available_items.append(item)
        candles_by_symbol[item.symbol] = load_candles_from_csv(item.csv_path, symbol=item.symbol)
    return available_items, candles_by_symbol, missing_items


def screen_rows(watchlist_path: str | Path, strategy_name: str) -> list[dict]:
    items, candles_by_symbol, _ = load_available_watchlist_candles(watchlist_path)
    metadata = {item.symbol: {"name": item.name, "market": item.market} for item in items}
    strategy = create_strategy(strategy_name)
    signals = screen_watchlist(candles_by_symbol, strategy)
    return [flatten_screen_row(signal_with_metadata(signal, metadata)) for signal in signals]


def flatten_screen_row(row: dict) -> dict:
    risk = row.get("risk", {})
    reasons = row.get("reasons", [])
    return {
        "symbol": row.get("symbol", ""),
        "name": row.get("name", ""),
        "market": row.get("market", ""),
        "signal": row.get("signal", ""),
        "confidence": row.get("confidence", 0.0),
        "stop_loss": risk.get("stop_loss"),
        "take_profit": risk.get("take_profit"),
        "max_position_pct": risk.get("max_position_pct"),
        "reason": reasons[0] if reasons else "",
        "reasons": reasons,
    }


def watchlist_status(watchlist_path: str | Path) -> dict:
    items, _, missing_items = load_available_watchlist_candles(watchlist_path)
    return {
        "total": len(items) + len(missing_items),
        "downloaded": len(items),
        "missing": len(missing_items),
    }


def filter_watchlist_rows(
    rows: list[dict],
    search_text: str,
    market_filter: str,
    signal_filter: str,
    min_confidence: float,
) -> list[dict]:
    search = search_text.strip().lower()
    return [
        row
        for row in rows
        if _matches_search(row, search)
        and (market_filter == "all" or row.get("market") == market_filter)
        and (signal_filter == "all" or row.get("signal") == signal_filter)
        and row.get("confidence", 0.0) >= min_confidence
    ]


def filter_screen_rows(rows: list[dict], signal_filter: str, min_confidence: float) -> list[dict]:
    return filter_watchlist_rows(rows, "", "all", signal_filter, min_confidence)


def _matches_search(row: dict, search: str) -> bool:
    if not search:
        return True
    return search in str(row.get("symbol", "")).lower() or search in str(row.get("name", "")).lower()


def write_filtered_watchlist(source_watchlist: str | Path, output_watchlist: str | Path, symbols: list[str]) -> dict:
    symbol_set = {symbol.upper() for symbol in symbols}
    items = [item for item in load_watchlist(resolve_dashboard_path(source_watchlist)) if item.symbol in symbol_set]
    return write_watchlist(resolve_dashboard_path(output_watchlist), items)


def download_symbols_from_watchlist(
    source_watchlist: str | Path,
    symbols: list[str],
    start_date: str,
    end_date: str,
    output_dir: str | Path,
) -> dict:
    filtered_path = Path("watchlists") / "_dashboard_filtered.csv"
    write_filtered_watchlist(source_watchlist, filtered_path, symbols)
    return refresh_watchlist_data(filtered_path, start_date, end_date, output_dir, max_symbols=None)


def backtest_symbol(
    watchlist_path: str | Path,
    symbol: str,
    strategy_name: str,
    initial_cash: float = 10_000.0,
) -> dict:
    _, candles_by_symbol, _ = load_available_watchlist_candles(watchlist_path)
    normalized_symbol = symbol.upper()
    candles = candles_by_symbol[normalized_symbol]
    strategy = create_strategy(strategy_name)
    result = LongOnlyBacktester(initial_cash=initial_cash).run(candles, strategy)
    return {
        "symbol": normalized_symbol,
        "candles": candles_to_chart_rows(candles),
        "backtest": result.as_dict(),
    }



def compare_symbol_strategies(
    watchlist_path: str | Path,
    symbol: str,
    initial_cash: float = 10_000.0,
) -> list[dict]:
    _, candles_by_symbol, _ = load_available_watchlist_candles(watchlist_path)
    return compare_strategies(candles_by_symbol[symbol.upper()], initial_cash=initial_cash)


def compare_watchlist_strategies(
    watchlist_path: str | Path,
    initial_cash: float = 10_000.0,
    top_n: int | None = 20,
) -> list[dict]:
    items, candles_by_symbol, _ = load_available_watchlist_candles(watchlist_path)
    rows = compare_watchlist(items, candles_by_symbol, initial_cash=initial_cash)
    return rows[:top_n] if top_n is not None and top_n > 0 else rows


def evaluate_symbol_ml(
    watchlist_path: str | Path,
    symbol: str,
    model_name: str,
    horizon: int,
    splits: int,
    threshold: float,
) -> dict:
    _, candles_by_symbol, _ = load_available_watchlist_candles(watchlist_path)
    return evaluate_time_series_model(candles_by_symbol[symbol.upper()], model_name, horizon, splits, threshold)


def rank_watchlist_ml_candidates(
    watchlist_path: str | Path,
    model_name: str,
    horizon: int,
    threshold: float,
    top_n: int | None = 20,
) -> list[dict]:
    items, candles_by_symbol, _ = load_available_watchlist_candles(watchlist_path)
    return rank_watchlist_ml(items, candles_by_symbol, model_name, horizon, threshold, top_n)


def candles_to_chart_rows(candles: list[Candle]) -> list[dict]:
    return [
        {
            "timestamp": candle.timestamp,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        for candle in candles
    ]


def create_candlestick_figure(candle_rows: list[dict], title: str = ""):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
    )
    figure.add_trace(
        go.Candlestick(
            x=[row["timestamp"] for row in candle_rows],
            open=[row["open"] for row in candle_rows],
            high=[row["high"] for row in candle_rows],
            low=[row["low"] for row in candle_rows],
            close=[row["close"] for row in candle_rows],
            name="K线 / Candles",
            increasing_line_color="#ef4444",
            decreasing_line_color="#22c55e",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=[row["timestamp"] for row in candle_rows],
            y=[row["volume"] for row in candle_rows],
            name="成交量 / Volume",
            marker_color="#94a3b8",
        ),
        row=2,
        col=1,
    )
    figure.update_layout(
        title=title,
        height=620,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="right",
        legend_x=1,
    )
    figure.update_yaxes(title_text="价格 / Price", row=1, col=1)
    figure.update_yaxes(title_text="成交量 / Volume", row=2, col=1)
    return figure


def ensure_default_watchlist(watchlist_path: str | Path, data_dir: str | Path) -> dict:
    watchlist = resolve_dashboard_path(watchlist_path)
    data_path = resolve_dashboard_path(data_dir)
    csv_base = Path(os.path.relpath(data_path, watchlist.parent))
    items = [
        WatchlistItem(
            symbol=symbol,
            name=name,
            market=market,
            csv_path=csv_base / f"{symbol}.csv",
        )
        for symbol, name, market in DEFAULT_A_SHARE_SYMBOLS
    ]
    return write_watchlist(watchlist, items)


def create_full_a_share_watchlist(output_path: str | Path, data_dir: str | Path, limit: int | None = None) -> dict:
    watchlist = resolve_dashboard_path(output_path)
    data_path = resolve_dashboard_path(data_dir)
    csv_base = Path(os.path.relpath(data_path, watchlist.parent))
    items = fetch_a_share_spot_universe()
    if limit is not None and limit > 0:
        items = items[:limit]
    normalized_items = [
        WatchlistItem(
            symbol=item.symbol,
            name=item.name,
            market=item.market,
            csv_path=csv_base / f"{item.symbol}.csv",
        )
        for item in items
    ]
    return write_watchlist(watchlist, normalized_items)


def refresh_watchlist_data(
    watchlist_path: str | Path,
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    max_symbols: int | None = None,
) -> dict:
    return download_a_share_watchlist(
        resolve_dashboard_path(watchlist_path),
        start_date,
        end_date,
        resolve_dashboard_path(output_dir),
        max_symbols=max_symbols,
    )


def default_end_date() -> str:
    return date.today().strftime("%Y%m%d")


def resolve_dashboard_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Market Monitor", layout="wide")
    _render_css(st)
    _render_header(st)

    config = _render_sidebar(st)
    watchlist_path = config["watchlist_path"]
    path = resolve_dashboard_path(watchlist_path)
    if not path.exists():
        st.warning(f"未找到股票池文件：{path}")
        st.info("请先在左侧创建示例股票池，或生成全A股票池，然后下载行情数据。")
        return

    try:
        rows = screen_rows(path, config["strategy_name"])
    except Exception as exc:
        st.error(f"筛选失败：{exc}")
        return

    status = watchlist_status(path)
    market_labels = industry_options(rows)
    selected_market_label = st.selectbox("行业板块", market_labels, index=0)
    market_filter = industry_value_from_label(selected_market_label)
    filtered_rows = filter_watchlist_rows(
        rows,
        config["search_text"],
        market_filter,
        config["signal_filter"],
        config["min_confidence"],
    )
    _render_summary_metrics(st, rows, filtered_rows, config["strategy_name"], status)

    tabs = st.tabs(["策略筛选", "个股K线", "回测分析", "策略对比", "股票池评分", "ML评估", "AI候选排序"])
    with tabs[0]:
        _render_screening_tab(st, filtered_rows)
        if filtered_rows:
            if st.button("下载当前筛选结果 / Download filtered rows", use_container_width=True):
                try:
                    symbols = [row["symbol"] for row in filtered_rows]
                    summary = download_symbols_from_watchlist(
                        path,
                        symbols,
                        config["start_date"],
                        config["end_date"],
                        config["data_dir"],
                    )
                    st.success(f"已下载 {summary['count']} 支筛选股票")
                except Exception as exc:
                    st.error(f"筛选结果下载失败：{exc}")

    if not filtered_rows:
        return

    selected_symbol, selected_label = _select_symbol(st, filtered_rows)
    try:
        details = backtest_symbol(path, selected_symbol, config["strategy_name"], config["initial_cash"])
    except Exception as exc:
        st.error(f"Failed to backtest {selected_symbol}: {exc}")
        return

    with tabs[1]:
        _render_symbol_detail_tab(st, selected_label, details["candles"])
    with tabs[2]:
        _render_backtest_tab(st, details["backtest"])
    with tabs[3]:
        _render_strategy_comparison_tab(st, path, selected_label, selected_symbol, config["initial_cash"])
    with tabs[4]:
        _render_watchlist_score_tab(st, path, config["initial_cash"], config["score_top_n"])
    with tabs[5]:
        _render_ml_evaluation_tab(st, path, selected_label, selected_symbol, config)
    with tabs[6]:
        _render_ai_candidate_tab(st, path, config)


def _render_css(st) -> None:
    st.markdown(
        """
        <style>
        .stApp {background: #f8fafc; color: #0f172a;}
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1520px;}
        .hero {padding: 26px 30px; border-radius: 22px; background: linear-gradient(135deg, #ffffff 0%, #eff6ff 54%, #e0f2fe 100%); color: #0f172a; margin-bottom: 22px; box-shadow: 0 14px 36px rgba(15,23,42,.08); border: 1px solid #dbeafe;}
        .hero h1 {margin: 0; font-size: 2.15rem; letter-spacing: .03em; color: #0f172a;}
        .hero p {margin: 10px 0 0 0; color: #475569; font-size: 1rem;}
        section[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e2e8f0; box-shadow: 8px 0 24px rgba(15,23,42,.04);}
        section[data-testid="stSidebar"] * {color: #0f172a;}
        div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e2e8f0; padding: 16px 18px; border-radius: 16px; box-shadow: 0 8px 24px rgba(15,23,42,.06);}
        div[data-testid="stMetric"] label {color: #64748b !important;}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {color: #0f172a;}
        .stButton > button {border-radius: 12px; border: 1px solid #bfdbfe; background: #2563eb; color: white; font-weight: 700; box-shadow: 0 6px 16px rgba(37,99,235,.20);}
        .stButton > button:hover {border-color: #2563eb; background: #1d4ed8; color: white;}
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {border-radius: 12px; border-color: #cbd5e1; background: #ffffff;}
        div[data-testid="stDataFrame"] {border-radius: 16px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 10px 24px rgba(15,23,42,.05);}
        .section-note {color: #64748b; font-size: 0.92rem; margin-top: -8px; margin-bottom: 12px;}
        h1, h2, h3 {color: #0f172a;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(st) -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>量化市场监控系统</h1>
          <p>全A股票池 · 策略筛选 · K线分析 · 回测验证</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(st) -> dict:
    with st.sidebar:
        st.header("控制台")
        watchlist_path = st.text_input("股票池文件", value="watchlists/a_share_all.csv")
        data_dir = st.text_input("行情数据目录", value="data/a_share")
        start_date = st.text_input("开始日期", value=DEFAULT_START_DATE)
        end_date = st.text_input("结束日期", value=default_end_date())
        selected_strategy_label = st.selectbox("交易策略", strategy_options(), index=strategy_options().index(STRATEGY_LABELS["ma_trend"]))
        strategy_name = strategy_key_from_label(selected_strategy_label)
        st.info(strategy_description(strategy_name))
        st.caption(STRATEGY_GUIDANCE)
        initial_cash = st.number_input("初始资金", min_value=1_000.0, value=10_000.0, step=1_000.0)
        signal_filter = st.selectbox("信号类型", ["全部信号", "买入候选", "观望"], index=0)
        search_text = st.text_input("搜索代码或名称", value="")
        min_confidence = st.slider("最低置信度", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        max_symbols = st.number_input("最多下载股票数（0为全部）", min_value=0, value=50, step=10)
        universe_limit = st.number_input("生成股票池上限（0为全部）", min_value=0, value=0, step=100)
        score_top_n = st.number_input("股票池评分Top N", min_value=1, value=20, step=5)
        selected_ml_model = st.selectbox("ML模型", available_models(), index=available_models().index("hist_gradient_boosting"))
        ml_horizon = st.number_input("ML预测周期", min_value=1, value=10, step=1)
        ml_splits = st.number_input("ML时间切分数", min_value=2, value=5, step=1)
        ml_threshold = st.number_input("ML收益阈值", value=0.0, step=0.005, format="%.4f")

        st.divider()
        if st.button("创建示例股票池", use_container_width=True):
            try:
                summary = ensure_default_watchlist(watchlist_path, data_dir)
                st.success(f"已创建 {summary['count']} 支股票")
            except Exception as exc:
                st.error(f"创建失败：{exc}")

        if st.button("生成全A股票池", use_container_width=True):
            try:
                limit = universe_limit if universe_limit > 0 else None
                summary = create_full_a_share_watchlist(watchlist_path, data_dir, limit)
                st.success(f"已创建 {summary['count']} 支股票")
            except Exception as exc:
                st.error(f"股票池获取失败：{exc}")

        if st.button("下载/刷新行情", use_container_width=True):
            try:
                limit = max_symbols if max_symbols > 0 else None
                summary = refresh_watchlist_data(watchlist_path, start_date, end_date, data_dir, limit)
                st.success(f"已下载 {summary['count']} 支股票")
            except Exception as exc:
                st.error(f"下载失败：{exc}")

    return {
        "watchlist_path": watchlist_path,
        "data_dir": data_dir,
        "start_date": start_date,
        "end_date": end_date,
        "strategy_name": strategy_name,
        "initial_cash": initial_cash,
        "signal_filter": _signal_value_from_label(signal_filter),
        "search_text": search_text,
        "min_confidence": min_confidence,
        "score_top_n": int(score_top_n),
        "ml_model": selected_ml_model,
        "ml_horizon": int(ml_horizon),
        "ml_splits": int(ml_splits),
        "ml_threshold": ml_threshold,
    }


def _render_summary_metrics(st, rows: list[dict], filtered_rows: list[dict], strategy_name: str, status: dict | None = None) -> None:
    buy_candidates = [row for row in filtered_rows if row["signal"] == "buy_candidate"]
    average_confidence = sum(row["confidence"] for row in filtered_rows) / len(filtered_rows) if filtered_rows else 0.0
    status = status or {"total": len(rows), "downloaded": len(rows), "missing": 0}
    metric_cols = st.columns(6)
    metric_cols[0].metric("股票池总数", status["total"])
    metric_cols[1].metric("已下载", status["downloaded"])
    metric_cols[2].metric("未下载", status["missing"])
    metric_cols[3].metric("当前显示", len(filtered_rows))
    metric_cols[4].metric("买入候选", len(buy_candidates))
    metric_cols[5].metric("平均置信度", f"{average_confidence:.2f}")
    st.caption(f"当前策略：{STRATEGY_LABELS.get(strategy_name, strategy_name)}")


def _render_screening_tab(st, rows: list[dict]) -> None:
    st.subheader("策略筛选结果")
    st.markdown('<div class="section-note">结果按信号优先级和置信度排序，可通过左侧条件过滤。</div>', unsafe_allow_html=True)
    display_columns = [
        "symbol",
        "name",
        "market",
        "signal",
        "confidence",
        "stop_loss",
        "take_profit",
        "max_position_pct",
        "reason",
    ]
    column_names = {
        "symbol": "代码",
        "name": "名称",
        "market": "行业",
        "signal": "信号",
        "confidence": "置信度",
        "stop_loss": "止损位",
        "take_profit": "止盈位",
        "max_position_pct": "建议仓位",
        "reason": "核心理由",
    }
    display_rows = [{column_names[key]: row.get(key) for key in display_columns} for row in rows]
    st.dataframe(display_rows, use_container_width=True, height=560)


def _select_symbol(st, rows: list[dict]) -> tuple[str, str]:
    symbol_labels = {f"{row['symbol']} {row.get('name', '')}".strip(): row["symbol"] for row in rows}
    selected_label = st.selectbox("选择标的 / Select Symbol", list(symbol_labels), key="selected_symbol")
    return symbol_labels[selected_label], selected_label


def _render_symbol_detail_tab(st, selected_label: str, candle_rows: list[dict]) -> None:
    st.subheader(f"{selected_label} K线 / Candlestick")
    if not candle_rows:
        st.info("No candle data available.")
        return
    try:
        st.plotly_chart(create_candlestick_figure(candle_rows, selected_label), use_container_width=True)
    except Exception as exc:
        st.warning(f"Candlestick chart unavailable, using fallback line chart: {exc}")
        st.line_chart(candle_rows, x="timestamp", y="close")
        st.bar_chart(candle_rows, x="timestamp", y="volume")
    st.subheader("最近K线 / Recent Candles")
    st.dataframe(candle_rows[-30:], use_container_width=True)


def _render_backtest_tab(st, backtest: dict) -> None:
    st.subheader("回测指标 / Backtest Metrics")
    metrics = backtest.get("metrics", {})
    backtest_cols = st.columns(5)
    backtest_cols[0].metric("Total Return", f"{backtest.get('total_return_pct', 0):.2f}%")
    backtest_cols[1].metric("Max Drawdown", f"{backtest.get('max_drawdown_pct', 0):.2f}%")
    backtest_cols[2].metric("Trades", backtest.get("trades", 0))
    backtest_cols[3].metric("Win Rate", _format_pct(metrics.get("win_rate_pct")))
    backtest_cols[4].metric("Annualized", _format_pct(metrics.get("annualized_return_pct")))

    st.subheader("权益曲线 / Equity Curve")
    equity_curve = backtest.get("equity_curve", [])
    if equity_curve:
        st.line_chart(equity_curve, x="timestamp", y="equity")
    else:
        st.info("No equity curve available.")

    st.subheader("交易记录 / Trade Records")
    trade_records = backtest.get("trade_records", [])
    if trade_records:
        st.dataframe(trade_records, use_container_width=True)
    else:
        st.info("No trades recorded.")


def _render_strategy_comparison_tab(st, watchlist_path: str | Path, selected_label: str, selected_symbol: str, initial_cash: float) -> None:
    st.subheader(f"{selected_label} 策略对比")
    st.markdown('<div class="section-note">对同一标的运行全部策略，并按综合score排序。</div>', unsafe_allow_html=True)
    try:
        with st.spinner("正在比较策略..."):
            rows = compare_symbol_strategies(watchlist_path, selected_symbol, initial_cash)
        st.dataframe(rows, use_container_width=True, height=420)
    except Exception as exc:
        st.error(f"策略对比失败：{exc}")


def _render_watchlist_score_tab(st, watchlist_path: str | Path, initial_cash: float, top_n: int) -> None:
    st.subheader("股票池策略评分")
    st.markdown('<div class="section-note">对已下载行情的股票池运行全部策略，展示score最高的标的/策略组合。</div>', unsafe_allow_html=True)
    try:
        with st.spinner("正在计算股票池评分..."):
            rows = compare_watchlist_strategies(watchlist_path, initial_cash, top_n)
        st.dataframe(rows, use_container_width=True, height=560)
    except Exception as exc:
        st.error(f"股票池评分失败：{exc}")


def _render_ml_evaluation_tab(st, watchlist_path: str | Path, selected_label: str, selected_symbol: str, config: dict) -> None:
    st.subheader(f"{selected_label} ML时间序列评估")
    st.markdown('<div class="section-note">使用TimeSeriesSplit评估监督学习baseline，避免随机K折造成时间泄漏。</div>', unsafe_allow_html=True)
    st.caption("该结果用于研究特征有效性，不构成交易建议。")
    try:
        with st.spinner("正在训练并验证ML模型..."):
            result = evaluate_symbol_ml(
                watchlist_path,
                selected_symbol,
                config["ml_model"],
                config["ml_horizon"],
                config["ml_splits"],
                config["ml_threshold"],
            )
        metrics = result.get("metrics", {})
        metric_cols = st.columns(5)
        metric_cols[0].metric("Accuracy", _format_metric(metrics.get("accuracy")))
        metric_cols[1].metric("Precision", _format_metric(metrics.get("precision")))
        metric_cols[2].metric("Recall", _format_metric(metrics.get("recall")))
        metric_cols[3].metric("ROC AUC", _format_metric(metrics.get("roc_auc")))
        metric_cols[4].metric("样本数", result.get("samples", 0))
        st.subheader("分折结果 / Fold Results")
        st.dataframe(result.get("folds", []), use_container_width=True, height=360)
        with st.expander("特征列 / Feature Columns"):
            st.write(result.get("feature_columns", []))
    except Exception as exc:
        st.error(f"ML评估失败：{exc}")


def _render_ai_candidate_tab(st, watchlist_path: str | Path, config: dict) -> None:
    st.subheader("AI候选排序")
    st.markdown('<div class="section-note">对股票池逐个训练轻量ML baseline，并按最新上涨概率排序。</div>', unsafe_allow_html=True)
    st.caption("该排序用于研究优先级，不构成交易建议。")
    try:
        with st.spinner("正在计算AI候选排序..."):
            rows = rank_watchlist_ml_candidates(
                watchlist_path,
                config["ml_model"],
                config["ml_horizon"],
                config["ml_threshold"],
                config["score_top_n"],
            )
        if rows:
            st.dataframe(rows, use_container_width=True, height=560)
        else:
            st.info("没有可用候选。请确认行情数据充足，或调小ML预测周期。")
    except Exception as exc:
        st.error(f"AI候选排序失败：{exc}")


    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def _format_metric(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
