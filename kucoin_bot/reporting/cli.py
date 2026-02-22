"""CLI reporting – dashboard and performance export."""

from __future__ import annotations

import datetime
import json
import logging
from typing import Dict, List

from kucoin_bot.services.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# Fallback epoch for synthetic bar-index timestamps (bar_index * 3600 seconds)
_SYNTHETIC_EPOCH = datetime.datetime(2000, 1, 1)


def print_dashboard(risk_mgr: RiskManager, strategies_active: Dict[str, str] | None = None) -> str:
    """Print a text-based dashboard of current state. Returns the text."""
    summary = risk_mgr.get_risk_summary()
    lines = [
        "=" * 60,
        "  KuCoin Trading Bot – Dashboard",
        "=" * 60,
        f"  Equity:        ${summary['equity']:,.2f}",
        f"  Peak Equity:   ${summary['peak_equity']:,.2f}",
        f"  Daily PnL:     ${summary['daily_pnl']:,.2f}",
        f"  Drawdown:      {summary['drawdown_pct']:.2f}%",
        f"  Exposure:      ${summary['total_exposure']:,.2f}",
        f"  Positions:     {summary['positions']}",
        f"  Circuit Brk:   {'ACTIVE' if summary['circuit_breaker'] else 'OK'}",
    ]
    if strategies_active:
        lines.append("-" * 60)
        lines.append("  Active Strategies:")
        for sym, strat in strategies_active.items():
            lines.append(f"    {sym}: {strat}")
    lines.append("=" * 60)
    text = "\n".join(lines)
    print(text)
    return text


def export_performance(risk_mgr: RiskManager, filepath: str = "performance.json") -> None:
    """Export performance metrics to JSON."""
    summary = risk_mgr.get_risk_summary()
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Performance exported to %s", filepath)


def export_backtest_report(
    result: "BacktestResult",  # type: ignore[name-defined]
    filepath: str = "backtest_report.json",
) -> dict:
    """Export a full backtest performance report with daily/weekly aggregation.

    Args:
        result: BacktestResult from BacktestEngine.run() or walk_forward().
        filepath: Output JSON path.

    Returns:
        The report dict (also written to filepath).
    """
    from kucoin_bot.backtest.engine import BacktestResult, BacktestTrade

    closed_trades: List[BacktestTrade] = [t for t in result.trades if t.side == "exit"]

    # Daily PnL aggregation (timestamp is bar index * 3600 or unix seconds)
    daily: Dict[str, float] = {}
    weekly: Dict[str, float] = {}
    for t in closed_trades:
        try:
            dt = datetime.datetime.utcfromtimestamp(t.timestamp)
        except (OSError, OverflowError, ValueError):
            # Synthetic timestamps (bar index * 3600 may be too small for utcfromtimestamp)
            dt = _SYNTHETIC_EPOCH + datetime.timedelta(seconds=t.timestamp)
        day_key = dt.strftime("%Y-%m-%d")
        week_key = dt.strftime("%Y-W%W")
        daily[day_key] = daily.get(day_key, 0.0) + t.pnl
        weekly[week_key] = weekly.get(week_key, 0.0) + t.pnl

    report = {
        "summary": {
            "initial_equity": result.initial_equity,
            "final_equity": result.final_equity,
            "total_return_pct": result.total_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "total_fees": result.total_fees,
            "expectancy": result.expectancy,
            "turnover": result.turnover,
        },
        "daily_pnl": daily,
        "weekly_pnl": weekly,
    }

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Backtest report exported to %s", filepath)
    return report
