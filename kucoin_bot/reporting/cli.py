"""CLI reporting – dashboard and performance export."""

from __future__ import annotations

import json
import logging
from typing import Dict

from kucoin_bot.services.risk_manager import RiskManager

logger = logging.getLogger(__name__)


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
