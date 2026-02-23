"""CLI reporting – dashboard and performance export."""

from __future__ import annotations

import json
import logging
import time
from typing import Dict

from kucoin_bot.services.risk_manager import RiskManager

logger = logging.getLogger(__name__)

_DASH_WIDTH = 70


def print_dashboard(
    risk_mgr: RiskManager,
    strategies_active: Dict[str, str] | None = None,
    cycle: int = 0,
) -> str:
    """Print a text-based dashboard of current state. Returns the text."""
    summary = risk_mgr.get_risk_summary()
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    header = f"  KuCoin Trading Bot – Dashboard  [cycle {cycle}]  {ts}"
    lines = [
        "=" * _DASH_WIDTH,
        header,
        "=" * _DASH_WIDTH,
        f"  Equity:     ${summary['equity']:>12,.2f}    Peak:     ${summary['peak_equity']:>12,.2f}",
        f"  Daily PnL:  ${summary['daily_pnl']:>+12,.2f}    Drawdown: {summary['drawdown_pct']:>8.2f}%",
        f"  Exposure:   ${summary['total_exposure']:>12,.2f}    Positions: {summary['positions']}   "
        f"Circuit: {'⚠ ACTIVE' if summary['circuit_breaker'] else '✓ OK'}",
    ]

    # Per-position details with unrealized PnL
    open_positions = {sym: pos for sym, pos in risk_mgr.positions.items() if pos.size > 0}
    if open_positions:
        lines.append("-" * _DASH_WIDTH)
        lines.append("  Open Positions:")
        for sym, pos in open_positions.items():
            cur = pos.current_price if pos.current_price > 0 else pos.entry_price
            upnl = (cur - pos.entry_price) * pos.size if pos.side == "long" else (pos.entry_price - cur) * pos.size
            lines.append(
                f"    {sym:<14} {pos.side.upper():<5}  sz={pos.size:.6f}"
                f"  entry={pos.entry_price:.4f}  cur={cur:.4f}"
                f"  uPnL={upnl:+.2f} USDT"
            )

    if strategies_active:
        lines.append("-" * _DASH_WIDTH)
        lines.append("  Active Strategies:")
        for sym, strat in strategies_active.items():
            lines.append(f"    {sym}: {strat}")
    lines.append("=" * _DASH_WIDTH)
    text = "\n".join(lines)
    print(text)
    return text


def export_performance(risk_mgr: RiskManager, filepath: str = "performance.json") -> None:
    """Export performance metrics to JSON."""
    summary = risk_mgr.get_risk_summary()
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Performance exported to %s", filepath)
