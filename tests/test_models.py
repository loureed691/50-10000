"""Tests for database models."""

from __future__ import annotations

import pytest
from kucoin_bot.models import init_db, Order, Trade, SignalSnapshot, PositionRecord


class TestModels:
    def test_init_db(self, tmp_path):
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        session_factory = init_db(db_url)
        with session_factory() as session:
            order = Order(
                symbol="BTC-USDT", side="buy", order_type="limit",
                quantity=0.001, price=30000.0,
            )
            session.add(order)
            session.commit()
            assert order.id is not None

    def test_signal_snapshot(self, tmp_path):
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        session_factory = init_db(db_url)
        with session_factory() as session:
            snap = SignalSnapshot(
                symbol="BTC-USDT", regime="trending_up",
                strategy_name="trend_following",
                signal_data='{"momentum": 0.5}',
                decision="entry_long", reason="test",
            )
            session.add(snap)
            session.commit()
            assert snap.id is not None

    def test_trade_record(self, tmp_path):
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        session_factory = init_db(db_url)
        with session_factory() as session:
            trade = Trade(
                symbol="BTC-USDT", side="buy",
                price=30000.0, quantity=0.001, fee=0.03,
            )
            session.add(trade)
            session.commit()
            result = session.query(Trade).first()
            assert result.price == 30000.0

    def test_position_record_insert(self, tmp_path):
        """PositionRecord can be inserted and queried."""
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        session_factory = init_db(db_url)
        with session_factory() as session:
            pos = PositionRecord(
                symbol="BTC-USDT", side="long", size=0.005,
                entry_price=30000.0, current_price=30100.0,
                leverage=1.0, account_type="trade",
            )
            session.add(pos)
            session.commit()
            result = session.query(PositionRecord).filter_by(symbol="BTC-USDT").first()
            assert result is not None
            assert result.side == "long"
            assert result.size == 0.005
            assert result.entry_price == 30000.0

    def test_position_record_upsert_and_delete(self, tmp_path):
        """PositionRecord can be updated and deleted to simulate close."""
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        session_factory = init_db(db_url)
        with session_factory() as session:
            pos = PositionRecord(
                symbol="ETH-USDT", side="short", size=0.1,
                entry_price=2000.0, current_price=1980.0,
            )
            session.add(pos)
            session.commit()

        # Update size (simulated partial close)
        with session_factory() as session:
            rec = session.query(PositionRecord).filter_by(symbol="ETH-USDT").first()
            rec.current_price = 1970.0
            session.commit()

        # Delete (position closed)
        with session_factory() as session:
            rec = session.query(PositionRecord).filter_by(symbol="ETH-USDT").first()
            session.delete(rec)
            session.commit()
            assert session.query(PositionRecord).filter_by(symbol="ETH-USDT").first() is None

    def test_position_record_unique_symbol(self, tmp_path):
        """PositionRecord enforces unique symbol constraint."""
        from sqlalchemy.exc import IntegrityError
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        session_factory = init_db(db_url)
        with session_factory() as session:
            session.add(PositionRecord(symbol="SOL-USDT", side="long", size=1.0, entry_price=100.0))
            session.commit()
        with pytest.raises(IntegrityError):
            with session_factory() as session:
                session.add(PositionRecord(symbol="SOL-USDT", side="short", size=0.5, entry_price=110.0))
                session.commit()


class TestDashboard:
    """Tests for the enhanced print_dashboard function."""

    def _make_risk_mgr(self) -> "RiskManager":
        from kucoin_bot.config import RiskConfig
        from kucoin_bot.services.risk_manager import RiskManager
        rm = RiskManager(config=RiskConfig())
        rm.update_equity(10_000.0)
        return rm

    def test_dashboard_basic(self, capsys):
        from kucoin_bot.reporting.cli import print_dashboard
        rm = self._make_risk_mgr()
        text = print_dashboard(rm, cycle=1)
        assert "cycle 1" in text
        assert "10,000.00" in text
        assert "UTC" in text

    def test_dashboard_shows_open_position_upnl(self):
        from kucoin_bot.reporting.cli import print_dashboard
        from kucoin_bot.services.risk_manager import PositionInfo
        rm = self._make_risk_mgr()
        rm.update_position("BTC-USDT", PositionInfo(
            symbol="BTC-USDT", side="long", size=0.001,
            entry_price=30000.0, current_price=31000.0,
        ))
        text = print_dashboard(rm, cycle=5)
        assert "BTC-USDT" in text
        assert "LONG" in text
        # unrealized PnL should be +1.00 USDT
        assert "+1.00" in text

    def test_dashboard_short_position_upnl(self):
        from kucoin_bot.reporting.cli import print_dashboard
        from kucoin_bot.services.risk_manager import PositionInfo
        rm = self._make_risk_mgr()
        rm.update_position("ETH-USDT", PositionInfo(
            symbol="ETH-USDT", side="short", size=0.1,
            entry_price=2000.0, current_price=1900.0,
        ))
        text = print_dashboard(rm, cycle=3)
        assert "SHORT" in text
        # unrealized PnL = (2000 - 1900) * 0.1 = +10.00
        assert "+10.00" in text

    def test_dashboard_shows_strategies(self):
        from kucoin_bot.reporting.cli import print_dashboard
        rm = self._make_risk_mgr()
        text = print_dashboard(rm, strategies_active={"BTC-USDT": "trend_following"}, cycle=2)
        assert "trend_following" in text

    def test_dashboard_returns_string(self):
        from kucoin_bot.reporting.cli import print_dashboard
        rm = self._make_risk_mgr()
        result = print_dashboard(rm)
        assert isinstance(result, str)
        assert len(result) > 0

