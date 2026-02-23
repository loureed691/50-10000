"""Tests for database models."""

from __future__ import annotations

from kucoin_bot.models import Order, SignalSnapshot, Trade, init_db


class TestModels:
    def test_init_db(self, tmp_path):
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        session_factory = init_db(db_url)
        with session_factory() as session:
            order = Order(
                symbol="BTC-USDT",
                side="buy",
                order_type="limit",
                quantity=0.001,
                price=30000.0,
            )
            session.add(order)
            session.commit()
            assert order.id is not None

    def test_signal_snapshot(self, tmp_path):
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        session_factory = init_db(db_url)
        with session_factory() as session:
            snap = SignalSnapshot(
                symbol="BTC-USDT",
                regime="trending_up",
                strategy_name="trend_following",
                signal_data='{"momentum": 0.5}',
                decision="entry_long",
                reason="test",
            )
            session.add(snap)
            session.commit()
            assert snap.id is not None

    def test_trade_record(self, tmp_path):
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        session_factory = init_db(db_url)
        with session_factory() as session:
            trade = Trade(
                symbol="BTC-USDT",
                side="buy",
                price=30000.0,
                quantity=0.001,
                fee=0.03,
            )
            session.add(trade)
            session.commit()
            result = session.query(Trade).first()
            assert result.price == 30000.0
