"""Database models for trades, orders, signals, and PnL tracking."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class Order(Base):
    """Persisted order record."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    exchange_order_id = Column(String(64), index=True)
    symbol = Column(String(32), nullable=False)
    side = Column(String(8), nullable=False)  # buy / sell
    order_type = Column(String(16), nullable=False)  # limit / market
    quantity = Column(Float, nullable=False)
    price = Column(Float)
    filled_qty = Column(Float, default=0.0)
    avg_fill_price = Column(Float)
    status = Column(String(16), default="pending")
    account_type = Column(String(16), default="trade")  # trade / margin / futures
    leverage = Column(Float, default=1.0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)


class Trade(Base):
    """Fill / execution record."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, index=True)
    symbol = Column(String(32), nullable=False)
    side = Column(String(8), nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    fee_currency = Column(String(16))
    realized_pnl = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)


class SignalSnapshot(Base):
    """Audit trail for signals that led to decisions."""

    __tablename__ = "signal_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    regime = Column(String(32))
    strategy_name = Column(String(64))
    signal_data = Column(Text)  # JSON blob of feature scores
    decision = Column(String(16))  # entry / exit / hold
    reason = Column(Text)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)


class BalanceRecord(Base):
    """Periodic balance snapshot."""

    __tablename__ = "balances"

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_type = Column(String(16))
    currency = Column(String(16))
    total = Column(Float)
    available = Column(Float)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)


class TransferRecord(Base):
    """Internal account transfer log."""

    __tablename__ = "transfers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    idempotency_key = Column(String(64), unique=True)
    from_account = Column(String(16))
    to_account = Column(String(16))
    currency = Column(String(16))
    amount = Column(Float)
    status = Column(String(16))
    timestamp = Column(DateTime, default=dt.datetime.utcnow)


class PnLRecord(Base):
    """Daily PnL tracking."""

    __tablename__ = "pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False)  # YYYY-MM-DD
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    total_equity = Column(Float, default=0.0)
    drawdown_pct = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)


def init_db(db_url: str = "sqlite:///kucoin_bot.db") -> sessionmaker:
    """Create tables and return a session factory."""
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
