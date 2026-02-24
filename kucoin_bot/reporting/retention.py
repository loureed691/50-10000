"""Database retention helpers – TTL-based cleanup for high-volume tables.

SignalSnapshots are written every slow cycle for every symbol and can grow
quickly.  ``purge_old_snapshots`` deletes rows older than a configurable
retention window (default 7 days) so the database stays bounded.

Usage::

    from kucoin_bot.models import init_db
    from kucoin_bot.reporting.retention import purge_old_snapshots

    Session = init_db(db_url)
    deleted = purge_old_snapshots(Session, days=7)

Environment variable ``SIGNAL_RETENTION_DAYS`` overrides the default.
"""

from __future__ import annotations

import datetime as dt
import logging
import os

from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

_DEFAULT_RETENTION_DAYS = 7


def purge_old_snapshots(
    session_factory: sessionmaker,
    days: int | None = None,
) -> int:
    """Delete ``SignalSnapshot`` rows older than *days* days.

    Returns the number of deleted rows.
    """
    from kucoin_bot.models import SignalSnapshot

    if days is None:
        days = int(os.getenv("SIGNAL_RETENTION_DAYS", str(_DEFAULT_RETENTION_DAYS)))

    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
    try:
        with session_factory() as sess:
            count = sess.query(SignalSnapshot).filter(SignalSnapshot.timestamp < cutoff).delete()
            sess.commit()
            logger.info("Purged %d signal snapshots older than %s (%d days)", count, cutoff.isoformat(), days)
            return count
    except Exception:
        logger.exception("Failed to purge old signal snapshots")
        return 0
