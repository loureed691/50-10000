# Changelog

## [Unreleased] — Hardening PR

### Breaking Changes
- `ALLOW_INTERNAL_TRANSFERS` default changed from `false` to `true` (set `ALLOW_INTERNAL_TRANSFERS=false` to restore old behavior)
- `KuCoinAPIError` exception is now raised on non-retryable HTTP errors (400, 401, 403, 404) instead of returning an error dict. Callers catching generic `Exception` are unaffected.
- Docker default `DB_URL` changed to `sqlite:////app/data/kucoin_bot.db` (persisted under mounted volume). Set `DB_URL` env var explicitly to override.
- `ExecutionEngine` now polls order status by default (`poll_fills=True`). Set `poll_fills=False` to restore old assume-filled behavior.

### Added
- **API client**: Typed `KuCoinAPIError` exception for non-2xx responses
- **API client**: Explicit `aiohttp.ClientTimeout` (30s total, 10s connect)
- **API client**: Proper `urllib.parse.urlencode` for query string construction (same string used for signing and request)
- **API client**: Jitter on 429 backoff; never retry auth/validation errors
- **API client**: Futures parity endpoints — klines, funding rate, positions list, open orders, cancel order, cancel all orders, get order, ticker
- **API client**: Spot `cancel_all_orders` endpoint
- **Market data**: Separate `get_klines_spot()` and `get_klines_futures()` with automatic routing based on market type
- **Market data**: Extended `MarketInfo` with futures-specific fields (`contract_multiplier`, `lot_size`, `tick_size`, `max_leverage`)
- **Market data**: Kline cache TTL now uses correct period per kline type (1min=60s, 5min=300s, 1hour=3600s, etc.)
- **Execution**: Order lifecycle polling — orders are polled for fill status instead of assumed filled
- **Execution**: Futures-aware `cancel_all` cancels both spot and futures orders
- **Execution**: Correct futures sizing — integer contracts aligned to `lot_size`, not reusing spot rounding logic
- **Execution**: `flatten_position` now accepts market info and sets `reduce_only=True` for futures
- **Startup reconciliation**: Fetches open futures positions and orders on restart, rebuilds risk manager state
- **Portfolio**: Transfer records persisted to DB via `TransferRecord` model with idempotency keys
- **CI**: Added ruff lint check and black format check to CI pipeline
- **CI**: mypy check is now mandatory (removed `continue-on-error`)
- **Tests**: 24 new tests — API client signatures, param encoding, execution sizing, order lifecycle, cancel_all, flatten

### Fixed
- Query string encoding: was using manual string concatenation, now uses `urllib.parse.urlencode` for consistent signing
- Kline time window: was always using 3600s (1hour) regardless of kline type
- Futures kline format: converted to spot-compatible format (OHLCV + turnover)
- mypy: All 15+ type errors fixed (no-any-return, no-redef, assignment)
- ruff: All 40 unused import warnings fixed
- Docker: DB file now persists across container restarts via mounted volume

### Changed
- Dev dependencies: Added `black`, `ruff`, `isort` to `[dev]` extras
- pyproject.toml: Added tool configs for black, ruff, isort, mypy
- Codebase reformatted with black (120 line length) and isort
