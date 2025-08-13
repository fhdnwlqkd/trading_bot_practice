CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS ohlcv_xrp_krw_1m(
  ts TIMESTAMPTZ PRIMARY KEY,
  open  DOUBLE PRECISION,
  high  DOUBLE PRECISION,
  low   DOUBLE PRECISION,
  close DOUBLE PRECISION,
  volume DOUBLE PRECISION,
  trade_count BIGINT
);

SELECT create_hypertable('ohlcv_xrp_krw_1m','ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ohlcv_xrp_1m_ts_desc ON ohlcv_xrp_krw_1m (ts DESC);

-- (선택) 5분 연속 집계 (Continuous Aggregate)
CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_ohlcv_xrp_krw_5m
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('5 minutes', ts) AS bucket,
  first(open, ts)  AS open,
  max(high)        AS high,
  min(low)         AS low,
  last(close, ts)  AS close,
  sum(volume)      AS volume,
  sum(trade_count) AS trade_count
FROM ohlcv_xrp_krw_1m
GROUP BY bucket
WITH NO DATA;

-- (선택) 자동 새로고침 정책: 최근 2일
SELECT add_continuous_aggregate_policy(
  'cagg_ohlcv_xrp_krw_5m',
  start_offset => INTERVAL '2 days',
  end_offset   => INTERVAL '1 minute',
  schedule_interval => INTERVAL '5 minutes'
);