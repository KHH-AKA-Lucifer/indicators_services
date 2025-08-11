
import time
import os
import numpy as np
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from psycopg.extras import execute_values

# ------------ CONFIG -------------
DB_DSN = os.getenv("DB_DSN", "postgresql://USER:PASSWORD@HOST:PORT/DBNAME")
HEAL_BUCKETS = int(os.getenv("HEAL_BUCKETS", "2"))
K_LEN = int(os.getenv("K_LEN", "14"))
D_LEN = int(os.getenv("D_LEN", "3"))
CCI_SHORT = int(os.getenv("CCI_SHORT", "3"))
CCI_LONG  = int(os.getenv("CCI_LONG", "9"))
SLEEP_SEC = float(os.getenv("SLEEP_SEC", "5"))

# List of (schema, interval, ca_table, ind_table, bucket_sql_interval)
ASSETS = [
    # gold 6h & 1d examples — edit/add others (silver, platinum, sgd, myr)
    ("gold", "6h", "ohlc_data_6hr_bid_xau_usd",  "indicators_6hr_bid_xau_usd",  "6 hours"),
    ("gold", "1d", "ohlc_data_daily_bid_xau_usd","indicators_daily_bid_xau_usd","1 day"),
]

# ------------ INDICATORS -------------
def compute_slowd(df, k_period=14, d_period=3, avg="sma",
                  high="High", low="Low", close="Close"):
    hh = df[high].rolling(window=k_period, min_periods=k_period).max()
    ll = df[low].rolling(window=k_period, min_periods=k_period).min()
    rng = (hh - ll).replace(0, np.nan)
    fastk = 100.0 * (df[close] - ll) / rng
    if avg == "ema":
        slowd = fastk.ewm(span=d_period, adjust=False, min_periods=d_period).mean()
    else:
        slowd = fastk.rolling(window=d_period, min_periods=d_period).mean()
    out = pd.DataFrame({"FastK": fastk, "SlowD": slowd})
    return out

def compute_cci(df, period=20, high="High", low="Low", close="Close", name=None):
    tp = (df[high] + df[low] + df[close]) / 3.0
    sma = tp.rolling(window=period, min_periods=period).mean()
    md  = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=False
    )
    denom = (0.015 * md).replace(0, np.nan)
    cci = (tp - sma) / denom
    return pd.DataFrame({ name or f"CCI{period}": cci })

# ------------ DB OPS -------------
def fetch_ca_window(cur, schema, ca_table, bucket_interval, longest_lookback):
    cur.execute(f"""
        SELECT date_time, open AS Open, high AS High, low AS Low, close AS Close
        FROM {schema}.{ca_table}
        WHERE date_time >= time_bucket(%s, now()) - %s::int * INTERVAL %s
        ORDER BY date_time
    """, (bucket_interval, longest_lookback + 5, bucket_interval))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["date_time","Open","High","Low","Close"])
    df = pd.DataFrame(rows)
    return df

def upsert_indicators(cur, schema, ind_table, recs):
    sql = f"""
    INSERT INTO {schema}.{ind_table}
        (date_time, slowd, cci3, cci9, updated_at)
    VALUES %s
    ON CONFLICT (date_time) DO UPDATE SET
        slowd = EXCLUDED.slowd,
        cci3  = EXCLUDED.cci3,
        cci9  = EXCLUDED.cci9,
        updated_at = now();
    """
    values = [(r["date_time"], r["slowd"], r["cci3"], r["cci9"], None) for r in recs]
    execute_values(cur, sql, values)

# ------------ MAIN LOOP -------------
def main():
    longest = max(K_LEN, CCI_LONG)
    with psycopg.connect(DB_DSN) as conn:
        conn.autocommit = False
        while True:
            try:
                with conn.cursor(row_factory=dict_row) as cur:
                    # single-writer advisory lock
                    cur.execute("SELECT pg_try_advisory_lock( hashtextextended('indicators_service', 101) )")
                    if not cur.fetchone()["pg_try_advisory_lock"]:
                        conn.rollback(); time.sleep(SLEEP_SEC); continue

                    for schema, interval, ca_table, ind_table, bucket_sql in ASSETS:
                        df = fetch_ca_window(cur, schema, ca_table, bucket_sql, longest)
                        if df.empty:
                            continue
                        df = df.set_index("date_time")

                        stoch = compute_slowd(df, k_period=K_LEN, d_period=D_LEN, avg="sma")
                        cci3  = compute_cci(df, period=CCI_SHORT, name="CCI3")
                        cci9  = compute_cci(df, period=CCI_LONG,  name="CCI9")

                        merged = pd.concat([stoch, cci3, cci9], axis=1)
                        # keep last forming + heal buckets
                        last_idx = merged.index[-(HEAL_BUCKETS+1):] if len(merged) >= (HEAL_BUCKETS+1) else merged.index
                        trimmed = merged.loc[last_idx, ["SlowD","CCI3","CCI9"]].rename(
                            columns={"SlowD":"slowd","CCI3":"cci3","CCI9":"cci9"}
                        ).reset_index()  # brings date_time back as column

                        recs = trimmed.to_dict(orient="records")
                        if recs:
                            upsert_indicators(cur, schema, ind_table, recs)

                    cur.execute("SELECT pg_advisory_unlock_all();")
                conn.commit()
            except Exception as e:
                conn.rollback()
                print("Error:", repr(e))
            time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()
