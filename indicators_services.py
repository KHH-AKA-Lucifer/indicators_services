# import requied libraries
import math
import asyncpg
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os, asyncio, urllib.parse

# ------------ Config ----------------------------
load_dotenv()

HEAL_BUCKETS = int(os.getenv("HEAL_BUCKETS", "2"))
K_LEN = int(os.getenv("K_LEN", "9"))
D_LEN = int(os.getenv("D_LEN", "3"))
CCI_SHORT = int(os.getenv("CCI_SHORT", "3"))
CCI_LONG  = int(os.getenv("CCI_LONG", "9"))
SLEEP_SEC = float(os.getenv("SLEEP_SEC", "5"))

# -------------- Signle Database Connection -------

def build_dsn() -> str:
    """
    Creating a single data base connection source.
    """
    dsn = os.getenv("DB_DSN")
    if dsn:
        return dsn

    host = os.getenv("host")
    port = os.getenv("port", "5432")
    user = os.getenv("user")
    pwd  = os.getenv("pass")
    db   = os.getenv("db")

    if not all([host, user, pwd, db]):
        raise RuntimeError("DB credentials missing: set DB_DSN or host/user/pass/db in .env")

    user_q = urllib.parse.quote_plus(user)
    pwd_q  = urllib.parse.quote_plus(pwd)
    return f"postgresql://{user_q}:{pwd_q}@{host}:{port}/{db}"

DB_DSN = build_dsn()

# list of (schema, interval, ca_table, ind_table, bucket_sql_interval)
ASSETS = [
    ("gold", "6h", "ohlc_data_6hr_bid_xau_usd",  "indicators_6hr_bid_xau_usd", "6 hours", "bucket_6h"),
    ("gold",      "1d", "ohlc_data_daily_bid_xau_usd",   "indicators_daily_bid_xau_usd",    "1 day",   "bucket_daily"),
    ("silver",    "6h", "ohlc_data_6hr_bid_xag_usd",     "indicators_6hr_bid_xag_usd",      "6 hours", "bucket_6h"),
    ("silver",    "1d", "ohlc_data_daily_bid_xag_usd",   "indicators_daily_bid_xag_usd",    "1 day",   "bucket_daily"),
    ("platinum",  "6h", "ohlc_data_6hr_bid_xpt_usd",     "indicators_6hr_bid_xpt_usd",      "6 hours", "bucket_6h"),
    ("platinum",  "1d", "ohlc_data_daily_bid_xpt_usd",   "indicators_daily_bid_xpt_usd",    "1 day",   "bucket_daily"),
    ("sgd",       "6h", "ohlc_data_6hr_bid_usd_sgd",     "indicators_6hr_bid_usd_sgd",      "6 hours", "bucket_6h"),
    ("sgd",       "1d", "ohlc_data_daily_bid_usd_sgd",   "indicators_daily_bid_usd_sgd",    "1 day",   "bucket_daily"),
    ("myr",       "6h", "ohlc_data_6hr_bid_usd_myr",     "indicators_6hr_bid_usd_myr",      "6 hours", "bucket_6h"),
    ("myr",       "1d", "ohlc_data_daily_bid_usd_myr",   "indicators_daily_bid_usd_myr",    "1 day",   "bucket_daily"),
]

# ------------ Indicators -------------
def calculate_slowD(df: pd.DataFrame, k_period: int = 9, d_period: int = 3) -> pd.DataFrame:
    """A stochastic function that calculates the Fast %K & Slow %D using EMA.
    
    Parameters
    ----------
    df: pd.DataFrame (Input dataframe containing OHLC data.)
    k_period: int, optional (Period to calculate the Fast %K <default is 9>.)
    d_period: int, optional (Period to calculate the Slow %D <default is 3>.)
    
    Returns
    -------
    pd.DataFrame (DataFrame that contains Fast %K, Fast %D (EMA), and Slow %D (EMA).)
    """

    # find the highest high market price in the k period
    df['highest_high'] = df['high'].rolling(window=k_period).max()

    # find the lowest low market price in the k period
    df['lowest_low'] = df['low'].rolling(window=k_period).min()

    # calculate Fast %K
    df['fastk'] = ((df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low'])) * 100

    # calculate Fast %D (EMA of Fast %K with period 1, which is just FastK itself)
    df['fastd'] = df['fastk']

    # calculate Slow %D (EMA of Fast %D with period d_period)
    df['slowd'] = df['fastd'].ewm(span=d_period, adjust=False).mean()

    # drop unecessary columns
    df.drop(columns=['highest_high', 'lowest_low'], inplace=True)

    # Return the dataframe with stochastic values
    return df


def calculate_cci(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """ A method that calculates commodity channel index.

        Parameters
        ----------
        df: pd.DataFrame (Input dataframe containing OHLC data.)
        period: int (lookback period)

        Returns
        -------
        pd.DataFrame (DataFrame that contains Commodity Channel Index (CCI).)
    """
        
    # calculate the typical price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

    # calculate the simple moving average (SMA) of the Typical Price
    sma = df['typical_price'].rolling(window=period).mean()

    # calculate the mean deviation manually
    mean_deviation = df['typical_price'].rolling(window=period).apply(
        lambda x: (np.abs(x - x.mean()).mean()), raw=True
    )

    # calculate the CCI
    df[f'CCI{period}'] = (df['typical_price'] - sma) / \
        (0.015 * mean_deviation)

    # return the resulted dataframe
    return df

# ------------ DB helper functions -------------
async def fetch_ca_window(conn, schema, ca_table, bucket_sql, lookback_buckets, bucket_col):
    # Build SQL query to fetch a window of OHLC data for a given asset and timeframe
    sql = f"""
        SELECT {bucket_col} AS date_time,
               (open)::double precision  AS open,
               (high)::double precision  AS high,
               (low)::double precision   AS low,
               (close)::double precision AS close
        FROM {schema}.{ca_table}
        WHERE {bucket_col} >= time_bucket(($1)::text::interval, now())
                              - ($2::int + 5) * (($1)::text::interval)
        ORDER BY {bucket_col}
    """
    # Execute the query and fetch results
    rows = await conn.fetch(sql, bucket_sql, lookback_buckets)
    if not rows:
        # Return empty DataFrame with expected columns if no data
        return pd.DataFrame(columns=["date_time","open","high","low","close"])
    # Convert asyncpg records to DataFrame
    return pd.DataFrame([dict(r) for r in rows])

async def upsert_indicator_row(conn, schema: str, ind_table: str, dt, slowd, cci3, cci9):
    """ 
    Upsert (insert or update) a row of indicator values for a given date_time
    If a row with the same date_time exists, update its values; otherwise, insert a new row
    """
    sql = f"""
        INSERT INTO {schema}.{ind_table} (date_time, slowd, cci3, cci9, updated_at)
        VALUES ($1, $2, $3, $4, now())
        ON CONFLICT (date_time) DO UPDATE SET
            slowd = EXCLUDED.slowd,
            cci3  = EXCLUDED.cci3,
            cci9  = EXCLUDED.cci9,
            updated_at = now();
    """
    await conn.execute(sql, dt, slowd, cci3, cci9)

# -------- Healing window -------------
def ema_extra_warmup(d_period: int, tol: float = 1e-3) -> int:
    # EMA alpha for pandas ewm(span=d, adjust=False)
    alpha = 2 / (d_period + 1)
    # m such that (1 - alpha)^m <= tol
    m = math.log(tol) / math.log(1 - alpha)
    return max(0, math.ceil(m))

# ------------ Core per-table work -------------
async def process_one_table(conn, schema: str,
                            interval_label: str,
                            ca_table: str,
                            ind_table: str,
                            bucket_sql: str,
                            bucket_col: str):
    # process one asset/table for a given interval and bucket
    # fetch a window of OHLC data for the asset
    WARMUP_EMA = ema_extra_warmup(D_LEN, tol=1e-3)  # e.g., ~10 for d=3, ~35 for d=10
    WARMUP_BASE = K_LEN + (D_LEN - 1)               # %K window plus D seed
    longest = max(WARMUP_BASE + WARMUP_EMA, CCI_LONG) + 5  # +5 cushion

    df = await fetch_ca_window(conn, schema, ca_table, bucket_sql, longest, bucket_col)
    if df.empty:
        # if no data, exit early
        return

    # set time as index for indicator calculations
    df = df.set_index("date_time")

    # calculate indicators using provided functions
    stoch_df = calculate_slowD(df.copy(), k_period=K_LEN, d_period=D_LEN)
    cci3_df  = calculate_cci(df.copy(), period=CCI_SHORT)
    cci9_df  = calculate_cci(df.copy(), period=CCI_LONG)

    # find columns case-insensitively and normalize names
    # find SlowD column
    try:
        slowd_src = next(c for c in stoch_df.columns if c.lower() == "slowd")
    except StopIteration:
        raise KeyError("Stochastic function did not produce a 'SlowD' column")

    # find CCI columns
    want_cci3 = f"cci{CCI_SHORT}"
    want_cci9 = f"cci{CCI_LONG}"
    try:
        cci3_src = next(c for c in cci3_df.columns if c.lower() == want_cci3)
    except StopIteration:
        raise KeyError(f"calculate_cci(period={CCI_SHORT}) did not produce '{want_cci3}' / 'CCI{CCI_SHORT}'")
    try:
        cci9_src = next(c for c in cci9_df.columns if c.lower() == want_cci9)
    except StopIteration:
        raise KeyError(f"calculate_cci(period={CCI_LONG}) did not produce '{want_cci9}' / 'CCI{CCI_LONG}'")

    # merge indicator columns into one DataFrame
    merged = df.join(stoch_df[[slowd_src]].rename(columns={slowd_src: "slowd"}), how="left")
    merged = merged.join(cci3_df[[cci3_src]].rename(columns={cci3_src: "cci3"}), how="left")
    merged = merged.join(cci9_df[[cci9_src]].rename(columns={cci9_src: "cci9"}), how="left")

    # select the most recent forming + heal window
    n = HEAL_BUCKETS + 1
    last_idx = merged.index[-n:] if len(merged) >= n else merged.index
    trimmed = merged.loc[last_idx, ["slowd", "cci3", "cci9"]].reset_index()  # includes date_time

    # upsert a few rows (forming + previous heal buckets) into the indicators table
    for _, row in trimmed.iterrows():
        await upsert_indicator_row(
            conn, schema, ind_table,
            row["date_time"], row["slowd"], row["cci3"], row["cci9"]
        )


# ------------ Main loop -------------
async def main():
    # open one connection for the service
    conn = await asyncpg.connect(dsn=DB_DSN)  # you can add server_settings={"application_name": "indicators_5s"}
    try:
        while True:
            # try to acquire the single-writer lock
            got = await conn.fetchval(
                "SELECT pg_try_advisory_lock( hashtextextended('indicators_service_async', 202) )"
            )
            if not got:
                await asyncio.sleep(SLEEP_SEC)
                continue

            try:
                # one transaction for the whole batch
                async with conn.transaction():
                    for schema, interval_label, ca_table, ind_table, bucket_sql, bucket_col in ASSETS:
                        await process_one_table(conn, schema, interval_label, ca_table, ind_table, bucket_sql, bucket_col)

            except Exception as e:
                # the async-with transaction rolls back automatically on exceptions
                # we still log and continue the loop
                print("ERROR in indicators loop:", e)

            finally:
                # ALWAYS release the advisory lock — success or error
                await conn.execute("SELECT pg_advisory_unlock_all();")

            # cadence
            await asyncio.sleep(SLEEP_SEC)

    finally:
        # close database connection on exit (systemd stop/kill, etc.)
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())