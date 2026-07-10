#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
import run_research as rr

# One synchronized native-tick hour per venue-day. The raw files remain full-day;
# only the analysis tensor is bounded to keep 10 ms resolution tractable on CI.
def fast_robust_z(x, window=None):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-15:
        scale = np.nanstd(x) + 1e-15
    return (x - med) / scale


def first_hour_aggregate(df: pd.DataFrame, resolution_ms: int) -> pd.DataFrame:
    day0 = int(pd.Timestamp(df["date"].iloc[0], tz="UTC").timestamp() * 1000)
    end = day0 + 60 * 60 * 1000
    d = df[(df.t_ms >= day0) & (df.t_ms < end)]
    if d.empty:
        d = df.iloc[: min(len(df), 100000)]
        day0 = int(d.t_ms.min() // resolution_ms * resolution_ms)
        end = int(d.t_ms.max()) + resolution_ms
    b = ((d.t_ms.to_numpy(np.int64) - day0) // resolution_ms).astype(np.int64)
    n = int((end - day0) // resolution_ms)
    sign = d.sign.to_numpy(float); qty = d.qty.to_numpy(float)
    notion = d.notional.to_numpy(float); price = d.price.to_numpy(float)
    count = np.bincount(b, minlength=n)[:n]
    signed_count = np.bincount(b, weights=sign, minlength=n)[:n]
    volume = np.bincount(b, weights=qty, minlength=n)[:n]
    signed_volume = np.bincount(b, weights=sign * qty, minlength=n)[:n]
    notional = np.bincount(b, weights=notion, minlength=n)[:n]
    signed_notional = np.bincount(b, weights=sign * notion, minlength=n)[:n]
    last = np.full(n, np.nan)
    for bi, p in zip(b, price):
        if 0 <= bi < n:
            last[bi] = p
    last = pd.Series(last).ffill().bfill().to_numpy()
    ret = np.r_[0.0, np.diff(np.log(last))]
    return pd.DataFrame({
        "t_ms": day0 + np.arange(n) * resolution_ms,
        "count": count,
        "signed_count": signed_count,
        "volume": volume,
        "signed_volume": signed_volume,
        "notional": notional,
        "signed_notional": signed_notional,
        "last": last,
        "return": ret,
    })

rr.robust_z = fast_robust_z
rr.aggregate_trades = first_hour_aggregate
rr.main()
