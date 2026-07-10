#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import run_research as rr

DATE = "2024-03-05"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
RESOLUTIONS = [50, 100, 250, 500, 1000]
N_PERM = 99
OUT = Path(__file__).resolve().parent.parent / "tmp_microfield_results_bybit"
OUT.mkdir(parents=True, exist_ok=True)
rr.OUT = OUT
rr.BYBIT_DATES = [DATE]
rr.BYBIT_SYMBOLS = SYMBOLS
rr.RESOLUTIONS_MS = RESOLUTIONS


def fast_robust_z(x, window=None):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-15:
        scale = np.nanstd(x) + 1e-15
    return (x - med) / scale


def parse_first_hour(path: Path, symbol: str, date: str) -> pd.DataFrame:
    start = int(pd.Timestamp(date, tz="UTC").timestamp() * 1000)
    end = start + 3_600_000
    rows = []
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        fields = [c.strip() for c in (reader.fieldnames or [])]
        fmap = {c.lower(): c for c in fields}
        time_key = next((fmap[k] for k in ["timestamp", "time", "trade_time_ms", "transact_time"] if k in fmap), fields[0])
        side_key = next((fmap[k] for k in ["side", "direction"] if k in fmap), None)
        price_key = next((fmap[k] for k in ["price", "p"] if k in fmap), None)
        size_key = next((fmap[k] for k in ["size", "qty", "quantity", "amount"] if k in fmap), None)
        id_key = next((fmap[k] for k in ["trdmatchid", "trade_id", "id"] if k in fmap), None)
        if price_key is None or size_key is None or side_key is None:
            raise ValueError(f"unknown Bybit schema: {fields}")
        for idx, r in enumerate(reader):
            try:
                t = rr.event_time_to_ms(r[time_key])
                if t < start:
                    continue
                if t >= end:
                    break
                price = float(r[price_key]); qty = float(r[size_key]); side = str(r[side_key]).lower()
                sign = 1.0 if side.startswith("b") else -1.0
                rows.append((t, price, qty, sign, r.get(id_key, str(idx)) if id_key else str(idx)))
            except Exception:
                continue
    df = pd.DataFrame(rows, columns=["t_ms", "price", "qty", "sign", "trade_id"])
    if df.empty:
        raise ValueError(f"empty first-hour parse: {path}")
    df = df.sort_values(["t_ms", "trade_id"], kind="stable").reset_index(drop=True)
    df["notional"] = df["price"] * df["qty"]
    df["signed_notional"] = df["sign"] * df["notional"]
    df["symbol"] = symbol
    df["date"] = date
    return df


def null99(source_imp, target, resolution_ms, n_perm=200):
    return original_null(source_imp, target, resolution_ms, N_PERM)

original_null = rr.circular_null
rr.robust_z = fast_robust_z
rr.parse_bybit = parse_first_hour
rr.circular_null = null99
result = rr.bybit_research()
(OUT / "BYBIT_RESULTS.json").write_text(json.dumps(result, indent=2, default=str))
(OUT / "RUN_CONFIG.json").write_text(json.dumps({
    "date": DATE,
    "symbols": SYMBOLS,
    "window_utc": [f"{DATE}T00:00:00Z", f"{DATE}T01:00:00Z"],
    "resolutions_ms": RESOLUTIONS,
    "circular_shift_null_replicates": N_PERM,
    "fdr": "Benjamini-Hochberg across response-return and response-flow tests per resolution/date output",
    "p2_rule": "directed edge only if q<0.05 at lag<=1000ms; temporal ordering alone is insufficient"
}, indent=2))
(OUT / "DONE.txt").write_text("ok\n")
print(json.dumps({"status": "ok", "dates": list(result["results"]), "manifest_entries": len(result["manifest"])}, indent=2))
