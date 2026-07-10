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
LAGS_MS = np.array([50, 100, 250, 500, 1000, 2000, 5000], dtype=int)
N_PERM = 199
SEED = 20260710
RNG = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parent.parent / "tmp_microfield_results_bybit_corrected"
RAW = Path(__file__).resolve().parent / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)


def robust_z(x):
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
            raise ValueError(f"unknown schema {fields}")
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
        raise ValueError(f"empty {symbol}")
    df = df.sort_values(["t_ms", "trade_id"], kind="stable").reset_index(drop=True)
    df["notional"] = df.price * df.qty
    df["signed_notional"] = df.sign * df.notional
    return df


def aggregate_hour(df: pd.DataFrame, date: str, resolution_ms: int) -> pd.DataFrame:
    start = int(pd.Timestamp(date, tz="UTC").timestamp() * 1000)
    n = 3_600_000 // resolution_ms
    b = ((df.t_ms.to_numpy(np.int64) - start) // resolution_ms).astype(np.int64)
    keep = (b >= 0) & (b < n)
    b = b[keep]
    sign = df.sign.to_numpy(float)[keep]
    qty = df.qty.to_numpy(float)[keep]
    notion = df.notional.to_numpy(float)[keep]
    price = df.price.to_numpy(float)[keep]
    count = np.bincount(b, minlength=n)
    signed_count = np.bincount(b, weights=sign, minlength=n)
    volume = np.bincount(b, weights=qty, minlength=n)
    signed_volume = np.bincount(b, weights=sign * qty, minlength=n)
    notional = np.bincount(b, weights=notion, minlength=n)
    signed_notional = np.bincount(b, weights=sign * notion, minlength=n)
    last = np.full(n, np.nan)
    for bi, p in zip(b, price):
        last[bi] = p
    last = pd.Series(last).ffill().bfill().to_numpy()
    ret = np.r_[0.0, np.diff(np.log(last))]
    return pd.DataFrame({"t_ms": start + np.arange(n) * resolution_ms, "count": count, "signed_count": signed_count, "volume": volume, "signed_volume": signed_volume, "notional": notional, "signed_notional": signed_notional, "last": last, "return": ret})


def detect(frame: pd.DataFrame, resolution_ms: int) -> pd.DataFrame:
    z_flow = robust_z(frame.signed_notional.to_numpy())
    z_activity = robust_z(np.log1p(frame.notional.to_numpy()))
    idx = np.flatnonzero((np.abs(z_flow) >= 6.0) & (z_activity >= 3.0))
    refractory = max(1, int(np.ceil(1000 / resolution_ms)))
    kept = []
    for i in idx:
        if kept and i - kept[-1] < refractory:
            if abs(z_flow[i]) > abs(z_flow[kept[-1]]):
                kept[-1] = int(i)
            continue
        kept.append(int(i))
    return pd.DataFrame({"idx": kept, "t_ms": frame.t_ms.iloc[kept].to_numpy(np.int64), "direction": np.sign(z_flow[kept]).astype(int), "z_flow": z_flow[kept], "signed_notional": frame.signed_notional.iloc[kept].to_numpy(float)})


def exclusive_impulses(impulses: dict[str, pd.DataFrame], resolution_ms: int) -> dict[str, pd.DataFrame]:
    radius = max(1, int(np.ceil(100 / resolution_ms)))
    index_sets = {s: set(df.idx.astype(int)) for s, df in impulses.items()}
    result = {}
    for source, df in impulses.items():
        keep = []
        for r in df.itertuples():
            common = False
            for other, idxs in index_sets.items():
                if other == source:
                    continue
                if any((r.idx + d) in idxs for d in range(-radius, radius + 1)):
                    common = True
                    break
            if not common:
                keep.append(r.Index)
        result[source] = df.loc[keep].reset_index(drop=True)
    return result


def residualize(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    syms = sorted(frames)
    returns = np.column_stack([frames[s]["return"].to_numpy(float) for s in syms])
    flows = np.column_stack([frames[s]["signed_notional"].to_numpy(float) for s in syms])
    out = {}
    for j, s in enumerate(syms):
        others = [k for k in range(len(syms)) if k != j]
        Xr = np.column_stack([np.ones(len(returns)), returns[:, others]])
        Xf = np.column_stack([np.ones(len(flows)), flows[:, others]])
        br = np.linalg.lstsq(Xr, returns[:, j], rcond=None)[0]
        bf = np.linalg.lstsq(Xf, flows[:, j], rcond=None)[0]
        d = frames[s].copy()
        d["resid_return"] = returns[:, j] - Xr @ br
        d["resid_flow"] = flows[:, j] - Xf @ bf
        out[s] = d
    return out


def responses(imp: pd.DataFrame, target: pd.DataFrame, resolution_ms: int):
    ret = target.resid_return.to_numpy(float)
    flow = target.resid_flow.to_numpy(float)
    cr = np.r_[0.0, np.cumsum(ret)]
    cf = np.r_[0.0, np.cumsum(flow)]
    obs_r = []; obs_f = []; event_r = []; event_f = []
    for lag in LAGS_MS:
        h = max(1, int(np.ceil(lag / resolution_ms)))
        vr = []; vf = []
        for row in imp.itertuples():
            i = int(row.idx); j = min(len(ret), i + h + 1)
            if i + 1 >= len(ret):
                continue
            vr.append(row.direction * (cr[j] - cr[i + 1]))
            vf.append(row.direction * (cf[j] - cf[i + 1]))
        event_r.append(np.asarray(vr, float)); event_f.append(np.asarray(vf, float))
        obs_r.append(float(np.mean(vr)) if vr else np.nan); obs_f.append(float(np.mean(vf)) if vf else np.nan)
    return np.asarray(obs_r), np.asarray(obs_f), event_r, event_f


def circular_null(imp: pd.DataFrame, target: pd.DataFrame, resolution_ms: int):
    ret = target.resid_return.to_numpy(float); flow = target.resid_flow.to_numpy(float); n = len(ret)
    idx0 = imp.idx.to_numpy(int); dirs = imp.direction.to_numpy(float)
    nr = np.zeros((N_PERM, len(LAGS_MS))); nf = np.zeros((N_PERM, len(LAGS_MS)))
    for p in range(N_PERM):
        shift = int(RNG.integers(max(10, n // 20), max(11, n - n // 20)))
        idxs = (idx0 + shift) % n
        for k, lag in enumerate(LAGS_MS):
            h = max(1, int(np.ceil(lag / resolution_ms)))
            offsets = np.arange(1, h + 1)
            vals_r = []; vals_f = []
            for i, d in zip(idxs, dirs):
                js = (i + offsets) % n
                vals_r.append(d * ret[js].sum()); vals_f.append(d * flow[js].sum())
            nr[p, k] = np.mean(vals_r) if vals_r else np.nan
            nf[p, k] = np.mean(vals_f) if vals_f else np.nan
    return nr, nf


def bh(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); x = p[order] * n / np.arange(1, n + 1); x = np.minimum.accumulate(x[::-1])[::-1]; out = np.empty(n); out[order] = np.clip(x, 0, 1); return out


def bootstrap_ci(values, reps=999):
    if len(values) < 2:
        return [None, None]
    means = np.empty(reps)
    for i in range(reps):
        means[i] = np.mean(RNG.choice(values, size=len(values), replace=True))
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


manifest = []
raw = {}
for sym in SYMBOLS:
    url = f"https://public.bybit.com/trading/{sym}/{sym}{DATE}.csv.gz"
    path = RAW / f"{sym}{DATE}.csv.gz"
    meta = rr.download(url, path, timeout=1200, retries=3)
    df = parse_first_hour(path, sym, DATE)
    manifest.append(meta | {"symbol": sym, "date": DATE, "parsed_first_hour_rows": int(len(df))})
    raw[sym] = df

records = []
summary = {"date": DATE, "window_utc": [f"{DATE}T00:00:00Z", f"{DATE}T01:00:00Z"], "resolutions": {}}
geometry = []
for res in RESOLUTIONS:
    frames = {s: aggregate_hour(raw[s], DATE, res) for s in SYMBOLS}
    initial = {s: detect(frames[s], res) for s in SYMBOLS}
    impulses = exclusive_impulses(initial, res)
    resid = residualize(frames)
    p_all = []
    local = []
    gain = np.zeros((len(SYMBOLS), len(SYMBOLS)))
    for a, source in enumerate(SYMBOLS):
        for b, target in enumerate(SYMBOLS):
            if source == target:
                continue
            imp = impulses[source]
            if len(imp) < 20:
                continue
            obs_r, obs_f, evr, evf = responses(imp, resid[target], res)
            nr, nf = circular_null(imp, resid[target], res)
            pr = (1 + (np.abs(nr) >= np.abs(obs_r)).sum(axis=0)) / (N_PERM + 1)
            pf = (1 + (np.abs(nf) >= np.abs(obs_f)).sum(axis=0)) / (N_PERM + 1)
            for k, lag in enumerate(LAGS_MS):
                rec = {"date": DATE, "resolution_ms": res, "source": source, "target": target, "lag_ms": int(lag), "n_initial_impulses": int(len(initial[source])), "n_exclusive_impulses": int(len(imp)), "response_resid_return": float(obs_r[k]), "response_resid_flow": float(obs_f[k]), "p_return": float(pr[k]), "p_flow": float(pf[k]), "return_ci95": bootstrap_ci(evr[k], 499), "flow_ci95": bootstrap_ci(evf[k], 499)}
                local.append(rec); p_all.extend([pr[k], pf[k]])
            k1 = int(np.argmin(np.abs(LAGS_MS - 1000)))
            excess = max(0.0, abs(obs_f[k1]) - np.median(np.abs(nf[:, k1])))
            gain[b, a] = excess / (np.median(np.abs(imp.signed_notional.to_numpy())) + 1e-12)
    q = bh(np.asarray(p_all)) if p_all else np.array([])
    qi = 0
    for rec in local:
        rec["q_return"] = float(q[qi]); rec["q_flow"] = float(q[qi + 1]); qi += 2
        rec["p2_return"] = bool(rec["q_return"] < .05 and rec["return_ci95"][0] is not None and not (rec["return_ci95"][0] <= 0 <= rec["return_ci95"][1]))
        rec["p2_flow"] = bool(rec["q_flow"] < .05 and rec["flow_ci95"][0] is not None and not (rec["flow_ci95"][0] <= 0 <= rec["flow_ci95"][1]))
        records.append(rec)
    eig = np.linalg.eigvals(gain)
    radius = float(np.max(np.abs(eig)))
    summary["resolutions"][str(res)] = {"bins": int(len(next(iter(frames.values())))), "initial_impulses": {s: int(len(initial[s])) for s in SYMBOLS}, "exclusive_impulses": {s: int(len(impulses[s])) for s in SYMBOLS}, "n_tests": int(len(local) * 2), "n_p2_return": int(sum(r["p2_return"] for r in local)), "n_p2_flow": int(sum(r["p2_flow"] for r in local)), "gain_spectral_radius": radius, "gain_matrix": gain.tolist()}
    geometry.append({"resolution_ms": res, "gain_matrix": gain.tolist(), "eigenvalues": [[float(x.real), float(x.imag)] for x in eig], "spectral_radius": radius})

response_df = pd.DataFrame(records)
response_df.to_csv(OUT / "CORRECTED_RESPONSE.csv", index=False)
(OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
(OUT / "GEOMETRY.json").write_text(json.dumps(geometry, indent=2))
(OUT / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2))
(OUT / "RUN_CONFIG.json").write_text(json.dumps({"seed": SEED, "date": DATE, "symbols": SYMBOLS, "resolutions_ms": RESOLUTIONS, "lags_ms": LAGS_MS.tolist(), "null_replicates": N_PERM, "exclusive_impulse_radius_ms": 100, "common_factor_control": "target return and signed flow residualized contemporaneously against the other two assets", "multiple_testing": "BH-FDR across all source-target-lag return and flow tests within each resolution", "P2_gate": "q<0.05 and event-bootstrap 95% CI excludes zero"}, indent=2))
(OUT / "DONE.txt").write_text("ok\n")
print(json.dumps(summary, indent=2))
