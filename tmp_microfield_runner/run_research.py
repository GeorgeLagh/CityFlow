#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import shutil
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "tmp_microfield_results"
RAW = ROOT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
SEED = 20260710
RNG = np.random.default_rng(SEED)
RESOLUTIONS_MS = [10, 25, 50, 100, 250, 500, 1000]
LAGS_MS = np.array([10, 25, 50, 100, 250, 500, 1000, 2000, 5000], dtype=int)
BYBIT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BYBIT_DATES = ["2022-11-09", "2024-03-05", "2025-01-30"]
QF_COMMIT = "024921eb507fcc0c4ffe3e0a96802724be1ae84a"
QF_BASE = "https://raw.githubusercontent.com/QF-Bench/QuantitativeFinance-Bench/" + QF_COMMIT + "/tasks/binance-btc-participation-tca/environment/data"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path, timeout: int = 600, retries: int = 3) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return {"url": url, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), "cached": True}
    tmp = path.with_suffix(path.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "market-metastability-microfield/3.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f, length=1024 * 1024)
            tmp.replace(path)
            return {"url": url, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), "cached": False}
        except Exception as exc:
            last = repr(exc)
            tmp.unlink(missing_ok=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed: {url}: {last}")


def robust_z(x: np.ndarray, window: int | None = None) -> np.ndarray:
    x = np.asarray(x, float)
    if window is None or window >= len(x):
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-15:
            scale = np.nanstd(x) + 1e-15
        return (x - med) / scale
    s = pd.Series(x)
    minp = max(20, window // 5)
    med = s.rolling(window, min_periods=minp, center=True).median()
    mad = (s - med).abs().rolling(window, min_periods=minp, center=True).median()
    scale = 1.4826 * mad
    fallback = np.nanmedian(scale[scale > 0])
    if not np.isfinite(fallback):
        fallback = np.nanstd(x) + 1e-15
    scale = scale.where(scale > 1e-15, fallback)
    return ((s - med) / scale).to_numpy()


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def event_time_to_ms(v) -> int:
    x = float(v)
    if x > 1e15:
        return int(round(x / 1000.0))
    if x > 1e12:
        return int(round(x))
    return int(round(x * 1000.0))


def parse_bybit(path: Path, symbol: str, date: str) -> pd.DataFrame:
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
                t = event_time_to_ms(r[time_key])
                price = float(r[price_key]); qty = float(r[size_key]); side = str(r[side_key]).lower()
                sign = 1.0 if side.startswith("b") else -1.0
                rows.append((t, price, qty, sign, r.get(id_key, str(idx)) if id_key else str(idx)))
            except Exception:
                continue
    df = pd.DataFrame(rows, columns=["t_ms", "price", "qty", "sign", "trade_id"])
    if df.empty:
        raise ValueError(f"empty parsed file: {path}")
    df = df.sort_values(["t_ms", "trade_id"], kind="stable").reset_index(drop=True)
    df["notional"] = df["price"] * df["qty"]
    df["signed_notional"] = df["sign"] * df["notional"]
    df["symbol"] = symbol; df["date"] = date
    return df


def aggregate_trades(df: pd.DataFrame, resolution_ms: int) -> pd.DataFrame:
    day0 = int(pd.Timestamp(df["date"].iloc[0], tz="UTC").timestamp() * 1000)
    b = ((df.t_ms.to_numpy(np.int64) - day0) // resolution_ms).astype(np.int64)
    good = b >= 0
    b = b[good]
    sign = df.sign.to_numpy(float)[good]; qty = df.qty.to_numpy(float)[good]
    notion = df.notional.to_numpy(float)[good]; price = df.price.to_numpy(float)[good]
    n = int(24 * 60 * 60 * 1000 // resolution_ms)
    count = np.bincount(b.clip(0, n - 1), minlength=n)[:n]
    signed_count = np.bincount(b.clip(0, n - 1), weights=sign, minlength=n)[:n]
    volume = np.bincount(b.clip(0, n - 1), weights=qty, minlength=n)[:n]
    signed_volume = np.bincount(b.clip(0, n - 1), weights=sign * qty, minlength=n)[:n]
    notional = np.bincount(b.clip(0, n - 1), weights=notion, minlength=n)[:n]
    signed_notional = np.bincount(b.clip(0, n - 1), weights=sign * notion, minlength=n)[:n]
    last = np.full(n, np.nan)
    for bi, p in zip(b, price):
        if 0 <= bi < n: last[bi] = p
    last = pd.Series(last).ffill().bfill().to_numpy()
    ret = np.r_[0.0, np.diff(np.log(last))]
    return pd.DataFrame({"t_ms": day0 + np.arange(n) * resolution_ms, "count": count, "signed_count": signed_count, "volume": volume, "signed_volume": signed_volume, "notional": notional, "signed_notional": signed_notional, "last": last, "return": ret})


def pick_impulses(frame: pd.DataFrame, resolution_ms: int, z_thr: float = 6.0, activity_thr: float = 3.0) -> pd.DataFrame:
    window = max(100, int(15 * 60 * 1000 / resolution_ms))
    z_flow = robust_z(frame.signed_notional.to_numpy(), window=window)
    z_activity = robust_z(np.log1p(frame.notional.to_numpy()), window=window)
    mask = (np.abs(z_flow) >= z_thr) & (z_activity >= activity_thr)
    idx = np.flatnonzero(mask)
    refractory = max(1, int(1000 / resolution_ms))
    kept = []; last = -10**18
    for i in idx:
        if i - last < refractory:
            if kept and abs(z_flow[i]) > abs(z_flow[kept[-1]]): kept[-1] = i; last = i
            continue
        kept.append(i); last = i
    out = frame.iloc[kept][["t_ms", "signed_notional", "notional", "return"]].copy()
    out["idx"] = kept; out["z_flow"] = z_flow[kept]; out["z_activity"] = z_activity[kept]
    out["direction"] = np.sign(out.z_flow).astype(int)
    return out.reset_index(drop=True)


def response_curve(source_imp: pd.DataFrame, target: pd.DataFrame, resolution_ms: int):
    lags_bins = np.maximum(1, np.ceil(LAGS_MS / resolution_ms).astype(int))
    ret = target["return"].to_numpy(float); signed = target["signed_notional"].to_numpy(float)
    cr = np.r_[0.0, np.cumsum(ret)]; cf = np.r_[0.0, np.cumsum(signed)]
    rr = []; ff = []; dirs = source_imp.direction.to_numpy(float); indices = source_imp.idx.to_numpy(int)
    for h in lags_bins:
        vr = []; vf = []
        for i, d in zip(indices, dirs):
            j = min(len(ret), i + h + 1)
            if i + 1 >= len(ret): continue
            vr.append(d * (cr[j] - cr[i + 1])); vf.append(d * (cf[j] - cf[i + 1]))
        rr.append(np.nanmean(vr) if vr else np.nan); ff.append(np.nanmean(vf) if vf else np.nan)
    return np.asarray(rr), np.asarray(ff)


def circular_null(source_imp: pd.DataFrame, target: pd.DataFrame, resolution_ms: int, n_perm: int = 200):
    lags_bins = np.maximum(1, np.ceil(LAGS_MS / resolution_ms).astype(int))
    ret = target["return"].to_numpy(float); signed = target["signed_notional"].to_numpy(float); n = len(ret)
    dirs = source_imp.direction.to_numpy(float); base_idx = source_imp.idx.to_numpy(int)
    nr = np.zeros((n_perm, len(lags_bins))); nf = np.zeros((n_perm, len(lags_bins)))
    for p in range(n_perm):
        shift = int(RNG.integers(max(10, n // 20), max(11, n - n // 20)))
        idxs = (base_idx + shift) % n
        for k, h in enumerate(lags_bins):
            vr = []; vf = []
            for i, d in zip(idxs, dirs):
                js = (np.arange(1, h + 1) + i) % n
                vr.append(d * ret[js].sum()); vf.append(d * signed[js].sum())
            nr[p, k] = np.mean(vr) if vr else np.nan; nf[p, k] = np.mean(vf) if vf else np.nan
    return nr, nf


def cross_asset_analysis(frames, impulses, date, res):
    rows = []; syms = sorted(frames); gain = np.zeros((len(syms), len(syms))); all_p = []; temp = []
    for a, src in enumerate(syms):
        for b, tgt in enumerate(syms):
            imp = impulses[src]
            if len(imp) < 8:
                obs_r = np.full(len(LAGS_MS), np.nan); obs_f = np.full(len(LAGS_MS), np.nan)
                p_r = np.ones(len(LAGS_MS)); p_f = np.ones(len(LAGS_MS)); nf = None
            else:
                obs_r, obs_f = response_curve(imp, frames[tgt], res); nr, nf = circular_null(imp, frames[tgt], res, 200)
                p_r = (1 + (np.abs(nr) >= np.abs(obs_r)).sum(axis=0)) / (len(nr) + 1)
                p_f = (1 + (np.abs(nf) >= np.abs(obs_f)).sum(axis=0)) / (len(nf) + 1)
            for k, lag in enumerate(LAGS_MS):
                temp.append({"date": date, "resolution_ms": res, "source": src, "target": tgt, "lag_ms": int(lag), "n_impulses": int(len(imp)), "response_return": float(obs_r[k]) if np.isfinite(obs_r[k]) else None, "response_signed_notional": float(obs_f[k]) if np.isfinite(obs_f[k]) else None, "p_return": float(p_r[k]), "p_flow": float(p_f[k])})
                all_p.extend([p_r[k], p_f[k]])
            k1 = int(np.argmin(np.abs(LAGS_MS - 1000)))
            if len(imp) >= 8 and nf is not None:
                excess = max(0.0, abs(obs_f[k1]) - np.nanmedian(np.abs(nf[:, k1])))
                gain[b, a] = excess / (np.nanmedian(np.abs(imp.signed_notional.to_numpy())) + 1e-12)
    qvals = bh_fdr(np.asarray(all_p)); qi = 0
    for rec in temp:
        rec["q_return"] = float(qvals[qi]); rec["q_flow"] = float(qvals[qi + 1]); qi += 2
        rec["significant"] = bool(rec["q_return"] < 0.05 or rec["q_flow"] < 0.05); rows.append(rec)
    return pd.DataFrame(rows), gain


def spectral_radius(m):
    return float(np.max(np.abs(np.linalg.eigvals(np.nan_to_num(m))))) if np.size(m) else float("nan")


def graph_geometry(gain, syms):
    w = np.maximum(gain, 0); sym = 0.5 * (w + w.T); n = len(syms)
    if n < 2 or np.all(sym <= 0): return {"symbols": syms, "gain": w.tolist(), "spectral_radius": spectral_radius(w), "lambda2": None, "mean_forman": None}
    adj = sym / (sym.max() + 1e-12); deg = adj.sum(axis=1); lap = np.diag(deg) - adj; eig = np.sort(np.linalg.eigvalsh(lap)); curv = []
    for i in range(n):
        for j in range(i + 1, n):
            wij = adj[i, j]
            if wij <= 0: continue
            term = 2.0
            for k in range(n):
                if k != j and adj[i, k] > 0: term -= math.sqrt(wij / adj[i, k])
                if k != i and adj[j, k] > 0: term -= math.sqrt(wij / adj[j, k])
            curv.append(term)
    return {"symbols": syms, "gain": w.tolist(), "spectral_radius": spectral_radius(w), "lambda2": float(eig[1]) if len(eig) > 1 else None, "laplacian_eigenvalues": eig.tolist(), "mean_forman": float(np.mean(curv)) if curv else None, "edge_forman": curv}


def build_cascades(impulses, significant_edges, max_lag_ms=2000):
    nodes = []
    for sym, df in impulses.items():
        for r in df.itertuples(): nodes.append({"sym": sym, "t": int(r.t_ms), "dir": int(r.direction), "z": float(r.z_flow)})
    nodes.sort(key=lambda x: x["t"]); n = len(nodes); parents = list(range(n)); depth = [0] * n; edges = []
    def find(x):
        while parents[x] != x: parents[x] = parents[parents[x]]; x = parents[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parents[rb] = ra
    recent = deque()
    for j, node in enumerate(nodes):
        while recent and node["t"] - nodes[recent[0]]["t"] > max_lag_ms: recent.popleft()
        candidates = [i for i in recent if nodes[i]["sym"] != node["sym"] and (nodes[i]["sym"], node["sym"]) in significant_edges and nodes[i]["dir"] == node["dir"]]
        if candidates:
            i = max(candidates, key=lambda x: (nodes[x]["t"], abs(nodes[x]["z"])))
            edges.append((i, j, node["t"] - nodes[i]["t"])); depth[j] = depth[i] + 1; union(i, j)
        recent.append(j)
    comps = defaultdict(list)
    for i in range(n): comps[find(i)].append(i)
    summaries = []
    for idxs in comps.values():
        if len(idxs) < 2: continue
        ts = [nodes[i]["t"] for i in idxs]; sy = sorted(set(nodes[i]["sym"] for i in idxs))
        summaries.append({"size": len(idxs), "duration_ms": max(ts) - min(ts), "assets": sy, "n_assets": len(sy), "max_depth": max(depth[i] for i in idxs), "start_ms": min(ts), "end_ms": max(ts)})
    summaries.sort(key=lambda x: (x["size"], x["max_depth"]), reverse=True)
    return {"n_nodes": n, "n_edges": len(edges), "n_cascades": len(summaries), "top_cascades": summaries[:50]}


def qf_microstructure():
    qpath = RAW / "qf_quotes.csv"; tpath = RAW / "qf_trades.csv"
    qmeta = download(QF_BASE + "/quotes.csv", qpath); tmeta = download(QF_BASE + "/trades.csv", tpath)
    q = pd.read_csv(qpath).sort_values("transaction_time").drop_duplicates("update_id")
    tr = pd.read_csv(tpath).sort_values("transact_time").drop_duplicates("agg_trade_id")
    q["mid"] = (q.best_bid_price + q.best_ask_price) / 2; q["spread"] = q.best_ask_price - q.best_bid_price
    q["imbalance"] = (q.best_bid_qty - q.best_ask_qty) / (q.best_bid_qty + q.best_ask_qty + 1e-12)
    q["depth"] = q.best_bid_qty + q.best_ask_qty; q["latency_ms"] = q.event_time - q.transaction_time
    tr["sign"] = np.where(tr.is_buyer_maker.astype(str).str.lower().eq("true"), -1.0, 1.0); tr["notional"] = tr.price * tr.quantity; tr["signed_notional"] = tr.sign * tr.notional
    start = max(int(q.transaction_time.min()), int(tr.transact_time.min())); end = min(int(q.transaction_time.max()), int(tr.transact_time.max()))
    grid = pd.DataFrame({"t_ms": np.arange(start, end + 1, dtype=np.int64)})
    qs = pd.merge_asof(grid, q.rename(columns={"transaction_time": "t_ms"}).sort_values("t_ms"), on="t_ms", direction="backward")
    b = (tr.transact_time.to_numpy(np.int64) - start).clip(0, len(grid) - 1)
    qs["trade_signed_notional"] = np.bincount(b, weights=tr.signed_notional, minlength=len(grid)); qs["trade_count"] = np.bincount(b, minlength=len(grid))
    qs["mid_return"] = np.r_[0.0, np.diff(np.log(qs.mid.to_numpy()))]; qs["d_depth"] = np.r_[0.0, np.diff(qs.depth.to_numpy())]; qs["d_imbalance"] = np.r_[0.0, np.diff(qs.imbalance.to_numpy())]
    z_depth = robust_z(qs.d_depth.to_numpy(), min(len(qs), 60000)); z_imb = robust_z(qs.d_imbalance.to_numpy(), min(len(qs), 60000)); z_flow = robust_z(qs.trade_signed_notional.to_numpy(), min(len(qs), 60000))
    idx = np.flatnonzero((np.abs(z_depth) >= 6) | (np.abs(z_imb) >= 6) | (np.abs(z_flow) >= 6)); kept = []; last = -999999
    for i in idx:
        if i - last >= 100: kept.append(i); last = i
    horizons = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000]; rows = []
    for h in horizons:
        vals = []; disp = []
        for i in kept:
            if i + h >= len(qs): continue
            d = np.sign(z_flow[i] if abs(z_flow[i]) >= 6 else (-z_depth[i] if abs(z_depth[i]) >= 6 else z_imb[i]))
            vals.append(d * (math.log(qs.mid.iloc[i + h]) - math.log(qs.mid.iloc[i]))); disp.append(abs(qs.imbalance.iloc[i + h] - qs.imbalance.iloc[i]))
        rows.append({"lag_ms": h, "signed_mid_response": float(np.mean(vals)) if vals else None, "imbalance_displacement": float(np.mean(disp)) if disp else None, "n": len(vals)})
    recovery = []; base_scale = np.nanmedian(np.abs(qs.d_imbalance.to_numpy())) + 1e-12
    for i in kept:
        level = qs.imbalance.iloc[i]; pre = qs.imbalance.iloc[max(0, i - 100)]
        for h in range(1, min(5000, len(qs) - i)):
            if abs(qs.imbalance.iloc[i + h] - pre) <= max(base_scale, 0.1 * abs(level - pre)): recovery.append(h); break
    baseline_rate = float(qs.trade_count.mean()); post_counts = [qs.trade_count.iloc[i + 1:i + 1001].sum() for i in kept if i + 1000 < len(qs)]
    branching = max(0.0, (float(np.mean(post_counts)) - 1000 * baseline_rate) / (float(np.mean(post_counts)) + 1e-12)) if post_counts else None
    latency = q.latency_ms.to_numpy(float); pd.DataFrame(rows).to_csv(OUT / "qf_bbo_response.csv", index=False)
    return {"source": {"quotes": qmeta, "trades": tmeta}, "quote_rows": int(len(q)), "trade_rows": int(len(tr)), "start_ms": start, "end_ms": end, "duration_s": (end - start) / 1000, "median_quote_interarrival_ms": float(np.median(np.diff(q.transaction_time))), "p99_quote_interarrival_ms": float(np.quantile(np.diff(q.transaction_time), .99)), "median_trade_interarrival_ms": float(np.median(np.diff(tr.transact_time))), "spread_summary": {"median": float(q.spread.median()), "p95": float(q.spread.quantile(.95)), "max": float(q.spread.max())}, "latency_ms": {"median": float(np.median(latency)), "p95": float(np.quantile(latency, .95)), "p99": float(np.quantile(latency, .99)), "max": float(np.max(latency))}, "n_liquidity_or_flow_shocks": len(kept), "response_curve": rows, "recovery_ms": {"n": len(recovery), "median": float(np.median(recovery)) if recovery else None, "p90": float(np.quantile(recovery, .9)) if recovery else None}, "branching_proxy_1s": branching, "raw_correlations": {"imbalance_next_100ms_return": float(pd.Series(qs.imbalance).corr(pd.Series(qs.mid_return).rolling(100).sum().shift(-100))), "signed_flow_next_100ms_return": float(pd.Series(qs.trade_signed_notional).corr(pd.Series(qs.mid_return).rolling(100).sum().shift(-100)))}}


def try_tardis_l2():
    url = "https://datasets.tardis.dev/v1/binance-futures/incremental_book_L2/2024/01/01/BTCUSDT.csv.gz"; path = RAW / "tardis_BTCUSDT_L2.csv.gz"
    try: meta = download(url, path, timeout=1200, retries=2)
    except Exception as exc: return {"status": "download_failed", "error": repr(exc), "url": url}
    bids = {}; asks = {}; rows = 0; snapshots = 0; first_t = None; last_t = None; sampled = []
    with gzip.open(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            rows += 1; t = int(r["timestamp"]); first_t = t if first_t is None else first_t; last_t = t
            if str(r["is_snapshot"]).lower() == "true": snapshots += 1
            book = bids if r["side"].lower() == "bid" else asks; p = float(r["price"]); a = float(r["amount"])
            if a == 0: book.pop(p, None)
            else: book[p] = a
            if rows % 5000 == 0 and bids and asks:
                bb = max(bids); ba = min(asks); bd = sum(v for p, v in bids.items() if p >= bb - 20); ad = sum(v for p, v in asks.items() if p <= ba + 20)
                sampled.append((t, bb, ba, bd, ad, len(bids), len(asks)))
    df = pd.DataFrame(sampled, columns=["timestamp_us", "best_bid", "best_ask", "bid_depth_20", "ask_depth_20", "n_bid_levels", "n_ask_levels"])
    if not df.empty:
        df["spread"] = df.best_ask - df.best_bid; df["imbalance_20"] = (df.bid_depth_20 - df.ask_depth_20) / (df.bid_depth_20 + df.ask_depth_20 + 1e-12); df.to_csv(OUT / "tardis_l2_sampled_state.csv", index=False)
    return {"status": "ok", "source": meta, "rows": rows, "snapshot_rows": snapshots, "first_timestamp_us": first_t, "last_timestamp_us": last_t, "duration_s": (last_t - first_t) / 1e6 if last_t and first_t else None, "sampled_states": len(df), "median_spread": float(df.spread.median()) if not df.empty else None, "median_abs_imbalance": float(df.imbalance_20.abs().median()) if not df.empty else None}


def bybit_research():
    manifest = []; all_results = {}; all_response = []; all_impulses = []; geometry = []; cascades = []
    for date in BYBIT_DATES:
        rawdfs = {}
        for sym in BYBIT_SYMBOLS:
            url = f"https://public.bybit.com/trading/{sym}/{sym}{date}.csv.gz"; path = RAW / f"{sym}{date}.csv.gz"
            try:
                meta = download(url, path, timeout=1200, retries=3); df = parse_bybit(path, sym, date); manifest.append(meta | {"symbol": sym, "date": date, "rows": len(df)}); rawdfs[sym] = df
            except Exception as exc: manifest.append({"url": url, "symbol": sym, "date": date, "error": repr(exc)})
        if len(rawdfs) < 2: all_results[date] = {"status": "insufficient_symbols", "available": sorted(rawdfs)}; continue
        date_result = {"symbols": {}, "resolutions": {}}
        for sym, df in rawdfs.items():
            dt = np.diff(df.t_ms.to_numpy(np.int64)); date_result["symbols"][sym] = {"rows": len(df), "start_ms": int(df.t_ms.min()), "end_ms": int(df.t_ms.max()), "duration_s": (df.t_ms.max() - df.t_ms.min()) / 1000, "median_interarrival_ms": float(np.median(dt)), "p99_interarrival_ms": float(np.quantile(dt, .99)), "total_notional": float(df.notional.sum()), "buy_share_notional": float(df.loc[df.sign > 0, "notional"].sum() / df.notional.sum())}
        for res in RESOLUTIONS_MS:
            frames = {sym: aggregate_trades(df, res) for sym, df in rawdfs.items()}; imps = {sym: pick_impulses(fr, res) for sym, fr in frames.items()}
            for sym, x in imps.items(): y = x.copy(); y["symbol"] = sym; y["date"] = date; y["resolution_ms"] = res; all_impulses.append(y)
            resp, gain = cross_asset_analysis(frames, imps, date, res); all_response.append(resp); syms = sorted(frames); geom = graph_geometry(gain, syms) | {"date": date, "resolution_ms": res}; geometry.append(geom)
            sig = set((r.source, r.target) for r in resp.itertuples() if r.significant and r.lag_ms <= 1000 and r.source != r.target); cas = build_cascades(imps, sig) | {"date": date, "resolution_ms": res, "significant_edges": [list(x) for x in sorted(sig)]}; cascades.append(cas)
            X = np.column_stack([frames[s].signed_notional.to_numpy() for s in syms]); X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-12); eig = np.linalg.eigvalsh(np.nan_to_num(np.corrcoef(X, rowvar=False))); coherence = float(eig[-1] / max(1e-12, eig.sum()))
            date_result["resolutions"][str(res)] = {"bins": len(next(iter(frames.values()))), "impulses": {s: len(imps[s]) for s in syms}, "coherence": coherence, "spectral_radius": geom["spectral_radius"], "mean_forman": geom["mean_forman"], "n_significant_directed_edges": len(sig), "cascades": cas["n_cascades"], "largest_cascade": cas["top_cascades"][0] if cas["top_cascades"] else None}
        all_results[date] = date_result
    if all_response: pd.concat(all_response, ignore_index=True).to_csv(OUT / "cross_asset_response.csv", index=False)
    if all_impulses: pd.concat(all_impulses, ignore_index=True).to_csv(OUT / "impulses.csv", index=False)
    pd.DataFrame([{k: v for k, v in g.items() if k not in ["gain", "edge_forman", "laplacian_eigenvalues", "symbols"]} for g in geometry]).to_csv(OUT / "geometry_summary.csv", index=False)
    (OUT / "geometry_full.json").write_text(json.dumps(geometry, indent=2)); (OUT / "cascades.json").write_text(json.dumps(cascades, indent=2))
    return {"manifest": manifest, "results": all_results, "geometry": geometry, "cascades": cascades}


def write_report(result):
    qf = result["qf_bbo"]; by = result["bybit"]; lines = ["# Phase M1 empirical microfield report", "", "Status: **real-data exploratory execution**. No synthetic observations enter the estimates.", "", "## Data actually processed", f"- Binance USD-M BTCUSDT BBO updates: {qf.get('quote_rows')} rows; aggregate trades: {qf.get('trade_rows')} rows; duration: {qf.get('duration_s')} s."]
    for m in by.get("manifest", []):
        lines.append(f"- FAILED {m['symbol']} {m['date']}: {m['error']}" if "error" in m else f"- Bybit {m['symbol']} {m['date']}: {m['rows']:,} native trades; compressed bytes {m['bytes']:,}; SHA-256 `{m['sha256']}`.")
    td = result.get("tardis_l2", {}); lines += [f"- Tardis L2: {td.get('status')}; rows={td.get('rows')}; sampled states={td.get('sampled_states')}.", "", "## BBO / liquidity field", f"- Median quote interarrival: {qf.get('median_quote_interarrival_ms')} ms; median trade interarrival: {qf.get('median_trade_interarrival_ms')} ms.", f"- Detected liquidity/order-flow shocks: {qf.get('n_liquidity_or_flow_shocks')}.", f"- Imbalance recovery median: {qf.get('recovery_ms', {}).get('median')} ms; p90: {qf.get('recovery_ms', {}).get('p90')} ms.", f"- One-second branching proxy: {qf.get('branching_proxy_1s')}.", "", "## Multi-asset excitation field"]
    for date, d in by.get("results", {}).items():
        lines.append(f"### {date}")
        if d.get("status"): lines.append(f"- {d}"); continue
        for res, rr in d.get("resolutions", {}).items(): lines.append(f"- {res} ms: impulses={rr['impulses']}; coherence={rr['coherence']:.4f}; gain spectral radius={rr['spectral_radius']:.4f}; significant directed edges={rr['n_significant_directed_edges']}; cascades={rr['cascades']}; largest={rr['largest_cascade']}.")
    lines += ["", "## Interpretation boundary", "P0 is a native trade or quote update. P1 is a robust local outlier in signed flow, activity, depth, or imbalance. P2 requires a directed response that survives circular-shift null calibration and FDR control. A temporal succession alone is never labelled causal.", "", "The report does not claim discovery of a universal market quantum. It tests whether localized events form reproducible, finite-lifetime, amplifying cascades at millisecond-to-second scales."]
    (OUT / "MICROFIELD_EMPIRICAL_REPORT.md").write_text("\n".join(lines))


def main():
    manifest = {"seed": SEED, "resolutions_ms": RESOLUTIONS_MS, "lags_ms": LAGS_MS.tolist(), "bybit_symbols": BYBIT_SYMBOLS, "bybit_dates": BYBIT_DATES, "started_utc": pd.Timestamp.utcnow().isoformat()}
    qf = qf_microstructure(); by = bybit_research(); td = try_tardis_l2(); result = {"run_manifest": manifest, "qf_bbo": qf, "bybit": by, "tardis_l2": td}; result["run_manifest"]["finished_utc"] = pd.Timestamp.utcnow().isoformat()
    (OUT / "RESULTS.json").write_text(json.dumps(result, indent=2, default=str)); write_report(result)
    prov = [qf["source"]["quotes"] | {"dataset": "qf_quotes"}, qf["source"]["trades"] | {"dataset": "qf_trades"}] + [m | {"dataset": "bybit_trades"} for m in by["manifest"]]
    if td.get("source"): prov.append(td["source"] | {"dataset": "tardis_l2"})
    (OUT / "PROVENANCE.json").write_text(json.dumps(prov, indent=2, default=str)); (OUT / "DONE.txt").write_text("ok\n")
    print(json.dumps({"status": "ok", "files": [p.name for p in OUT.iterdir()]}, indent=2))


if __name__ == "__main__": main()
