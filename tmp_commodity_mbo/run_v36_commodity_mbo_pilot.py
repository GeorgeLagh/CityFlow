#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("MM_V36_OUT", "tmp_commodity_mbo_results"))
OUT.mkdir(parents=True, exist_ok=True)

DATASET = "GLBX.MDP3"
SYMBOLS = [
    "CL.v.0",
    "NG.v.0",
    "GC.v.0",
    "HG.v.0",
    "ZC.v.0",
    "ZS.v.0",
    "LE.v.0",
    "HE.v.0",
]
START = os.environ.get("MM_V36_START", "2026-07-14T14:30:00Z")
END = os.environ.get("MM_V36_END", "2026-07-14T14:31:00Z")
MAX_COST_USD = float(os.environ.get("MM_V36_MAX_COST_USD", "5.00"))
RAW_PATH = OUT / "GLBX_MDP3_8COMMODITIES_1MIN.mbo.dbn.zst"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name: str, payload: Any) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def status_payload(status: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "status": status,
        "dataset": DATASET,
        "symbols": SYMBOLS,
        "start": START,
        "end": END,
        "max_cost_usd": MAX_COST_USD,
        **kwargs,
    }


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.reset_index()
    for candidate in ("ts_recv", "index"):
        if candidate in x.columns:
            x[candidate] = pd.to_datetime(x[candidate], utc=True, errors="coerce")
    if "ts_event" in x.columns:
        x["ts_event"] = pd.to_datetime(x["ts_event"], utc=True, errors="coerce")
    if "symbol" not in x.columns:
        x["symbol"] = x.get("instrument_id", "unknown").astype(str)
    return x


def event_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["symbol", "action", "side"] if c in df.columns]
    if not cols:
        return pd.DataFrame([{"records": len(df)}])
    g = df.groupby(cols, dropna=False).size().rename("records").reset_index()
    if "size" in df.columns:
        s = df.groupby(cols, dropna=False)["size"].sum().rename("size_sum").reset_index()
        g = g.merge(s, on=cols, how="left")
    return g.sort_values(cols).reset_index(drop=True)


def timestamp_audit(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_event" not in df.columns or "ts_recv" not in df.columns:
        return pd.DataFrame()
    delta = (df["ts_recv"] - df["ts_event"]).dt.total_seconds() * 1e9
    tmp = pd.DataFrame({"symbol": df["symbol"], "delta_ns": delta})
    rows = []
    for symbol, g in tmp.groupby("symbol"):
        a = g["delta_ns"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if not len(a):
            continue
        rows.append({
            "symbol": symbol,
            "n": int(len(a)),
            "median_ns": float(np.median(a)),
            "p01_ns": float(np.quantile(a, 0.01)),
            "p99_ns": float(np.quantile(a, 0.99)),
            "negative_fraction": float(np.mean(a < 0)),
        })
    return pd.DataFrame(rows)


def lifecycle_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"symbol", "order_id", "action", "ts_event"}
    if not required.issubset(df.columns):
        return pd.DataFrame(), pd.DataFrame()
    x = df[df["order_id"].fillna(0).astype("int64") != 0].copy()
    if x.empty:
        return pd.DataFrame(), pd.DataFrame()
    sort_cols = [c for c in ["symbol", "order_id", "ts_event", "sequence"] if c in x.columns]
    x = x.sort_values(sort_cols, kind="stable")
    rows = []
    for (symbol, order_id), g in x.groupby(["symbol", "order_id"], sort=False):
        actions = g["action"].astype(str).tolist()
        if "A" not in actions:
            continue
        add = g[g["action"].astype(str) == "A"].iloc[0]
        after = g[g["ts_event"] >= add["ts_event"]]
        terminal = after[after["action"].astype(str).isin(["C", "F"])]
        if "size" in after.columns:
            terminal = pd.concat([
                terminal,
                after[(after["action"].astype(str) == "M") & (after["size"].fillna(1) == 0)],
            ]).sort_values("ts_event", kind="stable").drop_duplicates()
        if terminal.empty:
            continue
        end = terminal.iloc[0]
        lifetime_ns = int((end["ts_event"] - add["ts_event"]).total_seconds() * 1e9)
        if lifetime_ns < 0:
            continue
        rows.append({
            "symbol": symbol,
            "order_id": int(order_id),
            "add_ts": add["ts_event"],
            "terminal_ts": end["ts_event"],
            "terminal_action": str(end["action"]),
            "side": str(add.get("side", "")),
            "price": float(add.get("price", np.nan)),
            "initial_size": float(add.get("size", np.nan)),
            "lifetime_ns": lifetime_ns,
            "event_count": int(len(after[after["ts_event"] <= end["ts_event"]])),
        })
    life = pd.DataFrame(rows)
    if life.empty:
        return life, pd.DataFrame()
    summary = life.groupby(["symbol", "terminal_action"]).agg(
        completed_orders=("order_id", "count"),
        median_lifetime_ns=("lifetime_ns", "median"),
        p10_lifetime_ns=("lifetime_ns", lambda s: s.quantile(0.10)),
        p90_lifetime_ns=("lifetime_ns", lambda s: s.quantile(0.90)),
        median_event_count=("event_count", "median"),
    ).reset_index()
    return life, summary


def multiscale_activity(df: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "ts_event", "action", "side"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    x = df.dropna(subset=["ts_event"]).copy()
    x["signed_size"] = x.get("size", 1.0).astype(float) * x["side"].map({"B": 1.0, "A": -1.0}).fillna(0.0)
    outputs = []
    for freq in ["1ms", "10ms", "100ms"]:
        for symbol, g in x.groupby("symbol"):
            z = g.set_index("ts_event")
            base = z.resample(freq).agg(
                event_count=("action", "size"),
                signed_size=("signed_size", "sum"),
            )
            for action in ["A", "C", "M", "T", "F", "R"]:
                base[f"action_{action}"] = (z["action"].astype(str) == action).resample(freq).sum()
            base = base.reset_index()
            base.insert(0, "symbol", symbol)
            base.insert(1, "resolution", freq)
            outputs.append(base)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def cross_asset_common_mode(activity: pd.DataFrame) -> dict[str, Any]:
    if activity.empty:
        return {"status": "unavailable"}
    x = activity[activity["resolution"] == "100ms"].copy()
    pivot = x.pivot_table(index="ts_event", columns="symbol", values="event_count", aggfunc="sum", fill_value=0)
    if pivot.shape[0] < 5 or pivot.shape[1] < 2:
        return {"status": "insufficient", "shape": list(pivot.shape)}
    a = np.log1p(pivot.to_numpy(float))
    a = a - a.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    energy = s * s
    explained = energy / energy.sum() if energy.sum() else np.zeros_like(energy)
    return {
        "status": "candidate_only",
        "symbols": list(pivot.columns),
        "n_bins": int(len(pivot)),
        "first_component_explained_fraction": float(explained[0]) if len(explained) else None,
        "first_loading": {str(sym): float(v) for sym, v in zip(pivot.columns, vt[0])} if len(vt) else {},
        "interpretation": "descriptive event-activity common mode; not a transmission operator",
    }


def main() -> int:
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if not key:
        write_json("PREFLIGHT.json", status_payload(
            "blocked_missing_databento_api_key",
            message="Add repository secret DATABENTO_API_KEY (32-character key beginning db-) and rerun workflow.",
            charged_usd=0.0,
        ))
        (OUT / "README.txt").write_text(
            "The pilot was launched but no Databento key was available. No data request was made and no charge occurred.\n",
            encoding="utf-8",
        )
        return 0

    try:
        import databento as db

        client = db.Historical(key)
        schemas = client.metadata.list_schemas(DATASET)
        dataset_range = client.metadata.get_dataset_range(DATASET)
        conditions = client.metadata.get_dataset_condition(DATASET, start_date=START[:10], end_date=END[:10])
        query = dict(
            dataset=DATASET,
            symbols=SYMBOLS,
            schema="mbo",
            stype_in="continuous",
            start=START,
            end=END,
        )
        cost = float(client.metadata.get_cost(**query))
        count = int(client.metadata.get_record_count(**query))
        billable_size = int(client.metadata.get_billable_size(**query))
        preflight = status_payload(
            "cost_estimated",
            schemas=schemas,
            dataset_range=dataset_range,
            dataset_condition=conditions,
            estimated_cost_usd=cost,
            estimated_records=count,
            estimated_billable_bytes=billable_size,
            charged_usd=0.0,
        )
        write_json("PREFLIGHT.json", preflight)
        if cost > MAX_COST_USD:
            write_json("BLOCKED_COST_CAP.json", status_payload(
                "blocked_cost_cap",
                estimated_cost_usd=cost,
                estimated_records=count,
                estimated_billable_bytes=billable_size,
                charged_usd=0.0,
                message="Preflight succeeded, but automatic download was not started because estimated cost exceeds cap.",
            ))
            return 0

        store = client.timeseries.get_range(**query, stype_out="raw_symbol", path=RAW_PATH)
        df = normalize_frame(store.to_df())
        df.head(1000).to_csv(OUT / "MBO_HEAD_1000.csv", index=False)
        pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}).to_csv(
            OUT / "MBO_SCHEMA.csv", index=False
        )
        event_summary(df).to_csv(OUT / "EVENT_SUMMARY.csv", index=False)
        timestamp_audit(df).to_csv(OUT / "TIMESTAMP_AUDIT.csv", index=False)
        life, life_summary = lifecycle_audit(df)
        if not life.empty:
            life.to_csv(OUT / "ORDER_LIFECYCLES.csv.gz", index=False, compression="gzip")
        life_summary.to_csv(OUT / "ORDER_LIFECYCLE_SUMMARY.csv", index=False)
        activity = multiscale_activity(df)
        if not activity.empty:
            activity.to_csv(OUT / "MULTISCALE_ACTIVITY.csv.gz", index=False, compression="gzip")
        common = cross_asset_common_mode(activity)
        write_json("COMMON_MODE_AUDIT.json", common)

        symbol_counts = df.groupby("symbol").size().astype(int).to_dict()
        actions = df["action"].astype(str).value_counts().astype(int).to_dict() if "action" in df else {}
        summary = status_payload(
            "pilot_downloaded_and_parsed",
            estimated_cost_usd=cost,
            estimated_records=count,
            estimated_billable_bytes=billable_size,
            actual_records=int(len(df)),
            actual_symbols=sorted(map(str, df["symbol"].dropna().unique())),
            records_by_symbol={str(k): int(v) for k, v in symbol_counts.items()},
            actions={str(k): int(v) for k, v in actions.items()},
            raw_file=str(RAW_PATH),
            raw_bytes=RAW_PATH.stat().st_size,
            raw_sha256=sha256(RAW_PATH),
            completed_order_lifecycles=int(len(life)),
            semantic_status={
                "native_event_stream": "validated_for_one_minute",
                "not_price_only": True,
                "full_book_state": "not_claimed_without session-start/snapshot replay",
                "transmission_operator": "not_claimed",
                "continuous_contract_handling": "volume-ranked mapping retained as raw expiry symbol",
            },
        )
        write_json("PILOT_SUMMARY.json", summary)
        (OUT / "DONE.txt").write_text("ok\n", encoding="utf-8")
        return 0
    except Exception as exc:
        write_json("ERROR.json", status_payload(
            "failed",
            error=repr(exc),
            traceback=traceback.format_exc(),
        ))
        return 1


if __name__ == "__main__":
    sys.exit(main())
