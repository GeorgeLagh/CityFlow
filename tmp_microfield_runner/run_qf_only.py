#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import run_research as rr


def fast_robust_z(x, window=None):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-15:
        scale = np.nanstd(x) + 1e-15
    return (x - med) / scale

rr.robust_z = fast_robust_z
out = Path(__file__).resolve().parent.parent / "tmp_microfield_results_qf"
out.mkdir(parents=True, exist_ok=True)
rr.OUT = out
result = rr.qf_microstructure()
(out / "QF_RESULTS.json").write_text(json.dumps(result, indent=2, default=str))
lines = [
    "# Millisecond BBO and trade-field execution",
    "",
    f"Quotes: {result['quote_rows']}; trades: {result['trade_rows']}; duration: {result['duration_s']} s.",
    f"Median quote interarrival: {result['median_quote_interarrival_ms']} ms.",
    f"Median trade interarrival: {result['median_trade_interarrival_ms']} ms.",
    f"Detected shocks: {result['n_liquidity_or_flow_shocks']}.",
    f"Median imbalance recovery: {result['recovery_ms']['median']} ms; p90: {result['recovery_ms']['p90']} ms.",
    f"One-second branching proxy: {result['branching_proxy_1s']}.",
    f"Correlations: {result['raw_correlations']}.",
]
(out / "QF_REPORT.md").write_text("\n".join(lines))
(out / "DONE.txt").write_text("ok\n")
print(json.dumps({"status":"ok","out":str(out)}, indent=2))
