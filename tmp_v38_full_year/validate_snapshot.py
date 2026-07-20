#!/usr/bin/env python3
import csv, json
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root / "SUMMARY_V38.json").read_text())
geometry = json.loads((root / "GEOMETRY_SUMMARY.json").read_text())
with (root / "STRICT_PROXY_DIRECTION_CANDIDATES.csv").open() as f:
    strict = list(csv.DictReader(f))

checks = {
    "raw_ticks": summary["raw_ticks"] == 118140732,
    "valid_days": summary["valid_days"] == 247,
    "events": summary["events"] == 5092,
    "daily_edges": summary["daily_edge_rows"] == 356672,
    "robust_cells": summary["operator_robust_same_sample"] == 56,
    "two_strict_proxy_directions": {r["source"] + "->" + r["target"] for r in strict} == {
        "coppercmdusd->brentcmdusd",
        "coppercmdusd->xagusd",
    },
    "activity_only": all(r["channel"] == "activity" and int(r["lag_s"]) == 5 for r in strict),
    "A7_quarantined": geometry["A7"] == "QUARANTINE",
    "canonical_fan_blocked": geometry["canonical_fan"] == "BLOCKED",
}
failed = [k for k, v in checks.items() if not v]
print(json.dumps({"status": "PASS" if not failed else "FAIL", "checks": checks}, indent=2))
if failed:
    raise SystemExit("failed checks: " + ", ".join(failed))
