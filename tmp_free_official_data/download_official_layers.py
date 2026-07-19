#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup

UA = "MarketMetastabilityFreeData/3.6.1 (+research; contact via repository)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})
TIMEOUT = 90
RETRIES = 4


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "download"


def request(url: str, *, stream: bool = False) -> requests.Response:
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = SESSION.get(url, timeout=TIMEOUT, stream=stream, allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < RETRIES:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {RETRIES} attempts: {url}: {last}")


def download(url: str, dest: Path) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with request(url, stream=True) as r, tmp.open("wb") as out:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                out.write(chunk)
    tmp.replace(dest)
    return {
        "url": url,
        "resolved_url": r.url,
        "file": str(dest),
        "bytes": dest.stat().st_size,
        "sha256": sha256(dest),
        "retrieved_at_utc": now_utc(),
        "status": "ok",
    }


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def page_links(url: str) -> list[tuple[str, str]]:
    r = request(url)
    soup = BeautifulSoup(r.text, "html.parser")
    out: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(r.url, a["href"])
        text = " ".join(a.get_text(" ", strip=True).split())
        out.append((text, href))
    return out


def run_cftc(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    year_now = dt.datetime.now().year
    for year in range(2018, year_now + 1):
        url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
        dest = root / "cftc" / f"fut_disagg_txt_{year}.zip"
        try:
            rec = download(url, dest)
            with zipfile.ZipFile(dest) as zf:
                rec["zip_members"] = zf.namelist()
                rec["zip_test"] = zf.testzip()
            rec["dataset"] = "CFTC Disaggregated COT futures only"
            rec["year"] = year
            out.append(rec)
        except Exception as exc:  # noqa: BLE001
            out.append({"dataset": "CFTC", "year": year, "url": url, "status": "failed", "error": repr(exc)})
    return out


def run_wasde(root: Path) -> list[dict[str, Any]]:
    page = "https://www.usda.gov/historical-wasde-report-data-3"
    links = page_links(page)
    selected: list[tuple[str, str]] = []
    for text, href in links:
        lower = href.lower()
        if lower.endswith((".csv", ".zip")) and ("wasde" in lower or "oce" in lower or "commodity" in lower):
            selected.append((text, href))
    # Some USDA file links are content-addressed and do not contain WASDE in the URL.
    if len(selected) < 30:
        selected = [(t, h) for t, h in links if h.lower().endswith((".csv", ".zip"))]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, (text, url) in enumerate(selected):
        if url in seen:
            continue
        seen.add(url)
        parsed = urllib.parse.urlparse(url)
        name = Path(parsed.path).name or f"wasde_{idx:03d}.dat"
        if not Path(name).suffix:
            name += ".csv"
        dest = root / "usda_wasde" / safe_name(name)
        try:
            rec = download(url, dest)
            rec.update({"dataset": "USDA historical WASDE vintage", "link_text": text, "source_page": page})
            out.append(rec)
        except Exception as exc:  # noqa: BLE001
            out.append({"dataset": "USDA WASDE", "url": url, "link_text": text, "status": "failed", "error": repr(exc)})
    return out


def load_eia_manifest() -> Any:
    urls = [
        "https://api.eia.gov/bulk/manifest.txt",
        "http://api.eia.gov/bulk/manifest.txt",
    ]
    last: Exception | None = None
    for url in urls:
        try:
            return request(url).json()
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"unable to load EIA bulk manifest: {last}")


def flatten_records(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, list):
        for x in obj:
            yield from flatten_records(x)
    elif isinstance(obj, dict):
        if any(k in obj for k in ("accessURL", "accessUrl", "access_url")):
            yield obj
        for value in obj.values():
            if isinstance(value, (list, dict)):
                yield from flatten_records(value)


def run_eia(root: Path) -> list[dict[str, Any]]:
    manifest = load_eia_manifest()
    save_json(root / "eia" / "bulk_manifest.json", manifest)
    targets = {
        "petroleum",
        "natural gas",
        "short-term energy outlook",
        "crude oil imports",
        "total energy",
    }
    candidates: list[dict[str, Any]] = []
    for rec in flatten_records(manifest):
        title = str(rec.get("title") or rec.get("data_set") or rec.get("identifier") or "").strip()
        low = title.lower()
        if any(t == low or t in low for t in targets):
            candidates.append(rec)
    # Deduplicate by access URL and prefer exact target names.
    by_url: dict[str, dict[str, Any]] = {}
    for rec in candidates:
        url = rec.get("accessURL") or rec.get("accessUrl") or rec.get("access_url")
        if url:
            by_url[str(url)] = rec
    out: list[dict[str, Any]] = []
    for idx, (url, meta) in enumerate(sorted(by_url.items())):
        path_name = Path(urllib.parse.urlparse(url).path).name or f"eia_{idx:02d}.zip"
        dest = root / "eia" / safe_name(path_name)
        try:
            rec = download(url.replace("http://", "https://"), dest)
            rec.update({"dataset": "EIA bulk", "manifest_metadata": meta})
            if zipfile.is_zipfile(dest):
                with zipfile.ZipFile(dest) as zf:
                    rec["zip_members"] = zf.namelist()
                    rec["zip_test"] = zf.testzip()
            out.append(rec)
        except Exception as exc:  # noqa: BLE001
            out.append({"dataset": "EIA bulk", "url": url, "status": "failed", "error": repr(exc), "metadata": meta})
    return out


def run_world_bank(root: Path) -> list[dict[str, Any]]:
    page = "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/world-bank-commodities-price-data-the-pink-sheet"
    links = page_links(page)
    wanted = [(t, h) for t, h in links if "CMO-Historical-Data-" in (t + " " + h) and h.lower().endswith((".xlsx", ".xls"))]
    out: list[dict[str, Any]] = []
    for text, url in wanted:
        name = Path(urllib.parse.urlparse(url).path).name or safe_name(text) + ".xlsx"
        dest = root / "world_bank" / safe_name(name)
        try:
            rec = download(url, dest)
            rec.update({"dataset": "World Bank Pink Sheet historical data", "link_text": text, "source_page": page})
            out.append(rec)
        except Exception as exc:  # noqa: BLE001
            out.append({"dataset": "World Bank Pink Sheet", "url": url, "status": "failed", "error": repr(exc)})
    return out


def extract_year(text: str) -> int | None:
    m = re.search(r"\b(20\d{2})\b", text)
    return int(m.group(1)) if m else None


def run_lme(root: Path) -> list[dict[str, Any]]:
    page = "https://www.lme.com/Market-data/Reports-and-data/Warehouse-and-stocks-reports/Warehouse-and-queue-data"
    links = page_links(page)
    wanted: list[tuple[str, str]] = []
    for text, href in links:
        year = extract_year(text + " " + href)
        if year is not None and year >= 2018 and "warehouse" in text.lower() and href.lower().endswith((".xlsx", ".xls")):
            wanted.append((text, href))
    out: list[dict[str, Any]] = []
    for idx, (text, url) in enumerate(wanted):
        name = Path(urllib.parse.urlparse(url).path).name or f"lme_warehouse_{idx:03d}.xlsx"
        dest = root / "lme" / safe_name(name)
        try:
            rec = download(url, dest)
            rec.update({"dataset": "LME warehouse company stocks and queue", "link_text": text, "source_page": page})
            out.append(rec)
        except Exception as exc:  # noqa: BLE001
            out.append({"dataset": "LME warehouse queue", "url": url, "link_text": text, "status": "failed", "error": repr(exc)})
    return out


def run_fred(root: Path) -> list[dict[str, Any]]:
    series = {
        "DTWEXBGS": "Nominal broad U.S. dollar index",
        "DFF": "Federal funds effective rate",
        "DGS10": "10-year Treasury constant maturity rate",
        "T10YIE": "10-year breakeven inflation rate",
        "VIXCLS": "CBOE volatility index",
        "CPIAUCSL": "Consumer price index",
        "PPIACO": "Producer price index, all commodities",
    }
    out: list[dict[str, Any]] = []
    for sid, title in series.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        dest = root / "fred" / f"{sid}.csv"
        try:
            rec = download(url, dest)
            rec.update({"dataset": "FRED macro control", "series_id": sid, "title": title})
            out.append(rec)
        except Exception as exc:  # noqa: BLE001
            out.append({"dataset": "FRED", "series_id": sid, "url": url, "status": "failed", "error": repr(exc)})
    return out


def moex_get(url: str) -> dict[str, Any]:
    return request(url).json()


def block_rows(payload: dict[str, Any], block: str) -> list[dict[str, Any]]:
    obj = payload.get(block, {})
    cols = obj.get("columns", [])
    return [dict(zip(cols, row)) for row in obj.get("data", [])]


def run_moex(root: Path) -> list[dict[str, Any]]:
    base = "https://iss.moex.com/iss/engines/futures/markets/forts"
    sec_url = f"{base}/securities.json?iss.meta=off&iss.only=securities"
    market_url = f"{base}/securities.json?iss.meta=off&iss.only=marketdata"
    records: list[dict[str, Any]] = []
    try:
        sec_payload = moex_get(sec_url)
        market_payload = moex_get(market_url)
        save_json(root / "moex" / "securities.json", sec_payload)
        save_json(root / "moex" / "marketdata.json", market_payload)
        securities = block_rows(sec_payload, "securities")
        pattern = re.compile(r"brent|oil|нефт|natural gas|газ|gold|золот|silver|сереб|copper|мед|soy|со[яи]|coffee|коф", re.I)
        selected = []
        for row in securities:
            hay = " ".join(str(row.get(k, "")) for k in ("SECID", "SHORTNAME", "SECNAME", "NAME", "ASSETCODE", "ASSETTYPE"))
            if pattern.search(hay):
                selected.append(row)
        save_json(root / "moex" / "selected_commodity_securities.json", selected)
        records.append({
            "dataset": "MOEX ISS derivatives metadata and delayed market data",
            "status": "ok",
            "securities_count": len(securities),
            "selected_count": len(selected),
            "retrieved_at_utc": now_utc(),
            "urls": [sec_url, market_url],
        })
        for row in selected[:80]:
            secid = str(row.get("SECID", "")).strip()
            if not secid:
                continue
            for block in ("trades", "orderbook"):
                url = f"{base}/securities/{urllib.parse.quote(secid)}/{block}.json?iss.meta=off"
                try:
                    payload = moex_get(url)
                    path = root / "moex" / block / f"{safe_name(secid)}.json"
                    save_json(path, payload)
                    records.append({
                        "dataset": f"MOEX ISS current delayed {block}",
                        "secid": secid,
                        "status": "ok",
                        "url": url,
                        "file": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                        "retrieved_at_utc": now_utc(),
                    })
                except Exception as exc:  # noqa: BLE001
                    records.append({"dataset": f"MOEX {block}", "secid": secid, "url": url, "status": "failed", "error": repr(exc)})
    except Exception as exc:  # noqa: BLE001
        records.append({"dataset": "MOEX", "status": "failed", "error": repr(exc)})
    return records


def run_small(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for name, fn in [
        ("cftc", run_cftc),
        ("wasde", run_wasde),
        ("eia", run_eia),
        ("world_bank", run_world_bank),
        ("lme", run_lme),
        ("fred", run_fred),
        ("moex", run_moex),
    ]:
        try:
            manifest.extend(fn(root))
        except Exception as exc:  # noqa: BLE001
            manifest.append({"dataset": name, "status": "failed_top_level", "error": repr(exc)})
    save_json(root / "MANIFEST.json", manifest)
    summary = {
        "retrieved_at_utc": now_utc(),
        "records": len(manifest),
        "ok": sum(r.get("status") == "ok" for r in manifest),
        "failed": sum(str(r.get("status", "")).startswith("failed") for r in manifest),
        "bytes_downloaded": sum(int(r.get("bytes", 0) or 0) for r in manifest if r.get("status") == "ok"),
        "source_classes": sorted({str(r.get("dataset")) for r in manifest}),
        "semantic_status": "official slow/medium/delayed control layers; not native CME/ICE MBO",
    }
    save_json(root / "SUMMARY.json", summary)
    return 0 if summary["ok"] else 1


def stream_filter_nass(category: str, root: Path) -> int:
    page = "https://www.nass.usda.gov/datasets/"
    links = page_links(page)
    pattern = re.compile(rf"qs\.{re.escape(category)}_\d{{8}}\.txt\.gz$", re.I)
    matches = [(t, h) for t, h in links if pattern.search(Path(urllib.parse.urlparse(h).path).name)]
    if not matches:
        raise RuntimeError(f"NASS bulk link not found for {category}")
    text, url = sorted(matches, key=lambda x: x[1])[-1]
    raw = root / "nass" / "raw" / Path(urllib.parse.urlparse(url).path).name
    rec = download(url, raw)
    commodities = {
        "crops": ["CORN", "SOYBEANS", "COFFEE"],
        "animals_products": ["CATTLE", "HOGS", "BEEF", "PORK"],
        "economics": ["CORN", "SOYBEANS", "COFFEE", "CATTLE", "HOGS", "BEEF", "PORK"],
        "environmental": ["CORN", "SOYBEANS", "COFFEE", "CATTLE", "HOGS"],
    }.get(category, [])
    target = root / "nass" / "filtered" / f"{category}_commodity_subset.txt.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    total = 0
    with gzip.open(raw, "rt", encoding="utf-8", errors="replace", newline="") as src, gzip.open(target, "wt", encoding="utf-8", newline="") as dst:
        header = src.readline()
        if not header:
            raise RuntimeError("empty NASS file")
        dst.write(header)
        upper_terms = [x.upper() for x in commodities]
        for line in src:
            total += 1
            upper = line.upper()
            if any(term in upper for term in upper_terms):
                dst.write(line)
                kept += 1
    result = {
        "dataset": "USDA NASS QuickStats bulk filtered",
        "category": category,
        "source_page": page,
        "source_link_text": text,
        "source_url": url,
        "raw_file": str(raw),
        "raw_bytes": raw.stat().st_size,
        "raw_sha256": rec["sha256"],
        "total_records_scanned": total,
        "kept_records": kept,
        "terms": commodities,
        "filtered_file": str(target),
        "filtered_bytes": target.stat().st_size,
        "filtered_sha256": sha256(target),
        "retrieved_at_utc": now_utc(),
        "status": "ok",
    }
    save_json(root / "nass" / f"{category}_SUMMARY.json", result)
    # Raw bulk is reproducible from the official URL and too large for artifact retention.
    raw.unlink(missing_ok=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["small", "nass"])
    ap.add_argument("--category", default="crops")
    ap.add_argument("--out", default="free_official_data")
    args = ap.parse_args()
    root = Path(args.out)
    if args.mode == "small":
        return run_small(root)
    return stream_filter_nass(args.category, root)


if __name__ == "__main__":
    sys.exit(main())
