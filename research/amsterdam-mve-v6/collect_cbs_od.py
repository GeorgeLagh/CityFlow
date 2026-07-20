from __future__ import annotations

import json
import re
from pathlib import Path

import cbsodata
import pandas as pd

OUT = Path('cbs_od_artifact')
OUT.mkdir(parents=True, exist_ok=True)


def dimension_key(table: str, dimension: str, title: str = 'Amsterdam') -> str:
    meta = pd.DataFrame(cbsodata.get_meta(table, dimension))
    title_col = next(c for c in ('Title', 'title') if c in meta.columns)
    key_col = next(c for c in ('Key', 'Identifier', 'key') if c in meta.columns)
    exact = meta[meta[title_col].astype(str).str.strip().eq(title)]
    if exact.empty:
        exact = meta[meta[title_col].astype(str).str.contains(title, case=False, na=False)]
    if exact.empty:
        raise RuntimeError(f'{table}/{dimension}: Amsterdam key absent')
    return str(exact.iloc[0][key_col])


def get_filtered(table: str, dimension: str, key: str) -> pd.DataFrame:
    rows = cbsodata.get_data(table, filters=f"{dimension} eq '{key}'")
    return pd.DataFrame(rows)


def normalize(table: str) -> pd.DataFrame:
    woon_key = dimension_key(table, 'WoonregioS')
    werk_key = dimension_key(table, 'WerkregioS')
    outgoing = get_filtered(table, 'WoonregioS', woon_key)
    incoming = get_filtered(table, 'WerkregioS', werk_key)
    d = pd.concat([outgoing, incoming], ignore_index=True, sort=False).drop_duplicates()
    if d.empty:
        raise RuntimeError(f'{table}: no Amsterdam rows after exact-key filtering')
    origin = next(c for c in ('WoonregioS', 'Woonregio') if c in d.columns)
    destination = next(c for c in ('WerkregioS', 'Werkregio') if c in d.columns)
    period = next(c for c in ('Perioden', 'Period') if c in d.columns)
    excluded = {origin, destination, period, 'ID'}
    candidates = []
    for c in d.columns:
        if c in excluded:
            continue
        values = pd.to_numeric(d[c], errors='coerce')
        score = int(values.notna().sum())
        if score and re.search(r'Banen|Werknemer|Jobs', c, re.I):
            candidates.append((score, c))
    if not candidates:
        for c in d.columns:
            if c in excluded:
                continue
            values = pd.to_numeric(d[c], errors='coerce')
            score = int(values.notna().sum())
            if score:
                candidates.append((score, c))
    if not candidates:
        raise RuntimeError(f'{table}: numeric trip measure absent; columns={d.columns.tolist()}')
    trip_col = max(candidates)[1]
    out = pd.DataFrame({
        'origin': d[origin].astype(str).str.strip(),
        'destination': d[destination].astype(str).str.strip(),
        'period_raw': d[period].astype(str),
        'trips': pd.to_numeric(d[trip_col], errors='coerce'),
        'table_id': table,
        'measure_column': trip_col,
    })
    out['year'] = pd.to_numeric(out['period_raw'].str.extract(r'(20\d{2})')[0], errors='coerce')
    out['period'] = pd.to_datetime(out['year'].astype('Int64').astype(str) + '-12-01', errors='coerce')
    out = out[out['trips'].notna() & out['period'].notna()].copy()
    out['purpose'] = 'commute_residence_to_work'
    out['temporal_semantics'] = 'annual_december_snapshot'
    return out


def main() -> None:
    frames = []
    diagnostics = []
    for table in ('83628NED', '85481NED'):
        try:
            d = normalize(table)
            frames.append(d)
            diagnostics.append({'table': table, 'status': 'pass', 'rows': len(d), 'years': sorted(d.year.astype(int).unique().tolist()), 'measure': sorted(d.measure_column.unique().tolist())})
        except Exception as exc:
            diagnostics.append({'table': table, 'status': 'fail', 'error': str(exc)})
    if not frames:
        raise RuntimeError(json.dumps(diagnostics, ensure_ascii=False))
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    path = OUT / 'cbs_commute_od_amsterdam.parquet'
    out.to_parquet(path, index=False)
    report = {'status': 'pass', 'rows': len(out), 'years': sorted(out.year.astype(int).unique().tolist()), 'tables': diagnostics, 'scope': 'commute-only OD'}
    (OUT / 'cbs_od_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
