from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, box
from shapely.ops import unary_union
import osmium

OUT = Path(os.environ.get('AMSTERDAM_OUT', 'amsterdam_empirical_inputs')).resolve()
RAW = OUT / 'raw'
DATA = OUT / 'data'
QUALITY = OUT / 'quality'
for p in (RAW, DATA, QUALITY):
    p.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({'User-Agent': 'Amsterdam-MVE-UTO-v6/6.0 research reproducibility run'})
BBOX = (4.728, 52.278, 5.016, 52.431)
EPOCHS = [2019, 2021, 2023, 2026]
SCALES = [100, 250, 500, 1000]
SOURCES: list[dict[str, Any]] = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(source_id: str, url: str, path: Path, status: str = 'downloaded') -> None:
    SOURCES.append({
        'source_id': source_id,
        'url': url,
        'status': status,
        'path': str(path.relative_to(OUT)) if path.exists() else str(path),
        'bytes': path.stat().st_size if path.exists() else None,
        'sha256': sha256(path) if path.exists() and path.is_file() else None,
        'retrieved_utc': pd.Timestamp.now(tz='UTC').isoformat(),
    })


def get_json(url: str, params: dict[str, Any] | None = None, retries: int = 6) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f'GET failed: {url}: {last}')


def download(url: str, path: Path, retries: int = 6) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with SESSION.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with part.open('wb') as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            part.replace(path)
            return path
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f'download failed: {url}: {last}')


def feature_collection(url: str, params: dict[str, Any] | None = None, max_pages: int = 10000) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    next_url = url
    next_params = params
    seen: set[str] = set()
    for _ in range(max_pages):
        marker = requests.Request('GET', next_url, params=next_params).prepare().url or next_url
        if marker in seen:
            raise RuntimeError(f'pagination cycle: {marker}')
        seen.add(marker)
        payload = get_json(next_url, next_params)
        features.extend(payload.get('features', []))
        nxt = next((x.get('href') for x in payload.get('links', []) if x.get('rel') == 'next'), None)
        if not nxt:
            break
        next_url, next_params = nxt, None
    else:
        raise RuntimeError(f'page ceiling reached for {url}')
    return features


def load_official_boundary() -> gpd.GeoDataFrame:
    url = 'https://api.data.amsterdam.nl/v1/gebieden/buurten/?_format=geojson'
    path = RAW / 'amsterdam_buurten.geojson'
    download(url, path)
    record('AMS-BUURTEN-2026-07-20', url, path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    g = gpd.GeoDataFrame.from_features(payload['features'])
    if g.crs is None:
        bounds = g.total_bounds
        g = g.set_crs('EPSG:28992' if abs(bounds[0]) > 1000 else 'EPSG:4326')
    g = g.to_crs('EPSG:28992')
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    union = unary_union(g.geometry)
    out = gpd.GeoDataFrame([{
        'carrier_id': 'amsterdam_official_buurten_union_2026_07_20',
        'boundary_version': 'amsterdam_gebieden_buurten_2026-07-20',
        'source_id': 'AMS-BUURTEN-2026-07-20',
        'status': 'official_versioned_reference',
    }], geometry=[union], crs='EPSG:28992')
    out.to_parquet(DATA / 'official_boundary.parquet', index=False)
    g.to_parquet(DATA / 'official_buurten.parquet', index=False)
    return out


def build_grids(boundary: gpd.GeoDataFrame) -> dict[int, gpd.GeoDataFrame]:
    geom = boundary.geometry.iloc[0]
    minx, miny, maxx, maxy = geom.bounds
    grids: dict[int, gpd.GeoDataFrame] = {}
    summary = []
    for scale in SCALES:
        x0 = math.floor(minx / scale) * scale
        y0 = math.floor(miny / scale) * scale
        recs = []
        seq = 0
        for x in np.arange(x0, maxx + scale, scale):
            for y in np.arange(y0, maxy + scale, scale):
                cell = box(float(x), float(y), float(x + scale), float(y + scale))
                active = cell.intersection(geom)
                if active.is_empty or active.area <= 0:
                    continue
                seq += 1
                recs.append({
                    'grid_id': f'g{scale}_{seq:06d}',
                    'carrier_id': 'amsterdam_official_buurten_union_2026_07_20',
                    'scale_m': scale,
                    'cell_area_m2': float(cell.area),
                    'active_area_m2': float(active.area),
                    'active_fraction': float(active.area / cell.area),
                    'geometry': cell,
                })
        grid = gpd.GeoDataFrame(recs, crs='EPSG:28992')
        grid.to_parquet(DATA / f'grid_{scale}m.parquet', index=False)
        grids[scale] = grid
        summary.append({'scale_m': scale, 'cell_count': len(grid)})
    (QUALITY / 'grid_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return grids


def load_crowdmonitor() -> Path:
    url = 'https://api.data.amsterdam.nl/v1/crowdmonitor/v1/passanten?_format=csv'
    path = RAW / 'crowdmonitor_passanten.csv'
    download(url, path)
    record('AMS-CROWDMONITOR-V1', url, path)
    return path


def load_bag(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    url = 'https://api.pdok.nl/kadaster/bag/ogc/v2/collections/pand/items'
    params = {'bbox': ','.join(map(str, BBOX)), 'limit': 1000, 'f': 'json'}
    feats = feature_collection(url, params)
    raw_path = RAW / 'bag_pand_amsterdam.geojson'
    raw_path.write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}, ensure_ascii=False), encoding='utf-8')
    record('BAG-PAND-2026-07-20', url, raw_path)
    g = gpd.GeoDataFrame.from_features(feats, crs='EPSG:4326').to_crs('EPSG:28992')
    if 'identificatie' in g.columns:
        g = g.drop_duplicates('identificatie', keep='last')
    status_col = next((c for c in ('status', 'pandstatus') if c in g.columns), None)
    if status_col:
        active = g[status_col].astype(str).str.lower().str.contains('in gebruik|bouw gestart|bouwvergunning', regex=True, na=False)
        g = g[active].copy()
    year_col = next((c for c in ('bouwjaar', 'oorspronkelijkBouwjaar', 'oorspronkelijk_bouwjaar') if c in g.columns), None)
    if year_col is None:
        raise RuntimeError('BAG construction year column absent')
    g['construction_year'] = pd.to_numeric(g[year_col], errors='coerce')
    g = g[g.geometry.notna() & ~g.geometry.is_empty & g['construction_year'].notna()].copy()
    g = g[g.intersects(boundary.geometry.iloc[0])].copy()
    g.to_parquet(DATA / 'bag_current_active.parquet', index=False)
    return g


def chunked_polygon_aggregate(objects: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, epoch: int) -> pd.DataFrame:
    objects = objects[['construction_year', 'geometry']].copy()
    objects['geometry'] = objects.geometry.make_valid()
    objects = objects[objects.geometry.notna() & ~objects.geometry.is_empty].copy()
    built_area: dict[str, float] = {}
    counts: dict[str, int] = {}
    lookup = grid.geometry
    for start in range(0, len(objects), 10000):
        chunk = objects.iloc[start:start + 10000].copy()
        cent = chunk.copy()
        cent.geometry = cent.geometry.centroid
        joined_count = gpd.sjoin(cent[['geometry']], grid[['grid_id', 'geometry']], predicate='within', how='inner')
        for gid, n in joined_count.groupby('grid_id').size().items():
            counts[gid] = counts.get(gid, 0) + int(n)
        joined = gpd.sjoin(chunk[['geometry']], grid[['grid_id', 'geometry']], predicate='intersects', how='inner')
        if joined.empty:
            continue
        cell_geom = gpd.GeoSeries(lookup.loc[joined['index_right']].to_numpy(), index=joined.index, crs=grid.crs)
        areas = joined.geometry.intersection(cell_geom).area
        sums = pd.DataFrame({'grid_id': joined['grid_id'].to_numpy(), 'area': areas.to_numpy()}).groupby('grid_id')['area'].sum()
        for gid, value in sums.items():
            built_area[gid] = built_area.get(gid, 0.0) + float(value)
    out = grid[['grid_id', 'carrier_id', 'scale_m', 'active_area_m2']].copy()
    out['morph_built_fraction'] = (out['grid_id'].map(built_area).fillna(0.0) / out['active_area_m2']).clip(0, 1)
    out['morph_building_count'] = out['grid_id'].map(counts).fillna(0).astype(int)
    out['observation_epoch'] = f'{epoch}-01-01'
    return out


class RoadHandler(osmium.SimpleHandler):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, Any]] = []

    def way(self, w: osmium.osm.Way) -> None:
        if 'highway' not in w.tags:
            return
        try:
            coords = [(n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            return
        if len(coords) < 2:
            return
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        if max(xs) < BBOX[0] or min(xs) > BBOX[2] or max(ys) < BBOX[1] or min(ys) > BBOX[3]:
            return
        self.rows.append({'osm_id': int(w.id), 'highway': str(w.tags.get('highway')), 'geometry': LineString(coords)})


def parse_osm(path: Path, boundary: gpd.GeoDataFrame, epoch: int) -> gpd.GeoDataFrame:
    h = RoadHandler()
    h.apply_file(str(path), locations=True)
    g = gpd.GeoDataFrame(h.rows, crs='EPSG:4326').to_crs('EPSG:28992')
    g = g[g.geometry.notna() & ~g.geometry.is_empty & g.intersects(boundary.geometry.iloc[0])].copy()
    g['observation_epoch'] = f'{epoch}-01-01'
    g.to_parquet(DATA / f'osm_network_{epoch}.parquet', index=False)
    return g


def chunked_line_aggregate(lines: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, epoch: int) -> pd.DataFrame:
    lengths: dict[str, float] = {}
    for start in range(0, len(lines), 20000):
        chunk = lines.iloc[start:start + 20000][['geometry']].copy()
        joined = gpd.sjoin(chunk, grid[['grid_id', 'geometry']], predicate='intersects', how='inner')
        if joined.empty:
            continue
        cell_geom = gpd.GeoSeries(grid.geometry.loc[joined['index_right']].to_numpy(), index=joined.index, crs=grid.crs)
        vals = joined.geometry.intersection(cell_geom).length
        sums = pd.DataFrame({'grid_id': joined['grid_id'].to_numpy(), 'length': vals.to_numpy()}).groupby('grid_id')['length'].sum()
        for gid, value in sums.items():
            lengths[gid] = lengths.get(gid, 0.0) + float(value)
    out = grid[['grid_id', 'carrier_id', 'scale_m', 'active_area_m2']].copy()
    out['network_length_density'] = out['grid_id'].map(lengths).fillna(0.0) / (out['active_area_m2'] / 1_000_000.0)
    out['observation_epoch'] = f'{epoch}-01-01'
    return out


def build_structural_theta(boundary: gpd.GeoDataFrame, grids: dict[int, gpd.GeoDataFrame]) -> Path:
    bag = load_bag(boundary)
    records: list[pd.DataFrame] = []
    for epoch in EPOCHS:
        subset = bag[bag['construction_year'] <= epoch].copy()
        for scale, grid in grids.items():
            agg = chunked_polygon_aggregate(subset, grid, epoch)
            for field, unit in [('morph_built_fraction', 'fraction'), ('morph_building_count', 'count')]:
                z = agg[['grid_id', 'carrier_id', 'scale_m', 'observation_epoch', field]].rename(columns={field: 'value'})
                z['field_id'] = field
                z['unit'] = unit
                z['source_id'] = 'BAG-PAND'
                z['source_family_id'] = 'BAG-PAND'
                z['temporal_semantics'] = 'surviving_stock_proxy'
                z['processing_version'] = 'mve-v6-github-1'
                records.append(z)
    osm_urls = {
        2019: 'https://download.geofabrik.de/europe/netherlands/noord-holland-190101.osm.pbf',
        2021: 'https://download.geofabrik.de/europe/netherlands/noord-holland-210101.osm.pbf',
        2023: 'https://download.geofabrik.de/europe/netherlands/noord-holland-230101.osm.pbf',
        2026: 'https://download.geofabrik.de/europe/netherlands/noord-holland-260101.osm.pbf',
    }
    for epoch, url in osm_urls.items():
        pbf = RAW / f'noord-holland-{str(epoch)[-2:]}0101.osm.pbf'
        download(url, pbf)
        record(f'OSM-{epoch}', url, pbf)
        roads = parse_osm(pbf, boundary, epoch)
        pbf.unlink(missing_ok=True)
        for scale, grid in grids.items():
            agg = chunked_line_aggregate(roads, grid, epoch)
            z = agg[['grid_id', 'carrier_id', 'scale_m', 'observation_epoch', 'network_length_density']].rename(columns={'network_length_density': 'value'})
            z['field_id'] = 'network_length_density'
            z['unit'] = 'm/km2_active_area'
            z['source_id'] = f'OSM-{epoch}'
            z['source_family_id'] = 'OSM'
            z['temporal_semantics'] = 'mapping_snapshot'
            z['processing_version'] = 'mve-v6-github-1'
            records.append(z)
    theta = pd.concat(records, ignore_index=True)
    theta = theta[['grid_id', 'carrier_id', 'scale_m', 'observation_epoch', 'field_id', 'value', 'unit', 'source_id', 'source_family_id', 'temporal_semantics', 'processing_version']]
    path = DATA / 'theta_structural_v6.parquet'
    theta.to_parquet(path, index=False)
    return path


def odata_rows(table: str, filter_expr: str) -> list[dict[str, Any]]:
    base = f'https://opendata.cbs.nl/ODataApi/OData/{table}/TypedDataSet'
    rows: list[dict[str, Any]] = []
    url: str | None = base
    params: dict[str, Any] | None = {'$filter': filter_expr, '$format': 'json'}
    while url:
        payload = get_json(url, params)
        rows.extend(payload.get('value', []))
        url = payload.get('odata.nextLink') or payload.get('@odata.nextLink')
        params = None
    return rows


def load_commute_od() -> Path:
    frames = []
    expr = "WoonregioS eq 'GM0363' or WerkregioS eq 'GM0363'"
    for table in ('83628NED', '85481NED'):
        try:
            rows = odata_rows(table, expr)
        except Exception as exc:
            SOURCES.append({'source_id': f'CBS-{table}', 'url': f'https://opendata.cbs.nl/ODataApi/OData/{table}', 'status': 'failed', 'error': str(exc), 'retrieved_utc': pd.Timestamp.now(tz='UTC').isoformat()})
            continue
        if not rows:
            continue
        d = pd.DataFrame(rows)
        d['table_id'] = table
        frames.append(d)
        raw = RAW / f'cbs_{table}_amsterdam.json'
        raw.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
        record(f'CBS-{table}', f'https://opendata.cbs.nl/ODataApi/OData/{table}', raw)
    if not frames:
        raise RuntimeError('CBS commute OD returned no rows')
    d = pd.concat(frames, ignore_index=True, sort=False)
    origin = next((c for c in ('WoonregioS', 'origin', 'origin_code') if c in d.columns), None)
    dest = next((c for c in ('WerkregioS', 'destination', 'destination_code') if c in d.columns), None)
    period = next((c for c in ('Perioden', 'period', 'year') if c in d.columns), None)
    trips = next((c for c in d.columns if re.search(r'BanenVanWerknemers|Werknemersbanen|Banen', c, re.I)), None)
    if not all((origin, dest, period, trips)):
        raise RuntimeError(f'CBS OD columns unresolved: {d.columns.tolist()}')
    out = pd.DataFrame({
        'origin': d[origin].astype(str),
        'destination': d[dest].astype(str),
        'period_raw': d[period].astype(str),
        'trips': pd.to_numeric(d[trips], errors='coerce'),
        'table_id': d['table_id'].astype(str),
    })
    out['year'] = pd.to_numeric(out['period_raw'].str.extract(r'(20\d{2})')[0], errors='coerce')
    out['period'] = pd.to_datetime(out['year'].astype('Int64').astype(str) + '-12-01', errors='coerce')
    out = out[out['trips'].notna() & out['period'].notna()].copy()
    out['purpose'] = 'commute_residence_to_work'
    out['temporal_semantics'] = 'annual_december_snapshot'
    path = DATA / 'cbs_commute_od_amsterdam.parquet'
    out.to_parquet(path, index=False)
    return path


def main() -> None:
    boundary = load_official_boundary()
    grids = build_grids(boundary)
    crowd = load_crowdmonitor()
    theta = build_structural_theta(boundary, grids)
    od = load_commute_od()
    pd.DataFrame(SOURCES).to_csv(QUALITY / 'source_provenance.csv', index=False)
    theta_df = pd.read_parquet(theta)
    od_df = pd.read_parquet(od)
    report = {
        'status': 'complete',
        'official_boundary': True,
        'crowdmonitor': str(crowd.relative_to(OUT)),
        'theta': str(theta.relative_to(OUT)),
        'od': str(od.relative_to(OUT)),
        'theta_rows': int(len(theta_df)),
        'theta_fields': sorted(theta_df['field_id'].unique().tolist()),
        'theta_epochs': sorted(theta_df['observation_epoch'].unique().tolist()),
        'theta_scales': sorted(int(x) for x in theta_df['scale_m'].unique().tolist()),
        'od_rows': int(len(od_df)),
        'scientific_limits': [
            'BAG epochs are surviving-stock proxies from current active geometry filtered by construction year.',
            'OSM epochs are mapping snapshots.',
            'Crowdmonitor is point flow, not origin-destination mobility.',
            'CBS OD is annual commute-only context.',
        ],
    }
    (QUALITY / 'collection_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
