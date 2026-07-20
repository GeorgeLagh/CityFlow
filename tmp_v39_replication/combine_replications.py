#!/usr/bin/env python3
from pathlib import Path
import json
root=Path('replications');years={}
for p in root.rglob('SUMMARY_REPLICATION.json'):
    d=json.loads(p.read_text());years[int(d['year'])]=d
if set(years)!={2023,2025}:
    raise SystemExit(f'missing years: {years.keys()}')
out=[]
for target in ['brentcmdusd','xagusd']:
    rows=[next(x for x in years[y]['preregistered_directions'] if x['target']==target) for y in [2023,2025]]
    both=all(x['year_replication_pass'] for x in rows)
    out.append({'source':'coppercmdusd','target':target,'channel':'activity','lag_s':5,'year_2023_pass':rows[0]['year_replication_pass'],'year_2025_pass':rows[1]['year_replication_pass'],'replicates_both_years':both,'status':'ADVANCE_TO_CROSS_VENUE' if both else 'FALSIFIED_OR_UNSTABLE'})
summary={'years':[2023,2025],'frozen_protocol':True,'directions':out,'rule':'advance only when the same sign, channel, lag, operator dominance, half-year stability and circular placebo pass in both independent years'}
Path('v39_combined').mkdir(exist_ok=True);Path('v39_combined/REPLICATION_SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
