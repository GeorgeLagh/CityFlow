#!/usr/bin/env python3
from __future__ import annotations
import ast
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
src = ROOT / 'tmp_microfield_results_bybit_corrected' / 'CORRECTED_RESPONSE.csv'
out = ROOT / 'tmp_microfield_results_bybit_summary'
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(src)
for c in ['p2_return','p2_flow']:
    df[c] = df[c].astype(str).str.lower().eq('true')
for c in ['return_ci95','flow_ci95']:
    df[c] = df[c].apply(lambda x: ast.literal_eval(x) if isinstance(x,str) else x)
p2 = df[df.p2_return | df.p2_flow].copy()
p2.to_csv(out/'P2_ROWS.csv', index=False)
rows=[]
for (res,source,target),g in df.groupby(['resolution_ms','source','target']):
    gr=g[g.p2_return]; gf=g[g.p2_flow]
    rows.append({
        'resolution_ms':int(res),'source':source,'target':target,
        'n_exclusive_impulses':int(g.n_exclusive_impulses.iloc[0]),
        'p2_return_lag_count':int(len(gr)),'p2_return_lags_ms':','.join(map(str,gr.lag_ms.astype(int).tolist())),
        'p2_return_min_lag_ms':int(gr.lag_ms.min()) if len(gr) else None,
        'p2_return_max_lag_ms':int(gr.lag_ms.max()) if len(gr) else None,
        'max_abs_resid_return':float(gr.response_resid_return.abs().max()) if len(gr) else None,
        'p2_flow_lag_count':int(len(gf)),'p2_flow_lags_ms':','.join(map(str,gf.lag_ms.astype(int).tolist())),
        'min_q_return':float(g.q_return.min()),'min_q_flow':float(g.q_flow.min())
    })
edges=pd.DataFrame(rows)
edges.to_csv(out/'P2_UNIQUE_EDGES.csv', index=False)
persistent=[]
for (source,target),g in edges.groupby(['source','target']):
    live=g[g.p2_return_lag_count>0]
    persistent.append({'source':source,'target':target,'active_resolutions_ms':live.resolution_ms.astype(int).tolist(),'n_active_resolutions':int(len(live)),'total_p2_return_lags':int(live.p2_return_lag_count.sum()),'any_p2_flow':bool((g.p2_flow_lag_count>0).any())})
summary={
    'rows_total':int(len(df)),'p2_rows_total':int(len(p2)),
    'p2_return_rows':int(df.p2_return.sum()),'p2_flow_rows':int(df.p2_flow.sum()),
    'p2_by_resolution':{str(int(k)):{'return':int(v.p2_return.sum()),'flow':int(v.p2_flow.sum()),'tested_rows':int(len(v))} for k,v in df.groupby('resolution_ms')},
    'persistent_directed_edges':persistent,
    'strict_p2_definition':'exclusive source P1; target contemporaneously residualized against other assets; circular-shift null N=199; BH-FDR q<0.05; event-bootstrap 95% CI excludes zero',
    'field_interpretation':'P2 rows are horizon-specific directed responses. Repeated lags from one source-target pair are one propagation channel with a response curve, not separate causal edges.'
}
(out/'P2_SUMMARY.json').write_text(json.dumps(summary,indent=2))
(out/'DONE.txt').write_text('ok\n')
print(json.dumps(summary,indent=2))
