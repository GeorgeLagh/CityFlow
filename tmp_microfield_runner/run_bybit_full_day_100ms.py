#!/usr/bin/env python3
from __future__ import annotations
import csv,gzip,json
from pathlib import Path
import numpy as np,pandas as pd
import run_research as rr

DATE='2024-03-05'; SYMBOLS=['BTCUSDT','ETHUSDT','SOLUSDT']; RES=100; N=864000; NPERM=199; SEED=20260710
LAGS=np.array([100,250,500,1000,2000,5000,10000],int); RNG=np.random.default_rng(SEED)
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/'tmp_microfield_results_bybit_fullday'; RAW=Path(__file__).resolve().parent/'raw'
OUT.mkdir(parents=True,exist_ok=True); RAW.mkdir(parents=True,exist_ok=True)

def rz(x):
 x=np.asarray(x,float); m=np.nanmedian(x); mad=np.nanmedian(np.abs(x-m)); s=1.4826*mad
 if not np.isfinite(s) or s<=1e-15:s=np.nanstd(x)+1e-15
 return (x-m)/s

def bh(p):
 p=np.asarray(p,float); n=len(p); o=np.argsort(p); x=p[o]*n/np.arange(1,n+1); x=np.minimum.accumulate(x[::-1])[::-1]; out=np.empty(n); out[o]=np.clip(x,0,1); return out

def boot(x,reps=499):
 if len(x)<2:return [None,None]
 vals=np.array([np.mean(RNG.choice(x,len(x),replace=True)) for _ in range(reps)]); return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def aggregate(path,sym):
 start=int(pd.Timestamp(DATE,tz='UTC').timestamp()*1000); count=np.zeros(N,np.int32); notional=np.zeros(N); signed=np.zeros(N); last=np.full(N,np.nan); rows=0
 with gzip.open(path,'rt',newline='') as f:
  rd=csv.DictReader(f); fields=rd.fieldnames or []; fm={c.lower():c for c in fields}; tk=next((fm[k] for k in ['timestamp','time','trade_time_ms','transact_time'] if k in fm),fields[0]); sk=next(fm[k] for k in ['side','direction'] if k in fm); pk=next(fm[k] for k in ['price','p'] if k in fm); qk=next(fm[k] for k in ['size','qty','quantity','amount'] if k in fm)
  for r in rd:
   try:
    t=rr.event_time_to_ms(r[tk]); b=(t-start)//RES
    if b<0:continue
    if b>=N:break
    p=float(r[pk]); q=float(r[qk]); sg=1.0 if str(r[sk]).lower().startswith('b') else -1.0; v=p*q
    count[b]+=1; notional[b]+=v; signed[b]+=sg*v; last[b]=p; rows+=1
   except Exception:continue
 last=pd.Series(last).ffill().bfill().to_numpy(); ret=np.r_[0.,np.diff(np.log(last))]
 return pd.DataFrame({'count':count,'notional':notional,'signed':signed,'last':last,'ret':ret}),rows

def detect(f):
 zf=rz(f.signed); za=rz(np.log1p(f.notional)); ids=np.flatnonzero((np.abs(zf)>=6)&(za>=3)); kept=[]
 for i in ids:
  if kept and i-kept[-1]<10:
   if abs(zf[i])>abs(zf[kept[-1]]):kept[-1]=int(i)
   continue
  kept.append(int(i))
 return pd.DataFrame({'idx':kept,'dir':np.sign(zf[kept]).astype(int),'z':zf[kept],'signed':f.signed.iloc[kept].to_numpy()})

manifest=[]; frames={}; rowcounts={}
for s in SYMBOLS:
 url=f'https://public.bybit.com/trading/{s}/{s}{DATE}.csv.gz'; p=RAW/f'{s}{DATE}.csv.gz'; meta=rr.download(url,p,timeout=1800,retries=3); f,nrows=aggregate(p,s); frames[s]=f; rowcounts[s]=nrows; manifest.append(meta|{'symbol':s,'date':DATE,'native_rows':nrows})
initial={s:detect(frames[s]) for s in SYMBOLS}; sets={s:set(initial[s].idx.astype(int)) for s in SYMBOLS}; exclusive={}
for s,d in initial.items():
 keep=[]
 for r in d.itertuples():
  common=any(any((r.idx+k) in sets[o] for k in [-1,0,1]) for o in SYMBOLS if o!=s)
  if not common:keep.append(r.Index)
 exclusive[s]=d.loc[keep].reset_index(drop=True)
# contemporaneous common-factor residuals
R=np.column_stack([frames[s].ret for s in SYMBOLS]); F=np.column_stack([frames[s].signed for s in SYMBOLS]); residual={}
for j,s in enumerate(SYMBOLS):
 oth=[k for k in range(3) if k!=j]; Xr=np.column_stack([np.ones(N),R[:,oth]]); Xf=np.column_stack([np.ones(N),F[:,oth]]); br=np.linalg.lstsq(Xr,R[:,j],rcond=None)[0]; bf=np.linalg.lstsq(Xf,F[:,j],rcond=None)[0]; residual[s]={'ret':R[:,j]-Xr@br,'flow':F[:,j]-Xf@bf}
records=[]; pall=[]
for source in SYMBOLS:
 imp=exclusive[source]; idx=imp.idx.to_numpy(int); dirs=imp.dir.to_numpy(float)
 for target in SYMBOLS:
  if source==target or len(idx)<20:continue
  ret=residual[target]['ret']; flow=residual[target]['flow']; cr=np.r_[0.,np.cumsum(ret)]; cf=np.r_[0.,np.cumsum(flow)]; obsr=[]; obsf=[]; evrs=[]; evfs=[]
  for lag in LAGS:
   h=int(np.ceil(lag/RES)); valid=idx+h+1<=N; ii=idx[valid]; dd=dirs[valid]; er=dd*(cr[ii+h+1]-cr[ii+1]); ef=dd*(cf[ii+h+1]-cf[ii+1]); evrs.append(er); evfs.append(ef); obsr.append(er.mean()); obsf.append(ef.mean())
  nr=np.zeros((NPERM,len(LAGS))); nf=np.zeros_like(nr)
  for p in range(NPERM):
   shift=int(RNG.integers(N//20,N-N//20)); ii=(idx+shift)%N
   for k,lag in enumerate(LAGS):
    h=int(np.ceil(lag/RES)); off=np.arange(1,h+1); js=(ii[:,None]+off[None,:])%N; nr[p,k]=np.mean(dirs*ret[js].sum(axis=1)); nf[p,k]=np.mean(dirs*flow[js].sum(axis=1))
  pr=(1+(np.abs(nr)>=np.abs(obsr)).sum(0))/(NPERM+1); pf=(1+(np.abs(nf)>=np.abs(obsf)).sum(0))/(NPERM+1)
  for k,lag in enumerate(LAGS):
   rci=boot(evrs[k]); fci=boot(evfs[k]); rec={'source':source,'target':target,'lag_ms':int(lag),'n_initial':len(initial[source]),'n_exclusive':len(imp),'response_resid_return':float(obsr[k]),'response_resid_flow':float(obsf[k]),'p_return':float(pr[k]),'p_flow':float(pf[k]),'return_ci95':rci,'flow_ci95':fci}; records.append(rec); pall.extend([pr[k],pf[k]])
q=bh(pall); qi=0
for r in records:
 r['q_return']=float(q[qi]);r['q_flow']=float(q[qi+1]);qi+=2;r['p2_return']=bool(r['q_return']<.05 and not(r['return_ci95'][0]<=0<=r['return_ci95'][1]));r['p2_flow']=bool(r['q_flow']<.05 and not(r['flow_ci95'][0]<=0<=r['flow_ci95'][1]))
df=pd.DataFrame(records);df.to_csv(OUT/'FULLDAY_RESPONSE.csv',index=False)
# Hourly stability of each detected channel at 500 ms.
hourly=[]
for r in df[(df.lag_ms==500)&df.p2_return].itertuples():
 imp=exclusive[r.source]; ret=residual[r.target]['ret']; h=5
 for hour in range(24):
  z=imp[(imp.idx>=hour*36000)&(imp.idx<(hour+1)*36000)]; idx=z.idx.to_numpy(int); dirs=z.dir.to_numpy(float); valid=idx+h<N; idx=idx[valid]; dirs=dirs[valid]
  val=float(np.mean(dirs*(np.add.reduceat(ret,np.ravel(np.column_stack([idx+1,idx+h+1])))[:0]))) if False else (float(np.mean([d*ret[i+1:i+h+1].sum() for i,d in zip(idx,dirs)])) if len(idx) else None)
  hourly.append({'source':r.source,'target':r.target,'hour_utc':hour,'n_impulses':len(idx),'response_500ms':val})
pd.DataFrame(hourly).to_csv(OUT/'HOURLY_STABILITY.csv',index=False)
summary={'date':DATE,'resolution_ms':RES,'native_rows':rowcounts,'bins':N,'initial_impulses':{s:len(initial[s]) for s in SYMBOLS},'exclusive_impulses':{s:len(exclusive[s]) for s in SYMBOLS},'tests':len(df)*2,'p2_return_rows':int(df.p2_return.sum()),'p2_flow_rows':int(df.p2_flow.sum()),'p2_channels':sorted(set((r.source,r.target) for r in df[df.p2_return].itertuples())),'provenance':manifest,'null_replicates':NPERM,'controls':['exclusive source impulses ±100ms','target residualized against other two assets','circular-shift null','BH-FDR','event bootstrap CI']}
(OUT/'FULLDAY_SUMMARY.json').write_text(json.dumps(summary,indent=2));(OUT/'DONE.txt').write_text('ok\n');print(json.dumps(summary,indent=2))
