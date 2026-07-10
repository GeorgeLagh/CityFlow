#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run_research as rr

SEED=20260710
RNG=np.random.default_rng(SEED)
N_PERM=499
HORIZONS=np.array([1,5,10,25,50,100,250,500,1000,2000,5000],dtype=int)
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'tmp_microfield_results_qf_null'
RAW=Path(__file__).resolve().parent/'raw'
OUT.mkdir(parents=True,exist_ok=True); RAW.mkdir(parents=True,exist_ok=True)


def rz(x):
    x=np.asarray(x,float); med=np.nanmedian(x); mad=np.nanmedian(np.abs(x-med)); s=1.4826*mad
    if not np.isfinite(s) or s<=1e-15: s=np.nanstd(x)+1e-15
    return (x-med)/s


def qbin(x,q=5):
    s=pd.Series(x)
    try: return pd.qcut(s.rank(method='first'),q,labels=False,duplicates='drop').fillna(0).astype(int).to_numpy()
    except Exception: return np.zeros(len(s),dtype=int)


def bh(p):
    p=np.asarray(p,float); n=len(p); order=np.argsort(p); x=p[order]*n/np.arange(1,n+1); x=np.minimum.accumulate(x[::-1])[::-1]; out=np.empty(n); out[order]=np.clip(x,0,1); return out


def boot_ci(x,reps=999):
    x=np.asarray(x,float)
    if len(x)<2:return [None,None]
    m=np.empty(reps)
    for i in range(reps):m[i]=np.mean(RNG.choice(x,size=len(x),replace=True))
    return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]

qpath=RAW/'qf_quotes.csv'; tpath=RAW/'qf_trades.csv'
qmeta=rr.download(rr.QF_BASE+'/quotes.csv',qpath); tmeta=rr.download(rr.QF_BASE+'/trades.csv',tpath)
q=pd.read_csv(qpath).sort_values('transaction_time').drop_duplicates('update_id')
tr=pd.read_csv(tpath).sort_values('transact_time').drop_duplicates('agg_trade_id')
q['mid']=(q.best_bid_price+q.best_ask_price)/2; q['spread']=q.best_ask_price-q.best_bid_price
q['imbalance']=(q.best_bid_qty-q.best_ask_qty)/(q.best_bid_qty+q.best_ask_qty+1e-12); q['depth']=q.best_bid_qty+q.best_ask_qty
tr['sign']=np.where(tr.is_buyer_maker.astype(str).str.lower().eq('true'),-1.0,1.0); tr['signed_notional']=tr.sign*tr.price*tr.quantity
start=max(int(q.transaction_time.min()),int(tr.transact_time.min())); end=min(int(q.transaction_time.max()),int(tr.transact_time.max()))
grid=pd.DataFrame({'t_ms':np.arange(start,end+1,dtype=np.int64)})
x=pd.merge_asof(grid,q.rename(columns={'transaction_time':'t_ms'}).sort_values('t_ms'),on='t_ms',direction='backward')
b=(tr.transact_time.to_numpy(np.int64)-start).clip(0,len(x)-1)
x['signed_flow']=np.bincount(b,weights=tr.signed_notional,minlength=len(x)); x['trade_count']=np.bincount(b,minlength=len(x))
x['mid_ret']=np.r_[0.0,np.diff(np.log(x.mid.to_numpy()))]; x['d_depth']=np.r_[0.0,np.diff(x.depth.to_numpy())]; x['d_imbalance']=np.r_[0.0,np.diff(x.imbalance.to_numpy())]
z_flow=rz(x.signed_flow); z_depth=rz(x.d_depth); z_imb=rz(x.d_imbalance)
score=np.maximum.reduce([np.abs(z_flow),np.abs(z_depth),np.abs(z_imb)])
idx=np.flatnonzero(score>=6.0); kept=[]
for i in idx:
    if kept and i-kept[-1]<100:
        if score[i]>score[kept[-1]]:kept[-1]=int(i)
        continue
    kept.append(int(i))
events=np.array(kept,dtype=int)
kind=np.argmax(np.column_stack([np.abs(z_flow[events]),np.abs(z_depth[events]),np.abs(z_imb[events])]),axis=1)
labels=np.array(['trade_flow','top_depth','bbo_imbalance'])[kind]
direction=np.where(kind==0,np.sign(z_flow[events]),np.where(kind==1,-np.sign(z_depth[events]),np.sign(z_imb[events]))).astype(int)
# State matching: spread, depth, absolute imbalance and trailing 100ms activity.
activity=pd.Series(x.trade_count).rolling(100,min_periods=1).sum().to_numpy()
state=np.column_stack([qbin(x.spread,4),qbin(x.depth,5),qbin(np.abs(x.imbalance),5),qbin(activity,5)])
code=state[:,0]+10*state[:,1]+100*state[:,2]+1000*state[:,3]
maxh=int(HORIZONS.max()); eligible=np.arange(maxh,len(x)-maxh)
blocked=np.zeros(len(x),dtype=bool)
for i in events: blocked[max(0,i-maxh):min(len(x),i+maxh+1)]=True
pool_by_code={}
for c in np.unique(code[eligible]):
    z=eligible[(code[eligible]==c)&(~blocked[eligible])]
    if len(z):pool_by_code[int(c)]=z
fallback=eligible[~blocked[eligible]]
logmid=np.log(x.mid.to_numpy(float)); imbalance=x.imbalance.to_numpy(float)
obs_ret=[]; obs_disp=[]; event_ret=[]; event_disp=[]
for h in HORIZONS:
    er=direction*(logmid[events+h]-logmid[events]); ed=np.abs(imbalance[events+h]-imbalance[events])
    event_ret.append(er); event_disp.append(ed); obs_ret.append(float(np.mean(er))); obs_disp.append(float(np.mean(ed)))
null_ret=np.zeros((N_PERM,len(HORIZONS))); null_disp=np.zeros_like(null_ret)
for p in range(N_PERM):
    controls=np.empty(len(events),dtype=int)
    for j,i in enumerate(events):
        pool=pool_by_code.get(int(code[i]),fallback)
        controls[j]=int(RNG.choice(pool))
    for k,h in enumerate(HORIZONS):
        null_ret[p,k]=np.mean(direction*(logmid[controls+h]-logmid[controls]))
        null_disp[p,k]=np.mean(np.abs(imbalance[controls+h]-imbalance[controls]))
obs_ret=np.asarray(obs_ret); obs_disp=np.asarray(obs_disp)
p_ret=(1+(np.abs(null_ret)>=np.abs(obs_ret)).sum(axis=0))/(N_PERM+1)
p_disp=(1+(null_disp>=obs_disp).sum(axis=0))/(N_PERM+1)
q_ret=bh(p_ret); q_disp=bh(p_disp)
rows=[]
for k,h in enumerate(HORIZONS):
    rows.append({'lag_ms':int(h),'n_events':int(len(events)),'signed_mid_response':float(obs_ret[k]),'signed_mid_response_bp':float(obs_ret[k]*1e4),'return_ci95':boot_ci(event_ret[k]),'matched_null_return_median':float(np.median(null_ret[:,k])),'p_return':float(p_ret[k]),'q_return':float(q_ret[k]),'imbalance_displacement':float(obs_disp[k]),'imbalance_ci95':boot_ci(event_disp[k]),'matched_null_displacement_median':float(np.median(null_disp[:,k])),'p_displacement':float(p_disp[k]),'q_displacement':float(q_disp[k]),'response_survives_gate':bool(q_ret[k]<.05 and not (boot_ci(event_ret[k],199)[0]<=0<=boot_ci(event_ret[k],199)[1]))})
pd.DataFrame(rows).to_csv(OUT/'QF_MATCHED_NULL_RESPONSE.csv',index=False)
event_table=pd.DataFrame({'idx':events,'t_ms':x.t_ms.iloc[events].to_numpy(np.int64),'kind':labels,'direction':direction,'score':score[events],'spread':x.spread.iloc[events].to_numpy(float),'depth':x.depth.iloc[events].to_numpy(float),'imbalance':x.imbalance.iloc[events].to_numpy(float),'signed_flow':x.signed_flow.iloc[events].to_numpy(float)})
event_table.to_csv(OUT/'QF_EVENTS.csv',index=False)
summary={'source':{'quotes':qmeta,'trades':tmeta},'seed':SEED,'null_replicates':N_PERM,'rows':{'quotes':int(len(q)),'trades':int(len(tr)),'milliseconds':int(len(x))},'events':{'total':int(len(events)),'by_kind':{k:int(v) for k,v in pd.Series(labels).value_counts().items()}},'matched_state_dimensions':['spread quartile','top depth quintile','absolute BBO imbalance quintile','trailing-100ms trade activity quintile'],'refractory_ms':100,'threshold_abs_robust_z':6.0,'response_gate':'BH-FDR q<0.05 plus bootstrap 95% CI excludes zero','significant_return_horizons_ms':[int(r['lag_ms']) for r in rows if r['response_survives_gate']],'significant_displacement_horizons_ms':[int(r['lag_ms']) for r in rows if r['q_displacement']<.05]}
(OUT/'QF_NULL_SUMMARY.json').write_text(json.dumps(summary,indent=2)); (OUT/'DONE.txt').write_text('ok\n'); print(json.dumps(summary,indent=2))
