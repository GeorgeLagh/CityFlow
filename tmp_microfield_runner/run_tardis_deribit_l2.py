#!/usr/bin/env python3
from __future__ import annotations
import csv
import gzip
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import run_research as rr

SEED=20260710
RNG=np.random.default_rng(SEED)
N_PERM=199
SAMPLE_US=100_000
WINDOW_US=30*60*1_000_000
HORIZONS_MS=np.array([100,250,500,1000,2000,5000,10000],dtype=int)
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'tmp_microfield_results_tardis_l2'
RAW=Path(__file__).resolve().parent/'raw'
OUT.mkdir(parents=True,exist_ok=True); RAW.mkdir(parents=True,exist_ok=True)
L2_URL='https://datasets.tardis.dev/v1/deribit/incremental_book_L2/2024/01/01/BTC-PERPETUAL.csv.gz'
TR_URL='https://datasets.tardis.dev/v1/deribit/trades/2024/01/01/BTC-PERPETUAL.csv.gz'

def rz(x):
    x=np.asarray(x,float); med=np.nanmedian(x); mad=np.nanmedian(np.abs(x-med)); s=1.4826*mad
    if not np.isfinite(s) or s<=1e-15:s=np.nanstd(x)+1e-15
    return (x-med)/s

def entropy(vals):
    a=np.asarray(vals,float); z=a.sum()
    if z<=0:return 0.0
    p=a/z; p=p[p>0]
    return float(-(p*np.log(p)).sum())

def qbin(x,q=5):
    try:return pd.qcut(pd.Series(x).rank(method='first'),q,labels=False,duplicates='drop').fillna(0).astype(int).to_numpy()
    except Exception:return np.zeros(len(x),dtype=int)

def bh(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p); x=p[o]*n/np.arange(1,n+1); x=np.minimum.accumulate(x[::-1])[::-1]; out=np.empty(n); out[o]=np.clip(x,0,1); return out

l2p=RAW/'tardis_deribit_BTC-PERPETUAL_L2_2024-01-01.csv.gz'; trp=RAW/'tardis_deribit_BTC-PERPETUAL_trades_2024-01-01.csv.gz'
try:
    l2meta=rr.download(L2_URL,l2p,timeout=1800,retries=3)
    trmeta=rr.download(TR_URL,trp,timeout=1200,retries=3)
except Exception as exc:
    (OUT/'DOWNLOAD_FAILURE.json').write_text(json.dumps({'status':'download_failed','error':repr(exc),'l2_url':L2_URL,'trades_url':TR_URL},indent=2)); (OUT/'DONE.txt').write_text('failed\n'); raise

bids={}; asks={}; samples=[]; first_ts=None; end_ts=None; current_bin=None; last_snapshot_group=None
agg={'signed_liq_delta':0.0,'removed_qty':0.0,'added_qty':0.0,'updates':0,'bid_delta':0.0,'ask_delta':0.0}
rows=0; snapshot_rows=0; crossed=0

def flush(bin_id):
    global agg
    if not bids or not asks:return
    bb=max(bids); ba=min(asks); mid=.5*(bb+ba)
    if bb>=ba:return
    bid50=sum(v for p,v in bids.items() if p>=mid*(1-0.0005)); ask50=sum(v for p,v in asks.items() if p<=mid*(1+0.0005))
    bid10=sum(v for p,v in bids.items() if p>=mid*(1-0.0001)); ask10=sum(v for p,v in asks.items() if p<=mid*(1+0.0001))
    topb=[bids[p] for p in sorted(bids,reverse=True)[:20]]; topa=[asks[p] for p in sorted(asks)[:20]]
    samples.append({'timestamp_us':int(bin_id*SAMPLE_US),'best_bid':bb,'best_ask':ba,'mid':mid,'spread':ba-bb,'bid_depth_10bp':bid10,'ask_depth_10bp':ask10,'imbalance_10bp':(bid10-ask10)/(bid10+ask10+1e-12),'bid_depth_50bp':bid50,'ask_depth_50bp':ask50,'imbalance_50bp':(bid50-ask50)/(bid50+ask50+1e-12),'bid_entropy_top20':entropy(topb),'ask_entropy_top20':entropy(topa),'n_bid_levels':len(bids),'n_ask_levels':len(asks),**agg})

with gzip.open(l2p,'rt',newline='') as f:
    reader=csv.DictReader(f)
    for r in reader:
        t=int(r['timestamp']); rows+=1
        if first_ts is None:
            first_ts=t; end_ts=first_ts+WINDOW_US
        if t>=end_ts:break
        is_snap=str(r['is_snapshot']).lower()=='true'
        if is_snap:
            snapshot_rows+=1
            if last_snapshot_group!=t:
                bids.clear(); asks.clear(); last_snapshot_group=t
        bin_id=t//SAMPLE_US
        if current_bin is None:current_bin=bin_id
        if bin_id!=current_bin:
            flush(current_bin)
            for missing in range(current_bin+1,bin_id):
                agg={'signed_liq_delta':0.0,'removed_qty':0.0,'added_qty':0.0,'updates':0,'bid_delta':0.0,'ask_delta':0.0}; flush(missing)
            agg={'signed_liq_delta':0.0,'removed_qty':0.0,'added_qty':0.0,'updates':0,'bid_delta':0.0,'ask_delta':0.0}; current_bin=bin_id
        side=r['side'].lower(); p=float(r['price']); amount=float(r['amount']); book=bids if side=='bid' else asks; old=book.get(p,0.0); delta=amount-old
        if amount==0:book.pop(p,None)
        else:book[p]=amount
        sgn=1.0 if side=='bid' else -1.0
        agg['signed_liq_delta']+=sgn*delta; agg['removed_qty']+=max(0.0,-delta); agg['added_qty']+=max(0.0,delta); agg['updates']+=1
        if side=='bid':agg['bid_delta']+=delta
        else:agg['ask_delta']+=delta
        if bids and asks and max(bids)>=min(asks):crossed+=1
if current_bin is not None:flush(current_bin)
state=pd.DataFrame(samples).sort_values('timestamp_us').reset_index(drop=True)
if state.empty:raise RuntimeError('no reconstructed states')
# Trades, first same 30-minute interval.
trade_bins={}; trade_rows=0
with gzip.open(trp,'rt',newline='') as f:
    for r in csv.DictReader(f):
        t=int(r['timestamp'])
        if t<first_ts:continue
        if t>=end_ts:break
        side=r['side'].lower(); sign=1.0 if side=='buy' else -1.0; amount=float(r['amount']); price=float(r['price']); b=t//SAMPLE_US
        rec=trade_bins.setdefault(b,[0,0.0,0.0]); rec[0]+=1; rec[1]+=amount; rec[2]+=sign*amount*price; trade_rows+=1
state['bin_id']=(state.timestamp_us//SAMPLE_US).astype(np.int64)
state['trade_count']=[trade_bins.get(int(b),[0,0,0])[0] for b in state.bin_id]
state['trade_amount']=[trade_bins.get(int(b),[0,0,0])[1] for b in state.bin_id]
state['signed_trade_notional']=[trade_bins.get(int(b),[0,0,0])[2] for b in state.bin_id]
state['log_mid']=np.log(state.mid); state['return']=state.log_mid.diff().fillna(0); state['d_imbalance_50bp']=state.imbalance_50bp.diff().fillna(0)
state.to_csv(OUT/'L2_100MS_STATE.csv.gz',index=False,compression='gzip')
# P1: local shocks in liquidity delta, removals, trades, or imbalance change.
features=np.column_stack([rz(state.signed_liq_delta),rz(state.removed_qty),rz(state.signed_trade_notional),rz(state.d_imbalance_50bp)])
score=np.max(np.abs(features),axis=1); kind_idx=np.argmax(np.abs(features),axis=1); kind_names=np.array(['signed_liquidity_delta','removed_liquidity','signed_trade_notional','imbalance_change'])
idx=np.flatnonzero(score>=6); kept=[]
for i in idx:
    if kept and i-kept[-1]<10:
        if score[i]>score[kept[-1]]:kept[-1]=int(i)
        continue
    kept.append(int(i))
maxh=int(HORIZONS_MS.max()//100); events=np.array([i for i in kept if maxh<=i<len(state)-maxh],dtype=int); kinds=kind_names[kind_idx[events]]
directions=np.where(kind_idx[events]==0,np.sign(features[events,0]),np.where(kind_idx[events]==1,-np.sign(state.bid_delta.iloc[events]-state.ask_delta.iloc[events]),np.where(kind_idx[events]==2,np.sign(features[events,2]),np.sign(features[events,3])))).astype(int)
# Matched-state controls: spread, total depth, abs imbalance, update intensity.
code=qbin(state.spread,4)+10*qbin(state.bid_depth_50bp+state.ask_depth_50bp,5)+100*qbin(np.abs(state.imbalance_50bp),5)+1000*qbin(state.updates,5)
elig=np.arange(maxh,len(state)-maxh); blocked=np.zeros(len(state),bool)
for i in events:blocked[max(0,i-1):min(len(state),i+2)]=True
pools={}
for c in np.unique(code[elig]):
    z=elig[(code[elig]==c)&(~blocked[elig])]
    if len(z):pools[int(c)]=z
fallback=elig[~blocked[elig]]
obs=[]; null=np.zeros((N_PERM,len(HORIZONS_MS)))
for k,hms in enumerate(HORIZONS_MS):
    h=int(hms//100); vals=directions*(state.log_mid.iloc[events+h].to_numpy()-state.log_mid.iloc[events].to_numpy()); obs.append(float(vals.mean()))
for p in range(N_PERM):
    controls=np.array([int(RNG.choice(pools.get(int(code[i]),fallback))) for i in events])
    for k,hms in enumerate(HORIZONS_MS):
        h=int(hms//100); null[p,k]=np.mean(directions*(state.log_mid.iloc[controls+h].to_numpy()-state.log_mid.iloc[controls].to_numpy()))
obs=np.asarray(obs); pv=(1+(np.abs(null)>=np.abs(obs)).sum(axis=0))/(N_PERM+1); qv=bh(pv)
response=pd.DataFrame({'lag_ms':HORIZONS_MS,'n_events':len(events),'signed_mid_response':obs,'signed_mid_response_bp':obs*1e4,'matched_null_median':np.median(null,axis=0),'p_value':pv,'q_value':qv})
response.to_csv(OUT/'L2_RESPONSE.csv',index=False)
pd.DataFrame({'idx':events,'timestamp_us':state.timestamp_us.iloc[events].to_numpy(np.int64),'kind':kinds,'direction':directions,'score':score[events]}).to_csv(OUT/'L2_EVENTS.csv',index=False)
summary={'status':'ok','source':{'l2':l2meta,'trades':trmeta},'window':{'first_timestamp_us':int(first_ts),'end_timestamp_us':int(end_ts),'duration_s':WINDOW_US/1e6,'sampling_ms':100},'raw':{'l2_rows_processed':rows,'snapshot_rows':snapshot_rows,'trade_rows_processed':trade_rows,'crossed_book_observations':crossed},'states':{'rows':int(len(state)),'median_spread':float(state.spread.median()),'median_levels':{'bid':float(state.n_bid_levels.median()),'ask':float(state.n_ask_levels.median())},'median_depth_50bp':{'bid':float(state.bid_depth_50bp.median()),'ask':float(state.ask_depth_50bp.median())},'median_entropy_top20':{'bid':float(state.bid_entropy_top20.median()),'ask':float(state.ask_entropy_top20.median())}},'events':{'raw_after_refractory':len(kept),'boundary_valid':len(events),'by_kind':{k:int(v) for k,v in pd.Series(kinds).value_counts().items()}},'null':{'replicates':N_PERM,'matched_dimensions':['spread','50bp total depth','absolute 50bp imbalance','book-update intensity'],'significant_horizons_ms':[int(h) for h,q in zip(HORIZONS_MS,qv) if q<.05]}}
(OUT/'TARDIS_L2_SUMMARY.json').write_text(json.dumps(summary,indent=2)); (OUT/'DONE.txt').write_text('ok\n'); print(json.dumps(summary,indent=2))
