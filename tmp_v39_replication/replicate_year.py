#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore', category=RuntimeWarning)
INSTR=['brentcmdusd','lightcmdusd','gascmdusd','coppercmdusd','xauusd','xagusd','soybeancmdusx','coffeecmdusx']
FAMILY={'brentcmdusd':'energy','lightcmdusd':'energy','gascmdusd':'energy','coppercmdusd':'industrial_metals','xauusd':'precious_metals','xagusd':'precious_metals','soybeancmdusx':'agriculture','coffeecmdusx':'agriculture'}
LAGS=[1,3,6,12,24];LAG_S={x:x*5 for x in LAGS};CHANNELS=['residual_return','spread_rel','activity','volume_imbalance','log_volume','quote_pressure']
DAY_MS=86400000;STEP=5000;SOURCE_START=9*3600*1000;SOURCE_END=18*3600*1000;START_MS=int(13.5*3600*1000);END_MS=17*3600*1000

def seed(k):return int.from_bytes(hashlib.sha256(repr(k).encode()).digest()[:8],'little')%(2**32)
def rz(a):
 a=np.asarray(a,float);o=np.full(a.shape,np.nan);x=a[np.isfinite(a)]
 if len(x)<10:return o
 med=np.median(x);scale=1.4826*np.median(np.abs(x-med))
 if not np.isfinite(scale) or scale<1e-15:
  q=np.quantile(x,[.25,.75]);scale=(q[1]-q[0])/1.349
 if not np.isfinite(scale) or scale<1e-15:scale=np.std(x)
 if not np.isfinite(scale) or scale<1e-15:return o
 o[np.isfinite(a)]=(a[np.isfinite(a)]-med)/scale;return o
def bh(p):
 p=np.asarray(p,float);o=np.full(len(p),np.nan);ok=np.isfinite(p)
 if not ok.any():return o
 idx=np.where(ok)[0];v=p[ok];s=np.argsort(v);r=v[s];a=np.minimum.accumulate((r*len(r)/np.arange(1,len(r)+1))[::-1])[::-1];a=np.clip(a,0,1);t=np.empty_like(a);t[s]=a;o[idx]=t;return o
def signflip(v,key,n=199):
 v=np.asarray(v,float);v=v[np.isfinite(v)]
 if len(v)<8:return np.nan
 obs=abs(v.mean());rng=np.random.default_rng(seed(key));ge=0
 for st in range(0,n,1000):
  k=min(1000,n-st);sg=rng.integers(0,2,size=(k,len(v)),dtype=np.int8)*2-1;ge+=int(np.sum(np.abs((sg*v).mean(1))>=obs-1e-15))
 return (ge+1)/(n+1)
def aggregate_file(p,inst,year,out):
 parts=[];raw=0;pb=pa=np.nan
 for ch in pd.read_csv(p,chunksize=1_000_000):
  raw+=len(ch);ts=ch.timestamp.to_numpy(np.int64);tod=ts%DAY_MS;m=(tod>=SOURCE_START)&(tod<SOURCE_END)
  if not m.any():continue
  ts=ts[m];bid=ch.bidPrice.to_numpy(float)[m];ask=ch.askPrice.to_numpy(float)[m];bv=ch.bidVolume.to_numpy(float)[m];av=ch.askVolume.to_numpy(float)[m]
  bm=np.empty(len(ts),np.int8);am=np.empty(len(ts),np.int8);bm[0]=0 if not np.isfinite(pb) else int(bid[0]!=pb);am[0]=0 if not np.isfinite(pa) else int(ask[0]!=pa)
  if len(ts)>1:bm[1:]=bid[1:]!=bid[:-1];am[1:]=ask[1:]!=ask[:-1]
  pb=bid[-1];pa=ask[-1];mid=(bid+ask)/2;spread=ask-bid;tot=bv+av;vi=np.divide(bv-av,tot,out=np.full_like(tot,np.nan),where=tot!=0)
  x=pd.DataFrame({'bin_ms':ts//5000*5000,'mid':mid,'spread':spread,'tick_count':1,'bid_moves':bm,'ask_moves':am,'bid_volume':bv,'ask_volume':av,'volume_total':tot,'volume_imbalance':vi})
  parts.append(x.groupby('bin_ms',sort=False).agg(mid=('mid','last'),spread=('spread','last'),tick_count=('tick_count','sum'),bid_moves=('bid_moves','sum'),ask_moves=('ask_moves','sum'),bid_volume=('bid_volume','last'),ask_volume=('ask_volume','last'),volume_total=('volume_total','last'),volume_imbalance=('volume_imbalance','last')).reset_index())
 if not parts:raise RuntimeError(f'no rows {p}')
 z=pd.concat(parts).groupby('bin_ms',sort=True).agg(mid=('mid','last'),spread=('spread','last'),tick_count=('tick_count','sum'),bid_moves=('bid_moves','sum'),ask_moves=('ask_moves','sum'),bid_volume=('bid_volume','last'),ask_volume=('ask_volume','last'),volume_total=('volume_total','last'),volume_imbalance=('volume_imbalance','last')).reset_index();month=int(p.name.split(f'-{year}-')[1].split('-')[0]);z.to_csv(out/f'{inst}-{month:02d}-5s.csv.gz',index=False,compression='gzip');return raw,len(z)
def aggregate(raw,agg,year):
 agg.mkdir(parents=True,exist_ok=True);rows=[]
 for inst in INSTR:
  files=sorted(raw.rglob(f'{inst}-{year}-??-tick-volume.csv.gz'))
  if len(files)!=12:raise RuntimeError(f'{inst}: expected 12 months, got {len(files)}')
  for p in files:
   month=int(p.name.split(f'-{year}-')[1].split('-')[0]);q=agg/f'{inst}-{month:02d}-5s.csv.gz'
   if q.exists():continue
   rr,b=aggregate_file(p,inst,year,agg);rows.append({'instrument':inst,'month':month,'raw_rows':rr,'bins':b})
 pd.DataFrame(rows).to_csv(agg/'AGGREGATION_LINEAGE.csv',index=False)
def load_month(agg,m):return {i:pd.read_csv(agg/f'{i}-{m:02d}-5s.csv.gz') for i in INSTR}
def build_day(d,day):
 grid=np.arange(day*DAY_MS+START_MS,day*DAY_MS+END_MS,STEP,dtype=np.int64);T=len(grid);arr={};cov={}
 for inst,x in d.items():
  q=x[(x.bin_ms>=grid[0])&(x.bin_ms<=grid[-1])].set_index('bin_ms').reindex(grid)
  for c in ['mid','spread','bid_volume','ask_volume','volume_total','volume_imbalance']:q[c]=q[c].ffill(limit=12)
  for c in ['tick_count','bid_moves','ask_moves']:q[c]=q[c].fillna(0.)
  cov[inst]=float(q.mid.notna().mean());mid=q.mid.to_numpy(float);vi=q.volume_imbalance.to_numpy(float);bp=q.bid_moves.to_numpy(float);ap=q.ask_moves.to_numpy(float)
  arr[inst]={'spread_rel':(q.spread/q.mid).to_numpy(float),'activity':np.log1p(q.tick_count.to_numpy(float)),'volume_imbalance':vi,'log_volume':np.log1p(q.volume_total.to_numpy(float)),'quote_pressure':(bp-ap)/(bp+ap+1.),'ret':np.r_[np.nan,np.diff(np.log(mid))],'vi_delta':np.r_[np.nan,np.diff(vi)]}
 if min(cov.values())<.65:return None,cov
 zr=np.column_stack([rz(arr[i]['ret']) for i in INSTR]);res=np.full_like(zr,np.nan)
 for j,inst in enumerate(INSTR):
  gm=np.nanmedian(zr[:,[k for k in range(8) if k!=j]],axis=1);fam=[k for k,x in enumerate(INSTR) if FAMILY[x]==FAMILY[inst] and k!=j];fm=np.nanmean(zr[:,fam],axis=1) if fam else np.zeros(T);y=zr[:,j];X=np.c_[np.ones(T),gm,fm];m=np.isfinite(y)&np.all(np.isfinite(X),1)
  if m.sum()>50:res[:,j]=y-X@np.linalg.lstsq(X[m],y[m],rcond=None)[0]
 M=np.nan_to_num(zr);valid=np.sum(np.isfinite(zr),1)>=6;s=np.linalg.svd(M[valid]-M[valid].mean(0),compute_uv=False) if valid.sum()>20 else np.array([]);pc=float(s[0]**2/np.sum(s**2)) if len(s) and np.sum(s**2)>0 else np.nan
 for j,inst in enumerate(INSTR):
  a=arr[inst];a['residual_return']=res[:,j];a['z_ret']=zr[:,j];a['z_spread']=rz(a['spread_rel']);a['z_activity']=rz(a['activity']);a['z_vi_delta']=rz(a['vi_delta']);a['z_log_volume']=rz(a['log_volume']);a['z_quote_pressure']=rz(a['quote_pressure'])
 return {'grid':grid,'arrays':arr,'pc1':pc},cov
def events(a):
 f=np.c_[a['z_ret'],a['z_spread'],a['z_activity'],a['z_vi_delta'],a['z_log_volume'],a['z_quote_pressure']];n=np.sum(np.isfinite(f),1);score=np.where(n>=4,np.sqrt(np.nansum(f*f,1)/n),np.nan);x=score[np.isfinite(score)]
 if len(x)<100:return [],score
 th=max(4.5,float(np.quantile(x,.9985)));sel=[];last=-999
 for i in np.where(score>=th)[0]:
  if i-last>=6 and i>=24 and i<len(score)-25:sel.append(int(i));last=i
 return sel[:60],score
def pm(x,i,L,ret=False):
 if ret:
  a=x[i-L+1:i+1];b=x[i+1:i+L+1]
  if np.isfinite(a).mean()<.8 or np.isfinite(b).mean()<.8:return None
  pre=np.nansum(a);post=np.nansum(b)
 else:
  v=[x[i-L],x[i],x[i+L]]
  if not np.all(np.isfinite(v)):return None
  pre=v[1]-v[0];post=v[2]-v[1]
 return pre,post,post-pre
def analyze(agg,out,year):
 out.mkdir(parents=True,exist_ok=True);D=[];E=[];G=[]
 for mo in range(1,13):
  d=load_month(agg,mo);days=sorted(set.intersection(*[set((x.bin_ms//DAY_MS).astype(int).unique()) for x in d.values()]))
  for day in days:
   b,c=build_day(d,day);date=pd.to_datetime(day*DAY_MS,unit='ms',utc=True).date().isoformat()
   if b is None:G.append({'date':date,'status':'skip_coverage','min_coverage':min(c.values())});continue
   cell={};times=[]
   for src in INSTR:
    idx,score=events(b['arrays'][src]);times+=idx
    for i in idx:
     E.append({'date':date,'timestamp':pd.to_datetime(b['grid'][i],unit='ms',utc=True).isoformat(),'source':src,'score':score[i]})
     for tgt in INSTR:
      if tgt==src:continue
      for L in LAGS:
       for ch in CHANNELS:
        r=pm(b['arrays'][tgt][ch],i,L,ch=='residual_return')
        if r is None:continue
        key=(src,tgt,LAG_S[L],ch);z=cell.setdefault(key,[[],[],[]]);[z[k].append(r[k]) for k in range(3)]
   for k,z in cell.items():
    v=np.asarray(z[2]);D.append({'date':date,'source':k[0],'target':k[1],'lag_s':k[2],'channel':k[3],'n_events':len(v),'daily_pre_mean':np.mean(z[0]),'daily_post_mean':np.mean(z[1]),'daily_mean':np.mean(v)})
   vc=pd.Series(times).value_counts();G.append({'date':date,'status':'ok','events':len(times),'common_event_fraction':float(vc[vc>=3].sum()/max(len(times),1)),'pc1_energy':b['pc1'],'min_coverage':min(c.values())})
 daily=pd.DataFrame(D);daily['date']=pd.to_datetime(daily.date);daily['month']=daily.date.dt.month;ev=pd.DataFrame(E);diag=pd.DataFrame(G)
 rows=[]
 for key,g in daily.groupby(['source','target','lag_s','channel']):
  v=g.daily_mean.to_numpy(float);mean=v.mean();sd=v.std(ddof=1);mm=g.groupby('month').daily_mean.mean();rows.append({'source':key[0],'target':key[1],'lag_s':key[2],'channel':key[3],'n_days':len(v),'n_events':int(g.n_events.sum()),'weight':mean,'effect_std':mean/sd if sd>0 else np.nan,'sign_consistency_days':max(np.mean(v>0),np.mean(v<0)),'monthly_sign_consistency':max(np.mean(mm>0),np.mean(mm<0)),'p_199':signflip(v,('op',year)+key,199)})
 op=pd.DataFrame(rows);op['q_value']=np.nan
 for _,idx in op.groupby(['channel','lag_s']).groups.items():op.loc[idx,'q_value']=bh(op.loc[idx,'p_199'])
 op['candidate_pass']=(op.n_days>=20)&(op.q_value<=.05)&(op.sign_consistency_days>=.6)&(op.effect_std.abs()>=.1)&(op.monthly_sign_consistency>=.58)
 daily.to_csv(out/'DAILY_EDGE_RESPONSES.csv.gz',index=False,compression='gzip');ev.to_csv(out/'P1_EVENTS.csv.gz',index=False,compression='gzip');diag.to_csv(out/'DAY_DIAGNOSTICS.csv',index=False);op.to_csv(out/'OPERATOR.csv',index=False)
 return daily,ev,diag,op
def circular_placebo(ev,agg,target,year):
 source_events=ev[ev.source=='coppercmdusd'].copy();source_events['timestamp']=pd.to_datetime(source_events.timestamp,utc=True);source_events['month']=source_events.timestamp.dt.month;days=[]
 for mo in range(1,13):
  x=pd.read_csv(agg/f'{target}-{mo:02d}-5s.csv.gz',usecols=['bin_ms','tick_count']);x['day']=(x.bin_ms//DAY_MS).astype(int);em=source_events[source_events.month==mo]
  for date,g in em.groupby(source_events.timestamp.dt.date.astype(str)):
   day=int(pd.Timestamp(date,tz='UTC').timestamp()*1000//DAY_MS);grid=np.arange(day*DAY_MS+START_MS,day*DAY_MS+END_MS,STEP,dtype=np.int64);q=x[x.day==day].set_index('bin_ms').reindex(grid);act=np.log1p(q.tick_count.fillna(0).to_numpy(float));idx=((g.timestamp.astype('int64')//1_000_000-(day*DAY_MS+START_MS))//STEP).astype(int).to_numpy();idx=idx[(idx>=1)&(idx<len(act)-1)]
   if not len(idx):continue
   curv=act[2:]-2*act[1:-1]+act[:-2];actual=float(np.mean(curv[idx-1]));shifts=np.array([z for z in range(-720,721,12) if abs(z)>=12],int);pool=[];n=len(act)
   for sh in shifts:
    j=((idx+sh-1)%(n-2))+1;pool.append(float(np.mean(curv[j-1])))
   days.append((actual,np.asarray(pool,float)))
 obs=float(np.mean([x[0] for x in days]));rng=np.random.default_rng(seed(('circular',year,target)));null=np.empty(9999)
 for k in range(9999):null[k]=np.mean([pool[rng.integers(0,len(pool))] for _,pool in days])
 centered=obs-float(null.mean());p=float((1+np.sum(np.abs(null-null.mean())>=abs(centered)-1e-15))/10000)
 return obs,float(null.mean()),centered,p
def targeted(daily,ev,agg,op,out,year):
 result=[]
 for target in ['brentcmdusd','xagusd']:
  f=daily[(daily.source=='coppercmdusd')&(daily.target==target)&(daily.channel=='activity')&(daily.lag_s==5)][['date','daily_mean','n_events']]
  r=daily[(daily.source==target)&(daily.target=='coppercmdusd')&(daily.channel=='activity')&(daily.lag_s==5)][['date','daily_mean']].rename(columns={'daily_mean':'reverse'});z=f.merge(r,on='date');diff=z.daily_mean-z.reverse
  q=op[(op.source=='coppercmdusd')&(op.target==target)&(op.channel=='activity')&(op.lag_s==5)];operator_candidate=bool(len(q) and q.iloc[0].candidate_pass)
  opneg=float(f.daily_mean.mean())<0;strong=abs(f.daily_mean.mean())>abs(z.reverse.mean());p0=signflip(f.daily_mean,('target',year,target),19999);pa=signflip(diff,('asym',year,target),19999)
  mid=pd.Timestamp(f'{year}-07-01');h1=float(f[f.date<mid].daily_mean.mean());h2=float(f[f.date>=mid].daily_mean.mean());actual,placebo,excess,pp=circular_placebo(ev,agg,target,year)
  passed=bool(operator_candidate and opneg and strong and p0<=.025 and pa<=.025 and h1<0 and h2<0 and pp<=.025)
  result.append({'year':year,'source':'coppercmdusd','target':target,'channel':'activity','lag_s':5,'n_days':len(f),'n_events':int(f.n_events.sum()),'effect':float(f.daily_mean.mean()),'reverse_effect':float(z.reverse.mean()),'asymmetry':float(diff.mean()),'operator_candidate':operator_candidate,'p_effect_19999':p0,'p_asymmetry_19999':pa,'half1_mean':h1,'half2_mean':h2,'circular_placebo_mean':placebo,'actual_minus_circular_placebo':excess,'p_actual_minus_circular_9999':pp,'operator_negative':opneg,'stronger_than_reverse':strong,'year_replication_pass':passed})
 pd.DataFrame(result).to_csv(out/'PREREGISTERED_DIRECTIONS.csv',index=False);return result
def main():
 a=argparse.ArgumentParser();a.add_argument('--year',type=int,required=True);a.add_argument('--raw',type=Path,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args();agg=x.out/'agg5s';res=x.out/'results';aggregate(x.raw,agg,x.year);daily,ev,diag,op=analyze(agg,res,x.year);t=targeted(daily,ev,agg,op,res,x.year)
 summaries=[json.loads(p.read_text()) for p in x.raw.rglob('SUMMARY.json')]
 summary={'year':x.year,'raw_ticks':int(sum(int(s['rows_total']) for s in summaries)),'valid_days':int((diag.status=='ok').sum()),'events':len(ev),'daily_edge_rows':len(daily),'operator_candidates':int(op.candidate_pass.sum()),'preregistered_directions':t,'protocol':'frozen v3.8 thresholds/lags/channels; no post-result tuning','semantic_status':'INDEPENDENT_YEAR_FREE_QUOTE_PROXY_REPLICATION'}
 (res/'SUMMARY_REPLICATION.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
