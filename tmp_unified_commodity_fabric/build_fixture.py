#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--universe',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 u=json.loads(a.universe.read_text())['commodities'];rng=np.random.default_rng(3701);start=pd.Timestamp('2024-01-08T13:00:00Z');n=7200;shocks=list(range(400,6801,200));signs=rng.choice([-1.,1.],len(shocks));common=np.zeros(n)
 for t,s in zip(shocks,signs):common[t:t+20]+=np.exp(-np.arange(20)/5)*s*.001
 delay={0:1,1:0,2:5,3:5,4:0,5:1,6:0,7:5};amp={0:.0045,1:.0035,2:.0028,3:.0028,4:.0045,5:.0035,6:.0045,7:.0028}
 for j,c in enumerate(u):
  directed=np.zeros(n)
  for t,s in zip(shocks,signs):
   q=t+delay[j];directed[q:q+15]+=np.exp(-np.arange(15)/4)*s*amp[j]
  ret=rng.normal(0,.00005,n)+.35*common+np.sin(np.arange(n)/300+j)*.00002+directed;price=(50+10*j)*np.exp(np.cumsum(ret));spread=np.maximum(price*(.0002+rng.lognormal(-10,.3,n)),1e-5);ask=price+spread/2;bid=price-spread/2;act=rng.poisson(1.5+20*np.abs(common),n)+1;bv=rng.lognormal(2,.4,n)*(1+4*np.maximum(common,0));av=rng.lognormal(2,.4,n)*(1+4*np.maximum(-common,0));rows=[]
  for i in range(n):
   for k in range(int(act[i])):rows.append((int((start+pd.Timedelta(seconds=i,milliseconds=100*k)).timestamp()*1000),ask[i],bid[i],av[i],bv[i]))
  d=a.out/c['instrument_id']/"2024";d.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows,columns=['timestamp','askPrice','bidPrice','askVolume','bidVolume']).to_csv(d/f"{c['instrument_id']}-fixture.csv.gz",index=False,compression='gzip')
 off=a.out/'official';off.mkdir(exist_ok=True);r=[]
 for c in u:
  for day in range(4):
   ref=pd.Timestamp('2024-01-01T00:00:00Z')+pd.Timedelta(days=day);r.append({'commodity_id':c['commodity_id'],'field':'fundamental_surprise_fixture','value':float(rng.normal()),'unit':'z','reference_time':ref,'available_time':ref+pd.Timedelta(days=2,hours=15,minutes=30),'source':'FIXTURE_OFFICIAL','venue':'OFFICIAL','quality_status':'VALID'})
 pd.DataFrame(r).to_csv(off/'official_fixture.csv',index=False);return 0
if __name__=='__main__':raise SystemExit(main())
