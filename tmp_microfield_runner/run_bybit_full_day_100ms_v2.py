#!/usr/bin/env python3
from pathlib import Path

base=Path(__file__).resolve().parent/'run_bybit_full_day_100ms.py'
source=base.read_text()
source=source.replace("OUT=ROOT/'tmp_microfield_results_bybit_fullday'", "OUT=ROOT/'tmp_microfield_results_bybit_fullday_v2'")
old="""def detect(f):
 zf=rz(f.signed); za=rz(np.log1p(f.notional)); ids=np.flatnonzero((np.abs(zf)>=6)&(za>=3)); kept=[]
 for i in ids:
  if kept and i-kept[-1]<10:
   if abs(zf[i])>abs(zf[kept[-1]]):kept[-1]=int(i)
   continue
  kept.append(int(i))
 return pd.DataFrame({'idx':kept,'dir':np.sign(zf[kept]).astype(int),'z':zf[kept],'signed':f.signed.iloc[kept].to_numpy()})
"""
new="""def detect(f):
 zf=np.zeros(len(f)); za=np.zeros(len(f))
 for hour in range(24):
  lo=hour*36000; hi=(hour+1)*36000; signed=f.signed.iloc[lo:hi].to_numpy(float); notion=f.notional.iloc[lo:hi].to_numpy(float); active=notion>0
  if active.sum()<20:continue
  sv=signed[active]; sm=np.median(sv); smad=np.median(np.abs(sv-sm)); ss=1.4826*smad
  if not np.isfinite(ss) or ss<=1e-15:ss=np.std(sv)+1e-15
  av=np.log1p(notion[active]); am=np.median(av); amad=np.median(np.abs(av-am)); ass=1.4826*amad
  if not np.isfinite(ass) or ass<=1e-15:ass=np.std(av)+1e-15
  local_zf=np.zeros(hi-lo); local_za=np.zeros(hi-lo); local_zf[active]=(signed[active]-sm)/ss; local_za[active]=(np.log1p(notion[active])-am)/ass
  zf[lo:hi]=local_zf; za[lo:hi]=local_za
 ids=np.flatnonzero((np.abs(zf)>=6)&(za>=3)); kept=[]
 for i in ids:
  if kept and i-kept[-1]<10:
   if abs(zf[i])>abs(zf[kept[-1]]):kept[-1]=int(i)
   continue
  kept.append(int(i))
 return pd.DataFrame({'idx':kept,'dir':np.sign(zf[kept]).astype(int),'z':zf[kept],'signed':f.signed.iloc[kept].to_numpy()})
"""
if old not in source:raise RuntimeError('detect block not found')
source=source.replace(old,new)
source=source.replace("'controls':['exclusive source impulses ±100ms'", "'baseline':'instrument-hour robust scaling on nonzero 100ms bins','controls':['exclusive source impulses ±100ms'")
namespace={'__file__':str(base),'__name__':'__main__'}
exec(compile(source,str(base),'exec'),namespace)
