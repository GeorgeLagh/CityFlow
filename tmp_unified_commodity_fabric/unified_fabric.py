#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

LAGS=(1,5,30); CHANNELS=("ret_1s","spread_rel","quote_activity","volume_imbalance")
BLOCKED={"order_id","queue_position","true_cancel_flow","full_depth_book","aggressor_side"}

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def rz(s:pd.Series)->pd.Series:
 x=pd.to_numeric(s,errors="coerce")
 if x.notna().sum()<3:return pd.Series(np.nan,index=s.index,dtype=float)
 m=x.median();d=(x-m).abs().median()
 return (x-m)/(1.4826*d) if np.isfinite(d) and d else pd.Series(np.nan,index=s.index,dtype=float)

def bh(p:Iterable[float])->np.ndarray:
 a=np.asarray(list(p),float);o=np.full(len(a),np.nan);ok=np.isfinite(a)
 if not ok.any():return o
 idx=np.where(ok)[0];v=a[ok];q=np.argsort(v);r=v[q]*len(v)/np.arange(1,len(v)+1);r=np.minimum.accumulate(r[::-1])[::-1];z=np.empty_like(r);z[q]=np.clip(r,0,1);o[idx]=z;return o

@dataclass(frozen=True)
class C: commodity_id:str;instrument_id:str;family:str;carrier:str;native_exchange:str

def universe(p:Path)->dict[str,C]:
 return {x["instrument_id"]:C(**x) for x in json.loads(p.read_text())["commodities"]}

def cols(cs):
 d={str(c).lower():str(c) for c in cs}
 def g(*n):return next((d[x.lower()] for x in n if x.lower() in d),None)
 return g("timestamp","time","ts_event"),g("askPrice","ask","ask_price"),g("bidPrice","bid","bid_price"),g("askVolume","ask_volume","ask_size"),g("bidVolume","bid_volume","bid_size")

def ts(s):
 if pd.api.types.is_numeric_dtype(s):
  x=pd.to_numeric(s,errors="coerce");return pd.to_datetime(x,unit="ms" if x.dropna().median()>1e11 else "s",utc=True,errors="coerce")
 return pd.to_datetime(s,utc=True,errors="coerce")

def reduce_file(p:Path,c:C):
 parts=[];n=0;vol=False;schema=[]
 for ch in pd.read_csv(p,compression="infer",chunksize=750000):
  n+=len(ch);schema=schema or list(ch);t,a,b,av,bv=cols(ch)
  if not t or not a or not b:raise ValueError(f"missing tick columns: {p}")
  x=pd.DataFrame({"t":ts(ch[t]),"a":pd.to_numeric(ch[a],errors="coerce"),"b":pd.to_numeric(ch[b],errors="coerce"),"av":pd.to_numeric(ch[av],errors="coerce") if av else np.nan,"bv":pd.to_numeric(ch[bv],errors="coerce") if bv else np.nan}).dropna(subset=["t","a","b"])
  vol=vol or x[["av","bv"]].notna().any().any();x["second"]=x.t.dt.floor("s");x["mid"]=(x.a+x.b)/2;x["spread"]=x.a-x.b;x["bm"]=x.b.diff().ne(0);x["am"]=x.a.diff().ne(0);x["vi"]=(x.bv-x.av)/(x.bv+x.av)
  parts.append(x.groupby("second").agg(mid=("mid","last"),spread=("spread","last"),quote_activity=("t","size"),bid_moves=("bm","sum"),ask_moves=("am","sum"),bid_volume=("bv","last"),ask_volume=("av","last"),volume_imbalance=("vi","last")).reset_index())
 if not parts:return pd.DataFrame(),{"file":str(p),"rows":n,"seconds":0}
 x=pd.concat(parts).sort_values("second").groupby("second").agg(mid=("mid","last"),spread=("spread","last"),quote_activity=("quote_activity","sum"),bid_moves=("bid_moves","sum"),ask_moves=("ask_moves","sum"),bid_volume=("bid_volume","last"),ask_volume=("ask_volume","last"),volume_imbalance=("volume_imbalance","last")).reset_index()
 x.insert(0,"commodity_id",c.commodity_id);x.insert(1,"instrument_id",c.instrument_id);x.insert(2,"family",c.family)
 if not vol:x[["bid_volume","ask_volume","volume_imbalance"]]=np.nan
 return x,{"file":str(p),"sha256":sha(p),"rows":n,"seconds":len(x),"columns":schema,"volume_observed":bool(vol),"min_time":str(x.second.min()),"max_time":str(x.second.max())}

def load_ticks(root:Path,u):
 fs=[];lin=[]
 for p in sorted(root.rglob("*.csv.gz")):
  k=next((k for k in u if k in str(p).lower()),None)
  if k:
   x,m=reduce_file(p,u[k]);lin.append(m)
   if not x.empty:fs.append(x)
 return (pd.concat(fs,ignore_index=True) if fs else pd.DataFrame()),lin

def align(x):
 out=[]
 for k,g in x.groupby("instrument_id"):
  meta=g.iloc[0];z=g.set_index("second")[["mid","spread","quote_activity","bid_moves","ask_moves","bid_volume","ask_volume","volume_imbalance"]].resample("1s").agg({"mid":"last","spread":"last","quote_activity":"sum","bid_moves":"sum","ask_moves":"sum","bid_volume":"last","ask_volume":"last","volume_imbalance":"last"});z[["mid","spread","bid_volume","ask_volume","volume_imbalance"]]=z[["mid","spread","bid_volume","ask_volume","volume_imbalance"]].ffill(limit=60);z["ret_1s"]=np.log(z.mid).diff();z["spread_rel"]=z.spread/z.mid;z["commodity_id"]=meta.commodity_id;z["instrument_id"]=k;z["family"]=meta.family;out.append(z.reset_index())
 return pd.concat(out,ignore_index=True)

def residual(x):
 for c in CHANNELS:x["z_"+c]=x.groupby(["instrument_id",x.second.dt.date])[c].transform(rz)
 a=x[["z_"+c for c in CHANNELS]].to_numpy(float);n=np.isfinite(a).sum(1);x["excitation_norm"]=np.where(n,np.sqrt(np.nansum(a*a,1)/np.maximum(n,1)),np.nan);x["global_return_mode"]=x.groupby("second").z_ret_1s.transform("mean");x["family_return_mode"]=x.groupby(["second","family"]).z_ret_1s.transform("mean");x["residual_return"]=x.z_ret_1s-x.family_return_mode;return x

def official(root:Path|None,byid):
 out=[]
 if root and root.exists():
  for p in root.rglob("*.csv"):
   x=pd.read_csv(p);need={"commodity_id","field","value","unit","reference_time","available_time"}
   if need-set(x):raise ValueError(f"official schema {p}")
   x.reference_time=pd.to_datetime(x.reference_time,utc=True);x.available_time=pd.to_datetime(x.available_time,utc=True)
   if (x.available_time<x.reference_time).any():raise ValueError(f"look-ahead {p}")
   for k,v in {"source":p.stem,"venue":"OFFICIAL","observation_layer":"external_field","quality_status":"VALID","source_hash":sha(p)}.items():x[k]=x[k] if k in x else v
   x["instrument_id"]=x.commodity_id.map(lambda q:byid[q].instrument_id if q in byid else "GLOBAL");x["family"]=x.commodity_id.map(lambda q:byid[q].family if q in byid else "macro");out.append(x[["commodity_id","instrument_id","family","source","venue","observation_layer","field","value","unit","reference_time","available_time","quality_status","source_hash"]])
 return pd.concat(out,ignore_index=True) if out else pd.DataFrame(columns=["commodity_id","instrument_id","family","source","venue","observation_layer","field","value","unit","reference_time","available_time","quality_status","source_hash"])

def events(x):
 r=[]
 for k,g in x.groupby("instrument_id"):
  last=None
  for _,z in g[g.excitation_norm>=6].sort_values("second").iterrows():
   if last is not None and (z.second-last).total_seconds()<30:continue
   d={c:float(z.get("z_"+c,np.nan)) for c in CHANNELS};r.append({"event_id":f"{k}:{z.second.isoformat()}","second":z.second,"source_instrument":k,"source_commodity":z.commodity_id,"source_family":z.family,"excitation_norm":z.excitation_norm,"dominant_channel":max(d,key=lambda q:abs(d[q]) if np.isfinite(d[q]) else -1),**{"source_z_"+q:v for q,v in d.items()}});last=z.second
 return pd.DataFrame(r)

def responses(x,e):
 if e.empty:return pd.DataFrame()
 look={k:g.set_index("second").sort_index() for k,g in x.groupby("instrument_id")};r=[]
 for _,v in e.iterrows():
  t=v.second
  for k,g in look.items():
   if k==v.source_instrument:continue
   for lag in LAGS:
    p,b,a=t-pd.Timedelta(seconds=lag),t,t+pd.Timedelta(seconds=lag)
    if p not in g.index or b not in g.index or a not in g.index:continue
    for c in ("residual_return","spread_rel","quote_activity","volume_imbalance"):
     q=[g.at[j,c] for j in (p,b,a)];q=[z.iloc[-1] if isinstance(z,pd.Series) else z for z in q]
     r.append({"event_id":v.event_id,"source":v.source_instrument,"target":k,"source_family":v.source_family,"target_family":g.family.iloc[0],"lag_s":lag,"channel":c,"pre_change":q[1]-q[0] if np.isfinite(q[0]) and np.isfinite(q[1]) else np.nan,"post_change":q[2]-q[1] if np.isfinite(q[1]) and np.isfinite(q[2]) else np.nan,"post_minus_pre":q[2]-2*q[1]+q[0] if all(np.isfinite(q)) else np.nan})
 return pd.DataFrame(r)

def perm(s,seed,n=999):
 a=s.dropna().to_numpy(float)
 if len(a)<5:return np.nan
 rng=np.random.default_rng(seed);o=abs(a.mean());return (1+sum(abs((a*rng.choice((-1.,1.),len(a))).mean())>=o for _ in range(n)))/(n+1)

def operator(r):
 if r.empty:return pd.DataFrame()
 keys=["source","target","source_family","target_family","lag_s","channel"];out=[]
 for key,g in r.groupby(keys):
  v=g.post_minus_pre.dropna()
  if len(v)>=5:out.append({**dict(zip(keys,key)),"n_events":len(v),"weight":v.mean(),"se":v.std(ddof=1)/math.sqrt(len(v)),"sign_consistency":max((v>0).mean(),(v<0).mean()),"p_value":perm(v,abs(hash(key))%(2**32))})
 z=pd.DataFrame(out)
 if z.empty:return z
 z["q_value"]=np.nan
 for _,idx in z.groupby(["channel","lag_s"]).groups.items():z.loc[idx,"q_value"]=bh(z.loc[idx,"p_value"])
 z["candidate_pass"]=(z.q_value<=.05)&(z.sign_consistency>=.65);z["semantic_status"]="FREE_DATA_PROXY_OPERATOR; DIRECTION_NOT_CAUSALLY_IDENTIFIED";return z

def write(x,p):p.parent.mkdir(parents=True,exist_ok=True);x.to_csv(p,index=False,compression="gzip" if p.suffix==".gz" else None)

def run(ticks:Path,off:Path|None,up:Path,out:Path):
 out.mkdir(parents=True,exist_ok=True);u=universe(up);raw,lin=load_ticks(ticks,u);f=residual(align(raw)) if not raw.empty else pd.DataFrame();o=official(off,{v.commodity_id:v for v in u.values()});e=events(f) if not f.empty else pd.DataFrame();r=responses(f,e);op=operator(r);vol=not f.empty and f.volume_imbalance.notna().any();ledger=pd.DataFrame([("midprice","VALID" if not f.empty else "PARTIAL"),("spread","VALID" if not f.empty else "PARTIAL"),("quote_activity","VALID" if not f.empty else "PARTIAL"),("volume_imbalance","PROXY_VALID" if vol else "PARTIAL")]+[(q,"BLOCKED_BY_DATA") for q in sorted(BLOCKED)],columns=["object","status"]);inst=f.instrument_id.nunique() if not f.empty else 0;edges=int(op.candidate_pass.sum()) if not op.empty else 0;fan=inst==8 and vol and edges>0;summary={"status":"complete","instruments":int(inst),"field_rows":len(f),"official_rows":len(o),"events":len(e),"responses":len(r),"operator_rows":len(op),"candidate_edges":edges,"free_fan_status":"NONCANONICAL_CANDIDATE" if fan else "BLOCKED","structural_validation":True,"completeness":{"eight_instrument_empirical_coverage":inst==8,"quoted_volume_channel_available":bool(vol),"official_external_fields_available":not o.empty}}
 for x,n in [(f,"FIELDS_1S.csv.gz"),(o,"OFFICIAL_OBSERVATIONS.csv.gz"),(e,"P1_EVENTS.csv.gz"),(r,"P2_RESPONSES.csv.gz"),(op,"TRANSMISSION_OPERATOR_CANDIDATE.csv")]:write(x,out/n)
 ledger.to_csv(out/"OBSERVABILITY_LEDGER.csv",index=False);(out/"LINEAGE.json").write_text(json.dumps({"tick_files":lin,"pipeline_version":"3.7"},indent=2,default=str));(out/"SUMMARY.json").write_text(json.dumps(summary,indent=2));return summary

def main():
 p=argparse.ArgumentParser();p.add_argument("--ticks-root",type=Path,required=True);p.add_argument("--official-root",type=Path);p.add_argument("--universe",type=Path,default=Path(__file__).parent/"schema/universe.json");p.add_argument("--out",type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.ticks_root,a.official_root,a.universe,a.out),indent=2))
if __name__=="__main__":main()
