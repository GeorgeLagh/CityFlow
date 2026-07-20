from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unified_fabric import BLOCKED, rz, official, C

def test_missing_channel_remains_missing():
    out=rz(pd.Series([np.nan,np.nan,np.nan]))
    assert out.isna().all()

def test_native_objects_are_blocked():
    assert {'order_id','queue_position','true_cancel_flow','full_depth_book','aggressor_side'} <= BLOCKED

def test_official_lookahead_is_rejected(tmp_path):
    p=tmp_path/'bad.csv'
    pd.DataFrame([{'commodity_id':'WTI','field':'stocks','value':1,'unit':'bbl','reference_time':'2024-01-10T00:00:00Z','available_time':'2024-01-09T00:00:00Z'}]).to_csv(p,index=False)
    byid={'WTI':C('WTI','lightcmdusd','energy','CFD_QUOTE_TICKS','NYMEX_PROXY')}
    with pytest.raises(ValueError):official(tmp_path,byid)
