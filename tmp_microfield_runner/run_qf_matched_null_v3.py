#!/usr/bin/env python3
from pathlib import Path

base = Path(__file__).resolve().parent / 'run_qf_matched_null_v2.py'
source = base.read_text()
source = source.replace("OUT=ROOT/'tmp_microfield_results_qf_null_v2'", "OUT=ROOT/'tmp_microfield_results_qf_null_v3'")
source = source.replace("for i in events:blocked[max(0,i-maxh):min(len(x),i+maxh+1)]=True", "for i in events:blocked[max(0,i-100):min(len(x),i+101)]=True")
source = source.replace("'boundary_exclusion_ms':maxh", "'boundary_exclusion_ms':maxh,'control_event_exclusion_radius_ms':100")
namespace = {'__file__': str(base), '__name__': '__main__'}
exec(compile(source, str(base), 'exec'), namespace)
