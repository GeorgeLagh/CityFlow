#!/usr/bin/env bash
set -euo pipefail
root="${1:-v37_fixture}"
rm -rf "$root"
python tmp_unified_commodity_fabric/build_fixture.py --out "$root/input" --universe tmp_unified_commodity_fabric/schema/universe.json
python tmp_unified_commodity_fabric/unified_fabric.py --ticks-root "$root/input" --official-root "$root/input/official" --out "$root/output"
python -m pytest -q tmp_unified_commodity_fabric/tests
python - "$root/output/SUMMARY.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['instruments']==8
assert p['events']>=100
assert p['responses']>0
assert p['candidate_edges']>0
assert p['free_fan_status']=='NONCANONICAL_CANDIDATE'
print('fixture semantic gates passed',p['candidate_edges'])
PY
