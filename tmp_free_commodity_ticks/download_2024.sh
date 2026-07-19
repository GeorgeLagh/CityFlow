#!/usr/bin/env bash
set -euo pipefail

instrument="${1:?instrument required}"
year="${2:-2024}"
out="free_commodity_ticks/${instrument}/${year}"
mkdir -p "$out"

python - "$year" > months.tsv <<'PY'
import calendar, sys
from datetime import date, timedelta
y=int(sys.argv[1])
for m in range(1,13):
    start=date(y,m,1)
    end=date(y+1,1,1) if m==12 else date(y,m+1,1)
    print(f"{m:02d}\t{start.isoformat()}\t{end.isoformat()}")
PY

printf 'instrument,year,month,from,to,status,rows,bytes,sha256,file\n' > "$out/MANIFEST.csv"

while IFS=$'\t' read -r month from to; do
  work="$(mktemp -d)"
  pushd "$work" >/dev/null
  set +e
  npx --yes dukascopy-node@1.49.0 -i "$instrument" -from "$from" -to "$to" -t tick -f csv > run.log 2>&1
  code=$?
  set -e
  csv="$(find . -maxdepth 2 -type f -name '*.csv' -printf '%s %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ $code -ne 0 || -z "$csv" || ! -s "$csv" ]]; then
    cp run.log "$OLDPWD/$out/${year}-${month}.log"
    printf '%s,%s,%s,%s,%s,failed,0,0,,\n' "$instrument" "$year" "$month" "$from" "$to" >> "$OLDPWD/$out/MANIFEST.csv"
    popd >/dev/null
    rm -rf "$work"
    continue
  fi
  rows=$(wc -l < "$csv")
  target="$OLDPWD/$out/${instrument}-${year}-${month}-tick.csv.gz"
  gzip -c -9 "$csv" > "$target"
  bytes=$(stat -c '%s' "$target")
  hash=$(sha256sum "$target" | awk '{print $1}')
  cp run.log "$OLDPWD/$out/${year}-${month}.log"
  printf '%s,%s,%s,%s,%s,ok,%s,%s,%s,%s\n' "$instrument" "$year" "$month" "$from" "$to" "$rows" "$bytes" "$hash" "$(basename "$target")" >> "$OLDPWD/$out/MANIFEST.csv"
  popd >/dev/null
  rm -rf "$work"
done < months.tsv

python - "$out/MANIFEST.csv" "$out/SUMMARY.json" <<'PY'
import csv,json,sys
p,out=sys.argv[1:]
rows=list(csv.DictReader(open(p)))
ok=[r for r in rows if r['status']=='ok']
summary={
  'instrument': rows[0]['instrument'] if rows else None,
  'year': int(rows[0]['year']) if rows else None,
  'months_requested': len(rows),
  'months_ok': len(ok),
  'rows_total': sum(int(r['rows']) for r in ok),
  'compressed_bytes_total': sum(int(r['bytes']) for r in ok),
  'source_semantics': 'Dukascopy bid/ask quote ticks with quoted volumes; CFD/spot carrier, not exchange-native futures MBO',
  'paid_api_used': False,
}
open(out,'w').write(json.dumps(summary,indent=2))
if len(ok)==0:
    raise SystemExit('no successful months')
PY
