# v3.5 multi-day replication runner

The runner reproduces the v3.1 full-day Bybit protocol on 20 preselected UTC days and forms a cross-day random-effects transmission-operator candidate.

Local checks:

```bash
python -m pip install numpy pandas scipy
python run_multiday_replication.py --self-test
```

Shard execution:

```bash
python run_multiday_replication.py --shard 0 --out results/shard_0
```

Aggregation:

```bash
python run_multiday_replication.py --aggregate collected --out results/aggregate
```

Raw exchange archives are removed after each instrument-day is aggregated. Only hashes, row counts and compact results are retained.
