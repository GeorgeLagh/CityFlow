# Market Metastability v3.7 — Unified Commodity Data Fabric

This layer joins free quote ticks, delayed exchange controls and official physical/fundamental observations without erasing their different semantics.

Core rules:

1. Every external observation has `reference_time` and `available_time`.
2. Point-in-time materialization uses only `available_time <= asof`.
3. Missing quoted volumes remain missing; they are never converted to zero.
4. CFD/spot quote ticks are proxy carriers and are not represented as CME/ICE MBO.
5. Order IDs, queue position, true cancellation flow, iceberg refresh, full depth and aggressor side remain `BLOCKED_BY_DATA`.
6. Common and family modes are removed before a pairwise residual operator is estimated.
7. The output operator is a free-data proxy candidate, not a causal exchange operator.
8. A1–A6 outputs are noncanonical operational candidates; A7 stays in quarantine and L5 is absent.

Canonical observation schema:

`commodity_id, instrument_id, family, source, venue, observation_layer, field, value, unit, reference_time, available_time, quality_status, source_hash`

Fixture run:

```bash
python build_fixture.py --out fixture --universe schema/universe.json
python unified_fabric.py --ticks-root fixture --official-root fixture/official --out result
```
