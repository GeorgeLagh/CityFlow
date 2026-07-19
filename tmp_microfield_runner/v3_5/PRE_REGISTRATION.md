# v3.5 pre-registration — cross-day event-conditioned transmission field

## Status

This protocol is frozen before the multi-day outputs are inspected. It extends the accepted v3.1 full-day Bybit protocol without changing the P1 or within-day P2 gates.

## Corpus

Twenty complete UTC days from 2024, divided into four fixed five-day shards. Instruments: BTCUSDT, ETHUSDT and SOLUSDT. Carrier: 100 ms. Native archives are downloaded from Bybit public trade data, hashed, streamed into the carrier, and deleted after aggregation.

The dates are fixed in `run_multiday_replication.py`. They span ordinary, high-activity and event-rich intervals. Failed or missing dates are reported, not silently replaced.

## P1

For each instrument and UTC hour, robust center and scale are estimated on active, nonzero-notional bins. A candidate requires:

- `|z_flow| >= 6`;
- `z_activity >= 3`;
- one-second refractory consolidation.

Source events within ±100 ms of a P1 in another instrument are removed.

## Within-day P2

For each directed source-target pair:

- target return and signed flow are contemporaneously residualized against the other two instruments;
- cumulative responses are tested at 100, 250, 500, 1,000, 2,000, 5,000 and 10,000 ms;
- 199 circular shifts form the alignment null;
- 499 event bootstraps form uncertainty intervals;
- BH-FDR is applied jointly to all return and flow hypotheses within a day;
- P2 requires `q < 0.05` and a bootstrap interval excluding zero.

## Cross-day operator candidate

Each source-target-lag-response cell is pooled with a random-effects meta-analysis. A lag is marked `replicated_candidate` only when all conditions hold:

- at least 15 completed dates;
- cross-cell BH-FDR `q < 0.05`;
- directional sign consistency at least 0.75;
- within-day P2 frequency at least 0.25;
- leave-one-day-out sign stability at least 0.80.

A channel is marked a replicated candidate only if at least two lags pass.

## Interpretation boundary

The output is a cross-day, event-conditioned transmission-operator candidate. It is not:

- a Hawkes branching matrix;
- proof of self-sustaining gain;
- a canonical L4 field;
- a basis for curvature or topology unless the replication gate passes;
- `Section`, `Landscape`, `Tempo`, obstruction, `Fan(t)` or a trading signal.

Negative results and high heterogeneity remain first-class outputs.
