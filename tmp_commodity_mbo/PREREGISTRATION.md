# v3.6 commodity MBO pilot — preregistration

## Purpose

Validate access to a native event-by-event CME Globex MBO stream for eight commodity futures families before any transmission or metastability claim.

## Universe

`CL.v.0`, `NG.v.0`, `GC.v.0`, `HG.v.0`, `ZC.v.0`, `ZS.v.0`, `LE.v.0`, `HE.v.0`.

The continuous symbols are used only as point-in-time resolvers of the volume-ranked active expiry. The returned raw expiry symbol and instrument identity must be retained. No back-adjusted or stitched price series is created.

## Pilot interval

2026-07-14 14:30:00–14:31:00 UTC, a common active interval for the selected day-session products.

## Input

- Dataset: `GLBX.MDP3`
- Schema: `mbo`
- Native event timestamps and sequence numbers
- Adds, cancels, modifications, fills, trades and reset records
- Individual order IDs, side, price and size when supplied by the feed

## Automatic spending gate

The workflow requests free metadata first. It downloads time-series data only if Databento's estimated charge does not exceed USD 5.00. If the key is missing or the cap is exceeded, it records a blocker and incurs no time-series charge.

## Outputs

- raw DBN/Zstd stream and SHA-256;
- exact schema and a 1,000-record inspection sample;
- event counts and sizes by instrument/action/side;
- timestamp audit;
- completed order-life-cycle candidates;
- 1 ms, 10 ms and 100 ms activity representations;
- descriptive cross-asset common-mode audit.

## Non-claims

This one-minute pilot does not establish a causal transmission operator, branching, gain, dissipation, metastability, full order-book geometry, Section/Landscape/Tempo, obstruction, Fan or trading logic. Full book state is not claimed unless a valid initial snapshot or session-start replay is included.
