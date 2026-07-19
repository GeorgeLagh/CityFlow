# v3.6 free commodity tick carrier

This execution branch downloads free Dukascopy tick histories for eight commodity-related instruments for calendar year 2024.

Universe:

- `brentcmdusd` — Brent CFD
- `lightcmdusd` — WTI/light crude CFD
- `gascmdusd` — natural gas CFD
- `coppercmdusd` — copper CFD
- `xauusd` — spot gold
- `xagusd` — spot silver
- `soybeancmdusx` — soybean CFD
- `coffeecmdusx` — coffee CFD

The files contain tick-time bid/ask prices and quoted volumes as exposed by Dukascopy. They are free, large and suitable for event-time excitation, spread, quote-intensity, common-mode and cross-family response experiments.

They are **not** CME/ICE native futures messages, do not contain exchange order IDs, queue position, add/cancel actions or full order-book depth, and must not be represented as MBO. The carrier therefore complements rather than replaces the blocked exchange-native MBO branch.

The workflow downloads each instrument in monthly chunks, stores compressed raw CSV files, computes SHA-256 and row counts, and uploads one artifact per instrument. No paid API or secret is used.
