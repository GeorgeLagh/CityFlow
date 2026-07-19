# v3.6.1 free official commodity data expansion

This execution-only branch collects additional **free official** data layers for the commodity-field project. It does not modify the canonical ontology, mathematics, L3 registry, agent layer, Fan contract, or L5.

## Sources

- CFTC annual Disaggregated Commitments of Traders, futures only, 2018–latest.
- USDA monthly historical WASDE vintages, April 2010–latest.
- USDA NASS QuickStats bulk files, streamed and filtered to corn, soybeans, coffee, cattle and hogs.
- EIA bulk Petroleum, Natural Gas, Short-Term Energy Outlook, Crude Oil Imports and Total Energy datasets; no API key.
- LME warehouse-company stocks and queue reports, 2018–latest.
- World Bank Pink Sheet monthly and annual historical workbooks.
- FRED daily macro controls: dollar index, rates, inflation expectations and volatility.
- MOEX ISS delayed derivatives metadata, market data, anonymous trades and order-book snapshots for dynamically detected commodity-related futures.

## Semantics

These are slow, medium-frequency, delayed-exchange or snapshot control layers. They complement the Dukascopy tick carrier. They do not create CME/ICE order IDs, exchange queue position, native cancellation flow, matching-engine timestamps, or a full historical MBO book.

Every downloaded or derived file receives source URL, retrieval timestamp, byte size and SHA-256 in a manifest. Downloads are best-effort and a failed source is recorded rather than silently reconstructed.
