# Market Metastability v3.8 — full-year free commodity proxy operator

## 1. Scope

v3.8 is the first full-year empirical execution of the free commodity-field architecture. It processes eight commodity-related quote carriers for calendar year 2024: Brent, WTI/light crude, natural gas, copper, gold, silver, soybean and coffee.

The carrier is Dukascopy bid/ask quote ticks with quoted bid and ask volumes. It is not CME/ICE futures MBO. All conclusions therefore concern a free-data quote-field proxy, not native exchange order flow.

## 2. Data actually processed

- 118,140,732 native ticks;
- 96 complete instrument-month files;
- all files contain `askPrice`, `bidPrice`, `askVolume` and `bidVolume`;
- 247 common valid trading days and one coverage rejection;
- common analysis window 13:30–17:00 UTC;
- 5-second operational state;
- 5,092 vector P1 excitations;
- 356,672 day-level source–target response records.

The state preserves midquote, spread, quote-update intensity, bid/ask movements, total quoted volume and quoted-volume imbalance. It does not reconstruct a book or individual orders.

## 3. Operator and primary result

Returns are residualized against leave-one-out global and family modes. Every other instrument is measured at 5, 15, 30, 60 and 120 seconds in residual return, relative spread, quote activity, quoted-volume imbalance, log quoted volume and quote pressure. Inference uses day blocks, sign-flip nulls and BH-FDR within channel × lag.

The initial annual screen produced 57 candidate lag cells. A 9,999-permutation same-sample robustness pass retained 56. None belongs to residual return, spread, quoted-volume imbalance or quote pressure. The robust cells are 47 quote-activity cells, all negative, and 9 log-volume cells with mixed sign.

Thus v3.8 does not find a reliable price-transmission operator. Its main empirical object is a common relaxation field: after large source excitations, quote-update activity in several targets tends to decrease relative to the preceding interval.

## 4. Directional gate

Four paired asymmetries survived the coarse screen. Two were removed because neither direction on the same pair/channel/lag passed the operator gate. Two remain as strict proxy-direction candidates.

### Copper → Brent, activity, 5 seconds

- annual post-minus-pre activity effect −0.5329;
- 94 days and 164 source events;
- first-half mean −0.6083, sign-flip p = 0.00150;
- second-half mean −0.4435, sign-flip p = 0.02285;
- circular-time placebo mean −0.1152;
- observed-minus-placebo −0.4177, p = 0.00060.

### Copper → silver, activity, 5 seconds

- annual post-minus-pre activity effect −0.5902;
- 94 days and 164 source events;
- first-half mean −0.7681, sign-flip p = 0.00005;
- second-half mean −0.3793, sign-flip p = 0.00315;
- circular-time placebo mean −0.0673;
- observed-minus-placebo −0.5229, p = 0.00010.

These statements mean only that extreme copper quote-field states are followed by a faster decline in Brent and silver quote activity than expected from matched circular shifts. They do not establish economic causation, exchange-to-exchange latency or order-flow transmission.

## 5. Point-in-time context and geometry

The official layer contains 8,802 normalized CFTC/EIA observations and 20,748 daily as-of records with zero look-ahead violations. Nine post-discovery regime tests examined positioning and petroleum fields; none survived exploratory FDR.

The 56 robust lag cells collapse to 15 unique directed pairs. Directed density is 0.268, reciprocity 0.667, largest strongly connected component 5/8, effective singular rank 2.438, first singular energy 0.702 and negative-edge fraction 0.933. Copper has the highest outgoing absolute strength. Soybean and coffee are isolated under the strict annual gate.

## 6. Architectural status

- F0 quote ticks: COMPLETE-PROXY.
- F1 multichannel field: COMPLETE-PROXY.
- P1 excitations and P2 responses: COMPLETE-PROXY.
- Annual operator: SUPPORTED only for activity/log-volume relaxation.
- Price transmission: NOT-SUPPORTED.
- Directed transport: PARTIAL-PROXY, two candidates.
- A1–A3: PROXY-OPERATIONAL.
- A4: CANDIDATE-SUPPORTED.
- A5: PARTIAL-PROXY.
- A6: CANDIDATE.
- A7: QUARANTINE.
- Canonical Fan: BLOCKED.
- Free-data Fan: CANDIDATE-ONLY.
- L5/trading: NOT-STARTED.

## 7. Next falsification step

The frozen v3.8 protocol must run on 2023 and 2025 without changing thresholds after seeing results. A direction advances only if its sign, 5-second lag and activity channel replicate in both years and survives a second carrier or venue. Hawkes/propagator fitting remains downstream of that gate.
