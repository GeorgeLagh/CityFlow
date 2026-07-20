# v3.9 frozen independent-year replication

## Frozen input universe

`brentcmdusd`, `lightcmdusd`, `gascmdusd`, `coppercmdusd`, `xauusd`, `xagusd`, `soybeancmdusx`, `coffeecmdusx`.

Independent calendar years: 2023 and 2025. The data carrier remains free Dukascopy bid/ask quote ticks with `bidVolume` and `askVolume`. It is not CME/ICE MBO.

## Frozen protocol

No thresholds may be changed after either year is inspected.

- source session retained: 09:00–18:00 UTC;
- common operator window: 13:30–17:00 UTC;
- operational state: 5 seconds;
- state forward-fill limit: 60 seconds;
- common-day coverage gate: 0.65 for every instrument;
- leave-one-out global and family return modes;
- P1 score threshold: `max(4.5, within-day 0.9985 quantile)`;
- event cooldown: 30 seconds;
- maximum 60 events per source/day;
- lags: 5, 15, 30, 60 and 120 seconds;
- channels: residual return, relative spread, quote activity, volume imbalance, log volume and quote pressure;
- symmetric pre/post response;
- initial operator: day sign flips and BH-FDR within channel × lag;
- original v3.8 candidate thresholds remain unchanged.

## Only preregistered directional hypotheses

1. `coppercmdusd -> brentcmdusd`, quote activity, 5 seconds, negative post-minus-pre effect.
2. `coppercmdusd -> xagusd`, quote activity, 5 seconds, negative post-minus-pre effect.

A candidate passes an independent year only when all conditions hold:

- the forward operator cell passes the frozen operator gate;
- the forward effect is negative;
- its absolute magnitude exceeds the reverse-direction effect;
- effect sign-flip p <= 0.025;
- paired directional-asymmetry sign-flip p <= 0.025;
- both half-year means are negative;
- circular-time placebo p <= 0.025.

A candidate advances to a cross-venue test only when it passes both 2023 and 2025. Failure in either year gives `FALSIFIED_OR_UNSTABLE`. No substitute lag, channel, target or source may be selected in v3.9.

## Non-claims

The replication cannot identify OrderID, true cancellations, queue position, full depth, iceberg refresh, aggressor side or matching-engine latency. A successful result remains a quote-field proxy and does not establish economic causation.
