# Point-in-time contract

Every non-tick observation has two different clocks:

- `reference_time`: the period or physical state described by the value;
- `available_time`: the first UTC instant when the model could legally know the value.

A feature at time `t` may use only rows with `available_time <= t`. Revisions remain separate observations and do not overwrite earlier vintages. A weekly inventory number referring to the previous week therefore enters the model only at its actual publication time.

Tick observations use their provider event timestamp as both event and availability time. This does not imply exchange matching-engine time.

The pipeline rejects official input where `available_time < reference_time`, preserves source hashes, and never forward-fills external releases into periods before publication.
