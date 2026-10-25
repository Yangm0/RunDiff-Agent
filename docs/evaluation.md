# Evaluation protocol

examples/cases/<case-id>/run_a and run_b are agent inputs. The sibling expectations.json is a test-only answer key and is loaded only after the case analysis and model calls finish. The catalog served by the API contains only case IDs and titles.

The runner records deterministic baseline identification and the selected multi-step mode separately. Per-case values include exact planted-change recall, key-difference precision, evidence validator status, unsupported claim count, tool success rate, call count, decision rounds, and latency. When credentials are configured, a one-shot live baseline is also attempted for Mock Mode evaluation; its status and token use are recorded when available.

Mock evaluation means graph-routing and tool-plumbing validation only. Live scores require successful API calls for the reported case and are never inferred from Mock Mode. This small synthetic suite is not a statistically meaningful model benchmark.

