# Strict Suite Map

This reference maps the strict HydroModelMCP scenario cards to their main test focus.

- `S01`: unsupported exact parameter locking safety gate
- `S02`: Caravan basin 80/20 calibration and validation
- `S03`: Caravan basin sensitivity plus storage audit
- `S04`: explicit-parameter validation without hidden fallback
- `S05`: calibration readiness fail-fast on incomplete CSV inputs
- `S06`: strict inference rejection plus manual mapping recovery
- `S07`: session cache cleanup and no hidden parameter reuse
- `S08`: fair-budget cross-model comparison on one Caravan basin

Authoritative scenario definitions live in `examples/strict-interaction-mcp-tests/`.
