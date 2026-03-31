# Harness Cases

This reference defines case IDs and minimum expectations for the `hydromodel-mcp-harness` skill.

## Case catalog

- `H-01` model discovery surface
- `H-02` workspace listing surface
- `H-03` readiness check for simulation
- `H-04` v2 simulation csv output
- `H-05` metrics computation path
- `H-06` simulation redis output path
- `H-07` calibration readiness stop rule
- `H-08` v2 low-cost calibration path
- `H-09` v2 validation path
- `H-10` ensemble parameters path
- `H-11` resources/templates/prompts surface
- `H-12` negative legacy simulation payload
- `H-13` negative legacy calibration payload
- `H-14` strict inference negative path
- `H-15` clear session cache
- `H-16` stage2 calibration auto split path
- `H-17` stage2 calibration period split path

## Universal assertions

For every executed case:

- record expected call sequence
- record actual call sequence
- mark first mismatch precisely
- classify severity (`P0`, `P1`, `P2`)
- stop downstream calls on blocking failure when `fail_fast=true`

## Suggested artifact output

- human report section in final response
- machine report path example: `./result/harness/harness_report_<timestamp>.json`
