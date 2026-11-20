# QRculus 2.0 자동 검증

| 항목 | 결과 | 근거 |
|---|---|---|
| T01 | PASS | 100 unique six-character payloads reproduced |
| T02 | PASS | 400 generated binary QR matrices, RGB dimensions and correct baseline records |
| T03 | PASS | All 2,000 mask orders and 15 prefix sizes verified |
| T04 | PASS | Exact #FF69B4 pixel counts and untouched sample quiet zones |
| T05 | PASS | Unique scattered cells and 4-neighbor growth at each cluster step |
| T06 | PASS | Nested masks; all level rows checked against shared seed and coordinate order in the raw audit |
| T07 | PASS | All original and 112,400 final RGB SHA-256 values reproduced |
| T08 | PASS | 400 distinct exact decodes at 0%; baseline evidence equals raw rows |
| T09 | PASS | 112,400 complete trial keys and decoder diagnostics verified |
| T10 | PASS | All 116 condition aggregates verified against raw measurements |
| T11 | PASS | Actual coverage recomputed from 441 cells, valid per-condition rates checked in T10 |
| T12 | PASS | Exact-match success and independent outcome flags checked for every attempt |
| T13 | PASS | 80 training and 20 validation payload IDs, disjoint and reproducible |
| T14 | PASS | All eight linear models match the training knots |
| T15 | PASS | Primitive and trapezoid integrals agree for all intervals |
| T16 | PASS | Four-point cubic interpolation, derivative minimum and model rejection checked |
| T17 | PASS | Eight curve MAEs and 120 prediction rows match held-out data; no extrapolation |
| T18 | PASS | Five readable PNGs with units, provenance, sample counts and legends |
| T19 | PASS | 3 A4 pages, embedded text 10.0–11.0 pt, headings, formulas and references |
| T20 | PASS | 40 original result and report files matched the Git baseline byte-for-byte |
