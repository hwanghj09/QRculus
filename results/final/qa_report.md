# 자동 검증 T01–T19

| ID | 결과 | 실행 근거 |
|---|---|---|
| T01 | PASS | 233 reserved / 208 eligible |
| T02 | PASS | Four 21×21 binary matrices |
| T03 | PASS | 290×290 grayscale, 40px white borders |
| T04 | PASS | Four original images decoded with OpenCV |
| T05 | PASS | 0%, 2%, 10% → 0, 4, 21; 0% pixels unchanged |
| T06 | PASS | Reserved modules and quiet zone preserved |
| T07 | PASS | Matrix hashes unchanged |
| T08 | PASS | Same seed yields same pixel hash |
| T09 | PASS | Shared coordinates across levels |
| T10 | PASS | Nested damage across the target grid |
| T11 | PASS | Exact-match classification |
| T12 | PASS | 144 attempts and 72 groups, including hashes and exact-match classifications |
| T13 | PASS | 7,200 attempts and 72 groups with complete keys |
| T14 | PASS | 72 aggregates and 7,200 pixel hashes reproduced |
| T15 | PASS | Four integrals and 480 supplementary attempts checked |
| T16 | PASS | 5u³ − 25u² + 100, interior u=5/3, local mean 58.75% |
| T17 | PASS | Four readable PNGs with explicit damage units and local validity caption |
| T18 | PASS | Eight sections, source URLs, run provenance and measured results |
| T19 | PASS | 3 A4 pages, rendered text 10.0–11.0 pt, title and identity labels, provided identity verified |
