# 기존 실험·문서 안내 (기록 보존)

2026년 미적분Ⅰ 제출용 보고서와 **직접 일치하는 문서**는 [../report/](../report/)에 있습니다. 이 문서는 이전 실험·개발 자료를 구분하기 위한 색인입니다.

| 경로 | 구분 | 현재 처리 |
| --- | --- | --- |
| [results/pink_v2](../results/pink_v2) | **현 보고서의 원본 실험** | 보존. `raw/trials.csv` (112,400행), `raw/mask_orders.csv`, 집계 CSV 및 검증 분리 정보를 삭제하면 재현 불가 |
| [src/qr_calculus/pink_*](../src/qr_calculus) | 분홍 가림 실험 재현·검증 코드 | 보존. 이번 보고서의 112,400회 결과 생성에 사용된 기존 구현 |
| [notebooks/QRculus_multivariate_colab.ipynb](../notebooks/QRculus_multivariate_colab.ipynb) | **후속 다변수 머신러닝** | 보존. 수학적 도함수 모델과는 별도 분석 |
| [reports/pink_v2_3pages.md](../reports/pink_v2_3pages.md) 및 [PDF](../reports/pink_v2_3pages.pdf) | 분홍 가림 2.0의 **이전 제출 형식** | 역사적 참고용. 현재 제출 PDF와 내용·수학 모델이 다름 |
| [reports/통합_탐구보고서.md](../reports/통합_탐구보고서.md) 및 [PDF](../reports/통합_탐구보고서.pdf) | 기존 통합 보고서 | 역사적 참고용 |
| [reports/보고서_3쪽_원고.md](../reports/보고서_3쪽_원고.md) 및 [PDF](../reports/보고서_3쪽.pdf) | 최초 QR 모듈 반전 연구 | 역사적 참고용. **분홍색 가림 실험과 다른 데이터** |
| [results/final](../results/final) | 최초 7,200회 모듈 반전 본실험 | 보존. 현재 보고서 결과에는 섞지 않음 |
| [results/pilot](../results/pilot) | 예비 실험 | 보존. 현재 보고서에 사용하지 않음 |
| [specs](../specs) | 과거 연구계획서·개발 명세 | 보존. 연구 과정 기록 |
| [tests](../tests) | 기존 실행 코드 단위·종단 테스트 | 보존. 이전 실험 재현성 유지 |

### 왜 무작정 삭제하지 않았나?

- `src/qr_calculus`와 원본 CSV는 최종 보고서의 **증거이자 재현 근거**입니다.
- QRculus 2.0 Colab은 29MB 원자료와 3.4MB 가림 위치 데이터를 직접 읽습니다. 삭제·경로 변경은 실행을 중단시킵니다.
- 기존 PDF는 현재 제출본과 다르지만 어떤 탐구 과정을 거쳤는지 보여 주며, Git 과거 이력과 비교할 수 있습니다.
- **눈에 보이는 시작점**을 새 [README](../README.md), [최종 보고서](../report/), [보고서 재현 코드](../report/reproduce.py)로 통일했습니다.

과거의 길었던 루트 README는 [정리 전 버전](https://github.com/hwanghj09/QRculus/blob/7a6d9800c0d9aec98f9d6fe41d73a9fd57448665/README.md)에서 그대로 볼 수 있습니다.
