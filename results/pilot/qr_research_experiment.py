"""QR 코드 손상 실험: 데이터 영역 모듈 반전 비율과 복원 성공률.
필요: pip install qrcode[pil] opencv-python numpy matplotlib
실행: python qr_research_experiment.py --trials 20 --out qr_results

고정 조건: 버전 1 (21x21), 메시지 TEST01, 마스크 패턴 1,
           동일한 데이터 영역 모듈 선택 순서, OpenCV QRCodeDetector.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import cv2
import qrcode
import matplotlib.pyplot as plt

LEVELS = {
    'L': qrcode.constants.ERROR_CORRECT_L,
    'M': qrcode.constants.ERROR_CORRECT_M,
    'Q': qrcode.constants.ERROR_CORRECT_Q,
    'H': qrcode.constants.ERROR_CORRECT_H,
}
MESSAGE = 'TEST01'
VERSION = 1
MODULES = 21
SCALE = 10
QUIET = 4


def data_coordinates():
    """버전1 QR의 위치·분리·타이밍·형식 정보 모듈은 손상 대상에서 제외한다."""
    reserved = np.zeros((MODULES, MODULES), dtype=bool)
    # 위치 탐지 무늬 및 분리 영역
    reserved[0:8, 0:8] = True
    reserved[0:8, 13:21] = True
    reserved[13:21, 0:8] = True
    # 형식 정보 비트
    reserved[8, 0:9] = True
    reserved[8, 13:21] = True
    reserved[0:9, 8] = True
    reserved[13:21, 8] = True
    # 타이밍 패턴, 고정된 검은 모듈
    reserved[6, :] = True
    reserved[:, 6] = True
    reserved[13, 8] = True
    return np.argwhere(~reserved)


def make_qr(level):
    qr = qrcode.QRCode(
        version=VERSION,
        error_correction=LEVELS[level],
        box_size=SCALE,
        border=QUIET,
        mask_pattern=1,
    )
    qr.add_data(MESSAGE)
    qr.make(fit=False)
    modules = np.asarray(qr.modules, dtype=np.uint8)
    if modules.shape != (MODULES, MODULES):
        raise RuntimeError('QR 버전이 변경되었습니다.')
    return modules


def render(modules):
    canvas = np.pad(modules, QUIET, constant_values=0)
    pixels = np.kron(canvas, np.ones((SCALE, SCALE), dtype=np.uint8))
    return np.asarray((1 - pixels) * 255, dtype=np.uint8)


def run(trials, out):
    out.mkdir(parents=True, exist_ok=True)
    coords = data_coordinates()
    possible = len(coords)
    # 데이터 영역 기준 변경 비율(%). 작은 QR에서 복원 실패 전환이 빠르므로 조밀하게 측정한다.
    rates = [x / 2 for x in range(0, 17)] + [10]
    detector = cv2.QRCodeDetector()
    records = []
    summary = []

    for level in LEVELS:
        base = make_qr(level)
        if detector.detectAndDecode(render(base))[0] != MESSAGE:
            raise RuntimeError(f'{level} 등급의 원본조차 인식되지 않습니다. 실행 환경을 확인하세요.')
        cv2.imwrite(str(out / f'original_{level}.png'), render(base))
        counts = {rate: 0 for rate in rates}
        for trial in range(trials):
            # 모든 등급에 동일한 시드와 변경 순서를 적용한다.
            order = np.random.default_rng(2026 + trial).permutation(possible)
            for rate in rates:
                flip_count = round(possible * rate / 100)
                selected = coords[order[:flip_count]]
                damaged = base.copy()
                if flip_count:
                    yy, xx = selected.T
                    damaged[yy, xx] ^= 1
                decoded, _, _ = detector.detectAndDecode(render(damaged))
                success = int(decoded == MESSAGE)
                counts[rate] += success
                records.append((level, trial, rate, flip_count, possible, flip_count / possible * 100, success))
                if trial == 0 and level == 'H' and rate in (0, 3, 6, 10):
                    cv2.imwrite(str(out / f'sample_H_{rate:g}pct.png'), render(damaged))
        for rate in rates:
            summary.append((level, rate, round(possible * rate / 100) / possible * 100, counts[rate], trials, counts[rate] / trials * 100))
        print(level, '완료:', [(r, counts[r]) for r in (0, 2, 4, 6, 8, 10)])

    with (out / 'trials.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['level', 'trial', 'target_pct', 'changed_modules', 'eligible_modules', 'actual_pct', 'correct'])
        writer.writerows(records)
    with (out / 'success_rate.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['level', 'target_pct', 'actual_pct', 'correct_count', 'trials', 'success_pct'])
        writer.writerows(summary)

    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    for level in LEVELS:
        items = [r for r in summary if r[0] == level and r[1] <= 10]
        ax.plot([r[1] for r in items], [r[5] for r in items], marker='o', ms=3, label=level)
    ax.set(xlabel='Target changed data-module ratio (%)', ylabel='Correct decoding (%)',
           title=f'QR code recovery after data-module flips (n={trials} each)',
           xlim=(0, 10), ylim=(-3, 103))
    ax.grid(True, alpha=.3)
    ax.legend(title='Error correction')
    fig.tight_layout()
    fig.savefig(out / 'success_curves.png', dpi=180)
    plt.close(fig)

    # 각 구간을 직선으로 연결하면 정적분은 사다리꼴 넓이의 합으로 정확히 계산 가능.
    print('\n0~10% 구간 평균 복원 성공률 — 구간별 일차함수의 정적분')
    for level in LEVELS:
        arr = sorted([r for r in summary if r[0] == level and r[1] <= 10], key=lambda r: r[1])
        xs = np.array([r[1] for r in arr], dtype=float)
        ys = np.array([r[5] for r in arr], dtype=float)
        average = float(np.sum((ys[:-1] + ys[1:]) / 2 * (xs[1:] - xs[:-1])) / (xs[-1] - xs[0]))
        slopes = np.diff(ys) / np.diff(xs)
        k = int(np.argmin(slopes))
        print(f'{level}: 평균 {average:.2f}%, 최급감 구간 {xs[k]:g}~{xs[k+1]:g}%p, 기울기 {slopes[k]:.2f}')
    print('\n저장 위치:', out.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20, help='조건별 반복 횟수')
    parser.add_argument('--out', type=Path, default=Path('qr_results'))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials는 1 이상이어야 합니다.')
    run(args.trials, args.out)
