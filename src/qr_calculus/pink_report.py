"""Three-page Korean report composed from the recorded v2 results."""

from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import LEVELS, read_json
from .experiment import read_csv
from .pink_experiment import REPO, ensure_result_destination
from .pink_figures import FIGURES
from .pink_payloads import MASK_TYPES
from .report import CSS, identity_values, render_block

TITLE = "무작위 QR코드의 분홍색 가림 면적과 가림 형태에 따른 판독 성공률"
SECTIONS = ("동기·목적", "실험 설계", "탐구 내용", "국소 삼차함수", "정적분·결과",
            "알게 된 점·느낀 점", "한계·제언", "참고문헌·출처")
SHAPE_KO = {"scattered": "분산형", "clustered": "집중형"}
SOURCES = (
    ("DENSO WAVE, 오류 복원 기능", "https://www.qrcode.com/ko/about/error_correction.html"),
    ("DENSO WAVE, QR코드 정보량과 버전", "https://www.qrcode.com/ko/about/version.html"),
    ("OpenCV, QRCodeDetector API", "https://docs.opencv.org/4.x/de/dc3/classcv_1_1QRCodeDetector.html"),
    ("qrcode, PyPI", "https://pypi.org/project/qrcode/"),
    ("NumPy, Random Generator API", "https://numpy.org/doc/stable/reference/random/generator.html"),
    ("QRculus 2.0, 연구 계획과 실행 자료", "https://github.com/hwanghj09/QRculus/tree/main/results/pink_v2"),
)
LEGACY_REPORTS = {"보고서_3쪽.pdf", "보고서_3쪽_원고.md", "참고문헌.md"}


def formula(coefficients, powers):
    terms = []
    for exponent, coefficient in zip(powers, coefficients):
        if coefficient:
            sign = "−" if coefficient < 0 else "+"
            terms.append((sign, f"{abs(coefficient):.6g}{exponent}"))
    if not terms:
        return "0"
    sign, term = terms[0]
    return (sign if sign == "−" else "") + term + "".join(f" {s} {t}" for s, t in terms[1:])


def model_label(key):
    level, shape = key.split("/")
    return f"{level} {SHAPE_KO[shape]}"


def cubic_blocks(key, cubic):
    x0, x1 = cubic["x0"], cubic["x1"]
    equations = ["d=" + f"{cubic['equations'][0][1]:.3f}"]
    for u, y in cubic["equations"][1:]:
        equations.append(f"a({u:.4f})³+b({u:.4f})²+c({u:.4f})+d≈{y:.3f}")
    p = formula(cubic["coefficients"], ("u³", "u²", "u", ""))
    p_prime = formula(cubic["derivative_coefficients"], ("u²", "u", ""))
    low, high = cubic["endpoint_slopes"]
    vertex = cubic["vertex_u"]
    comparison = f"P′의 양 끝값 {low:.2f}, {high:.2f}"
    if vertex is not None and 0 < vertex < x1 - x0:
        comparison += f"와 꼭짓점 u=−b/(3a)≈{vertex:.4f}에서의 도함수 값"
    return [
        ("p", f"{model_label(key)}의 급락 구간을 감싸는 네 학습점을 택했다. "
         f"u=x−{x0:.4f}이고 유효 범위는 x={x0:.4f}~{x1:.4f}%다. 다음 네 식의 u는 반올림했다."),
        ("math", "   ·   ".join(equations[:2])),
        ("math", "   ·   ".join(equations[2:])),
        ("p", "u>0에서 q=(y−d)/u=au²+bu+c. 인접 q값의 기울기를 e₁₂, e₂₃라 하면 "
         "a=(e₂₃−e₁₂)/(u₃−u₁), b=e₁₂−a(u₁+u₂), c=q₁−au₁²−bu₁이다."),
        ("math", f"P(u)≈{p}   ·   P′(u)≈{p_prime}"),
        ("p", (
            f"{comparison}을 비교했다. "
            f"최소는 x={cubic['min_slope_pct']:.4f}%의 {cubic['min_slope']:.2f}%p/%p다. "
            + ("범위·감소 조건을 만족해 모델 추정으로 사용했다." if cubic["valid"] else
               f"성공률 범위가 {cubic['model_min_pct']:.2f}~{cubic['model_max_pct']:.2f}%이고 "
               f"최대 도함수가 {cubic['max_derivative']:.2f}이므로 채택 조건을 충족하지 못했다.")
        )),
    ]


def report_pages(directory, student_id, student_name):
    meta = read_json(directory / "run_metadata.json")
    calculations = read_json(directory / "processed/models.json")
    if meta.get("actual_calls") != 112_400 or calculations.get("run_id") != meta["run_id"]:
        raise ValueError("Report requires the complete pink experiment and matching analysis")
    records = calculations["curves"]
    models = read_csv(directory / "processed/model_summary.csv")
    holdout = read_csv(directory / "processed/holdout_metrics.csv")
    summary = read_csv(directory / "processed/summary.csv")
    coverage = read_csv(directory / "processed/finder_coverage.csv")
    model_map = {(r["level"], r["mask_type"]): r for r in models}
    selected = min(models, key=lambda row: float(row["steepest_slope"]))
    selected_key = f"{selected['level']}/{selected['mask_type']}"
    best_key = next((key for key in (selected_key, *records) if records[key]["cubic"]["valid"]), selected_key)
    cubic = records[best_key]["cubic"]
    ceiling = records[selected_key]["piecewise"]["range_pct"][1]
    results = [["등급", "분산형 평균", "집중형 평균", "집중−분산(%p)"]]
    for level in LEVELS:
        scatter, cluster = [float(model_map[level, shape]["mean_success_pct"]) for shape in MASK_TYPES]
        results.append([level, f"{scatter:.2f}%", f"{cluster:.2f}%", f"{cluster - scatter:+.2f}"])
    best = max(models, key=lambda row: float(row["mean_success_pct"]))
    min_mae, max_mae = min(float(r["linear_mae_pp"]) for r in holdout), max(float(r["linear_mae_pp"]) for r in holdout)
    all_n = sum(int(r["trials"]) for r in summary)
    missed = sum(int(r["undetected_failure_count"]) for r in summary)
    detected_failure = sum(int(r["detected_failure_count"]) for r in summary)
    scope = {r["finder_coverage"]: r for r in coverage if r["level"] == "H" and
             r["mask_type"] == "clustered" and int(r["target_pct"]) == 5}
    finder_note = ""
    if "zero" in scope and "at_least_one" in scope:
        zero, covered = scope["zero"], scope["at_least_one"]
        finder_note = (f" H 집중형(설정 5%)에서 탐지 무늬가 하나도 가려지지 않은 "
                       f"{zero['trials']}회와 한 칸 이상 가려진 {covered['trials']}회의 미검출 실패율은 "
                       f"{float(zero['undetected_failure_pct']):.2f}%, {float(covered['undetected_failure_pct']):.2f}%였다.")
    versions = meta["environment"]["library_versions"]
    seg = {key: float(selected[key]) for key in
           ("steepest_start_pct", "steepest_end_pct", "steepest_start_success_pct",
            "steepest_end_success_pct", "steepest_slope")}

    return [
        [
            ("identity", f"학번: {student_id}　 이름: {student_name}"),
            ("title", TITLE),
            ("h", "동기·목적"),
            ("p", "QR코드가 비슷한 넓이로 가려져도 읽기 결과가 달라지는 이유를 살펴보았다. "
             "가림률·형태·오류 정정 등급별 판독 성공률을 측정하고, 일차함수의 기울기와 정적분 평균을 비교했다."),
            ("p", "가림률이 증가하면 성공률은 대체로 감소하고, 가림 형태에 따라 성공률이 달라질 것으로 가설을 세웠다. "
             "탐지 무늬가 가려지면 검출 실패가 증가할 가능성도 관찰했다."),
            ("p", "QR 오류 정정 등급 L/M/Q/H의 약 7/15/25/30%는 코드워드 기준이다. "
             "이번 가림률은 QR 심볼 441개 모듈에 대한 면적 비율이다."),
            ("h", "실험 설계"),
            ("table", [
                ["변인", "설정"],
                ["독립", "0~40%의 15단계, 분산형·집중형, L/M/Q/H"],
                ["통제", "버전 1, 마스크 1, 영문 대문자·숫자 6자리, 290×290 RGB"],
                ["처리", "441셀 중 k=round(441×목표률/100)개를 #FF69B4로 가림"],
                ["종속", "OpenCV가 검출한 위치와 원문 완전 일치를 각각 기록"],
                ["표본", "QR 100개(학습 80, 검증 20), 각 QR·형태별 중첩 마스크 10개"],
            ]),
            ("p", "흰 여백은 보존했다. 분산형은 전체 셀 순열, 집중형은 상하좌우 이웃에서 무작위로 자라는 순서를 썼다. "
             "같은 원문·반복·형태는 같은 셀을 등급별로 가렸다. 0%는 원문·등급당 한 번으로 400회이며 비영 조건은 112,000회였다."),
            ("figure", FIGURES[0], "그림 1. 원본과 같은 QR에 적용한 두 가림 형태. 목표 20%, 실제 19.95%.", "damage"),
        ],
        [
            ("h", "탐구 내용"),
            ("figure", FIGURES[1], "그림 2. 점은 100개 QR의 실측, 선은 학습 80개 QR의 구간별 일차함수다.", ""),
            ("p", "실제 가림률 x=100k/441을 사용했다. 인접 측정점 (xᵢ,yᵢ), (xᵢ₊₁,yᵢ₊₁)을 잇는 두 점의 기울기를 "
             "mᵢ=(yᵢ₊₁−yᵢ)/(xᵢ₊₁−xᵢ)로 구했다."),
            ("math", "Rᵢ(x)=yᵢ+mᵢ(x−xᵢ)   ·   열린 구간에서 Rᵢ′(x)=mᵢ"),
            ("p", f"가장 큰 하락 기울기는 {model_label(selected_key)}의 "
             f"{seg['steepest_start_pct']:.4f}~{seg['steepest_end_pct']:.4f}% 구간이었다. "
             f"학습 성공률 {seg['steepest_start_success_pct']:.3f}%와 {seg['steepest_end_success_pct']:.3f}%를 대입했다."),
            ("math", f"R(x)≈{seg['steepest_start_success_pct']:.3f}+({seg['steepest_slope']:.3f})(x−{seg['steepest_start_pct']:.4f})"),
            ("h", "국소 삼차함수"),
            *cubic_blocks(best_key, cubic),
            ("p", "이 모델은 선택한 네 점 사이에서만 계산했다. 측정은 정수 개수의 셀에서 이루어졌다."),
        ],
        [
            ("h", "정적분·결과"),
            ("p", "인접 구간에서 원시함수 yᵢ(x−xᵢ)+(mᵢ/2)(x−xᵢ)²의 양 끝값 차를 구했다. "
             f"각 넓이를 더해 공통 길이 {ceiling:.4f}%로 나눴다."),
            ("math", "∫Rᵢ(x)dx=(yᵢ+yᵢ₊₁)(xᵢ₊₁−xᵢ)/2   ·   평균=구간 적분의 합/범위 길이"),
            ("table", results),
            ("p", f"적분 평균은 {model_label(best['level'] + '/' + best['mask_type'])}이 "
             f"{float(best['mean_success_pct']):.2f}%로 가장 높았다. 112,400회 중 미검출 실패는 "
             f"{missed:,}회({100 * missed / all_n:.2f}%), 검출 후 판독 실패는 {detected_failure:,}회"
             f"({100 * detected_failure / all_n:.2f}%)였다.{finder_note}"),
            ("h", "알게 된 점·느낀 점"),
            ("p", "기울기는 가장 빠르게 바뀌는 구간을, 적분 평균은 같은 가림 범위의 전체 성능을 보여 준다. "
             "검출과 정확 판독을 나누어 기록하니 서로 다른 실패 유형을 수치로 비교할 수 있었다."),
            ("h", "한계·제언"),
            ("p", f"검증 20개 QR에서 일차 모델의 MAE는 곡선별 {min_mae:.2f}~{max_mae:.2f}%p였다. "
             f"위 국소 후보는 {cubic['same_k_holdout_mae_pp']:.2f}%p였다. 검증은 같은 가림률에서 했으므로 "
             "중간 가림률의 연속 추정은 별도 검증이 필요하다."),
            ("p", "문자열 길이·색·판독기를 고정했고 같은 QR을 10개 마스크에 반복했다. 기능 무늬 가림과 미검출의 관계는 "
             "관측 비교이며 인과를 판정하려면 위치를 통제한 추가 실험이 필요하다. 다른 색과 판독기도 비교할 수 있다."),
            ("h", "참고문헌·출처"),
            ("p", f"본실험 raw/trials.csv 및 processed/summary.csv. 실행 {meta['run_id']}. "
             f"Python {meta['environment']['python']}, OpenCV {versions['opencv-python']}, NumPy {versions['numpy']}."),
            ("sources", SOURCES),
        ],
    ]


def export_pink_report(directory, pdf, manuscript=None, student_id=None, student_name=None, preview=False):
    from weasyprint import CSS as Stylesheet, HTML

    ensure_result_destination(directory)
    if student_id is None and student_name is None:
        preview = True
    student_id, student_name = identity_values(student_id, student_name, preview)
    manuscript = manuscript or pdf.with_suffix(".md")
    for out in (pdf, manuscript):
        resolved = out.resolve()
        legacy = [REPO / "reports" / name for name in LEGACY_REPORTS]
        if resolved in legacy or any(root in resolved.parents for root in
                                     (REPO / "results/final", REPO / "results/pilot")):
            raise ValueError("Use a separate PDF and manuscript path for the pink experiment")
    if pdf.resolve() == manuscript.resolve():
        raise ValueError("PDF and manuscript require distinct paths")
    pages = report_pages(directory, student_id, student_name)
    html, markdown = [], []
    for page in pages:
        blocks = [render_block(block, directory, manuscript) for block in page]
        html.append("<section>" + "\n".join(a for a, _ in blocks) + "</section>")
        markdown.append("\n\n".join(b for _, b in blocks))
    doc = HTML(string=f'<html lang="ko"><head><meta charset="utf-8"><title>{escape(TITLE)}</title>'
                      f'</head><body>{"".join(html)}</body></html>')
    rendered = doc.render(stylesheets=[Stylesheet(string=CSS)])
    if len(rendered.pages) != 3:
        raise RuntimeError(f"Pink report layout expected 3 pages, rendered {len(rendered.pages)}")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    manuscript.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pink-report-", dir=pdf.parent) as stage:
        tmp = Path(stage) / "report.pdf"
        rendered.write_pdf(tmp)
        tmp.replace(pdf)
    manuscript.write_text("\n\n---\n\n".join(markdown) + "\n", encoding="utf-8")
    print(f"{pdf} and {manuscript} · {len(rendered.pages)} A4 pages", flush=True)
