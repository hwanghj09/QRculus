"""Compose the Korean manuscript and render the three-page PDF."""

import os
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

from .config import LEVELS, actual_pct, read_json
from .experiment import read_csv
from .metrics import derivative

TITLE = "QR코드의 손상률에 따른 복원 성공률의 함수적 관계 분석"
REQUIRED_SECTIONS = (
    "탐구 주제", "동기·목적", "탐구 방법", "탐구 내용",
    "결과 및 알게 된 점", "느낀 점·제언", "탐구 한계", "참고문헌·출처",
)
SOURCES = (
    ("DENSO WAVE, 오류 복원 기능", "https://www.qrcode.com/ko/about/error_correction.html"),
    ("DENSO WAVE, QR코드 정보량과 버전", "https://www.qrcode.com/ko/about/version.html"),
    ("ISO/IEC 18004:2024, QR code bar code symbology specification", "https://www.iso.org/standard/83389.html"),
    ("qrcode 8.2, PyPI", "https://pypi.org/project/qrcode/8.2/"),
    ("OpenCV, QRCodeDetector API", "https://docs.opencv.org/4.x/de/dc3/classcv_1_1QRCodeDetector.html"),
    ("NumPy, Random Generator API", "https://numpy.org/doc/stable/reference/random/generator.html"),
)

CSS = """
@page {
  size: A4;
  margin: 14mm 16mm 15mm;
  @bottom-left { content: "미적분Ⅰ 심화탐구"; color: #475568; font-size: 10pt; }
  @bottom-right { content: counter(page) " / " counter(pages); color: #475568; font-size: 10pt; }
}
html { font: 10.5pt/1.44 "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic", sans-serif; color: #172739; }
body { margin: 0; }
section + section { break-before: page; }
h1, h2, h3 { font-size: 11pt; }
h1 { margin: 1.2mm 0 4mm; padding: 0 0 3mm; border-bottom: 1.4pt solid #28776f; }
h2, h3 { color: #14615a; margin: 3mm 0 1mm; break-after: avoid; }
section > h2:first-child { border-bottom: 1.4pt solid #28776f; padding-bottom: 2mm; }
p { margin: 0 0 1.7mm; overflow-wrap: anywhere; }
p.identity { font-weight: bold; margin: 0 0 1.6mm; }
p.math { background: #eef5f3; padding: 1.2mm 2.6mm; margin: 1.4mm 0; }
table { margin: 1.5mm 0 2mm; width: 100%; border-collapse: collapse; break-inside: avoid; }
th, td { border-bottom: 0.5pt solid #d6e1e7; text-align: left; padding: 0.7mm 1.3mm; }
th { background: #e5f0ee; }
figure { margin: 2mm 0; text-align: center; break-inside: avoid; }
figure img { width: 100%; max-height: 69mm; object-fit: contain; }
figure .damage { max-height: 43mm; }
figcaption { color: #364d5b; font-size: 10pt; }
ol { margin: 0; padding-left: 6mm; }
li { margin-bottom: .5mm; }
a { color: #14615a; text-decoration: underline; }
"""


def identity_values(student_id, student_name, preview):
    if preview:
        if student_id or student_name:
            raise ValueError("--preview cannot be combined with student identity")
        return "________", "________"
    if student_id is None and student_name is None and sys.stdin.isatty():
        student_id, student_name = input("학번: "), input("이름: ")
    if not student_id or not student_name or not student_id.strip() or not student_name.strip():
        raise ValueError("Provide the actual --student-id and --student-name, or use --preview")
    if any(ord(character) < 32 for character in student_id + student_name):
        raise ValueError("Use single-line identity fields")
    return student_id.strip(), student_name.strip()


def report_pages(input_dir, student_id, student_name):
    config = read_json(input_dir / "config.json")
    env = read_json(input_dir / "environment.json")
    details = read_json(input_dir / "analysis_details.json")
    if config["trials"] != 100 or config["seed_base"] != 2026 or details["run_id"] != config["run_id"]:
        raise ValueError("Report requires the final 100-trial design and its matching analysis")
    models = details["levels"]
    measurements = {
        (float(r["target_pct"]), r["level"]): float(r["success_pct"])
        for r in read_csv(input_dir / "success_rate.csv")
    }
    observation_table = [["설정 %", "실제 %", "L", "M", "Q", "H"]]
    for target, k in ((0, 0), (2, 4), (3, 6), (4, 8), (5, 10), (6, 12), (10, 21)):
        observation_table.append([f"{target:g}", f"{actual_pct(k):.3f}",
                                  *[f"{measurements[target, level]:.0f}%" for level in LEVELS]])

    result_table = [["등급", "가장 빠른 감소 구간", "기울기¹", "정적분 평균"]]
    for level in LEVELS:
        model = models[level]["piecewise"]
        seg = model["segments"][model["steepest_index"]]
        result_table.append([level, f"{seg['start_pct']:.3f}–{seg['end_pct']:.3f}%",
                             f"{seg['slope']:.2f}", f"{model['mean_success_pct']:.2f}%"])

    cubic = models["H"]["cubic"]
    if cubic:
        y0, y1, y2, y3 = [point["success_pct"] for point in cubic["points"]]
        k0 = cubic["k0"]
        x0, x1 = cubic["range_pct"]
        lo, hi = [derivative(cubic["coefficients"], u) for u in (0, 3)]
        validation = cubic["validation"]
        cubic_blocks = [
            ("p", f"H의 최급락 구간을 포함하는 k={k0}~{k0 + 3}을 골랐다. u=k−{k0}, 0≤u≤3이고 "
             f"x=100({k0}+u)/208이다. 네 점의 성공률은 {y0:g}, {y1:g}, {y2:g}, {y3:g}%다."),
            ("math", f"d={y0:g}, a+b+c+d={y1:g}, 8a+4b+2c+d={y2:g}, 27a+9b+3c+d={y3:g}"),
            ("p", f"이웃 식을 빼면 6a=y₃−3y₂+3y₁−y₀, 6a+2b=y₂−2y₁+y₀, c=y₁−y₀−a−b다."),
            ("math", f"P(u)={cubic['equation']}     P′(u)={cubic['derivative_equation']}"),
        ]
        if cubic["valid"]:
            minimum = (f"꼭짓점 u=−b/(3a)={cubic['min_derivative_u']:.2f}에서 최소이고"
                       if cubic["interior_minimum"] else
                       f"끝점 u={cubic['min_derivative_u']:.2f}에서 최소이고")
            cubic_blocks.extend([
                ("p", f"P′의 양 끝값 {lo:.2f}, {hi:.2f}와 구간 내 꼭짓점을 비교했다. {minimum} "
                 f"P′={cubic['min_derivative']:.2f}%p/모듈, x={cubic['min_derivative_pct']:.3f}%다. "
                 f"유효 범위는 x={x0:.3f}~{x1:.3f}%다."),
                ("p", f"[au⁴/4+bu³/3+cu²/2+du]₀³을 3으로 나눈 국소 평균은 "
                 f"{cubic['local_mean_success_pct']:.2f}%다. 이 값은 위 네 점이 정한 구간에 적용된다."),
            ])
        else:
            cubic_blocks.append(("p", f"H 삼차 후보는 {', '.join(cubic['issues'])}으로 해석에서 제외했다."))
        validation_text = (
            f"보조 실험은 각 k에서 새 시드 {validation['first_seed']}~{validation['last_seed']}로 "
            f"{validation['trials_per_k']}회씩 시행했다. H의 평균절대오차는 {validation['mae_pp']:.2f}%p다. "
            "이 오차는 같은 정수 k에서의 재측정 차이를 보여 준다. 네 점 사이 연속 구간의 예측은 검증 범위 밖이다."
        )
    else:
        cubic_blocks = [("p", "H의 최급락 구간을 포함하는 연속 네 모듈 수가 없어 국소 삼차 후보를 만들 수 없었다.")]
        validation_text = "국소 후보가 있는 등급의 같은 k 보조 실험 오차는 분석표에 기록했다."

    high_piecewise = models["H"]["piecewise"]
    high_segment = high_piecewise["segments"][high_piecewise["steepest_index"]]
    example = (f"H에서 x={high_segment['start_pct']:.3f}~{high_segment['end_pct']:.3f}%일 때 "
               f"Rᵢ(x)≈{high_segment['start_success_pct']:g}+({high_segment['slope']:.2f})"
               f"(x−{high_segment['start_pct']:.3f}). 이 구간의 적분은 {high_segment['area']:.2f}다.")
    ranking = sorted(LEVELS, key=lambda level: models[level]["piecewise"]["mean_success_pct"], reverse=True)
    means = [models[level]["piecewise"]["mean_success_pct"] for level in ranking]
    excluded = [level for level in LEVELS if models[level]["cubic"] and not models[level]["cubic"]["valid"]]
    exclusion_text = ""
    if excluded:
        exclusion_text = f" {', '.join(excluded)}의 국소 후보는 범위 초과 또는 구간 내 증가가 있어 해석에서 제외했다."
    libraries = env["library_versions"]
    return [
        [
            ("identity", f"학번: {student_id}　 이름: {student_name}"),
            ("title", TITLE),
            ("h", "탐구 주제"),
            ("p", "QR코드에서 데이터·오류 정정 모듈을 뒤집을 때 OpenCV의 복원 성공률을 측정한다. "
             "다항함수의 미분과 정적분으로 급락 구간, 국소 최대 감소 속도, 등급별 구간 평균을 구한다."),
            ("h", "동기·목적"),
            ("p", "오류 정정 등급이 실제 판독 결과에 얼마나 영향을 주는지 확인하고 싶었다. "
             "컴퓨터공학에서 생성기와 판독기의 동작을 통제 실험으로 관찰하고, 실측값의 변화를 미적분으로 설명하는 문제다."),
            ("p", "QR코드의 위치 탐지·타이밍·형식 정보는 판독에 쓰인다. DENSO WAVE의 약 7/15/25/30%는 "
             "코드워드 단위의 복원 능력이다. 이 실험에서 반전한 개별 모듈 비율과는 단위가 다르다."),
            ("h", "탐구 방법"),
            ("table", [
                ["변인", "실험 설정"],
                ["독립", "오류 정정 L/M/Q/H, 설정 손상률 0~8%를 0.5% 간격으로 측정하고 10% 추가"],
                ["종속", "각 조건 100회에서 TEST01을 정확히 판독한 비율"],
                ["통제", "버전 1(21×21), TEST01, 마스크 1, 모듈 10 px, 흰 여백 4모듈, OpenCV"],
                ["손상 대상", "예약 모듈 233개와 여백 보존, 데이터·오류 정정 모듈 208개만 XOR 반전"],
            ]),
            ("p", "① 원본 생성과 판독 확인 → ② 반복 t에서 seed=2026+t로 208개 좌표의 순열 생성 → "
             "③ 앞의 k개를 반전 → ④ 정확히 TEST01이면 성공. 각 t에서 등급들이 좌표를 공유하고 손상 집합은 중첩된다."),
            ("math", "k=round(208×설정률/100)     실제 손상률 x=100k/208 (%)"),
            ("p", f"총 4×18×100=7,200회이다. 0%는 모든 등급에서 100/100이었다. 설정 10%는 k=21로 "
             f"실제 {actual_pct(21):.4f}%다. 20회 예비 자료는 참고용으로 보관했다."),
            ("figure", "figure_01_origin_and_damage.png",
             "그림 1. H의 원본과 같은 반복에서 찍은 손상 예시. 실제 손상률의 분모는 208개.", "damage"),
        ],
        [
            ("h", "탐구 내용"),
            ("figure", "figure_02_success_by_level.png",
             "그림 2. 실제 손상률별 복원 성공률. 표식은 측정값이고 선은 구간별 일차 모델이다.", ""),
            ("table", observation_table),
            ("p", "인접 두 점 (xᵢ,yᵢ), (xᵢ₊₁,yᵢ₊₁)을 잇는 직선의 기울기는 mᵢ=(yᵢ₊₁−yᵢ)/(xᵢ₊₁−xᵢ)다."),
            ("math", "Rᵢ(x)=yᵢ+mᵢ(x−xᵢ)   (xᵢ≤x≤xᵢ₊₁)     내부 도함수 Rᵢ′(x)=mᵢ"),
            ("p", "정적분은 [yᵢ(x−xᵢ)+(mᵢ/2)(x−xᵢ)²]의 양 끝값 차이다. "
             "mᵢ를 대입하면 (yᵢ+yᵢ₊₁)(xᵢ₊₁−xᵢ)/2다. 각 면적을 더해 공통 구간 길이 100×21/208로 나눈다."),
            ("p", example),
            ("h", "국소 삼차함수 유도"),
            *cubic_blocks,
        ],
        [
            ("h", "결과 및 알게 된 점"),
            ("table", result_table),
            ("p", "¹ 성공률 %p / 실제 손상률 %p. 정적분 평균은 모든 등급에서 "
             f"0~{actual_pct(21):.4f}%의 공통 범위로 계산했다."),
            ("p", f"평균은 {' > '.join(ranking)} 순이었다. {ranking[0]}와 {ranking[-1]}의 차이는 "
             f"{means[0] - means[-1]:.2f}%p였다. 기울기는 가장 급한 변화 구간을, 정적분 평균은 공통 범위의 "
             f"전체 성능을 보여 준다.{exclusion_text}"),
            ("p", validation_text),
            ("h", "탐구 한계"),
            ("p", "좌표 선택 편차를 줄이기 위해 같은 반복의 순열을 공유했다. 예약 모듈과 여백을 보존하고, "
             "반올림 오차를 처리하기 위해 설정률과 실제율을 모두 기록했다. 보간의 과도한 진동은 "
             "도함수 부호와 극값으로 확인했다."),
            ("p", "메시지·마스크·버전·판독기를 하나씩 고정했다. 모듈 반전만 다루므로 얼룩, 가림, 회전, "
             "촬영 조건에 따른 성능은 별도 측정이 필요하다. 조건당 100회여서 성공률의 해상도는 1%p다."),
            ("h", "느낀 점·제언"),
            ("p", "기울기가 가장 큰 폭으로 내려가는 시점과 일정 범위에서의 성능은 서로 다른 질문임을 알았다. "
             "다음에는 메시지·마스크·QR 버전을 바꾸고, 집중 손상과 분산 손상 및 다른 판독기를 비교한다."),
            ("h", "참고문헌·출처"),
            ("p", f"본실험 원자료 trials.csv, 집계 success_rate.csv, 분석 analysis_details.json. 실행 ID {config['run_id']}. "
             f"Python {env['python']}, OpenCV {libraries['opencv-python']}, qrcode {libraries['qrcode']}, "
             f"NumPy {libraries['numpy']}. 세부 환경은 environment.json."),
            ("sources", SOURCES),
        ],
    ]


def render_block(block, input_dir, manuscript):
    kind, content, *extra = block
    value = escape(content) if isinstance(content, str) else None
    if kind in ("title", "h"):
        tag = "h1" if kind == "title" else "h2"
        return f"<{tag}>{value}</{tag}>", f"{'#' if kind == 'title' else '##'} {content}"
    if kind == "table":
        header, *rows = content
        rendered = []
        for index, row in enumerate(content):
            tag = "th" if index == 0 else "td"
            rendered.append("<tr>" + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in row) + "</tr>")
        def markdown_row(row):
            return "| " + " | ".join(row) + " |"

        markdown = "\n".join(markdown_row(row) for row in [header, ["---"] * len(header), *rows])
        return "<table>" + "".join(rendered) + "</table>", markdown
    if kind == "figure":
        caption, css_class = extra
        image = (input_dir / "figures" / content).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        relative = Path(os.path.relpath(image, manuscript.parent.resolve())).as_posix()
        return (
            f'<figure><img class="{css_class}" src="{image.as_uri()}" alt="{escape(caption)}">'
            f"<figcaption>{escape(caption)}</figcaption></figure>",
            f"![{caption}]({relative})\n\n{caption}",
        )
    if kind == "sources":
        html = "<ol>" + "".join(f'<li><a href="{url}">{escape(label)}</a></li>' for label, url in content) + "</ol>"
        markdown = "\n".join(f"{index}. {label} <{url}>" for index, (label, url) in enumerate(content, 1))
        return html, markdown
    return f'<p class="{kind}">{value}</p>', f"`{content}`" if kind == "math" else content


def export_report(input_dir, pdf, manuscript=None, student_id=None, student_name=None, preview=False):
    from weasyprint import CSS as Stylesheet, HTML

    student_id, student_name = identity_values(student_id, student_name, preview)
    manuscript = manuscript or pdf.with_name(f"{pdf.stem}_원고.md")
    if pdf.resolve() == manuscript.resolve():
        raise ValueError("PDF and manuscript require distinct output paths")
    pages = report_pages(input_dir, student_id, student_name)
    html_pages, markdown_pages = [], []
    for page in pages:
        formatted = [render_block(block, input_dir, manuscript) for block in page]
        html_pages.append("<section>" + "\n".join(html for html, _ in formatted) + "</section>")
        markdown_pages.append("\n\n".join(markdown for _, markdown in formatted))
    document = HTML(string=f'<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">'
                    f'<title>{TITLE}</title></head><body>{"".join(html_pages)}</body></html>')
    rendered = document.render(stylesheets=[Stylesheet(string=CSS)])
    if len(rendered.pages) != len(pages):
        raise RuntimeError(f"Report overflow: expected {len(pages)} pages, rendered {len(rendered.pages)}")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    manuscript.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".qr-report-", dir=pdf.parent) as staging:
        temporary_pdf = Path(staging) / "report.pdf"
        rendered.write_pdf(temporary_pdf)
        temporary_pdf.replace(pdf)
    manuscript.write_text("\n\n---\n\n".join(markdown_pages) + "\n", encoding="utf-8")
    return manuscript, len(pages)
