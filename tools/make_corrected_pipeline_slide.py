from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import math
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/samehissa/Downloads/SS-VIRULEX-workspace")
OUT_SVG = ROOT / "ss_virulex_corrected_pipeline_slide.svg"
OUT_PNG = ROOT / "ss_virulex_corrected_pipeline_slide.png"
OUT_PDF = ROOT / "ss_virulex_corrected_pipeline_slide.pdf"
OUT_CAPTION = ROOT / "ss_virulex_corrected_pipeline_slide_caption.md"

W, H = 1920, 1080

COLORS = {
    "bg": "#fbfaf6",
    "grid": "#efece6",
    "ink": "#111827",
    "muted": "#4b5563",
    "dataset": "#ffc400",
    "concept_lane": "#a8c792",
    "concept_box": "#3f7f06",
    "concept_text": "#ffffff",
    "stat_lane": "#d8a7a7",
    "stat_box": "#7b191d",
    "stat_text": "#ffffff",
    "process_lane": "#fee8a3",
    "process_box": "#ffc400",
    "model_lane": "#ffe8a9",
    "panel": "#ffc400",
    "line": "#111111",
    "green_line": "#3f7f06",
    "red_line": "#7b191d",
    "feed_line": "#6b7280",
    "note": "#6b7280",
}


@dataclass(frozen=True)
class Box:
    key: str
    x: int
    y: int
    w: int
    h: int
    fill: str
    text_color: str
    lines: tuple[str, ...]
    font_size: int = 21
    radius: int = 28
    bold: bool = False

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    @property
    def top(self) -> tuple[int, int]:
        return (self.cx, self.y)

    @property
    def bottom(self) -> tuple[int, int]:
        return (self.cx, self.y + self.h)

    @property
    def left(self) -> tuple[int, int]:
        return (self.x, self.cy)

    @property
    def right(self) -> tuple[int, int]:
        return (self.x + self.w, self.cy)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def svg_text_lines(box: Box) -> str:
    family = "Arial, Helvetica, sans-serif"
    weight = 700 if box.bold else 400
    line_height = int(box.font_size * 1.18)
    y0 = box.y + (box.h - (len(box.lines) * line_height)) // 2 + box.font_size
    parts = []
    for idx, line in enumerate(box.lines):
        parts.append(
            f'<text x="{box.cx}" y="{y0 + idx * line_height}" '
            f'font-family="{family}" font-size="{box.font_size}" font-weight="{weight}" '
            f'fill="{box.text_color}" text-anchor="middle">{escape(line)}</text>'
        )
    return "\n".join(parts)


def svg_box(box: Box) -> str:
    return (
        f'<rect x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}" rx="{box.radius}" '
        f'fill="{box.fill}"/>\n{svg_text_lines(box)}'
    )


def svg_container(x: int, y: int, w: int, h: int, fill: str, radius: int = 14) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" opacity="0.78"/>'


def svg_path(points: list[tuple[int, int]], color: str = COLORS["line"], width: int = 4, dashed: bool = False) -> str:
    d = f"M {points[0][0]} {points[0][1]} " + " ".join(f"L {x} {y}" for x, y in points[1:])
    dash = ' stroke-dasharray="9 8"' if dashed else ""
    opacity = ' opacity="0.68"' if dashed else ""
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash}{opacity} marker-end="url(#arrow)"/>'
    )


def draw_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str, width: int = 4, dashed: bool = False) -> None:
    if dashed:
        for start, end in zip(points, points[1:]):
            draw_dashed_segment(draw, start, end, color, width)
    else:
        draw.line(points, fill=color, width=width, joint="curve")
    draw_arrowhead(draw, points[-2], points[-1], color)


def draw_dashed_segment(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    width: int,
    dash: int = 10,
    gap: int = 8,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pos = 0.0
    while pos < length:
        seg_end = min(pos + dash, length)
        draw.line(
            [(x1 + dx * pos, y1 + dy * pos), (x1 + dx * seg_end, y1 + dy * seg_end)],
            fill=color,
            width=width,
        )
        pos += dash + gap


def draw_arrowhead(draw: ImageDraw.ImageDraw, previous: tuple[int, int], end: tuple[int, int], color: str) -> None:
    x1, y1 = previous
    x2, y2 = end
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 14
    spread = math.radians(28)
    left = (x2 - size * math.cos(angle - spread), y2 - size * math.sin(angle - spread))
    right = (x2 - size * math.cos(angle + spread), y2 - size * math.sin(angle + spread))
    draw.polygon([end, left, right], fill=color)


def draw_box(draw: ImageDraw.ImageDraw, box: Box) -> None:
    draw.rounded_rectangle([box.x, box.y, box.x + box.w, box.y + box.h], radius=box.radius, fill=box.fill)
    fnt = font(box.font_size, box.bold)
    line_height = int(box.font_size * 1.18)
    y0 = box.y + (box.h - len(box.lines) * line_height) // 2
    for idx, line in enumerate(box.lines):
        bbox = draw.textbbox((0, 0), line, font=fnt)
        tx = box.cx - (bbox[2] - bbox[0]) / 2
        draw.text((tx, y0 + idx * line_height), line, fill=box.text_color, font=fnt)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int = 18, color: str = COLORS["muted"]) -> None:
    fnt = font(size)
    draw.text(xy, text, fill=color, font=fnt)


def build_layout() -> tuple[list[Box], list[str]]:
    boxes = [
        Box("dataset", 170, 90, 290, 86, COLORS["dataset"], COLORS["ink"], ("Dataset",), 38, 34, True),
        Box(
            "concept_input",
            78,
            278,
            230,
            150,
            COLORS["concept_box"],
            COLORS["concept_text"],
            ("completed concept-label", "CSV + Med-MICN RN50", "checkpoint + CT images"),
            20,
        ),
        Box(
            "concept_reason",
            78,
            475,
            230,
            125,
            COLORS["concept_box"],
            COLORS["concept_text"],
            ("ConceptEmbedding +", "ConceptReasoningLayer", "inference"),
            20,
        ),
        Box(
            "concept_output",
            68,
            650,
            250,
            150,
            COLORS["concept_box"],
            COLORS["concept_text"],
            ("8 concept scores +", "task COVID probability +", "neural-symbolic", "COVID probability"),
            20,
        ),
        Box(
            "stat_input",
            405,
            275,
            250,
            105,
            COLORS["stat_box"],
            COLORS["stat_text"],
            ("augmented train +", "original val/test", "CT images"),
            20,
        ),
        Box("mobilenet", 405, 420, 250, 78, COLORS["stat_box"], COLORS["stat_text"], ("custom", "MobileNetV2"), 21),
        Box("xai", 405, 540, 250, 70, COLORS["stat_box"], COLORS["stat_text"], ("xai_feature_layer", "128-D vector"), 20),
        Box("descriptors", 405, 660, 250, 78, COLORS["stat_box"], COLORS["stat_text"], ("26 statistical", "descriptors"), 21),
        Box("fused", 815, 380, 230, 58, COLORS["process_box"], COLORS["ink"], ("36 fused features",), 18),
        Box("zero", 815, 495, 230, 58, COLORS["process_box"], COLORS["ink"], ("zero-fraction", "filtering"), 18),
        Box("mi", 800, 610, 260, 70, COLORS["process_box"], COLORS["ink"], ("mutual-information", "ranking"), 18),
        Box("subsets", 835, 750, 190, 48, COLORS["process_box"], COLORS["ink"], ("feature subsets",), 18),
        Box("rules", 1160, 360, 300, 112, COLORS["process_box"], COLORS["ink"], ("Decision Tree / RuleFit", "rule models", "rule artifacts / readable path"), 18),
        Box("strong", 1160, 625, 300, 58, COLORS["process_box"], COLORS["ink"], ("stronger classifier grids",), 17),
        Box(
            "calibration",
            1142,
            715,
            335,
            92,
            COLORS["process_box"],
            COLORS["ink"],
            ("sigmoid calibration +", "validation threshold selection"),
            18,
        ),
        Box("calib_output", 1168, 835, 285, 58, COLORS["process_box"], COLORS["ink"], ("calibrated predictions + metrics",), 16),
        Box(
            "visual_input",
            1510,
            245,
            330,
            105,
            COLORS["process_box"],
            COLORS["ink"],
            ("original test CT image +", "MobileNetV2 activation maps +", "Med-MICN concept scores"),
            19,
        ),
        Box("overlay", 1540, 425, 270, 70, COLORS["process_box"], COLORS["ink"], ("concept-aware", "SFMOV overlay"), 18),
        Box(
            "panel",
            1518,
            680,
            330,
            118,
            COLORS["panel"],
            COLORS["ink"],
            ("combined explanation panel", "Decision Tree path + overlay", "+ top concept scores"),
            20,
            32,
            False,
        ),
    ]
    notes = [
        "Calibration is applied to stronger classifiers, not to Decision Tree or RuleFit.",
        "Concept-aware SFMOV uses original test CT images, MobileNetV2 maps, and Med-MICN concept scores.",
    ]
    return boxes, notes


def build_svg() -> str:
    boxes, notes = build_layout()
    by_key = {box.key: box for box in boxes}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto" markerUnits="strokeWidth">',
        f'<path d="M2,2 L12,7 L2,12 Z" fill="{COLORS["line"]}"/>',
        "</marker>",
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>',
    ]
    for x in range(0, W, 95):
        parts.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
    for y in range(0, H, 95):
        parts.append(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
    parts.extend(
        [
            '<text x="70" y="52" font-family="Arial, Helvetica, sans-serif" font-size="30" font-weight="700" fill="#111827">SS-VIRULEX corrected implementation pipeline</text>',
            '<text x="70" y="84" font-family="Arial, Helvetica, sans-serif" font-size="18" fill="#4b5563">Workspace-grounded flow from COVID-CT inputs to fused prediction, rules, and concept-aware explanation outputs</text>',
            svg_container(58, 250, 282, 580, COLORS["concept_lane"]),
            svg_container(380, 250, 300, 580, COLORS["stat_lane"]),
            svg_container(765, 350, 340, 490, COLORS["process_lane"]),
            svg_container(1115, 325, 390, 590, COLORS["model_lane"]),
            svg_container(1480, 220, 390, 315, COLORS["process_lane"]),
        ]
    )
    arrows = [
        ([by_key["dataset"].bottom, (193, 230), by_key["concept_input"].top], COLORS["green_line"], False),
        ([by_key["dataset"].bottom, (530, 230), by_key["stat_input"].top], COLORS["red_line"], False),
        ([by_key["concept_input"].bottom, by_key["concept_reason"].top], COLORS["green_line"], False),
        ([by_key["concept_reason"].bottom, by_key["concept_output"].top], COLORS["green_line"], False),
        ([by_key["stat_input"].bottom, by_key["mobilenet"].top], COLORS["red_line"], False),
        ([by_key["mobilenet"].bottom, by_key["xai"].top], COLORS["red_line"], False),
        ([by_key["xai"].bottom, by_key["descriptors"].top], COLORS["red_line"], False),
        ([by_key["concept_output"].bottom, (193, 890), (770, 890), (770, 410), by_key["fused"].left], COLORS["line"], False),
        ([by_key["descriptors"].bottom, (530, 890), (770, 890), (770, 410), by_key["fused"].left], COLORS["line"], False),
        ([by_key["fused"].bottom, by_key["zero"].top], COLORS["line"], False),
        ([by_key["zero"].bottom, by_key["mi"].top], COLORS["line"], False),
        ([by_key["mi"].bottom, by_key["subsets"].top], COLORS["line"], False),
        ([by_key["subsets"].right, (1110, 774), (1110, 416), by_key["rules"].left], COLORS["line"], False),
        ([by_key["subsets"].right, (1110, 774), (1110, 750), by_key["calibration"].left], COLORS["line"], False),
        ([by_key["strong"].bottom, by_key["calibration"].top], COLORS["line"], False),
        ([by_key["calibration"].bottom, by_key["calib_output"].top], COLORS["line"], False),
        ([by_key["rules"].right, (1490, 416), (1490, 710), by_key["panel"].left], COLORS["line"], False),
        ([by_key["dataset"].right, (1680, 136), (1680, 220), by_key["visual_input"].top], COLORS["line"], False),
        ([by_key["concept_output"].right, (350, 725), (350, 930), (1680, 930), (1680, 535), by_key["visual_input"].bottom], COLORS["feed_line"], True),
        ([by_key["mobilenet"].right, (705, 459), (705, 190), (1675, 190), by_key["visual_input"].top], COLORS["feed_line"], True),
        ([by_key["visual_input"].bottom, by_key["overlay"].top], COLORS["line"], False),
        ([by_key["overlay"].bottom, (1675, 600), by_key["panel"].top], COLORS["line"], False),
    ]
    for pts, color, dashed in arrows:
        parts.append(svg_path(pts, color=color, width=3 if dashed else 4, dashed=dashed))
    for box in boxes:
        parts.append(svg_box(box))
    parts.extend(
        [
            f'<text x="70" y="1008" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="{COLORS["note"]}">Note: {escape(notes[0])}</text>',
            f'<text x="70" y="1034" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="{COLORS["note"]}">Note: {escape(notes[1])}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def build_png() -> Image.Image:
    boxes, notes = build_layout()
    by_key = {box.key: box for box in boxes}
    image = Image.new("RGB", (W, H), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    for x in range(0, W, 95):
        draw.line([(x, 0), (x, H)], fill=COLORS["grid"], width=1)
    for y in range(0, H, 95):
        draw.line([(0, y), (W, y)], fill=COLORS["grid"], width=1)
    draw_label(draw, (70, 24), "SS-VIRULEX corrected implementation pipeline", 30, COLORS["ink"])
    draw_label(
        draw,
        (70, 60),
        "Workspace-grounded flow from COVID-CT inputs to fused prediction, rules, and concept-aware explanation outputs",
        18,
        COLORS["muted"],
    )
    for rect in [
        (58, 250, 340, 830, COLORS["concept_lane"]),
        (380, 250, 680, 830, COLORS["stat_lane"]),
        (765, 350, 1105, 840, COLORS["process_lane"]),
        (1115, 325, 1505, 915, COLORS["model_lane"]),
        (1480, 220, 1870, 535, COLORS["process_lane"]),
    ]:
        draw.rounded_rectangle(rect[:4], radius=14, fill=rect[4])
    arrows = [
        ([by_key["dataset"].bottom, (193, 230), by_key["concept_input"].top], COLORS["green_line"], False),
        ([by_key["dataset"].bottom, (530, 230), by_key["stat_input"].top], COLORS["red_line"], False),
        ([by_key["concept_input"].bottom, by_key["concept_reason"].top], COLORS["green_line"], False),
        ([by_key["concept_reason"].bottom, by_key["concept_output"].top], COLORS["green_line"], False),
        ([by_key["stat_input"].bottom, by_key["mobilenet"].top], COLORS["red_line"], False),
        ([by_key["mobilenet"].bottom, by_key["xai"].top], COLORS["red_line"], False),
        ([by_key["xai"].bottom, by_key["descriptors"].top], COLORS["red_line"], False),
        ([by_key["concept_output"].bottom, (193, 890), (770, 890), (770, 410), by_key["fused"].left], COLORS["line"], False),
        ([by_key["descriptors"].bottom, (530, 890), (770, 890), (770, 410), by_key["fused"].left], COLORS["line"], False),
        ([by_key["fused"].bottom, by_key["zero"].top], COLORS["line"], False),
        ([by_key["zero"].bottom, by_key["mi"].top], COLORS["line"], False),
        ([by_key["mi"].bottom, by_key["subsets"].top], COLORS["line"], False),
        ([by_key["subsets"].right, (1110, 774), (1110, 416), by_key["rules"].left], COLORS["line"], False),
        ([by_key["subsets"].right, (1110, 774), (1110, 750), by_key["calibration"].left], COLORS["line"], False),
        ([by_key["strong"].bottom, by_key["calibration"].top], COLORS["line"], False),
        ([by_key["calibration"].bottom, by_key["calib_output"].top], COLORS["line"], False),
        ([by_key["rules"].right, (1490, 416), (1490, 710), by_key["panel"].left], COLORS["line"], False),
        ([by_key["dataset"].right, (1680, 136), (1680, 220), by_key["visual_input"].top], COLORS["line"], False),
        ([by_key["concept_output"].right, (350, 725), (350, 930), (1680, 930), (1680, 535), by_key["visual_input"].bottom], COLORS["feed_line"], True),
        ([by_key["mobilenet"].right, (705, 459), (705, 190), (1675, 190), by_key["visual_input"].top], COLORS["feed_line"], True),
        ([by_key["visual_input"].bottom, by_key["overlay"].top], COLORS["line"], False),
        ([by_key["overlay"].bottom, (1675, 600), by_key["panel"].top], COLORS["line"], False),
    ]
    for pts, color, dashed in arrows:
        draw_arrow(draw, pts, color, width=3 if dashed else 4, dashed=dashed)
    for box in boxes:
        draw_box(draw, box)
    draw_label(draw, (70, 982), f"Note: {notes[0]}", 16, COLORS["note"])
    draw_label(draw, (70, 1010), f"Note: {notes[1]}", 16, COLORS["note"])
    return image


def write_caption() -> None:
    OUT_CAPTION.write_text(
        "\n".join(
            textwrap.dedent(
                """
                Suggested caption: Corrected SS-VIRULEX implementation pipeline for COVID-CT classification and explanation.
                The figure separates the Med-MICN concept/probability branch from the MobileNetV2 statistical branch,
                shows late fusion into 36 model features, applies zero-fraction filtering and mutual-information
                feature subset selection, separates rule models from calibrated stronger classifiers, and represents
                concept-aware SFMOV as a visual branch driven by original test CT images, MobileNetV2 activation maps,
                and Med-MICN concept scores.
                """
            ).strip().splitlines()
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT_SVG.write_text(build_svg(), encoding="utf-8")
    image = build_png()
    image.save(OUT_PNG)
    image.save(OUT_PDF, "PDF", resolution=160.0)
    write_caption()
    print(OUT_SVG)
    print(OUT_PNG)
    print(OUT_PDF)
    print(OUT_CAPTION)


if __name__ == "__main__":
    main()
