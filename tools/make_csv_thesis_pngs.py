#!/usr/bin/env python3
import csv
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE_W = 2400
PAGE_H = 1700
MARGIN = 70


SPECS = [
    {
        "path": ROOT / "03_SS-VIRULEX-outputs/concept_branch/labels/covid_concept_labels.csv",
        "output": ROOT / "covid_concept_labels_thesis_table.png",
        "title": "COVID-CT Concept Label Table",
        "subtitle": "Thesis-ready PNG preview of manual/pipeline concept labels and representative rows.",
        "groups": [
            ("Metadata / label columns", lambda h: h in {"image_id", "file_name", "class_name", "true_label", "split", "image_path"}),
            ("Concept label columns", lambda h: h.startswith("concept_")),
        ],
        "sample_columns": [
            ("image_id", "image_id", 380, False),
            ("class", "class_name", 105, False),
            ("split", "split", 90, False),
            ("peripheral_ggo", "concept_peripheral_ground_glass_opacities_label", 150, True),
            ("bilateral", "concept_bilateral_involvement_label", 105, True),
            ("multilobar", "concept_multilobar_distribution_label", 120, True),
            ("crazy_paving", "concept_crazy_paving_pattern_label", 140, True),
            ("absence_lobar", "concept_absence_of_lobar_consolidation_label", 145, True),
            ("localized_diffuse", "concept_localized_or_diffuse_presentation_label", 165, True),
            ("increased_density", "concept_increased_density_in_the_lung_label", 165, True),
            ("ground_glass", "concept_ground_glass_appearance_label", 140, True),
        ],
        "caption": "Concept-label matrix for original COVID-CT images.",
        "note": "This label table contains original image-level concept labels; augmented processing rows appear in the probabilities and fused-feature tables.",
    },
    {
        "path": ROOT / "03_SS-VIRULEX-outputs/concept_branch/probabilities/covid_concept_probabilities.csv",
        "output": ROOT / "covid_concept_probabilities_thesis_table.png",
        "title": "COVID-CT Concept Probability Table",
        "subtitle": "Thesis-ready PNG preview of Med-MICN diagnosis probabilities, concept scores, and predicted concept labels.",
        "groups": [
            ("Metadata / source columns", lambda h: h in {"image_id", "file_name", "class_name", "true_label", "split", "image_path", "resolved_image_path", "source", "source_filepath", "concept_key"}),
            ("Ground-truth concept label columns", lambda h: h.startswith("concept_") and h.endswith("_label") and not h.startswith("concept_pred_")),
            ("Diagnosis probability columns", lambda h: h.startswith("task_") or h.startswith("neural_")),
            ("Concept score columns", lambda h: h.startswith("concept_score_")),
            ("Predicted concept label columns", lambda h: h.startswith("concept_pred_")),
        ],
        "sample_columns": [
            ("image_id", "image_id", 350, False),
            ("class", "class_name", 100, False),
            ("split", "split", 80, False),
            ("source", "source", 95, False),
            ("task_p_covid", "task_probability_covid", 125, True),
            ("neural_p_covid", "neural_probability_covid", 135, True),
            ("peripheral_ggo", "concept_score_peripheral_ground_glass_opacities", 140, True),
            ("bilateral", "concept_score_bilateral_involvement", 115, True),
            ("multilobar", "concept_score_multilobar_distribution", 120, True),
            ("crazy_paving", "concept_score_crazy_paving_pattern", 130, True),
            ("absence_lobar", "concept_score_absence_of_lobar_consolidation", 140, True),
            ("localized_diffuse", "concept_score_localized_or_diffuse_presentation", 150, True),
            ("increased_density", "concept_score_increased_density_in_the_lung", 150, True),
            ("ground_glass", "concept_score_ground_glass_appearance", 130, True),
        ],
        "caption": "Med-MICN concept-probability and diagnosis-probability table for the COVID-CT processing set.",
        "note": "Numeric probabilities and concept scores are rounded for display; the CSV retains full precision.",
    },
    {
        "path": ROOT / "03_SS-VIRULEX-outputs/fusion/fused_features_covid.csv",
        "output": ROOT / "fused_features_covid_thesis_table.png",
        "title": "COVID-CT Fused Feature Table",
        "subtitle": "Thesis-ready PNG preview of statistical, concept-score, and Med-MICN diagnosis features after fusion.",
        "groups": [
            ("Metadata / label columns", lambda h: h in {"image_id", "file_name", "class_name", "true_label", "split", "image_path"}),
            ("Statistical descriptor columns", lambda h: h.startswith("stat_")),
            ("Concept score columns", lambda h: h.startswith("concept_score_")),
            ("Med-MICN diagnosis probability columns", lambda h: h.startswith("med_")),
        ],
        "sample_columns": [
            ("image_id", "image_id", 330, False),
            ("class", "class_name", 100, False),
            ("split", "split", 80, False),
            ("stat_mean", "stat_mean", 105, True),
            ("stat_std", "stat_std_dev", 105, True),
            ("stat_skew", "stat_skewness", 110, True),
            ("stat_entropy", "stat_entropy", 120, True),
            ("stat_iqr", "stat_iqr", 100, True),
            ("concept_bilateral", "concept_score_bilateral_involvement", 140, True),
            ("concept_crazy_paving", "concept_score_crazy_paving_pattern", 155, True),
            ("concept_absence_lobar", "concept_score_absence_of_lobar_consolidation", 165, True),
            ("concept_ground_glass", "concept_score_ground_glass_appearance", 155, True),
            ("med_task_prob", "med_task_prob_covid", 130, True),
            ("med_neural_prob", "med_neural_prob_covid", 140, True),
        ],
        "caption": "Fused SS-VIRULEX feature matrix combining statistical, concept, and diagnostic probability features.",
        "note": "The full fused table contains 26 statistical features, eight concept scores, and two Med-MICN diagnostic probability features.",
    },
]


def esc(text):
    text = str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in text)


def approx_chars(width, size, mono=False):
    return max(4, int(width / (size * (0.61 if mono else 0.52))))


def truncate(text, width, size, mono=False):
    text = str(text)
    limit = approx_chars(width, size, mono)
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)] + "..."


def fmt_value(value, numeric):
    if not numeric:
        return str(value)
    try:
        number = float(value)
        if number in (0.0, 1.0):
            return str(int(number))
        return f"{number:.4f}"
    except (TypeError, ValueError):
        return str(value)


class PdfCanvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.ops = []

    def y(self, top_y):
        return self.height - top_y

    def color(self, hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def fill(self, color):
        r, g, b = self.color(color)
        self.ops.append(f"{r:.4f} {g:.4f} {b:.4f} rg")

    def stroke(self, color):
        r, g, b = self.color(color)
        self.ops.append(f"{r:.4f} {g:.4f} {b:.4f} RG")

    def rect(self, x, y, w, h, fill=None, stroke=None, line_width=1):
        if fill:
            self.fill(fill)
        if stroke:
            self.stroke(stroke)
        self.ops.append(f"{line_width} w")
        self.ops.append(f"{x:.2f} {self.height - y - h:.2f} {w:.2f} {h:.2f} re")
        if fill and stroke:
            self.ops.append("B")
        elif fill:
            self.ops.append("f")
        else:
            self.ops.append("S")

    def line(self, x1, y1, x2, y2, color="#d9dee7", line_width=1):
        self.stroke(color)
        self.ops.append(f"{line_width} w")
        self.ops.append(f"{x1:.2f} {self.y(y1):.2f} m {x2:.2f} {self.y(y2):.2f} l S")

    def text(self, text, x, y, size=20, font="F1", color="#1f2937", max_width=None, mono=False):
        if max_width:
            text = truncate(text, max_width, size, mono)
        self.fill(color)
        self.ops.append(f"BT /{font} {size:.2f} Tf {x:.2f} {self.y(y):.2f} Td ({esc(text)}) Tj ET")

    def stream(self):
        return "\n".join(self.ops)


def write_pdf(path, canvas):
    content = canvas.stream()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {canvas.width} {canvas.height}] "
            f"/Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R >> >> "
            f"/Contents 7 0 R >>"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        f"<< /Length {len(content.encode())} >>\nstream\n{content}\nendstream".encode(),
    ]

    out = b"%PDF-1.4\n"
    offsets = [0]
    for idx, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    Path(path).write_bytes(out)


def load_csv(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def representative_rows(rows, limit=10):
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("split", ""), row.get("class_name", ""))].append(row)
    wanted = [
        ("train", "COVID", 2),
        ("train", "NonCOVID", 2),
        ("val", "COVID", 2),
        ("val", "NonCOVID", 2),
        ("test", "COVID", 1),
        ("test", "NonCOVID", 1),
    ]
    sample = []
    for split, cls, count in wanted:
        sample.extend(groups.get((split, cls), [])[:count])
    if len(sample) < limit:
        seen = {id(row) for row in sample}
        for row in rows:
            if id(row) not in seen:
                sample.append(row)
            if len(sample) >= limit:
                break
    return sample[:limit]


def draw_card(c, x, y, w, h, label, value, note):
    c.rect(x, y, w, h, fill="#f8fafc", stroke="#cbd5e1", line_width=1.4)
    c.text(label.upper(), x + 28, y + 40, size=18, font="F2", color="#64748b", max_width=w - 56)
    c.text(value, x + 28, y + 86, size=42, font="F2", color="#0f172a", max_width=w - 56)
    c.text(note, x + 28, y + 122, size=18, font="F1", color="#475569", max_width=w - 56)


def concept_count(headers):
    return sum(1 for h in headers if h.startswith("concept_") and h != "concept_key")


def draw_column_inventory(c, headers, spec):
    x = MARGIN
    y = 405
    w = PAGE_W - 2 * MARGIN
    h = 520
    c.rect(x, y, w, h, fill="#ffffff", stroke="#cbd5e1", line_width=1.4)
    c.rect(x, y, w, 58, fill="#eaf2ff", stroke="#cbd5e1", line_width=1.0)
    c.text("Column Inventory", x + 24, y + 38, size=26, font="F2", color="#0f172a")
    c.text(f"All {len(headers)} CSV columns are grouped by provenance or output type.", x + 300, y + 37, size=18, color="#475569")

    compact = len(headers) > 30
    group_size = 16 if compact else 19
    item_size = 12.0 if compact else 15.5
    row_h = 21 if compact else 32
    columns = 5 if compact else 3
    cur_y = y + 94
    for group_title, predicate in spec["groups"]:
        names = [h for h in headers if predicate(h)]
        if not names:
            continue
        c.text(f"{group_title} ({len(names)})", x + 28, cur_y, size=group_size, font="F2", color="#334155")
        cur_y += 21 if compact else 28

        col_w = (w - 56) / columns
        for idx, name in enumerate(names):
            col = idx % columns
            row = idx // columns
            tx = x + 28 + col * col_w
            ty = cur_y + row * row_h
            c.text(name, tx, ty, size=item_size, font="F3", color="#1f2937", max_width=col_w - 22, mono=True)
        cur_y += ((len(names) + columns - 1) // columns) * row_h + (12 if compact else 28)
        if cur_y > y + h - 24:
            break


def draw_sample_table(c, rows, spec):
    x = MARGIN
    y = 985
    w = PAGE_W - 2 * MARGIN
    title_h = 62
    header_h = 56
    row_h = 44
    table_h = title_h + header_h + row_h * len(rows)

    c.rect(x, y, w, table_h, fill="#ffffff", stroke="#cbd5e1", line_width=1.4)
    c.rect(x, y, w, title_h, fill="#f8fafc", stroke="#cbd5e1", line_width=1.0)
    c.text("Representative Rows", x + 24, y + 40, size=26, font="F2", color="#0f172a")
    c.text("Displayed values are sampled for readability; full CSV remains the source of truth.", x + 300, y + 39, size=18, color="#475569")

    columns = spec["sample_columns"]
    scale = w / sum(width for _, _, width, _ in columns)
    scaled = [(label, field, width * scale, numeric) for label, field, width, numeric in columns]

    c.rect(x, y + title_h, w, header_h, fill="#dbeafe", stroke="#cbd5e1", line_width=1.0)
    cx = x
    for label, _, cw, _ in scaled:
        c.line(cx, y + title_h, cx, y + title_h + header_h + row_h * len(rows), color="#cbd5e1", line_width=0.8)
        c.text(label, cx + 10, y + title_h + 36, size=16, font="F2", color="#1e293b", max_width=cw - 18)
        cx += cw
    c.line(x + w, y + title_h, x + w, y + title_h + header_h + row_h * len(rows), color="#cbd5e1", line_width=0.8)

    for ridx, row in enumerate(rows):
        ry = y + title_h + header_h + ridx * row_h
        c.rect(x, ry, w, row_h, fill="#ffffff" if ridx % 2 == 0 else "#f8fafc")
        c.line(x, ry + row_h, x + w, ry + row_h, color="#e2e8f0", line_width=0.8)
        cx = x
        for _, field, cw, numeric in scaled:
            value = fmt_value(row.get(field, ""), numeric)
            c.text(value, cx + 10, ry + 29, size=14.5, font="F3" if numeric else "F1", color="#111827", max_width=cw - 18, mono=numeric)
            cx += cw


def build_pdf(spec, pdf_path):
    headers, rows = load_csv(spec["path"])
    split_counts = Counter(row.get("split", "") for row in rows)
    class_counts = Counter(row.get("class_name", "") for row in rows)

    c = PdfCanvas(PAGE_W, PAGE_H)
    c.rect(0, 0, PAGE_W, PAGE_H, fill="#ffffff")
    c.rect(0, 0, PAGE_W, 24, fill="#2563eb")
    c.text(spec["title"], MARGIN, 86, size=48, font="F2", color="#0f172a")
    c.text(f"Source: {spec['path'].relative_to(ROOT)}", MARGIN, 122, size=22, color="#334155")
    c.text(spec["subtitle"], MARGIN, 154, size=20, color="#64748b")

    card_y = 205
    card_w = (PAGE_W - 2 * MARGIN - 3 * 28) / 4
    draw_card(c, MARGIN, card_y, card_w, 135, "Data rows", f"{len(rows):,}", "rows in CSV")
    draw_card(c, MARGIN + card_w + 28, card_y, card_w, 135, "Columns", str(len(headers)), f"{concept_count(headers)} concept-related columns")
    draw_card(
        c,
        MARGIN + 2 * (card_w + 28),
        card_y,
        card_w,
        135,
        "Split rows",
        f"{split_counts['train']:,} / {split_counts['val']:,} / {split_counts['test']:,}",
        "train / val / test",
    )
    draw_card(
        c,
        MARGIN + 3 * (card_w + 28),
        card_y,
        card_w,
        135,
        "Class rows",
        f"{class_counts['COVID']:,} / {class_counts['NonCOVID']:,}",
        "COVID / NonCOVID",
    )

    draw_column_inventory(c, headers, spec)
    draw_sample_table(c, representative_rows(rows), spec)

    c.line(MARGIN, 1585, PAGE_W - MARGIN, 1585, color="#cbd5e1", line_width=1.0)
    c.text(spec["note"], MARGIN, 1620, size=18, color="#475569", max_width=PAGE_W - 2 * MARGIN)
    c.text(f"Suggested caption: {spec['caption']}", MARGIN, 1650, size=18, font="F2", color="#334155", max_width=PAGE_W - 2 * MARGIN)
    write_pdf(pdf_path, c)


def render_png(spec):
    if not spec["path"].exists():
        raise FileNotFoundError(spec["path"])
    pdf_path = spec["output"].with_suffix(".tmp.pdf")
    try:
        build_pdf(spec, pdf_path)
        result = subprocess.run(
            ["sips", "-s", "format", "png", str(pdf_path), "--out", str(spec["output"])],
            text=True,
            capture_output=True,
            check=True,
        )
        if result.stderr:
            print(result.stderr.strip())
    finally:
        if pdf_path.exists():
            pdf_path.unlink()
    return spec["output"]


def main():
    for spec in SPECS:
        print(render_png(spec))


if __name__ == "__main__":
    main()
