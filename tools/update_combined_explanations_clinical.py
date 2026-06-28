#!/usr/bin/env python3
import csv
import json
import math
import shutil
import subprocess
import textwrap
import zlib
from pathlib import Path

from make_labeled_concept_aware_heatmaps import decode_png_to_rgb


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "03_SS-VIRULEX-outputs/final_results/combined_explanations"
HEATMAP_CASES = ROOT / "03_SS-VIRULEX-outputs/visual_explanations/heatmaps/heatmap_cases_covid.csv"
PREDICTIONS = ROOT / "03_SS-VIRULEX-outputs/final_results/predictions/predictions_covid.csv"
FUSED = ROOT / "03_SS-VIRULEX-outputs/fusion/fused_features_covid.csv"
CONCEPT_PROBS = ROOT / "03_SS-VIRULEX-outputs/concept_branch/probabilities/covid_concept_probabilities.csv"
FEATURE_SETS = ROOT / "03_SS-VIRULEX-outputs/features/04_combined_zfmis_feature_sets.json"
CALIBRATION = ROOT / "03_SS-VIRULEX-outputs/summaries/05f_combined_calibrated_threshold_results.csv"

MANIFEST = OUT_DIR / "clinical_reporting_manifest.csv"
README = OUT_DIR / "README.md"
BACKUP_DIR = OUT_DIR / "legacy_layout_backup"

PAGE_W = 3240
PAGE_H = 1800
MARGIN = 90

STAT_ALIAS = {
    "entropy": "stat_entropy",
    "shannon_entropy": "stat_shannon_entropy",
    "mean_abs_dev": "stat_mean_abs_dev",
    "iqr": "stat_iqr",
    "signal_to_noise": "stat_signal_to_noise",
    "coef_of_var": "stat_coef_of_var",
    "geometric_mean": "stat_geometric_mean",
}


def esc(text):
    text = str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in text)


def rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))


class PdfCanvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.ops = []
        self.image = None

    def y(self, top_y):
        return self.height - top_y

    def fill(self, color):
        r, g, b = rgb(color)
        self.ops.append(f"{r:.4f} {g:.4f} {b:.4f} rg")

    def stroke(self, color):
        r, g, b = rgb(color)
        self.ops.append(f"{r:.4f} {g:.4f} {b:.4f} RG")

    def rect(self, x, y, w, h, fill=None, stroke=None, line_width=1):
        if fill:
            self.fill(fill)
        if stroke:
            self.stroke(stroke)
        self.ops.append(f"{line_width:.2f} w")
        self.ops.append(f"{x:.2f} {self.height - y - h:.2f} {w:.2f} {h:.2f} re")
        if fill and stroke:
            self.ops.append("B")
        elif fill:
            self.ops.append("f")
        else:
            self.ops.append("S")

    def line(self, x1, y1, x2, y2, color="#d5dce8", line_width=1):
        self.stroke(color)
        self.ops.append(f"{line_width:.2f} w")
        self.ops.append(f"{x1:.2f} {self.y(y1):.2f} m {x2:.2f} {self.y(y2):.2f} l S")

    def text(self, text, x, y, size=22, font="F1", color="#1f2937", max_chars=None):
        text = str(text)
        if max_chars and len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        self.fill(color)
        self.ops.append(f"BT /{font} {size:.2f} Tf {x:.2f} {self.y(y):.2f} Td ({esc(text)}) Tj ET")

    def wrapped_text(self, text, x, y, width_chars, size=20, font="F1", color="#1f2937", line_h=None, max_lines=None):
        line_h = line_h or size * 1.25
        lines = []
        for part in str(text).splitlines():
            lines.extend(textwrap.wrap(part, width=width_chars, replace_whitespace=False) or [""])
        if max_lines and len(lines) > max_lines:
            lines = lines[: max_lines - 1] + ["..."]
        for idx, line in enumerate(lines):
            self.text(line, x, y + idx * line_h, size=size, font=font, color=color)
        return y + len(lines) * line_h

    def image_from_png(self, path, x, y, w, h):
        img_w, img_h, rgb_data = decode_png_to_rgb(path)
        scale = min(w / img_w, h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        draw_x = x + (w - draw_w) / 2
        draw_y = y + (h - draw_h) / 2
        compressed = zlib.compress(rgb_data, 9)
        self.image = {
            "width": img_w,
            "height": img_h,
            "data": compressed,
        }
        self.ops.append(
            f"q {draw_w:.2f} 0 0 {draw_h:.2f} {draw_x:.2f} {self.height - draw_y - draw_h:.2f} cm /Im1 Do Q"
        )

    def stream(self):
        return "\n".join(self.ops)


def write_pdf(path, canvas):
    content = canvas.stream()
    has_image = canvas.image is not None
    resources = "/Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R >>"
    content_obj = 8 if has_image else 7
    if has_image:
        resources += " /XObject << /Im1 7 0 R >>"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {canvas.width} {canvas.height}] "
            f"/Resources << {resources} >> /Contents {content_obj} 0 R >>"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
    ]
    if has_image:
        image = canvas.image
        objs.append(
            (
                f"<< /Type /XObject /Subtype /Image /Width {image['width']} /Height {image['height']} "
                f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
                f"/Length {len(image['data'])} >>\nstream\n"
            ).encode()
            + image["data"]
            + b"\nendstream"
        )
    objs.append(f"<< /Length {len(content.encode())} >>\nstream\n{content}\nendstream".encode())

    out = b"%PDF-1.4\n"
    offsets = [0]
    for idx, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


def read_csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def label_from_binary(value):
    return "COVID" if str(value).strip() == "1" else "NonCOVID"


def display_label(label):
    return "Non-COVID" if label == "NonCOVID" else label


def fmt(value, digits=4):
    number = as_float(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def feature_column(name):
    return name if name.startswith("concept_score_") or name.startswith("med_") else STAT_ALIAS.get(name, name)


def feature_label(name):
    col = feature_column(name)
    if col.startswith("concept_score_"):
        return "Concept: " + col.removeprefix("concept_score_").replace("_", " ")
    if col.startswith("med_"):
        return "Diagnosis prob.: " + col.replace("_", " ")
    if col.startswith("stat_"):
        return "Statistical: " + col.removeprefix("stat_").replace("_", " ")
    return col.replace("_", " ")


def feature_source(name):
    col = feature_column(name)
    if col.startswith("concept_score_"):
        return "Concept"
    if col.startswith("med_"):
        return "Diagnostic probability"
    return "Statistical"


def selected_feature_rows(fused_row, selected):
    rows = []
    for name in selected:
        col = feature_column(name)
        rows.append(
            {
                "feature": feature_label(name),
                "source": feature_source(name),
                "value": fmt(fused_row.get(col, ""), 4),
            }
        )
    return rows


def top_concepts(concept_row, n=5):
    rows = []
    for key, value in concept_row.items():
        if key.startswith("concept_score_"):
            rows.append((key.removeprefix("concept_score_").replace("_", " "), as_float(value)))
    return sorted(rows, key=lambda item: item[1], reverse=True)[:n]


def rule_trace(fused_row):
    med_task = as_float(fused_row.get("med_task_prob_covid"))
    iqr = as_float(fused_row.get("stat_iqr"))
    multilobar = as_float(fused_row.get("concept_score_multilobar_distribution"))
    mad = as_float(fused_row.get("stat_mean_abs_dev"))
    lines = []

    if med_task <= 0.08:
        lines.append(f"med_task_prob_covid <= 0.08 (value {med_task:.4f})")
        if med_task <= 0.0:
            lines.append(f"med_task_prob_covid <= 0.00 (value {med_task:.4f})")
            pred = "NonCOVID"
        else:
            lines.append(f"med_task_prob_covid > 0.00 (value {med_task:.4f})")
            if iqr <= 0.58:
                lines.append(f"stat_iqr <= 0.58 (value {iqr:.4f})")
                pred = "NonCOVID"
            else:
                lines.append(f"stat_iqr > 0.58 (value {iqr:.4f})")
                pred = "COVID"
    else:
        lines.append(f"med_task_prob_covid > 0.08 (value {med_task:.4f})")
        if med_task <= 0.89:
            lines.append(f"med_task_prob_covid <= 0.89 (value {med_task:.4f})")
            if multilobar <= 0.59:
                lines.append(f"concept_multilobar <= 0.59 (value {multilobar:.4f})")
                pred = "NonCOVID"
            else:
                lines.append(f"concept_multilobar > 0.59 (value {multilobar:.4f})")
                pred = "COVID"
        else:
            lines.append(f"med_task_prob_covid > 0.89 (value {med_task:.4f})")
            if mad <= 0.24:
                lines.append(f"stat_mean_abs_dev <= 0.24 (value {mad:.4f})")
            else:
                lines.append(f"stat_mean_abs_dev > 0.24 (value {mad:.4f})")
            pred = "COVID"
    return pred, lines


def load_threshold():
    for row in read_csv_rows(CALIBRATION):
        if row["classifier"] == "logistic_regression" and row["feature_set"] == "selected_9":
            return as_float(row["threshold"], 0.63)
    return 0.63


def draw_section_header(c, x, y, w, title, number, accent="#0f766e"):
    c.rect(x, y, w, 54, fill="#f8fafc", stroke="#d5dce8", line_width=1.2)
    c.rect(x, y, 54, 54, fill=accent)
    c.text(str(number), x + 19, y + 36, size=24, font="F2", color="#ffffff")
    c.text(title, x + 72, y + 36, size=26, font="F2", color="#0f172a", max_chars=64)


def draw_badge(c, x, y, w, label, fill):
    c.rect(x, y, w, 46, fill=fill)
    c.text(label, x + 18, y + 31, size=22, font="F2", color="#ffffff", max_chars=34)


def draw_summary(c, x, y, w, h, title, value, note, color="#0f766e"):
    c.rect(x, y, w, h, fill="#ffffff", stroke="#d5dce8", line_width=1.3)
    c.rect(x, y, 10, h, fill=color)
    c.text(title.upper(), x + 28, y + 38, size=18, font="F2", color="#64748b")
    c.text(value, x + 28, y + 86, size=34, font="F2", color="#0f172a", max_chars=28)
    c.text(note, x + 28, y + 124, size=18, color="#475569", max_chars=42)


def draw_probability_bar(c, x, y, w, p_covid, threshold):
    c.rect(x, y, w, 28, fill="#e5e7eb", stroke="#cbd5e1", line_width=1.0)
    fill_w = max(0, min(w, w * p_covid))
    bar_color = "#dc2626" if p_covid >= threshold else "#2563eb"
    c.rect(x, y, fill_w, 28, fill=bar_color)
    tx = x + w * threshold
    c.line(tx, y - 8, tx, y + 42, color="#111827", line_width=2)
    c.text("0", x, y + 62, size=15, color="#64748b")
    c.text("1", x + w - 8, y + 62, size=15, color="#64748b")
    c.text(f"threshold {threshold:.2f}", tx - 52, y - 16, size=15, color="#111827")


def render_case(case, pred, fused_row, concept_row, selected, threshold, out_png):
    image_id = case["image_id"]
    true_label = case["class_name"]
    final_prediction = label_from_binary(pred["ss_virulex_calibrated_threshold_logistic_regression_prediction"])
    p_covid = as_float(pred["ss_virulex_calibrated_threshold_logistic_regression_probability_positive"])
    status = "Concordant" if true_label == final_prediction else "Discordant"
    status_color = "#16a34a" if status == "Concordant" else "#f97316"
    rule_pred, rule_lines = rule_trace(fused_row)
    selected_rows = selected_feature_rows(fused_row, selected)
    concepts = top_concepts(concept_row, 5)
    heatmap_path = Path(case["ss_virulex_heatmap_path"])

    c = PdfCanvas(PAGE_W, PAGE_H)
    c.rect(0, 0, PAGE_W, PAGE_H, fill="#ffffff")
    c.rect(0, 0, PAGE_W, 24, fill="#0f766e")
    c.text("SS-VIRULEX Final Outputs / Reporting", MARGIN, 78, size=50, font="F2", color="#0f172a")
    c.text("Clinical-style explanation report for thesis documentation; research output only, not for standalone clinical diagnosis.", MARGIN, 116, size=22, color="#475569")
    c.wrapped_text(f"Case ID: {image_id}", MARGIN, 150, width_chars=126, size=19, color="#334155", max_lines=2)

    draw_badge(c, 2380, 54, 330, f"Reference: {display_label(true_label)}", "#dc2626" if true_label == "COVID" else "#2563eb")
    draw_badge(c, 2730, 54, 370, f"Prediction: {display_label(final_prediction)}", status_color)
    c.text(f"p(COVID)={p_covid:.4f}", 2380, 126, size=22, font="F2", color="#0f172a")
    c.text(f"Concordance: {status}", 2660, 126, size=22, font="F2", color=status_color)

    summary_y = 210
    card_w = (PAGE_W - 2 * MARGIN - 3 * 28) / 4
    draw_summary(c, MARGIN, summary_y, card_w, 130, "Selected fused features", "selected_9", "Final calibrated model input", "#0f766e")
    draw_summary(c, MARGIN + card_w + 28, summary_y, card_w, 130, "Rule explanation", display_label(rule_pred), "Decision Tree selected_15", "#7c3aed")
    draw_summary(c, MARGIN + 2 * (card_w + 28), summary_y, card_w, 130, "Calibrated prediction", display_label(final_prediction), f"Threshold {threshold:.2f}", status_color)
    draw_summary(c, MARGIN + 3 * (card_w + 28), summary_y, card_w, 130, "Concept-aware heatmap", "Available", "Explanation files linked", "#0891b2")

    left_x, left_w = MARGIN, 1040
    mid_x, mid_w = left_x + left_w + 40, 1000
    right_x, right_w = mid_x + mid_w + 40, PAGE_W - MARGIN - (mid_x + mid_w + 40)
    top_y = 380

    c.rect(left_x, top_y, left_w, 600, fill="#ffffff", stroke="#d5dce8", line_width=1.2)
    draw_section_header(c, left_x, top_y, left_w, "Selected Fused Features", 1, "#0f766e")
    c.text("ZFMIS-selected feature set used by the final calibrated logistic model.", left_x + 28, top_y + 88, size=18, color="#475569")
    table_y = top_y + 122
    c.rect(left_x + 24, table_y, left_w - 48, 40, fill="#e0f2fe")
    c.text("Feature", left_x + 42, table_y + 27, size=17, font="F2", color="#0f172a")
    c.text("Source", left_x + 660, table_y + 27, size=17, font="F2", color="#0f172a")
    c.text("Value", left_x + 880, table_y + 27, size=17, font="F2", color="#0f172a")
    row_y = table_y + 40
    for idx, row in enumerate(selected_rows):
        c.rect(left_x + 24, row_y + idx * 43, left_w - 48, 43, fill="#ffffff" if idx % 2 == 0 else "#f8fafc")
        c.text(row["feature"], left_x + 42, row_y + idx * 43 + 28, size=15.5, color="#111827", max_chars=66)
        c.text(row["source"], left_x + 660, row_y + idx * 43 + 28, size=15.5, color="#334155", max_chars=24)
        c.text(row["value"], left_x + 880, row_y + idx * 43 + 28, size=15.5, font="F3", color="#111827")

    c.rect(left_x, 1010, left_w, 520, fill="#ffffff", stroke="#d5dce8", line_width=1.2)
    draw_section_header(c, left_x, 1010, left_w, "Rule-Based Explanations", 2, "#7c3aed")
    c.wrapped_text(
        "Readable trace from the best fused Decision Tree. This is an interpretable comparator, not the final calibrated model.",
        left_x + 28,
        1096,
        width_chars=106,
        size=18,
        color="#475569",
        max_lines=2,
        line_h=24,
    )
    c.text(f"Rule prediction: {display_label(rule_pred)}", left_x + 28, 1162, size=22, font="F2", color="#0f172a")
    c.text(f"Decision-tree p(COVID): {fmt(pred.get('ss_virulex_decision_tree_probability_positive', ''), 4)}", left_x + 360, 1162, size=20, color="#334155")
    y = 1210
    for idx, line in enumerate(rule_lines[:5]):
        prefix = "IF" if idx == 0 else "AND"
        c.rect(left_x + 30, y - 23, 58, 30, fill="#f3e8ff")
        c.text(prefix, left_x + 45, y, size=15, font="F2", color="#6d28d9")
        y = c.wrapped_text(line, left_x + 104, y, width_chars=86, size=17, color="#111827", line_h=23, max_lines=2) + 8

    c.rect(mid_x, top_y, mid_w, 1150, fill="#ffffff", stroke="#d5dce8", line_width=1.2)
    draw_section_header(c, mid_x, top_y, mid_w, "Concept-Aware Heatmaps and Explanation Files", 4, "#0891b2")
    c.text("Concept-aware SFMOV overlay with top Med-MICN concept scores.", mid_x + 28, top_y + 88, size=18, color="#475569")
    c.rect(mid_x + 45, top_y + 120, mid_w - 90, 820, fill="#f8fafc", stroke="#d5dce8", line_width=1.0)
    c.image_from_png(heatmap_path, mid_x + 55, top_y + 130, mid_w - 110, 800)
    c.text("Explanation files", mid_x + 45, top_y + 980, size=22, font="F2", color="#0f172a")
    c.wrapped_text("Concept-aware PNG: visual_explanations/heatmaps/ss_virulex_concept_aware/", mid_x + 45, top_y + 1018, 82, size=17, color="#334155", max_lines=2)
    c.wrapped_text("Combined report PNG: final_results/combined_explanations/", mid_x + 45, top_y + 1068, 82, size=17, color="#334155", max_lines=2)

    c.rect(right_x, top_y, right_w, 500, fill="#ffffff", stroke="#d5dce8", line_width=1.2)
    draw_section_header(c, right_x, top_y, right_w, "Calibrated Probabilistic Predictions", 3, status_color)
    c.wrapped_text(
        "Final operating model: calibrated logistic regression using selected_9 fused features.",
        right_x + 28,
        top_y + 88,
        width_chars=78,
        size=18,
        color="#475569",
        max_lines=2,
        line_h=24,
    )
    c.text("Reference diagnosis", right_x + 28, top_y + 146, size=18, font="F2", color="#64748b")
    c.text(display_label(true_label), right_x + 28, top_y + 188, size=34, font="F2", color="#0f172a")
    c.text("Model prediction", right_x + 360, top_y + 146, size=18, font="F2", color="#64748b")
    c.text(display_label(final_prediction), right_x + 360, top_y + 188, size=34, font="F2", color="#0f172a")
    c.text(f"p(COVID) = {p_covid:.4f}", right_x + 28, top_y + 252, size=28, font="F2", color="#0f172a")
    draw_probability_bar(c, right_x + 28, top_y + 292, right_w - 56, p_covid, threshold)
    c.text(f"Classification rule: COVID if calibrated p(COVID) >= {threshold:.2f}; otherwise Non-COVID.", right_x + 28, top_y + 398, size=18, color="#334155", max_chars=90)
    c.text(f"Concordance with reference: {status}", right_x + 28, top_y + 442, size=22, font="F2", color=status_color)

    c.rect(right_x, 910, right_w, 620, fill="#ffffff", stroke="#d5dce8", line_width=1.2)
    c.text("Top Detected Concepts", right_x + 28, 955, size=28, font="F2", color="#0f172a")
    c.text("Highest Med-MICN concept probabilities for this case.", right_x + 28, 992, size=18, color="#475569")
    bar_x = right_x + 280
    bar_w = right_w - 340
    y = 1044
    colors = ["#0f766e", "#2563eb", "#f97316", "#7c3aed", "#0891b2"]
    for idx, (concept, score) in enumerate(concepts):
        c.wrapped_text(concept, right_x + 28, y + 4, 26, size=16, color="#111827", max_lines=2, line_h=20)
        c.rect(bar_x, y, bar_w, 26, fill="#e5e7eb")
        c.rect(bar_x, y, bar_w * max(0, min(1, score)), 26, fill=colors[idx % len(colors)])
        c.text(f"{score:.3f}", bar_x + bar_w + 12, y + 20, size=16, font="F3", color="#111827")
        y += 84

    c.rect(right_x, 1560, right_w, 90, fill="#f8fafc", stroke="#d5dce8", line_width=1.0)
    c.text("Clinical reporting note", right_x + 28, 1592, size=19, font="F2", color="#0f172a")
    c.wrapped_text("Use as explainable AI evidence only. Heatmaps are model-derived saliency overlays and are not expert-validated localization.", right_x + 28, 1622, 88, size=15.5, color="#475569", max_lines=2)

    c.line(MARGIN, 1690, PAGE_W - MARGIN, 1690, color="#d5dce8", line_width=1)
    c.text("Sources: fused_features_covid.csv | predictions_covid.csv | 06_combined_best_decision_tree_rules.txt | heatmap_cases_covid.csv", MARGIN, 1726, size=17, color="#475569")
    c.text("Disclaimer: research/thesis reporting artifact only; not validated for independent clinical diagnosis or treatment decisions.", MARGIN, 1760, size=17, font="F2", color="#334155")

    tmp_pdf = out_png.with_suffix(".tmp.pdf")
    try:
        write_pdf(tmp_pdf, c)
        subprocess.run(["sips", "-s", "format", "png", str(tmp_pdf), "--out", str(out_png)], check=True, text=True, capture_output=True)
    finally:
        if tmp_pdf.exists():
            tmp_pdf.unlink()

    return {
        "image_id": image_id,
        "reference_diagnosis": true_label,
        "calibrated_prediction": final_prediction,
        "calibrated_probability_covid": f"{p_covid:.6f}",
        "threshold": f"{threshold:.2f}",
        "concordance": status,
        "selected_feature_set": "selected_9",
        "rule_model": "Decision Tree selected_15",
        "rule_prediction": rule_pred,
        "concept_aware_heatmap": str(heatmap_path),
        "combined_explanation": str(out_png),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BACKUP_DIR.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        for path in OUT_DIR.glob("*_combined_explanation.png"):
            shutil.copy2(path, BACKUP_DIR / path.name)

    cases = {row["image_id"]: row for row in read_csv_rows(HEATMAP_CASES)}
    preds = {row["image_id"]: row for row in read_csv_rows(PREDICTIONS)}
    fused = {row["image_id"]: row for row in read_csv_rows(FUSED) if row.get("split") == "test"}
    concepts = {row["image_id"]: row for row in read_csv_rows(CONCEPT_PROBS) if row.get("split") == "test"}
    selected = json.loads(FEATURE_SETS.read_text(encoding="utf-8"))["selected_9"]
    threshold = load_threshold()

    manifest_rows = []
    for image_id, case in sorted(cases.items()):
        out_png = OUT_DIR / f"{image_id}_combined_explanation.png"
        manifest_rows.append(render_case(case, preds[image_id], fused[image_id], concepts[image_id], selected, threshold, out_png))

    with MANIFEST.open("w", newline="") as handle:
        fieldnames = list(manifest_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    README.write_text(
        "These combined explanation PNGs are clinical-style SS-VIRULEX final reporting panels. "
        "Each panel contains four final output sections: selected fused features, rule-based "
        "explanation, calibrated probabilistic prediction, and concept-aware heatmap/explanation "
        "file references. The panels use reference diagnosis and model prediction wording and "
        "include a research-use-only disclaimer. Original pre-update panels are preserved in "
        "`legacy_layout_backup/`.\n\n"
        "Primary manifest: `clinical_reporting_manifest.csv`.\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(manifest_rows)} clinical-style combined explanation panels")
    print(OUT_DIR)
    print(MANIFEST)


if __name__ == "__main__":
    main()
