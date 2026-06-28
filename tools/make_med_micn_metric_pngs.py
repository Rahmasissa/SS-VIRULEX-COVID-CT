#!/usr/bin/env python3
import csv
import subprocess
from pathlib import Path

from make_csv_thesis_pngs import PdfCanvas, write_pdf


ROOT = Path(__file__).resolve().parents[1]
METRICS_CSV = ROOT / "03_SS-VIRULEX-outputs/01_Med-MICN/outputs/covid_ct_med_micn/test_metrics.csv"
CONCEPT_CSV = ROOT / "03_SS-VIRULEX-outputs/01_Med-MICN/outputs/covid_ct_med_micn/test_concept_metrics.csv"

METRICS_PNG = ROOT / "med_micn_test_metrics_table.png"
CONCEPT_PNG = ROOT / "med_micn_test_concept_metrics_table.png"

PAGE_W = 2400
PAGE_H = 1500
MARGIN = 70


def rel(path):
    return str(path.relative_to(ROOT))


def fmt(value):
    if value in ("", None):
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def read_one_row(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def draw_title(c, title, source, subtitle):
    c.rect(0, 0, PAGE_W, PAGE_H, fill="#ffffff")
    c.rect(0, 0, PAGE_W, 24, fill="#2563eb")
    c.text(title, MARGIN, 86, size=48, font="F2", color="#0f172a")
    c.text(f"Source: {source}", MARGIN, 122, size=22, color="#334155")
    c.text(subtitle, MARGIN, 154, size=20, color="#64748b")


def draw_card(c, x, y, w, h, label, value, note):
    c.rect(x, y, w, h, fill="#f8fafc", stroke="#cbd5e1", line_width=1.4)
    c.text(label.upper(), x + 28, y + 40, size=18, font="F2", color="#64748b", max_width=w - 56)
    c.text(value, x + 28, y + 86, size=42, font="F2", color="#0f172a", max_width=w - 56)
    c.text(note, x + 28, y + 122, size=18, font="F1", color="#475569", max_width=w - 56)


def draw_table(c, x, y, columns, rows, title, note=None, row_h=62, title_h=62, header_h=58):
    w = PAGE_W - 2 * MARGIN
    table_h = title_h + header_h + row_h * len(rows)
    c.rect(x, y, w, table_h, fill="#ffffff", stroke="#cbd5e1", line_width=1.4)
    c.rect(x, y, w, title_h, fill="#f8fafc", stroke="#cbd5e1", line_width=1.0)
    c.text(title, x + 24, y + 40, size=26, font="F2", color="#0f172a")
    if note:
        c.text(note, x + 500, y + 39, size=18, color="#475569", max_width=w - 530)

    scale = w / sum(col["width"] for col in columns)
    scaled = [{**col, "width": col["width"] * scale} for col in columns]

    c.rect(x, y + title_h, w, header_h, fill="#dbeafe", stroke="#cbd5e1", line_width=1.0)
    cx = x
    for col in scaled:
        c.line(cx, y + title_h, cx, y + title_h + header_h + row_h * len(rows), color="#cbd5e1", line_width=0.8)
        c.text(col["label"], cx + 12, y + title_h + 37, size=18, font="F2", color="#1e293b", max_width=col["width"] - 22)
        cx += col["width"]
    c.line(x + w, y + title_h, x + w, y + title_h + header_h + row_h * len(rows), color="#cbd5e1", line_width=0.8)

    for ridx, row in enumerate(rows):
        ry = y + title_h + header_h + ridx * row_h
        c.rect(x, ry, w, row_h, fill="#ffffff" if ridx % 2 == 0 else "#f8fafc")
        c.line(x, ry + row_h, x + w, ry + row_h, color="#e2e8f0", line_width=0.8)
        cx = x
        for col in scaled:
            value = row.get(col["key"], "")
            c.text(value, cx + 12, ry + 38, size=18, font="F1", color="#111827", max_width=col["width"] - 22)
            cx += col["width"]


def render_metrics_pdf(pdf_path):
    row = read_one_row(METRICS_CSV)
    c = PdfCanvas(PAGE_W, PAGE_H)
    draw_title(
        c,
        "Med-MICN Aggregate Test Metrics",
        rel(METRICS_CSV),
        "Task classifier, neural-symbolic output, and aggregate concept encoder metrics from the test split.",
    )

    card_y = 205
    card_w = (PAGE_W - 2 * MARGIN - 3 * 28) / 4
    draw_card(c, MARGIN, card_y, card_w, 135, "Split", row["split"], "evaluation subset")
    draw_card(c, MARGIN + card_w + 28, card_y, card_w, 135, "Loss", fmt(row["loss"]), "saved test loss")
    draw_card(c, MARGIN + 2 * (card_w + 28), card_y, card_w, 135, "Best accuracy", fmt(row["concept_accuracy"]), "concept aggregate")
    draw_card(c, MARGIN + 3 * (card_w + 28), card_y, card_w, 135, "Best AUC", fmt(row["task_auc"]), "task classifier")

    table_rows = [
        {
            "output": "Task classifier",
            "accuracy": fmt(row["task_accuracy"]),
            "precision": fmt(row["task_precision_macro"]),
            "recall": fmt(row["task_recall_macro"]),
            "f1": fmt(row["task_f1_macro"]),
            "auc": fmt(row["task_auc"]),
        },
        {
            "output": "Neural-symbolic layer",
            "accuracy": fmt(row["neural_accuracy"]),
            "precision": fmt(row["neural_precision_macro"]),
            "recall": fmt(row["neural_recall_macro"]),
            "f1": fmt(row["neural_f1_macro"]),
            "auc": fmt(row["neural_auc"]),
        },
        {
            "output": "Concept encoder aggregate",
            "accuracy": fmt(row["concept_accuracy"]),
            "precision": "-",
            "recall": "-",
            "f1": f"{fmt(row['concept_f1_macro'])} macro / {fmt(row['concept_f1_micro'])} micro",
            "auc": "-",
        },
    ]
    columns = [
        {"label": "Output", "key": "output", "width": 460},
        {"label": "Accuracy", "key": "accuracy", "width": 250},
        {"label": "Macro precision", "key": "precision", "width": 310},
        {"label": "Macro recall / Bal. accuracy", "key": "recall", "width": 430},
        {"label": "Macro F1", "key": "f1", "width": 380},
        {"label": "ROC AUC", "key": "auc", "width": 230},
    ]
    draw_table(c, MARGIN, 430, columns, table_rows, "Aggregate Branch Metrics", "Values rounded to four decimals for thesis display.", row_h=76)

    c.line(MARGIN, 1280, PAGE_W - MARGIN, 1280, color="#cbd5e1", line_width=1.0)
    c.text("Interpretation note: macro recall is used as the balanced-accuracy equivalent for the Med-MICN branch comparison.", MARGIN, 1320, size=19, color="#475569", max_width=PAGE_W - 2 * MARGIN)
    c.text("Suggested caption: Med-MICN branch aggregate metrics on the COVID-CT test split.", MARGIN, 1360, size=20, font="F2", color="#334155")
    write_pdf(pdf_path, c)


def render_concept_pdf(pdf_path):
    rows = read_rows(CONCEPT_CSV)
    c = PdfCanvas(PAGE_W, PAGE_H)
    draw_title(
        c,
        "Med-MICN Per-Concept Test Metrics",
        rel(CONCEPT_CSV),
        "Per-concept positive rates, accuracy, precision, recall, and F1 from the test split.",
    )

    best_f1 = max(rows, key=lambda r: float(r["f1"]))
    lowest_f1 = min(rows, key=lambda r: float(r["f1"]))
    card_y = 205
    card_w = (PAGE_W - 2 * MARGIN - 3 * 28) / 4
    draw_card(c, MARGIN, card_y, card_w, 135, "Concepts", str(len(rows)), "evaluated outputs")
    draw_card(c, MARGIN + card_w + 28, card_y, card_w, 135, "Highest F1", fmt(best_f1["f1"]), best_f1["concept"])
    draw_card(c, MARGIN + 2 * (card_w + 28), card_y, card_w, 135, "Lowest F1", fmt(lowest_f1["f1"]), lowest_f1["concept"])
    draw_card(c, MARGIN + 3 * (card_w + 28), card_y, card_w, 135, "Rows", str(len(rows)), "CSV records")

    table_rows = [
        {
            "concept": r["concept"],
            "true_rate": fmt(r["positive_rate_true"]),
            "pred_rate": fmt(r["positive_rate_pred"]),
            "accuracy": fmt(r["accuracy"]),
            "precision": fmt(r["precision"]),
            "recall": fmt(r["recall"]),
            "f1": fmt(r["f1"]),
        }
        for r in rows
    ]
    columns = [
        {"label": "Concept", "key": "concept", "width": 610},
        {"label": "True positive rate", "key": "true_rate", "width": 280},
        {"label": "Pred. positive rate", "key": "pred_rate", "width": 300},
        {"label": "Accuracy", "key": "accuracy", "width": 230},
        {"label": "Precision", "key": "precision", "width": 230},
        {"label": "Recall", "key": "recall", "width": 230},
        {"label": "F1", "key": "f1", "width": 190},
    ]
    draw_table(c, MARGIN, 430, columns, table_rows, "Per-Concept Metrics", "Values rounded to four decimals for thesis display.", row_h=64)

    c.line(MARGIN, 1280, PAGE_W - MARGIN, 1280, color="#cbd5e1", line_width=1.0)
    c.text("Interpretation note: high recall with prediction rates near 1.0000 can indicate broad positive prediction rather than strong concept discrimination.", MARGIN, 1320, size=19, color="#475569", max_width=PAGE_W - 2 * MARGIN)
    c.text("Suggested caption: Per-concept Med-MICN metrics on the COVID-CT test split.", MARGIN, 1360, size=20, font="F2", color="#334155")
    write_pdf(pdf_path, c)


def convert_pdf(pdf_path, png_path):
    result = subprocess.run(
        ["sips", "-s", "format", "png", str(pdf_path), "--out", str(png_path)],
        text=True,
        capture_output=True,
        check=True,
    )
    if result.stderr:
        print(result.stderr.strip())


def render(render_pdf, png_path):
    pdf_path = png_path.with_suffix(".tmp.pdf")
    try:
        render_pdf(pdf_path)
        convert_pdf(pdf_path, png_path)
    finally:
        if pdf_path.exists():
            pdf_path.unlink()
    return png_path


def main():
    print(render(render_metrics_pdf, METRICS_PNG))
    print(render(render_concept_pdf, CONCEPT_PNG))


if __name__ == "__main__":
    main()
