#!/usr/bin/env python3
import csv
import struct
import subprocess
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEATMAP_DIR = ROOT / "03_SS-VIRULEX-outputs/visual_explanations/heatmaps/ss_virulex_concept_aware"
CASE_CSV = ROOT / "03_SS-VIRULEX-outputs/visual_explanations/heatmaps/heatmap_cases_covid.csv"
PRED_CSV = ROOT / "03_SS-VIRULEX-outputs/final_results/predictions/predictions_covid.csv"
OUT_DIR = HEATMAP_DIR / "diagnosis_labeled"
MANIFEST = OUT_DIR / "diagnosis_labeled_manifest.csv"

BANNER_H = 132


def pdf_escape(text):
    text = str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in text)


def hex_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def set_fill(color):
    r, g, b = hex_rgb(color)
    return f"{r:.4f} {g:.4f} {b:.4f} rg"


def text_op(text, x, y, size=24, font="F1", color="#111827"):
    return "\n".join([
        set_fill(color),
        f"BT /{font} {size:.2f} Tf {x:.2f} {y:.2f} Td ({pdf_escape(text)}) Tj ET",
    ])


def rect_op(x, y, w, h, fill):
    return "\n".join([
        set_fill(fill),
        f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re f",
    ])


def truncate(text, limit):
    text = str(text)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def paeth(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png_to_rgb(path):
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG: {path}")

    pos = 8
    width = height = bit_depth = color_type = None
    idat = []
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk)
        elif chunk_type == b"IDAT":
            idat.append(chunk)
        elif chunk_type == b"IEND":
            break

    if bit_depth != 8:
        raise ValueError(f"Unsupported PNG bit depth {bit_depth}: {path}")
    channels_by_type = {0: 1, 2: 3, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError(f"Unsupported PNG color type {color_type}: {path}")

    channels = channels_by_type[color_type]
    stride = width * channels
    raw = zlib.decompress(b"".join(idat))
    rows = []
    prev = bytearray(stride)
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        scan = bytearray(raw[offset + 1:offset + 1 + stride])
        offset += 1 + stride

        for i in range(stride):
            left = scan[i - channels] if i >= channels else 0
            up = prev[i]
            upper_left = prev[i - channels] if i >= channels else 0
            if filter_type == 1:
                scan[i] = (scan[i] + left) & 0xFF
            elif filter_type == 2:
                scan[i] = (scan[i] + up) & 0xFF
            elif filter_type == 3:
                scan[i] = (scan[i] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                scan[i] = (scan[i] + paeth(left, up, upper_left)) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"Unsupported PNG filter {filter_type}: {path}")

        if color_type == 6:
            rgb = bytearray(width * 3)
            for px in range(width):
                r, g, b, a = scan[px * 4:px * 4 + 4]
                inv = 255 - a
                rgb[px * 3] = (r * a + 255 * inv) // 255
                rgb[px * 3 + 1] = (g * a + 255 * inv) // 255
                rgb[px * 3 + 2] = (b * a + 255 * inv) // 255
            rows.append(bytes(rgb))
        elif color_type == 2:
            rows.append(bytes(scan))
        else:
            rgb = bytearray(width * 3)
            for px, gray in enumerate(scan):
                rgb[px * 3:px * 3 + 3] = bytes([gray, gray, gray])
            rows.append(bytes(rgb))
        prev = scan

    return width, height, b"".join(rows)


def write_pdf_with_image(pdf_path, image_path, labels):
    img_w, img_h, rgb = decode_png_to_rgb(image_path)
    page_w = img_w
    page_h = img_h + BANNER_H
    image_stream = zlib.compress(rgb, 9)

    true_color = "#dc2626" if labels["true_diagnosis"] == "COVID" else "#2563eb"
    pred_color = "#16a34a" if labels["status"] == "Correct" else "#f97316"

    ops = [
        rect_op(0, img_h, page_w, BANNER_H, "#ffffff"),
        rect_op(0, img_h + BANNER_H - 10, page_w, 10, "#2563eb"),
        rect_op(24, img_h + 76, 310, 36, true_color),
        rect_op(352, img_h + 76, 350, 36, pred_color),
        text_op("Concept-aware SFMOV heatmap", 24, img_h + 42, 22, "F2", "#0f172a"),
        text_op(f"TRUE DIAGNOSIS: {labels['true_diagnosis']}", 42, img_h + 88, 16, "F2", "#ffffff"),
        text_op(f"FINAL PREDICTION: {labels['prediction_label']}", 372, img_h + 88, 16, "F2", "#ffffff"),
        text_op(f"pCOVID={labels['p_covid']}", 720, img_h + 96, 16, "F1", "#334155"),
        text_op(f"Status: {labels['status']}", 720, img_h + 74, 16, "F1", "#334155"),
        text_op(truncate(labels["image_id"], 96), 24, img_h + 18, 16, "F1", "#475569"),
        f"q {img_w:.2f} 0 0 {img_h:.2f} 0 0 cm /Im1 Do Q",
    ]
    content = "\n".join(ops)

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w} {page_h}] "
            f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> /XObject << /Im1 6 0 R >> >> "
            f"/Contents 7 0 R >>"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        (
            f"<< /Type /XObject /Subtype /Image /Width {img_w} /Height {img_h} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(image_stream)} >>\nstream\n"
        ).encode() + image_stream + b"\nendstream",
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
    pdf_path.write_bytes(out)


def load_cases():
    with CASE_CSV.open(newline="") as handle:
        return {row["image_id"]: row for row in csv.DictReader(handle)}


def load_predictions():
    with PRED_CSV.open(newline="") as handle:
        return {row["image_id"]: row for row in csv.DictReader(handle)}


def label_from_binary(value):
    if str(value).strip() == "1":
        return "COVID"
    if str(value).strip() == "0":
        return "NonCOVID"
    return "Unavailable"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    predictions = load_predictions()
    written = []

    for image_id, case in sorted(cases.items()):
        source = Path(case["ss_virulex_heatmap_path"])
        if not source.exists():
            source = HEATMAP_DIR / f"{image_id}_concept_aware.png"
        if not source.exists():
            raise FileNotFoundError(source)

        pred = predictions.get(image_id, {})
        true_label = case["class_name"]
        pred_label = label_from_binary(pred.get("ss_virulex_calibrated_threshold_logistic_regression_prediction", ""))
        p_covid_raw = pred.get("ss_virulex_calibrated_threshold_logistic_regression_probability_positive", "")
        try:
            p_covid = f"{float(p_covid_raw):.4f}"
        except (TypeError, ValueError):
            p_covid = "Unavailable"
        status = "Correct" if pred_label == true_label else ("Unavailable" if pred_label == "Unavailable" else "Mismatch")

        out_png = OUT_DIR / f"{image_id}_concept_aware_diagnosis_labeled.png"
        tmp_pdf = OUT_DIR / f"{image_id}_concept_aware_diagnosis_labeled.tmp.pdf"
        labels = {
            "image_id": image_id,
            "true_diagnosis": true_label,
            "prediction_label": pred_label,
            "p_covid": p_covid,
            "status": status,
        }
        try:
            write_pdf_with_image(tmp_pdf, source, labels)
            subprocess.run(
                ["sips", "-s", "format", "png", str(tmp_pdf), "--out", str(out_png)],
                text=True,
                capture_output=True,
                check=True,
            )
        finally:
            if tmp_pdf.exists():
                tmp_pdf.unlink()

        written.append({
            "image_id": image_id,
            "true_diagnosis": true_label,
            "final_prediction": pred_label,
            "p_covid": p_covid,
            "status": status,
            "source_heatmap": str(source),
            "diagnosis_labeled_heatmap": str(out_png),
        })

    with MANIFEST.open("w", newline="") as handle:
        fieldnames = [
            "image_id",
            "true_diagnosis",
            "final_prediction",
            "p_covid",
            "status",
            "source_heatmap",
            "diagnosis_labeled_heatmap",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(written)

    print(f"Wrote {len(written)} labeled heatmaps")
    print(OUT_DIR)
    print(MANIFEST)


if __name__ == "__main__":
    main()
