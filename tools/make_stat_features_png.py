#!/usr/bin/env python3
import csv
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "03_SS-VIRULEX-outputs/statistical_branch/features/covid_statistical_features.csv"
PNG_PATH = ROOT / "covid_statistical_features_thesis_table.png"

PAGE_W = 2400
PAGE_H = 1700
MARGIN = 70


def esc(text):
    text = str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in text)


def approx_chars(width, size, mono=False):
    factor = 0.61 if mono else 0.52
    return max(4, int(width / (size * factor)))


def truncate(text, width, size, mono=False):
    text = str(text)
    limit = approx_chars(width, size, mono)
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)] + "..."


class PdfCanvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.ops = []

    def y(self, top_y):
        return self.height - top_y

    def color(self, hex_color):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        return r, g, b

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


def load_rows():
    with CSV_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def fmt_num(value):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def representative_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["class_name"])].append(row)

    wanted = [
        ("train", "COVID", 2),
        ("train", "NonCOVID", 2),
        ("val", "COVID", 2),
        ("val", "NonCOVID", 2),
        ("test", "COVID", 1),
        ("test", "NonCOVID", 1),
    ]
    sample = []
    for split, cls, n in wanted:
        sample.extend(groups.get((split, cls), [])[:n])
    return sample


def draw_card(c, x, y, w, h, label, value, note):
    c.rect(x, y, w, h, fill="#f8fafc", stroke="#cbd5e1", line_width=1.4)
    c.text(label.upper(), x + 28, y + 40, size=18, font="F2", color="#64748b", max_width=w - 56)
    c.text(value, x + 28, y + 86, size=44, font="F2", color="#0f172a", max_width=w - 56)
    c.text(note, x + 28, y + 122, size=18, font="F1", color="#475569", max_width=w - 56)


def draw_column_inventory(c, headers):
    y = 405
    x = MARGIN
    w = PAGE_W - (2 * MARGIN)
    h = 480
    c.rect(x, y, w, h, fill="#ffffff", stroke="#cbd5e1", line_width=1.4)
    c.rect(x, y, w, 58, fill="#eaf2ff", stroke="#cbd5e1", line_width=1.0)
    c.text("Column Inventory", x + 24, y + 38, size=26, font="F2", color="#0f172a")
    c.text("All 32 CSV columns are shown here; statistical descriptor columns are prefixed with stat_.", x + 300, y + 37, size=18, color="#475569")

    metadata = [h for h in headers if not h.startswith("stat_")]
    stat_cols = [h for h in headers if h.startswith("stat_")]

    c.text("Metadata / label columns", x + 28, y + 94, size=20, font="F2", color="#334155")
    chip_x = x + 28
    chip_y = y + 118
    chip_h = 34
    for name in metadata:
        chip_w = max(120, min(360, len(name) * 13 + 30))
        if chip_x + chip_w > x + w - 28:
            chip_x = x + 28
            chip_y += 44
        c.rect(chip_x, chip_y, chip_w, chip_h, fill="#f1f5f9", stroke="#cbd5e1", line_width=0.8)
        c.text(name, chip_x + 14, chip_y + 23, size=16, font="F3", color="#1e293b", max_width=chip_w - 24, mono=True)
        chip_x += chip_w + 12

    c.text("Statistical descriptor columns", x + 28, y + 214, size=20, font="F2", color="#334155")
    columns = 4
    col_w = (w - 56) / columns
    for idx, name in enumerate(stat_cols):
        col = idx % columns
        row = idx // columns
        tx = x + 28 + col * col_w
        ty = y + 250 + row * 34
        c.text(name, tx, ty, size=17, font="F3", color="#1f2937", max_width=col_w - 20, mono=True)


def draw_sample_table(c, rows):
    x = MARGIN
    y = 935
    w = PAGE_W - (2 * MARGIN)
    header_h = 56
    row_h = 44
    title_h = 62
    table_h = title_h + header_h + row_h * len(rows)

    c.rect(x, y, w, table_h, fill="#ffffff", stroke="#cbd5e1", line_width=1.4)
    c.rect(x, y, w, title_h, fill="#f8fafc", stroke="#cbd5e1", line_width=1.0)
    c.text("Representative Feature Rows", x + 24, y + 40, size=26, font="F2", color="#0f172a")
    c.text("Numeric values are rounded to four decimals for display.", x + 410, y + 39, size=18, color="#475569")

    cols = [
        ("image_id", 410),
        ("class", 120),
        ("split", 100),
        ("mean", 126),
        ("std_dev", 126),
        ("skewness", 136),
        ("kurtosis", 136),
        ("entropy", 130),
        ("energy", 130),
        ("iqr", 116),
        ("snr", 116),
        ("rms", 116),
        ("shannon", 136),
    ]
    scale = w / sum(width for _, width in cols)
    cols = [(name, width * scale) for name, width in cols]

    c.rect(x, y + title_h, w, header_h, fill="#dbeafe", stroke="#cbd5e1", line_width=1.0)
    cx = x
    for name, cw in cols:
        c.line(cx, y + title_h, cx, y + title_h + header_h + row_h * len(rows), color="#cbd5e1", line_width=0.8)
        c.text(name, cx + 10, y + title_h + 36, size=17, font="F2", color="#1e293b", max_width=cw - 18)
        cx += cw
    c.line(x + w, y + title_h, x + w, y + title_h + header_h + row_h * len(rows), color="#cbd5e1", line_width=0.8)

    field_map = {
        "image_id": "image_id",
        "class": "class_name",
        "split": "split",
        "mean": "stat_mean",
        "std_dev": "stat_std_dev",
        "skewness": "stat_skewness",
        "kurtosis": "stat_kurtosis",
        "entropy": "stat_entropy",
        "energy": "stat_energy",
        "iqr": "stat_iqr",
        "snr": "stat_signal_to_noise",
        "rms": "stat_root_mean_square",
        "shannon": "stat_shannon_entropy",
    }

    for ridx, row in enumerate(rows):
        ry = y + title_h + header_h + ridx * row_h
        c.rect(x, ry, w, row_h, fill="#ffffff" if ridx % 2 == 0 else "#f8fafc")
        c.line(x, ry + row_h, x + w, ry + row_h, color="#e2e8f0", line_width=0.8)
        cx = x
        for name, cw in cols:
            field = field_map[name]
            value = row[field]
            if field.startswith("stat_"):
                value = fmt_num(value)
            c.text(value, cx + 10, ry + 29, size=15, font="F3" if field.startswith("stat_") else "F1", color="#111827", max_width=cw - 18, mono=field.startswith("stat_"))
            cx += cw


def build_pdf(pdf_path):
    headers, rows = load_rows()
    split_counts = Counter(row["split"] for row in rows)
    class_counts = Counter(row["class_name"] for row in rows)
    stat_cols = [h for h in headers if h.startswith("stat_")]

    c = PdfCanvas(PAGE_W, PAGE_H)
    c.rect(0, 0, PAGE_W, PAGE_H, fill="#ffffff")
    c.rect(0, 0, PAGE_W, 24, fill="#2563eb")
    c.text("COVID-CT Statistical Feature Table", MARGIN, 86, size=48, font="F2", color="#0f172a")
    c.text("Source: 03_SS-VIRULEX-outputs/statistical_branch/features/covid_statistical_features.csv", MARGIN, 122, size=22, color="#334155")
    c.text("Thesis-ready PNG preview of the CSV structure and representative rows.", MARGIN, 154, size=20, color="#64748b")

    card_y = 205
    card_w = (PAGE_W - 2 * MARGIN - 3 * 28) / 4
    draw_card(c, MARGIN, card_y, card_w, 135, "Data rows", f"{len(rows):,}", "processing rows in CSV")
    draw_card(c, MARGIN + (card_w + 28), card_y, card_w, 135, "Columns", str(len(headers)), f"{len(stat_cols)} statistical descriptors")
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

    draw_column_inventory(c, headers)
    draw_sample_table(c, representative_rows(rows))

    c.line(MARGIN, 1585, PAGE_W - MARGIN, 1585, color="#cbd5e1", line_width=1.0)
    c.text("Display note: the full CSV contains all 1,257 rows and original numeric precision; this PNG is a readable thesis figure, not a replacement for the raw artifact.", MARGIN, 1620, size=18, color="#475569")
    c.text("Suggested caption: Extracted statistical feature matrix for the COVID-CT processing set.", MARGIN, 1650, size=18, font="F2", color="#334155")
    write_pdf(pdf_path, c)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    pdf_path = PNG_PATH.with_suffix(".tmp.pdf")
    try:
        build_pdf(pdf_path)
        result = subprocess.run(
            ["sips", "-s", "format", "png", str(pdf_path), "--out", str(PNG_PATH)],
            text=True,
            capture_output=True,
            check=True,
        )
        if result.stderr:
            print(result.stderr.strip())
    finally:
        if pdf_path.exists():
            pdf_path.unlink()
    print(PNG_PATH)


if __name__ == "__main__":
    main()
