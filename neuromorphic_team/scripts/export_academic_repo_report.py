#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import FontFace
from PIL import Image


IMAGE_RE = re.compile(r"^!\[(?P<caption>.*)\]\((?P<path>.*)\)\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
ORDERED_RE = re.compile(r"^\d+\.\s+(.*)$")


def clean_inline(text: str) -> str:
    text = text.replace("`", "")
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("*", "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    block: list[str] = []
    index = start
    while index < len(lines) and lines[index].lstrip().startswith("|"):
        block.append(lines[index].rstrip())
        index += 1

    rows: list[list[str]] = []
    for row_index, line in enumerate(block):
        cells = [clean_inline(cell.strip()) for cell in line.strip().strip("|").split("|")]
        if row_index == 1 and all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows, index


class AcademicPDF(FPDF):
    def __init__(self, title: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.doc_title = title
        self.set_title(title)
        self.set_author("Vision AI Project")
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(18, 18, 18)

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8.5)
        self.set_text_color(90, 90, 90)
        self.cell(0, 5, self.doc_title, new_x="LMARGIN", new_y="NEXT", align="R")
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-10)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 4, f"Page {self.page_no()}", align="C")


def add_title_page(pdf: AcademicPDF, title: str, subtitle: str) -> None:
    pdf.add_page()
    pdf.set_y(42)
    pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(0, 10, clean_inline(title), align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11.5)
    pdf.set_text_color(70, 70, 70)
    pdf.multi_cell(0, 6, clean_inline(subtitle), align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(18)
    pdf.set_font("Helvetica", "", 11)
    intro = (
        "This report provides a project-centered summary of the dataset, preprocessing pipeline, "
        "architectural choices, experimental protocol, and quantitative findings currently preserved "
        "in the local workspace."
    )
    pdf.multi_cell(0, 6, intro, align="J")


def add_heading(pdf: AcademicPDF, text: str, level: int) -> None:
    if level <= 2 and pdf.get_y() > 245:
        pdf.add_page()
    size_map = {2: 15, 3: 12.5, 4: 11.5, 5: 11, 6: 11}
    size = size_map.get(level, 11)
    pdf.ln(2 if level <= 3 else 1)
    pdf.set_font("Helvetica", "B", size)
    pdf.multi_cell(0, 7 if level <= 3 else 6, clean_inline(text))
    pdf.ln(0.5)


def add_paragraph(pdf: AcademicPDF, text: str) -> None:
    if not text.strip():
        return
    pdf.set_font("Helvetica", "", 10.5)
    pdf.multi_cell(0, 5.4, clean_inline(text), align="J")
    pdf.ln(1.2)


def add_list_item(pdf: AcademicPDF, text: str, *, label: str) -> None:
    pdf.set_font("Helvetica", "", 10.5)
    clean_text = clean_inline(text)
    line_height = 5.3
    text_width = pdf.epw - 8
    try:
        preview = pdf.multi_cell(
            text_width,
            line_height,
            clean_text,
            dry_run=True,
            output="LINES",
        )
        estimated_height = max(1, len(preview)) * line_height
    except TypeError:
        estimated_height = line_height * 2

    remaining = pdf.h - pdf.b_margin - pdf.get_y()
    if remaining < estimated_height + 3:
        pdf.add_page()

    x = pdf.get_x()
    y = pdf.get_y()
    pdf.set_xy(pdf.l_margin + 2, y)
    pdf.cell(6, line_height, label)
    pdf.set_xy(pdf.l_margin + 8, y)
    pdf.multi_cell(text_width, line_height, clean_text, align="J")
    pdf.ln(0.2)
    pdf.set_x(x)


def add_table(pdf: AcademicPDF, rows: list[list[str]]) -> None:
    if not rows:
        return
    col_count = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (col_count - len(row)) for row in rows]
    weights = []
    for col in range(col_count):
        max_len = max(len(row[col]) for row in normalized_rows)
        weights.append(max(10, min(34, max_len)))
    total = float(sum(weights))
    col_widths = [pdf.epw * weight / total for weight in weights]

    pdf.set_font("Helvetica", "", 8.8)
    with pdf.table(
        col_widths=col_widths,
        line_height=5,
        text_align="LEFT",
        headings_style=FontFace(emphasis="BOLD"),
        cell_fill_color=(255, 255, 255),
        cell_fill_mode="ROWS",
        borders_layout="SINGLE_TOP_LINE",
    ) as table:
        for row_index, row_values in enumerate(normalized_rows):
            row = table.row()
            for value in row_values:
                row.cell(value)
    pdf.ln(1.5)


def add_image(pdf: AcademicPDF, path: Path, caption: str) -> None:
    if not path.exists():
        add_paragraph(pdf, f"[Missing figure: {path}]")
        return

    with Image.open(path) as image:
        width_px, height_px = image.size

    max_w = pdf.epw
    remaining_h = pdf.h - pdf.b_margin - pdf.get_y() - 14
    if remaining_h < 55:
        pdf.add_page()
        remaining_h = pdf.h - pdf.b_margin - pdf.get_y() - 14

    width_mm = max_w
    height_mm = width_mm * height_px / width_px
    if height_mm > remaining_h:
        height_mm = remaining_h
        width_mm = height_mm * width_px / height_px

    x = pdf.l_margin + (pdf.epw - width_mm) / 2
    y = pdf.get_y()
    pdf.image(str(path), x=x, y=y, w=width_mm, h=height_mm)
    pdf.set_y(y + height_mm + 2)
    pdf.set_font("Helvetica", "I", 8.8)
    pdf.multi_cell(0, 4.5, clean_inline(caption), align="C")
    pdf.ln(1.5)


def render_markdown(pdf: AcademicPDF, markdown_path: Path) -> None:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Markdown file is empty: {markdown_path}")

    first_heading = ""
    subtitle = ""
    start_index = 0
    for index, line in enumerate(lines):
        heading_match = HEADING_RE.match(line)
        if heading_match and len(heading_match.group(1)) == 1:
            first_heading = heading_match.group(2).strip()
            start_index = index + 1
            break
    for index in range(start_index, len(lines)):
        candidate = lines[index].strip()
        if candidate:
            subtitle = candidate
            start_index = index + 1
            break

    add_title_page(pdf, first_heading or markdown_path.stem, subtitle)

    paragraph_parts: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_parts
        if paragraph_parts:
            add_paragraph(pdf, " ".join(part.strip() for part in paragraph_parts))
            paragraph_parts = []

    index = start_index
    while index < len(lines):
        raw = lines[index].rstrip()
        stripped = raw.strip()

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        heading_match = HEADING_RE.match(raw)
        if heading_match:
            flush_paragraph()
            add_heading(pdf, heading_match.group(2), len(heading_match.group(1)))
            index += 1
            continue

        image_match = IMAGE_RE.match(stripped)
        if image_match:
            flush_paragraph()
            image_path = (markdown_path.parent / image_match.group("path")).resolve()
            add_image(pdf, image_path, image_match.group("caption"))
            index += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            rows, index = parse_table(lines, index)
            add_table(pdf, rows)
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            add_list_item(pdf, stripped[2:], label="-")
            index += 1
            continue

        ordered_match = ORDERED_RE.match(stripped)
        if ordered_match:
            flush_paragraph()
            label = stripped.split(".", 1)[0] + "."
            add_list_item(pdf, ordered_match.group(1), label=label)
            index += 1
            continue

        paragraph_parts.append(stripped)
        index += 1

    flush_paragraph()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the academic project report markdown to PDF.")
    parser.add_argument(
        "--input",
        default="Analytics/Vision_AI_Project_Academic_Report.md",
        help="Path to the academic markdown report.",
    )
    parser.add_argument(
        "--output",
        default="Analytics/Vision_AI_Project_Academic_Report.pdf",
        help="Path to the output PDF.",
    )
    return parser.parse_args()


def extract_document_title(markdown_path: Path) -> str:
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            return clean_inline(match.group(2))
    return markdown_path.stem.replace("_", " ")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = AcademicPDF(title=extract_document_title(input_path))
    render_markdown(pdf, input_path)
    pdf.output(str(output_path))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
