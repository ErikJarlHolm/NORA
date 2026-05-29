"""
PDF Report Generator – lag PDF-rapporter med tekst, tabeller og bilder.

Tilgjengelig som verktøy for NORA-agenten slik at hun kan generere
profesjonelle PDF-rapporter og lagre dem lokalt.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fpdf import FPDF

log = logging.getLogger(__name__)

# Standard output folder
DEFAULT_OUTPUT_FOLDER = Path(
    r"C:\Users\erikholm\OneDrive - Atea\Documents\Kunder\Atea AI Norge\Agent tallknusing"
)

# Windows system fonts for Unicode support (Norwegian characters etc.)
_FONT_DIR = Path(r"C:\Windows\Fonts")
_FONT_REGULAR = str(_FONT_DIR / "arial.ttf")
_FONT_BOLD = str(_FONT_DIR / "arialbd.ttf")
_FONT_ITALIC = str(_FONT_DIR / "ariali.ttf")


class NoraPDF(FPDF):
    """Custom PDF class with NORA branding and full Unicode support."""

    def __init__(self):
        super().__init__()
        self.add_font("NFont", "", _FONT_REGULAR)
        self.add_font("NFont", "B", _FONT_BOLD)
        self.add_font("NFont", "I", _FONT_ITALIC)

    def header(self):
        self.set_font("NFont", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "NORA \u2013 Numerical Operations & Results Assistant", align="R")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("NFont", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Side {self.page_no()}/{{nb}}", align="C")


def create_pdf_report(
    title: str,
    content: str,
    filename: str = "rapport.pdf",
    output_folder: Optional[str] = None,
    images: Optional[list[str]] = None,
    table_data: Optional[list[list[str]]] = None,
    table_headers: Optional[list[str]] = None,
) -> str:
    """
    Lag en PDF-rapport og lagre lokalt.

    Args:
        title: Hovedtittel på rapporten
        content: Brødtekst (støtter avsnitt separert med dobbel linjeskift)
        filename: Filnavn for PDF-en
        output_folder: Mappe å lagre i (standard: OneDrive-mappen)
        images: Liste med filstier til PNG-bilder som skal inkluderes
        table_data: Tabellrader (liste av lister med strengverdier)
        table_headers: Kolonneoverskrifter for tabellen

    Returns:
        Full filsti til den lagrede PDF-filen.
    """
    folder = Path(output_folder) if output_folder else DEFAULT_OUTPUT_FOLDER
    folder.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".pdf"):
        filename += ".pdf"

    filepath = folder / filename

    pdf = NoraPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("NFont", "B", 20)
    pdf.set_text_color(25, 25, 112)
    pdf.multi_cell(0, 12, title, align="C")
    pdf.ln(8)

    # Horizontal line
    pdf.set_draw_color(25, 25, 112)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)

    # Content paragraphs
    pdf.set_font("NFont", "", 11)
    pdf.set_text_color(30, 30, 30)

    paragraphs = content.split("\n\n")
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check if paragraph is a heading (starts with ## or ###)
        if para.startswith("### "):
            pdf.set_font("NFont", "B", 12)
            pdf.set_text_color(50, 50, 50)
            pdf.ln(4)
            pdf.multi_cell(0, 7, para[4:])
            pdf.ln(2)
            pdf.set_font("NFont", "", 11)
            pdf.set_text_color(30, 30, 30)
        elif para.startswith("## "):
            pdf.set_font("NFont", "B", 14)
            pdf.set_text_color(25, 25, 112)
            pdf.ln(6)
            pdf.multi_cell(0, 8, para[3:])
            pdf.ln(3)
            pdf.set_font("NFont", "", 11)
            pdf.set_text_color(30, 30, 30)
        else:
            pdf.multi_cell(0, 6, para)
            pdf.ln(4)

    # Table
    if table_data:
        pdf.ln(6)
        _add_table(pdf, table_headers or [], table_data)

    # Images
    if images:
        for img_path in images:
            img = Path(img_path)
            if img.exists() and img.suffix.lower() in (".png", ".jpg", ".jpeg"):
                pdf.ln(8)
                available_width = pdf.w - pdf.l_margin - pdf.r_margin
                try:
                    pdf.image(str(img), x=pdf.l_margin, w=available_width)
                except Exception as e:
                    log.warning("Kunne ikke legge til bilde %s: %s", img_path, e)
                    pdf.set_font("NFont", "I", 9)
                    pdf.cell(0, 8, f"[Bilde ikke tilgjengelig: {img.name}]")
                    pdf.set_font("NFont", "", 11)

    pdf.output(str(filepath))
    log.info("PDF lagret: %s", filepath)
    return str(filepath)


def _add_table(pdf: FPDF, headers: list[str], rows: list[list[str]]) -> None:
    """Add a formatted table to the PDF."""
    if not rows:
        return

    num_cols = len(headers) if headers else len(rows[0])
    available_width = pdf.w - pdf.l_margin - pdf.r_margin
    col_width = available_width / num_cols
    row_height = 7

    # Header row
    if headers:
        pdf.set_font("NFont", "B", 10)
        pdf.set_fill_color(25, 25, 112)
        pdf.set_text_color(255, 255, 255)
        for header in headers:
            pdf.cell(col_width, row_height, str(header), border=1, align="C", fill=True)
        pdf.ln(row_height)

    # Data rows
    pdf.set_font("NFont", "", 9)
    pdf.set_text_color(30, 30, 30)
    for i, row in enumerate(rows):
        if i % 2 == 0:
            pdf.set_fill_color(240, 240, 250)
        else:
            pdf.set_fill_color(255, 255, 255)
        for cell_value in row:
            pdf.cell(col_width, row_height, str(cell_value), border=1, align="C", fill=True)
        pdf.ln(row_height)
