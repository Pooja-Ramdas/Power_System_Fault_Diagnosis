"""
Converts a downloaded PDF (official CEN single-line diagram) into a PNG
usable as the image modality input. Deliberately simple -- just renders the
first page at a decent resolution. If a diagram spans multiple pages,
render_pdf_page() lets you pick a different page number.
"""


def render_pdf_page(pdf_path, png_path, page_number=0, dpi=200):
    """Uses pdfplumber (already a project dependency) to rasterize one PDF
    page to PNG -- no poppler/ImageMagick system dependency required."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        if page_number >= len(pdf.pages):
            page_number = 0
        page = pdf.pages[page_number]
        im = page.to_image(resolution=dpi)
        im.save(png_path)
    return png_path
