import os
import config


def render_pdf_page(pdf_path, png_path, page_number=0, max_long_edge=getattr(config, "IMAGE_MAX_LONG_EDGE_PX", 1200)):
    """Uses pdfplumber to rasterize one PDF page to PNG with resolution capped
    so that the long edge does not exceed `max_long_edge` pixels."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        if page_number >= len(pdf.pages):
            page_number = 0
        page = pdf.pages[page_number]
        max_dim_pts = max(page.width, page.height)
        if max_dim_pts > 0 and max_long_edge:
            dpi = int(min(200, max(72, (max_long_edge / max_dim_pts) * 72)))
        else:
            dpi = 150
        im = page.to_image(resolution=dpi)
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        im.save(png_path, format="PNG", optimize=True)
    return png_path

