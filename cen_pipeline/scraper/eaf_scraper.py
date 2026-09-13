"""
Scrapes the CEN EAF (Estudios de Analisis de Falla) archive: one entry per
real fault event, each with a title like

    EAF 570/2025: Falla en linea 110 kV Cardones - Planta Matta; 31-12-2025, 13:23 horas

and a "Descargar PDF" link to the full report. This module lists all entries
for a given year and downloads + lightly cleans the report text.
"""
import os
import re
import time
import requests
from bs4 import BeautifulSoup

TITLE_RE = re.compile(
    r"EAF\s+(\d+)\s*/\s*(\d{4})\s*:\s*(.+?)\s*;\s*(\d{2}-\d{2}-\d{4})\s*,\s*(\d{2}:\d{2})\s*horas",
    re.IGNORECASE | re.DOTALL,
)

USER_AGENT = "Mozilla/5.0 (academic research script; multimodal fault diagnosis project)"


def new_session(bypass_cloudflare=True, headless=False):
    """Returns a requests.Session ready to call www.coordinador.cl.

    www.coordinador.cl sits behind a Cloudflare Turnstile challenge (see
    scraper/cf_bypass.py) -- plain requests get a 403 "Just a moment..."
    page. By default this solves that challenge once via a real Chrome
    session and returns a plain requests.Session loaded with the resulting
    cookies, which is enough for subsequent plain requests.get() calls.
    Set bypass_cloudflare=False to get the old plain-requests behaviour
    (useful for calling other, non-Cloudflare-protected domains like the
    InfoTecnica API).
    """
    if bypass_cloudflare:
        try:
            import config
            from scraper.cf_bypass import get_cloudflare_cleared_session
            return get_cloudflare_cleared_session(
                "https://www.coordinador.cl/", headless=headless,
                version_main=getattr(config, "CHROME_MAJOR_VERSION", None),
            )
        except Exception as e:
            print(f"[eaf_scraper] Cloudflare bypass unavailable/failed ({e}); "
                  f"falling back to a plain session, which will likely 403 "
                  f"against www.coordinador.cl. Run: pip install "
                  f"undetected-chromedriver selenium")
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


_session = new_session  # backwards-compat alias


def _get_with_cf_retry(session, url, timeout=30):
    """GET that, on a Cloudflare 403, re-solves the challenge once (mutating
    `session`'s cookies/headers in place) and retries -- cf_clearance
    cookies are time-limited, so a long run can hit this mid-way through."""
    resp = session.get(url, timeout=timeout)
    if resp.status_code == 403:
        print(f"[eaf_scraper] got 403 on {url} -- re-solving Cloudflare challenge ...")
        import config
        from scraper.cf_bypass import get_cloudflare_cleared_session
        fresh = get_cloudflare_cleared_session(
            "https://www.coordinador.cl/",
            version_main=getattr(config, "CHROME_MAJOR_VERSION", None),
        )
        session.cookies.update(fresh.cookies)
        session.headers.update(fresh.headers)
        resp = session.get(url, timeout=timeout)
    return resp


def list_eaf_entries(year, listing_url_template, max_pages=40, delay=1.0, session=None):
    """Returns a list of dicts: eaf_id, eaf_num, year, description, date, time, pdf_url."""
    session = session or new_session()
    base_url = listing_url_template.format(year=year)
    entries, seen_urls = [], set()
    page = 1
    while page <= max_pages:
        url = base_url if page == 1 else f"{base_url}/?page={page}"
        resp = _get_with_cf_retry(session, url, timeout=30)
        if resp.status_code != 200:
            break
        soup = BeautifulSoup(resp.text, "html.parser")

        pdf_links = [
            a for a in soup.find_all("a", href=True)
            if "descargar pdf" in a.get_text(strip=True).lower()
        ]
        if not pdf_links:
            break

        found_this_page = 0
        for a in pdf_links:
            pdf_url = a["href"]
            if pdf_url in seen_urls:
                continue
            # Walk up the DOM looking for the block containing the EAF title text
            title_text, node = "", a
            for _ in range(5):
                node = node.find_parent()
                if node is None:
                    break
                text = node.get_text(" ", strip=True)
                if TITLE_RE.search(text):
                    title_text = text
                    break
            m = TITLE_RE.search(title_text)
            if not m:
                continue
            eaf_num, eaf_year, description, date_str, time_str = m.groups()
            entries.append({
                "eaf_id": f"EAF-{int(eaf_num):04d}-{eaf_year}",
                "eaf_num": int(eaf_num),
                "year": int(eaf_year),
                "description": description.strip(),
                "date": date_str,
                "time": time_str,
                "pdf_url": pdf_url,
            })
            seen_urls.add(pdf_url)
            found_this_page += 1

        if found_this_page == 0:
            break
        page += 1
        time.sleep(delay)
    return entries


def download_pdf(url, dest_path, session=None, timeout=60):
    session = session or new_session()
    r = _get_with_cf_retry(session, url, timeout=timeout)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)
    return dest_path


def extract_report_text(pdf_path, max_pages=6):
    """
    Pulls text from the first `max_pages` pages and stops once it hits the
    raw SCADA/protection-relay annexes (ANEXO ...), which are log dumps
    rather than narrative text and would just bloat + confuse the text
    modality / translation step.
    """
    import pdfplumber
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            t = page.extract_text() or ""
            parts.append(t)
            if i > 0 and re.search(r"\bANEXO\b", t.upper()):
                break
    return "\n".join(parts).strip()