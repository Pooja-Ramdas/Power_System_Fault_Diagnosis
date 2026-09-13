"""
Standalone diagnostic -- figures out why eaf_scraper.list_eaf_entries()
found 0 entries. Doesn't touch any other file.

Run:  python debug_eaf_page.py
"""
import config
import scraper.eaf_scraper as eaf_scraper
from scraper.eaf_scraper import TITLE_RE, new_session

print("=== DEBUG SCRIPT VERSION: cf-bypass-v2 ===")
print(f"eaf_scraper module loaded from: {eaf_scraper.__file__}")
print(f"new_session function: {new_session}")
print(f"new_session accepts bypass_cloudflare arg: "
      f"{'bypass_cloudflare' in new_session.__code__.co_varnames}")
print()

url = config.EAF_LISTING_URL_TEMPLATE.format(year=2025)
print(f"Fetching: {url}\n")

s = new_session()  # now routes through the Cloudflare bypass
r = s.get(url, timeout=30)

print(f"status_code: {r.status_code}")
print(f"final url (after redirects): {r.url}")
print(f"content-length: {len(r.text)} chars")
print(f"'Descargar PDF' literally in raw HTML: {'descargar pdf' in r.text.lower()}")
print(f"'EAF ' literally in raw HTML: {'eaf ' in r.text.lower()}")
print(f"looks like a JS shell (has <div id=\"root\"> or similar with little else): "
      f"{'id=\"root\"' in r.text or 'id=\"app\"' in r.text}")

# does the TITLE_RE pattern match anywhere in the raw text at all?
m = TITLE_RE.search(r.text)
print(f"TITLE_RE finds a match in raw HTML: {bool(m)}")
if m:
    print("  matched:", m.group(0)[:150])

print("\n--- first 1500 characters of the raw response ---\n")
print(r.text[:1500])

print("\n--- trying a couple of alternate pagination URL shapes for page 2 ---")
for alt in [
    url.rstrip("/") + "/?page=2",
    url.rstrip("/") + "/page/2/",
    url.rstrip("/") + "?page=2",
]:
    r2 = s.get(alt, timeout=30)
    same_as_page1 = (r2.text == r.text)
    print(f"  {alt}\n    status={r2.status_code} same_content_as_page1={same_as_page1} "
          f"len={len(r2.text)}")