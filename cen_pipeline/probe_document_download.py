import requests

BASE_API = "https://api-infotecnica.coordinador.cl/v1"
DOC_ID = 5101

s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0"
})

urls = [
    f"{BASE_API}/documentos/{DOC_ID}/",
    f"{BASE_API}/documentos/{DOC_ID}",
    f"{BASE_API}/documentos/{DOC_ID}/download/",
    f"{BASE_API}/documentos/{DOC_ID}/download",
    f"{BASE_API}/documentos/{DOC_ID}/archivo/",
    f"{BASE_API}/documentos/{DOC_ID}/archivo",
    f"{BASE_API}/documentos/{DOC_ID}/file/",
    f"{BASE_API}/documentos/{DOC_ID}/file",
    f"{BASE_API}/documentos/{DOC_ID}/documento/",
    f"{BASE_API}/documentos/{DOC_ID}/documento",
]

print("=== DOCUMENT DOWNLOAD PROBE ===")

for url in urls:
    try:
        r = s.get(url, timeout=30, allow_redirects=False)

        print("\nGET", url)
        print("status:", r.status_code)
        print("content-type:", r.headers.get("content-type"))
        print("location:", r.headers.get("location"))

        if "application/json" in r.headers.get("content-type", ""):
            try:
                print("JSON:")
                print(r.json())
            except Exception:
                print("Could not parse JSON")

    except Exception as e:
        print("ERROR:", type(e).__name__, e)

print("\n=== PROBE FINISHED ===")