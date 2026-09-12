import requests
import sys

print("START", flush=True)

url = "https://api-infotecnica.coordinador.cl/v1/subestaciones/199/documentos/5101/"

print("URL:", url, flush=True)
print("Sending request...", flush=True)

try:
    r = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=(10, 20)
    )

    print("REQUEST FINISHED", flush=True)
    print("STATUS:", r.status_code, flush=True)
    print("CONTENT TYPE:", r.headers.get("content-type"), flush=True)
    print("LENGTH:", len(r.content), flush=True)
    print("BODY:", r.text[:2000], flush=True)

except requests.exceptions.Timeout:
    print("REQUEST TIMED OUT", flush=True)

except requests.exceptions.RequestException as e:
    print("REQUEST ERROR:", repr(e), flush=True)

except Exception as e:
    print("OTHER ERROR:", repr(e), flush=True)

print("END", flush=True)