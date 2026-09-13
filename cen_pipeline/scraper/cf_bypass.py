"""
www.coordinador.cl sits behind a Cloudflare Turnstile challenge (confirmed
via debug_eaf_page.py: HTTP 403, "Just a moment..." title, and
challenges.cloudflare.com in the response's CSP). Plain `requests` cannot
solve this on its own -- it requires a real browser's JS engine. This module
launches one real (undetected) Chrome session, waits for the challenge to
clear, harvests the resulting cookies (including cf_clearance) and the
browser's own User-Agent, and hands back a plain `requests.Session()`
pre-loaded with both -- which is enough for ordinary `requests.get()` calls
to keep working for the rest of the run, without needing a browser for
every single request.

Requires:
    pip install undetected-chromedriver selenium
and an actual Chrome/Chromium installed on this machine.

cf_clearance cookies are time-limited (commonly ~30 min-ish, site-configurable
and not guaranteed). For a long run across many cases, eaf_scraper.py calls
back into this module automatically if it ever sees a 403 mid-run.
"""
import time
import requests

try:
    from curl_cffi import requests as cf_requests
except ImportError:
    cf_requests = None


def get_cloudflare_cleared_session(challenge_url, user_agent_override=None,
                                    headless=False, wait_seconds=30, version_main=None):
    import undetected_chromedriver as uc

    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")

    print(f"[cf_bypass] launching Chrome to clear Cloudflare on {challenge_url} ..."
          + (f" (pinned to Chrome {version_main})" if version_main else ""))
    driver = uc.Chrome(options=options, version_main=version_main)
    try:
        driver.get(challenge_url)
        deadline = time.time() + wait_seconds
        cleared = False
        while time.time() < deadline:
            if "just a moment" not in driver.title.lower():
                cleared = True
                break
            time.sleep(1)
        if not cleared:
            raise TimeoutError(
                f"Cloudflare challenge did not clear within {wait_seconds}s. "
                f"If a visible Chrome window opened, check whether it's "
                f"showing a checkbox to click by hand (set headless=False so "
                f"you can see/solve it), then re-run. If it's still stuck, "
                f"CEN may have tightened their challenge -- tell me what the "
                f"Chrome window shows."
            )
        # small extra pause so the clearance cookie is fully set
        time.sleep(1.5)
        cookies = driver.get_cookies()
        ua = driver.execute_script("return navigator.userAgent;")
        print("[cf_bypass] challenge cleared.")
    finally:
        driver.quit()

    if cf_requests is not None:
        # Plain `requests` has a TLS/HTTP2 fingerprint that Cloudflare can
        # distinguish from a real browser even with a valid clearance
        # cookie -- curl_cffi impersonates Chrome's fingerprint so the
        # cookie actually gets honoured on replay.
        session = cf_requests.Session(impersonate="chrome124")
    else:
        print("[cf_bypass] curl_cffi not installed -- falling back to plain "
              "requests, which may still get 403'd by Cloudflare even with "
              "a valid cookie (TLS fingerprint mismatch). Run: "
              "pip install curl_cffi")
        session = requests.Session()
    session.headers.update({"User-Agent": user_agent_override or ua})
    for c in cookies:
        try:
            session.cookies.set(c["name"], c["value"], domain=c.get("domain"))
        except Exception:
            pass
    return session
