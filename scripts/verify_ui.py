"""Playwright UI audit for the Phase 3 web UI.

Usage:
    pip install playwright
    playwright install chromium
    python scripts/verify_ui.py [base_url]     # default http://127.0.0.1:8000

Checks (headless Chromium):
  - no console errors / page errors on home + project pages (favicon included)
  - home: tabs switch, client-side validation errors render, project list renders
  - failed project: tracker marks the FAILED stage red (not rendering/ready)
  - ready project: clip cards render with <video>, score badge, download links
  - hash routing: deep link works, back-link returns home

Exit code 0 = all checks passed. Requires the server to be running.
"""
from __future__ import annotations

import sys
import time

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

failures: list[str] = []
passed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed
    if ok:
        passed += 1
        print(f"  ok  {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"FAIL  {name}  {detail}")


def api(path: str):
    import json
    import urllib.request

    with urllib.request.urlopen(BASE + path, timeout=10) as resp:
        return json.loads(resp.read())


def wait_for_project(kind: str, timeout_s: float = 120) -> dict | None:
    """Find a project in the wanted terminal state, else return None."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for p in api("/projects"):
            if p["status"] == kind:
                return p
        time.sleep(2)
    return None


def audit(page) -> None:
    errors: list[str] = []
    page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
            if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

    # ------------------------------------------------------------ home page
    print("home page:")
    page.goto(f"{BASE}/#/")
    page.wait_for_selector("#create-form")
    check("title", page.title() == "clipper", page.title())
    check("form present", page.locator("#create-form").count() == 1)
    check("upload tab hidden initially",
          not page.locator("#f-file").is_visible())

    page.get_by_role("button", name="Upload file").click()
    check("upload tab switches", page.locator("#f-file").is_visible()
          and not page.locator("#f-url").is_visible())
    page.get_by_role("button", name="Link or path").click()

    # client-side validation error renders inline
    page.fill("#f-clips", "99")
    page.fill("#f-url", "https://example.com/v.mp4")
    page.get_by_role("button", name="Create clips").click()
    page.wait_for_selector("#form-error:not(.hidden)")
    check("clips range validation shows",
          "between 1 and 20" in page.locator("#form-error").text_content())

    # empty url validation
    page.fill("#f-clips", "5")
    page.fill("#f-url", "")
    page.get_by_role("button", name="Create clips").click()
    check("empty url validation shows",
          "Enter a video" in page.locator("#form-error").text_content())

    # list renders or shows the empty state
    page.wait_for_selector("#proj-list")
    page.wait_for_function(
        "document.querySelector('#proj-list').children.length > 0", timeout=15000)
    cards = page.locator("#proj-list .proj-card").count()
    empty = page.locator("#proj-list .empty").count()
    check("project list or empty state renders", cards > 0 or empty > 0,
          f"cards={cards} empty={empty}")

    # ------------------------------------------------------------ failed project
    failed = wait_for_project("failed", timeout_s=5)
    if failed:
        print(f"failed project page ({failed['title']!r}):")
        page.goto(f"{BASE}/#/project/{failed['id']}")
        page.wait_for_selector(".tracker-step")
        page.wait_for_timeout(300)  # allow poll to settle
        red = page.locator(".tracker-step.failed .tracker-label").text_content()
        check("failed stage is the worker's stage",
              red == (failed.get("stage") or "queued"),
              f"red={red!r} stage={failed.get('stage')!r}")
        check("failed stage is not 'ready'",
              red != "ready", f"red={red!r}")
        check("retry button visible",
              page.get_by_role("button", name="Retry project").is_visible())
        check("error box shows server detail",
              "yt-dlp" in (page.locator(".proj-error-box").text_content() or "")
              or (failed.get("error") or "") in (page.locator(".proj-error-box").text_content() or ""))
    else:
        print("failed project: none present, skipped")

    # ------------------------------------------------------------ ready project
    print("ready project page:")
    ready = wait_for_project("ready", timeout_s=5)
    if ready and ready.get("clips"):
        page.goto(f"{BASE}/#/project/{ready['id']}")
        page.wait_for_selector(".clip-card")
        n = len(ready["clips"])
        check("clip cards render", page.locator(".clip-card").count() == n,
              f"expected {n}")
        check("videos use stream_url",
              all(ready["clips"][0]["stream_url"] in
                  (page.locator(".clip-video").first.get_attribute("src") or "")
                  for _ in [0]))
        check("score badges render",
              page.locator(".score-badge").count() >= 1)
        check("download links point at download_url",
              page.locator(".btn-dl").first.get_attribute("href")
              == ready["clips"][0]["download_url"])
        check("download-all button present",
              page.get_by_role("button", name="Download all").is_visible())
        # first video has metadata (Range streaming works in-browser)
        page.wait_for_function(
            "document.querySelector('.clip-video').readyState >= 1", timeout=15000)
        check("first video loads metadata (Range ok)", True)
    else:
        print("ready project with clips: none present, skipped")

    # ------------------------------------------------------------ routing
    print("routing:")
    page.goto(f"{BASE}/#/project/nonexistent0000000000000000000000")
    page.wait_for_selector(".empty")
    check("unknown project shows not-found state",
          "not found" in page.locator(".empty").text_content().lower())
    page.goto(f"{BASE}/#/nope")
    page.wait_for_selector(".empty")
    check("unknown route shows not-found state", True)
    page.goto(f"{BASE}/#/project/{ready['id'] if ready else ''}")
    page.get_by_role("link", name="← Back to projects").click()
    page.wait_for_selector("#create-form")
    check("back link returns home", "#/" in page.url or page.url == f"{BASE}/")

    # ------------------------------------------------------------ console
    print("console:")
    check("no console/page errors", not errors, "; ".join(errors[:5]))


def main() -> int:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            audit(page)
        finally:
            browser.close()
    print(f"\n{passed} passed, {len(failures)} failed")
    for f in failures:
        print(f"  FAIL {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
