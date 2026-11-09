"""
Export a slide (a 1920x1080 <section> as written for the Slides artifact) into the
repo as a standalone folder that opens in any browser, plus PNG, PDF and notes.

    python scripts/export_slide.py <slide.html> <out_dir> <blob_id=image.png> ...

Each `/_blob/<id>` image in the slide is replaced by the local image given for
that id, copied to <out_dir>/img/. Rendering uses Microsoft Edge (or Chrome) in
headless mode.
"""
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
FONTS = ("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700"
         "&family=IBM+Plex+Sans:wght@400;600&display=swap")

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1920">
<title>{title}</title>
<link rel="stylesheet" href="{fonts}">
<style>
  @page {{ size: 1920px 1080px; margin: 0 }}
  html, body {{ margin: 0; padding: 0; background: #11161C }}
  section {{ width: 1920px; height: 1080px; box-sizing: border-box; position: relative;
             overflow: hidden }}
  h1, h2, h3, p, ul {{ margin: 0 }}
  ul {{ padding-left: 36px }}
  aside {{ display: none }}
</style>
</head>
<body>
{section}
</body>
</html>
"""


def main(slide, out_dir, *maps):
    slide, out = Path(slide), Path(out_dir)
    (out / "img").mkdir(parents=True, exist_ok=True)
    html = slide.read_text(encoding="utf-8")
    for m in maps:
        blob, src = m.split("=", 1)
        name = Path(src).name
        shutil.copy(src, out / "img" / name)
        html = html.replace(f"/_blob/{blob}", f"img/{name}")
    left = re.findall(r"/_blob/[0-9a-f]+", html)
    if left:
        raise SystemExit(f"no local image given for: {sorted(set(left))}")

    notes = re.search(r"<aside>(.*?)</aside>", html, re.S)
    title = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S)
    title = re.sub(r"<[^>]+>", "", title.group(1)).strip() if title else slide.stem
    if notes:
        (out / "speaker-notes.md").write_text(
            f"# Speaker notes: {title}\n\n{notes.group(1).strip()}\n", encoding="utf-8")
    page = out / "slide.html"
    page.write_text(PAGE.format(title=title, fonts=FONTS, section=html.strip()),
                    encoding="utf-8")

    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if not browser:
        print("No Edge/Chrome found: wrote slide.html only.")
        return
    url = page.resolve().as_uri()
    written = []
    # A throwaway profile, so the render does not hand off to an open browser window.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        common = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                  f"--user-data-dir={profile}", "--no-first-run",
                  "--virtual-time-budget=8000", "--window-size=1920,1080"]
        for name, flags in (("slide.png", [f"--screenshot={(out / 'slide.png').resolve()}"]),
                            ("slide.pdf", ["--no-pdf-header-footer",
                                           f"--print-to-pdf={(out / 'slide.pdf').resolve()}"])):
            (out / name).unlink(missing_ok=True)
            subprocess.run(common + flags + [url], capture_output=True, timeout=120)
            for _ in range(40):                  # the browser can finish writing after it returns
                if (out / name).exists() and (out / name).stat().st_size > 0:
                    written.append(name)
                    break
                time.sleep(0.5)
    missing = {"slide.png", "slide.pdf"} - set(written)
    print(f"wrote {out}: slide.html, {', '.join(written)}"
          + (", speaker-notes.md" if notes else "")
          + (f"  (FAILED: {', '.join(sorted(missing))})" if missing else ""))


if __name__ == "__main__":
    main(*sys.argv[1:])
