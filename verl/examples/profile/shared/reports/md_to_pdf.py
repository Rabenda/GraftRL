#!/usr/bin/env python3
"""Convert direction3 markdown (with relative images) to PDF via Playwright."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright


def md_to_html(md_text: str, base_dir: Path) -> str:
    def repl(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        if src.startswith(("http://", "https://", "file://")):
            return m.group(0)
        p = (base_dir / src).resolve()
        uri = p.as_uri() if p.exists() else src
        return f"![{alt}]({uri})"

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md_text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "nl2br"])
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <style>
    @page {{ margin: 18mm 14mm; }}
    body {{
      font-family: "Noto Sans CJK SC", "Noto Sans SC", "PingFang SC",
                   "Microsoft YaHei", "WenQuanYi Micro Hei", sans-serif;
      font-size: 11pt; line-height: 1.5; color: #111; max-width: 100%;
    }}
    h1 {{ font-size: 20pt; margin-top: 0; }}
    h2 {{ font-size: 15pt; margin-top: 1.4em; border-bottom: 1px solid #ccc; padding-bottom: 0.25em; }}
    h3 {{ font-size: 12.5pt; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0.6em 0; font-size: 9.5pt; page-break-inside: avoid; }}
    th, td {{ border: 1px solid #bbb; padding: 5px 7px; vertical-align: top; }}
    th {{ background: #f0f0f0; }}
    img {{ max-width: 100%; height: auto; }}
    td img {{ max-width: 200px; }}
    code {{ background: #f5f5f5; padding: 1px 4px; font-size: 9pt; }}
    blockquote {{ border-left: 3px solid #888; padding-left: 12px; color: #444; margin-left: 0; }}
    pre {{ background: #f5f5f5; padding: 10px; font-size: 8.5pt; white-space: pre-wrap; word-break: break-all; }}
    hr {{ border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }}
  </style>
</head>
<body>{body}</body>
</html>"""


def convert(md_path: Path, pdf_path: Path | None = None) -> Path:
    md_path = md_path.resolve()
    pdf_path = (pdf_path or md_path.with_suffix(".pdf")).resolve()
    base = md_path.parent
    html = md_to_html(md_path.read_text(encoding="utf-8"), base)
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(pdf_path),
            format="A4",
            margin={"top": "18mm", "bottom": "18mm", "left": "14mm", "right": "14mm"},
            print_background=True,
        )
        browser.close()
    return pdf_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("markdown", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()
    out = convert(args.markdown, args.output)
    print(out)


if __name__ == "__main__":
    main()
