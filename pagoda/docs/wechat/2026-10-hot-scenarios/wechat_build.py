#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert article.md -> WeChat-ready HTML (all styles inline, 677px column).

Usage:  python wechat_build.py
Output: article-wechat.html  (title line H1 is extracted, not repeated in body)
"""
import re
from pathlib import Path

import markdown

HERE = Path(__file__).parent
MD = (HERE / "article.md").read_text(encoding="utf-8")

fm = re.match(r"---\n([\s\S]*?)\n---\n", MD)
DIGEST = ""
if fm:
    meta = dict(re.findall(r'(\w+):\s*"?([^"\n]+?)"?\n', fm.group(1) + "\n"))
    DIGEST = meta.get("digest", "")
    MD = MD[fm.end():]

m = re.match(r"# (.+?)\n", MD)
TITLE = (m.group(1).strip() if m else "") or meta.get("title", "")
BODY = MD[m.end():] if m else MD

html = markdown.markdown(BODY, extensions=["tables", "fenced_code"])

INK = "#2B3452"; PRIMARY = "#4F6BD8"; ACCENT = "#C98A1B"; GRAY = "#8A8F98"

STYLES = {
    "h2": (f"font-size:18px;font-weight:700;color:{INK};margin:34px 0 6px;"
           f"padding-left:10px;border-left:4px solid {PRIMARY};line-height:1.4;"),
    "p": ("font-size:15.5px;color:#3f3f3f;line-height:1.85;letter-spacing:0.3px;"
          "margin:14px 0;text-align:justify;"),
    "strong": f"color:{INK};font-weight:700;",
    "em": "color:#5a6272;",
    "blockquote": (f"background:#F6F7FB;border-left:3px solid {ACCENT};margin:16px 0;"
                   "padding:12px 14px;border-radius:0 8px 8px 0;"),
    "code": ("background:#F3F4F8;color:#C2185B;padding:2px 5px;border-radius:4px;"
             "font-family:Consolas,Menlo,monospace;font-size:13.5px;"),
    "pre": ("background:#282C34;border-radius:10px;padding:16px 14px;margin:18px 0;"
            "overflow-x:auto;line-height:1.65;"),
    "table": ("width:100%;border-collapse:collapse;margin:18px 0;font-size:13.5px;"),
    "th": (f"background:#EEF1F8;color:{INK};font-weight:700;padding:9px 8px;"
           "border:1px solid #E2E6EF;text-align:left;"),
    "td": "padding:9px 8px;border:1px solid #E9ECF2;color:#3f3f3f;",
    "ul": "padding-left:1.5em;margin:12px 0;",
    "ol": "padding-left:1.5em;margin:12px 0;",
    "li": ("font-size:15.5px;color:#3f3f3f;line-height:1.8;letter-spacing:0.3px;"
           "margin:6px 0;"),
    "a": f"color:{PRIMARY};text-decoration:none;",
    "hr": "border:none;border-top:1px solid #E5E7EB;margin:28px 0;",
}

IMG_STYLE = "width:100%;border-radius:10px;margin:10px 0 2px;display:block;"
def cap(mobj):
    alt, src = mobj.group(1), mobj.group(2)
    img = f'<img alt="{alt}" src="{src}" style="{IMG_STYLE}"/>'
    if alt == "题图":
        return img
    return (img +
            f'<p style="font-size:12.5px;color:{GRAY};text-align:center;margin:2px 0 18px;'
            f'letter-spacing:0.3px;">{alt}</p>')
html = re.sub(r'<img alt="([^"]*)" src="([^"]+)"[^>]*/?>', cap, html)

for tag, style in STYLES.items():
    html = re.sub(rf"<{tag}(?![a-z])", f'<{tag} style="{style}"', html)

# code inside pre must not inherit the inline-code pink style
def fix_pre(mobj):
    seg = mobj.group(0)
    seg = re.sub(r'<code[^>]*>',
                 '<code style="background:transparent;color:#ABB2BF;padding:0;'
                 'font-family:Consolas,Menlo,monospace;font-size:13px;">', seg)
    return seg
html = re.sub(r'<pre[\s\S]*?</pre>', fix_pre, html)
# blockquote paragraphs: softer color
html = re.sub(r'(<blockquote[^>]*>)\s*<p style="[^"]*">',
              r'\1<p style="font-size:14.5px;color:#6A7382;line-height:1.75;margin:0;">', html)

page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>preview</title></head><body>
<section style="max-width:677px;margin:0 auto;padding:6px 4px;
font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue','PingFang SC',
'Hiragino Sans GB','Microsoft YaHei',sans-serif;">
{html}
</section></body></html>"""

(HERE / "article-wechat.html").write_text(page, encoding="utf-8")
(HERE / "meta.json").write_text(
    '{\n  "title": ' + __import__("json").dumps(TITLE, ensure_ascii=False) + ',\n'
    '  "digest": ' + __import__("json").dumps(DIGEST, ensure_ascii=False) + '\n}\n',
    encoding="utf-8")
print("title:", TITLE)
print("->", HERE / "article-wechat.html")
