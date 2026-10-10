#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REPL driver v2 for WeChat MP publish. One command per line:

  shot <name>          screenshot active page -> _publish_shots/<name>.png
  pages                list tabs
  switch <i>           switch active tab (brings to front)
  new <url>            open url in new tab and switch to it
  goto <url>
  click <text>         click exact text
  pclick <text>        click partial text
  css <sel>            click css selector
  eval <js>            evaluate JS on active page, print result
  fill <sel> | <text>
  key <key>
  sleep <seconds>
  url
  quit
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[4]
PROFILE = ROOT / "docs-build" / "edge-profile"
OUT = Path(__file__).parent / "_publish_shots"
OUT.mkdir(exist_ok=True)

lock = PROFILE / "lockfile"
if lock.exists():
    lock.unlink()

pw = sync_playwright().start()
ctx = pw.chromium.launch_persistent_context(
    user_data_dir=str(PROFILE),
    channel="msedge",
    headless=False,
    args=["--start-maximized"],
    no_viewport=True,
)
page = ctx.pages[0] if ctx.pages else ctx.new_page()
page.goto("https://mp.weixin.qq.com/", wait_until="domcontentloaded", timeout=60000)
time.sleep(3)
print("[open]", page.url, flush=True)
print("[repl-ready]", flush=True)

for line in sys.stdin:
    cmd = line.strip()
    if not cmd:
        continue
    try:
        verb, _, rest = cmd.partition(" ")
        if verb == "quit":
            break
        elif verb == "pages":
            for i, p in enumerate(ctx.pages):
                mark = "*" if p == page else " "
                print(f"[page]{mark}{i}: {p.url[:120]}", flush=True)
        elif verb == "switch":
            page = ctx.pages[int(rest)]
            page.bring_to_front()
            time.sleep(0.5)
            print("[switched]", int(rest), page.url[:120], flush=True)
        elif verb == "new":
            page = ctx.new_page()
            page.goto(rest, wait_until="domcontentloaded", timeout=60000)
            time.sleep(2)
            print("[new]", page.url, flush=True)
        elif verb == "shot":
            page.screenshot(path=str(OUT / f"{rest}.png"), timeout=15000)
            print("[shot]", rest, flush=True)
        elif verb == "goto":
            page.goto(rest, wait_until="domcontentloaded", timeout=60000)
            time.sleep(2)
            print("[goto]", page.url, flush=True)
        elif verb == "click":
            page.click(f'text="{rest}"', timeout=10000)
            time.sleep(1)
            print("[clicked]", rest, flush=True)
        elif verb == "pclick":
            page.click(f"text={rest}", timeout=10000)
            time.sleep(1)
            print("[pclicked]", rest, flush=True)
        elif verb == "css":
            page.click(rest, timeout=10000)
            time.sleep(1)
            print("[css-clicked]", rest, flush=True)
        elif verb == "eval":
            r = page.evaluate(rest)
            print("[eval]", repr(r)[:3000], flush=True)
        elif verb == "fill":
            sel, _, txt = rest.partition(" | ")
            page.fill(sel, txt, timeout=10000)
            print("[filled]", sel, flush=True)
        elif verb == "key":
            page.keyboard.press(rest)
            time.sleep(0.5)
            print("[key]", rest, flush=True)
        elif verb == "sleep":
            time.sleep(float(rest))
            print("[slept]", rest, flush=True)
        elif verb == "url":
            print("[url]", page.url, flush=True)
        else:
            print("[?] unknown:", verb, flush=True)
    except Exception as e:
        print("[err]", str(e)[:800].replace("\n", " | "), flush=True)

ctx.close()
pw.stop()
print("[bye]", flush=True)
