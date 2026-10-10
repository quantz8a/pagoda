#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assist publishing the hot-scenarios article to WeChat MP.

Reuses the logged-in Edge profile (docs-build/edge-profile), opens the MP
backend, checks login, opens the draft box, and screenshots each step.
The final publish click + admin QR scan stays with the human.

Run: python mp_publish_assist.py
"""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

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

print("[step] open mp.weixin.qq.com ...", flush=True)
page.goto("https://mp.weixin.qq.com/", wait_until="domcontentloaded", timeout=60000)
time.sleep(4)
page.screenshot(path=str(OUT / "01-home.png"))
print("[url]", page.url, flush=True)

if "loginpage" in page.url:
    print("[state] NOT logged in - login page shown, need QR scan", flush=True)
else:
    print("[state] logged in", flush=True)
    try:
        page.click("text=草稿箱", timeout=8000)
    except Exception as e:
        print("[warn] draftbox nav click failed:", e, flush=True)
    time.sleep(4)
    page.screenshot(path=str(OUT / "02-draftbox.png"))
    print("[url]", page.url, flush=True)

print("[ready] browser left open. shots in", OUT, flush=True)
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    pass
ctx.close()
pw.stop()
