#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a WeChat Official Account DRAFT from article-wechat.html.

Draft (草稿箱) is the review boundary: this script NEVER publishes.
After it runs, open mp.weixin.qq.com -> 草稿箱 (or the 订阅号助手 app),
review the rendered article, and press 发布 yourself.

Credentials come from env vars — never hardcode them:
  set WECHAT_APPID=wx...            (公众平台 -> 设置与开发 -> 基本配置)
  set WECHAT_APPSECRET=...
Also add this machine's public IP to 基本配置 -> IP白名单 first.
"""
import json, os, re, sys
from pathlib import Path

import requests

HERE = Path(__file__).parent
API = "https://api.weixin.qq.com"


def token(appid, secret):
    r = requests.get(API + "/cgi-bin/token", params=dict(
        grant_type="client_credential", appid=appid, secret=secret), timeout=20)
    d = r.json()
    if "access_token" not in d:
        sys.exit(f"[token] failed: {d}  (check AppID/AppSecret and IP whitelist)")
    return d["access_token"]


def upload_cover(tok, path):
    """Permanent material -> media_id for the draft cover."""
    r = requests.post(API + "/cgi-bin/material/add_material",
                      params=dict(access_token=tok, type="image"),
                      files={"media": (path.name, path.read_bytes(), "image/png")},
                      timeout=60)
    d = r.json()
    if "media_id" not in d:
        sys.exit(f"[cover] failed: {d}")
    return d["media_id"]


def upload_content_image(tok, path):
    """Inline image -> mmbiz URL usable inside article content."""
    r = requests.post(API + "/cgi-bin/media/uploadimg",
                      params=dict(access_token=tok),
                      files={"media": (path.name, path.read_bytes(), "image/png")},
                      timeout=60)
    d = r.json()
    if "url" not in d:
        sys.exit(f"[img {path.name}] failed: {d}")
    return d["url"]


def main():
    appid = os.environ.get("WECHAT_APPID")
    secret = os.environ.get("WECHAT_APPSECRET")
    if not appid or not secret:
        sys.exit("set WECHAT_APPID / WECHAT_APPSECRET first")

    html = (HERE / "article-wechat.html").read_text(encoding="utf-8")
    meta = json.loads((HERE / "meta.json").read_text(encoding="utf-8"))
    tok = token(appid, secret)
    print("[ok] access_token")

    cover = upload_cover(tok, HERE / "images" / "fig0-hero.png")
    print("[ok] cover media_id:", cover)

    def repl(m):
        src = m.group(1)
        if src.startswith("http"):
            return m.group(0)
        url = upload_content_image(tok, (HERE / src).resolve())
        print("[ok] content image:", src, "->", url[:60] + "...")
        return f'src="{url}"'
    html = re.sub(r'src="([^"]+)"', repl, html)

    r = requests.post(API + "/cgi-bin/draft/add",
                      params=dict(access_token=tok),
                      json={"articles": [{
                          "title": meta["title"],
                          "author": "pagoda",
                          "digest": meta["digest"],
                          "content": html,
                          "thumb_media_id": cover,
                          "need_open_comment": 1,
                          "only_fans_can_comment": 0,
                      }]}, timeout=60)
    d = r.json()
    if "media_id" not in d:
        sys.exit(f"[draft] failed: {d}")
    print("\n[done] draft created:", d["media_id"])
    print("review it: mp.weixin.qq.com -> 草稿箱  (or 订阅号助手 app)")
    print("publishing stays manual — press 发布 yourself after review.")


if __name__ == "__main__":
    main()
