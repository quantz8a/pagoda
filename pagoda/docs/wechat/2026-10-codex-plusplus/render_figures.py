#!/usr/bin/env python3
"""Short WeChat figures: why pagoda, then how Codex++ plugs in."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "images"
FONT_PATH = (
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/"
    "86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc"
)
SC = {"regular": 3, "medium": 7, "semibold": 11, "light": 15}

NAVY = (23, 36, 64)
GOLD = (232, 168, 74)
INK = (28, 39, 54)
MUTED = (100, 112, 128)
BLUE = (47, 102, 214)
GREEN = (32, 148, 104)
GRAY_BAR = (186, 194, 204)
CREAM = (246, 247, 250)
LINE = (226, 230, 236)
WHITE = (255, 255, 255)


def font(size, weight="regular"):
    return ImageFont.truetype(FONT_PATH, size, index=SC[weight])


def rr(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw, xy, s, f, fill, anchor="lt"):
    draw.text(xy, s, font=f, fill=fill, anchor=anchor)


def save(im, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    im.save(path, "PNG", optimize=True)
    print(path, im.size, path.stat().st_size)


def new(w, h, color=WHITE):
    im = Image.new("RGB", (w, h), color)
    return im, ImageDraw.Draw(im)


def bar(draw, x, y, w, h, value, vmax, fill, label, number):
    rr(draw, (x, y, x + w, y + h), 16, CREAM)
    bw = int((w - 28) * value / vmax)
    rr(draw, (x + 14, y + 46, x + 14 + max(bw, 24), y + h - 18), 10, fill)
    text(draw, (x + 22, y + 22), label, font(22, "medium"), MUTED)
    text(draw, (x + w - 22, y + 22), number, font(22, "semibold"), INK, "rt")


def hero():
    w, h = 1080, 640
    im, d = new(w, h, NAVY)
    text(d, (540, 78), "同一台 M3 Ultra · 同一份 Qwen3.8-27B 8-bit", font(24, "medium"), GOLD, "mm")
    text(d, (540, 168), "为什么用 pagoda", font(64, "semibold"), WHITE, "mm")
    cards = [
        ("23.5", "tok/s", "热解码", "mlx-serve 23.3"),
        ("50.9", "tok/s", "四个已知 token", "mlx-serve 22.4"),
        ("98%", "少做", "八个 Agent 的前向", "32,368 → 624"),
    ]
    cw, ch, gap = 300, 250, 24
    x0 = (w - (cw * 3 + gap * 2)) // 2
    y0 = 250
    for i, (big, unit, a, b) in enumerate(cards):
        x = x0 + i * (cw + gap)
        rr(d, (x, y0, x + cw, y0 + ch), 22, (32, 48, 82))
        text(d, (x + cw / 2, y0 + 78), big, font(64, "semibold"), GOLD, "mm")
        text(d, (x + cw / 2, y0 + 128), unit, font(22, "medium"), (214, 222, 234), "mm")
        text(d, (x + cw / 2, y0 + 176), a, font(22, "medium"), WHITE, "mm")
        text(d, (x + cw / 2, y0 + 210), b, font(18, "regular"), (170, 182, 200), "mm")
    text(d, (540, 580), "pagoda", font(22, "light"), (150, 162, 180), "mm")
    save(im, "fig0-hero.png")


def decode():
    w, h = 1080, 520
    im, d = new(w, h)
    text(d, (48, 40), "单流热解码", font(40, "semibold"), INK)
    text(d, (48, 96), "权重已在 GPU 上 · 贪心 · 2026-10-17", font(22, "regular"), MUTED)
    bar(d, 48, 170, 984, 120, 23.5, 28, BLUE, "pagoda", "23.5 tok/s")
    bar(d, 48, 320, 984, 120, 23.3, 28, GRAY_BAR, "mlx-serve", "23.3 tok/s")
    save(im, "fig1-decode.png")


def verify():
    w, h = 1080, 560
    im, d = new(w, h)
    text(d, (48, 40), "四个已知 token，一次算完", font(40, "semibold"), INK)
    text(d, (48, 96), "logit 绝对差是 0 · 2026-10-18", font(22, "regular"), MUTED)
    bar(d, 48, 180, 984, 130, 50.9, 56, GREEN, "pagoda 一次验证", "50.9 tok/s")
    bar(d, 48, 340, 984, 130, 22.4, 56, GRAY_BAR, "mlx-serve 逐条", "22.4 tok/s")
    text(d, (1032, 520), "约 2.28 倍", font(22, "semibold"), GREEN, "rm")
    save(im, "fig2-verify.png")


def share():
    w, h = 1080, 640
    im, d = new(w, h)
    text(d, (48, 40), "八个 Agent，一段系统提示", font(40, "semibold"), INK)
    text(d, (48, 96), "树干大约 3968 token · 各生成 32 token", font(22, "regular"), MUTED)
    # left pile
    rr(d, (48, 170, 500, 560), 24, CREAM)
    text(d, (274, 220), "各自重算", font(24, "medium"), MUTED, "mm")
    text(d, (274, 320), "32,368", font(64, "semibold"), INK, "mm")
    text(d, (274, 390), "次前向", font(24, "regular"), MUTED, "mm")
    for i in range(8):
        x = 88 + (i % 4) * 96
        y = 450 + (i // 4) * 36
        rr(d, (x, y, x + 80, y + 24), 6, GRAY_BAR)
    # right
    rr(d, (540, 170, 1032, 560), 24, (226, 244, 235))
    text(d, (786, 220), "树干只付一次", font(24, "medium"), GREEN, "mm")
    text(d, (786, 320), "624", font(72, "semibold"), GREEN, "mm")
    text(d, (786, 400), "次前向", font(24, "regular"), GREEN, "mm")
    rr(d, (620, 470, 952, 520), 12, GREEN)
    text(d, (786, 495), "少 98%", font(28, "semibold"), WHITE, "mm")
    save(im, "fig3-share.png")


def wire():
    w, h = 1080, 420
    im, d = new(w, h, NAVY)
    boxes = [
        (60, "Codex++"),
        (400, "pagoda"),
        (740, "Qwen3.8"),
    ]
    for x, name in boxes:
        rr(d, (x, 120, x + 280, 280), 24, (32, 48, 82))
        text(d, (x + 140, 200), name, font(36, "semibold"), WHITE, "mm")
    text(d, (360, 200), "→", font(48, "semibold"), GOLD, "mm")
    text(d, (700, 200), "→", font(48, "semibold"), GOLD, "mm")
    text(d, (540, 80), "Responses  ·  127.0.0.1:8000", font(26, "medium"), GOLD, "mm")
    text(d, (540, 340), "wire_api = responses", font(24, "regular"), (186, 198, 214), "mm")
    save(im, "fig4-wire.png")


if __name__ == "__main__":
    hero()
    decode()
    verify()
    share()
    wire()
