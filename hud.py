"""Интерфейс поверх видео (на русском): светящийся мозг из 139k точек,
«эмоции», решения мозга, всплывающие события, подписи в 3D."""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = "C:/Windows/Fonts/bahnschrift.ttf"
F = {s: ImageFont.truetype(FONT, s) for s in (14, 16, 18, 20, 24, 30, 44, 64)}

GROUPS = {  # цвет группы нейронов
    "зрение": (90, 200, 255),
    "органы чувств": (255, 215, 90),
    "центральный мозг": (190, 160, 255),
    "команды телу (DN)": (255, 80, 150),
    "мотонейроны": (255, 150, 50),
}


def group_of(super_class):
    sc = str(super_class)
    if sc in ("optic", "visual_projection", "visual_centrifugal"):
        return "зрение"
    if sc in ("sensory", "sensory_ascending"):
        return "органы чувств"
    if sc == "descending":
        return "команды телу (DN)"
    if sc == "motor":
        return "мотонейроны"
    return "центральный мозг"


class BrainCloud:
    """Все нейроны мухи точками (вид спереди, координаты FlyWire).
    Активные нейроны вспыхивают и медленно гаснут."""
    def __init__(self, brain_ids, w=560, h=300, annot_path="data/neuron_annotations.tsv"):
        a = pd.read_csv(annot_path, sep="\t", usecols=["root_id", "pos_x", "pos_y", "super_class"],
                        low_memory=False).drop_duplicates("root_id").set_index("root_id")
        a = a.reindex(brain_ids)
        ok = a.pos_x.notna().values
        x, y = a.pos_x.values, a.pos_y.values
        x0, x1 = np.nanpercentile(x, [0.3, 99.7]); y0, y1 = np.nanpercentile(y, [0.3, 99.7])
        s = min((w - 20) / (x1 - x0), (h - 20) / (y1 - y0))
        px = ((x - x0) * s + (w - (x1 - x0) * s) / 2)
        py = ((y - y0) * s + (h - (y1 - y0) * s) / 2)
        self.ok = ok & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        self.px = np.nan_to_num(px).astype(int).clip(0, w - 1)
        self.py = np.nan_to_num(py).astype(int).clip(0, h - 1)
        self.w, self.h = w, h
        grp = [group_of(v) for v in a.super_class.values]
        self.col = np.array([GROUPS[g] for g in grp], np.float32) / 255
        # фон: плотность всех нейронов, тусклым цветом группы
        base = np.zeros((h, w, 3), np.float32)
        np.add.at(base, (self.py[self.ok], self.px[self.ok]), self.col[self.ok] * 0.05)
        self.base = np.clip(base ** 0.6 * 0.8, 0, 0.45)
        self.act = np.zeros(len(brain_ids), np.float32)

    def update(self, spikes_hz, decay=0.55):
        self.act = self.act * decay + np.clip(spikes_hz / 60.0, 0, 3)

    def render(self):
        img = self.base.copy()
        idx = np.nonzero((self.act > 0.05) & self.ok)[0]
        glow = np.zeros_like(img)
        np.add.at(glow, (self.py[idx], self.px[idx]), self.col[idx] * np.minimum(self.act[idx, None], 2.0))
        g = Image.fromarray((np.clip(glow, 0, 1) * 255).astype(np.uint8))
        halo = np.asarray(g.filter(ImageFilter.GaussianBlur(4)), np.float32) / 255 * 3.0
        img = np.clip(img + glow * 1.5 + halo, 0, 1)
        return Image.fromarray((img * 255).astype(np.uint8)), len(idx)


def panel(draw, box, r=14, fill=(12, 14, 20, 170)):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=(255, 255, 255, 40), width=1)


def gauge(draw, x, y, w, label, val, color, vmax=1.0, text=None):
    draw.text((x, y), label, font=F[16], fill=(225, 228, 235, 255))
    t = text if text is not None else f"{val:.0f}"
    tw = draw.textlength(t, font=F[16])
    draw.text((x + w - tw, y), t, font=F[16], fill=(255, 255, 255, 255))
    y += 22
    draw.rounded_rectangle((x, y, x + w, y + 10), radius=5, fill=(255, 255, 255, 30))
    f = float(np.clip(val / vmax, 0, 1))
    if f > 0.01:
        draw.rounded_rectangle((x, y, x + max(int(w * f), 10), y + 10), radius=5, fill=(*color, 255))


def steer_gauge(draw, x, y, w, val):
    """Руль: центр = прямо, влево/вправо = куда тянут DNa01/02."""
    draw.text((x, y), "Руль (DNa01/02)", font=F[16], fill=(225, 228, 235, 255))
    lab = "поворот влево" if val > 0.12 else "поворот вправо" if val < -0.12 else "прямо"
    tw = draw.textlength(lab, font=F[16])
    draw.text((x + w - tw, y), lab, font=F[16], fill=(255, 255, 255, 255))
    y += 22
    cx = x + w // 2
    draw.rounded_rectangle((x, y, x + w, y + 10), radius=5, fill=(255, 255, 255, 30))
    d = int(np.clip(val, -1, 1) * w / 2)
    if abs(d) > 2:
        draw.rounded_rectangle((min(cx, cx - d), y, max(cx, cx - d), y + 10), radius=5, fill=(255, 170, 60, 255))
    draw.line((cx, y - 3, cx, y + 13), fill=(255, 255, 255, 180), width=2)


class Popups:
    def __init__(self):
        self.items = []  # (text, color, t0, dur, pos|None)
        self.used = set()

    def add(self, text, color, t, dur=1.6, pos=None, big=True, key=None):
        if key is not None:
            if key in self.used:
                return
            self.used.add(key)
        self.items.append(dict(text=text, color=color, t0=t, dur=dur, pos=pos, big=big))

    def draw(self, draw, t, W, H):
        alive = []
        slot = 0
        for it in reversed(self.items):
            age = (t - it["t0"]) / it["dur"]
            if age > 1:
                continue
            alive.append(it)
            a = int(255 * min(1, (1 - age) * 3, age * 8 + 0.2))
            font = F[44] if it["big"] else F[30]
            tw = draw.textlength(it["text"], font=font)
            if it["pos"] is None:
                x, y = W / 2 - tw / 2, 160 + 62 * slot - 10 * age
                slot += 1
            else:
                x, y = it["pos"][0] - tw / 2, it["pos"][1] - 40 - 60 * age
            draw.text((x + 2, y + 2), it["text"], font=font, fill=(0, 0, 0, a // 2))
            draw.text((x, y), it["text"], font=font, fill=(*it["color"], a))
        self.items = alive[::-1]


def label3d(draw, xy, text, color=(255, 255, 255)):
    if xy is None:
        return
    x, y = xy
    tw = draw.textlength(text, font=F[18])
    draw.rounded_rectangle((x - tw / 2 - 8, y - 34, x + tw / 2 + 8, y - 8), radius=8, fill=(10, 12, 18, 150))
    draw.polygon([(x - 5, y - 8), (x + 5, y - 8), (x, y - 2)], fill=(10, 12, 18, 150))
    draw.text((x - tw / 2, y - 31), text, font=F[18], fill=(*color, 255))
