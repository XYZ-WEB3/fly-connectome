"""Вертикальный кадр TikTok 1080x1920: сцена сверху + мозг и показатели снизу.

    img = compose(scene_rgb, hud_state, t)        # uint8 (1920, 1080, 3)

scene_rgb — уже отрендеренная 3D-сцена 1080x1150 (RGB uint8; другой размер будет отмасштабирован).
hud_state — словарь ОДНОГО кадра: воркеры рисуют кадры вразнобой, поэтому compose не хранит
состояние между кадрами (только кэши картинок). Ключи (все необязательные):
  title, fly_no                 — заголовок «МУХА №N смотрит: <title>»
  act_idx, act_val              — разреженная УЖЕ затухшая активность мозга (индексы FlyBrain,
                                  act > 0.05; то же, что hud.BrainCloud.act: act = act*0.55 + clip(hz/60, 0, 3))
  n_active                      — сколько нейронов горит (иначе len(act_idx))
  fear_hz, interest, hearing_hz, hunger (0..1), steer (-1..1, + = влево), gf_hz, mn9_hz, groom_hz
                                — сырые показатели (Гц); шкалы — GAUGES ниже
  g_<key>, pk_<key>             — сглаженное значение и «пик» (их считает prepare_timeline)
  captions = [(text, sub, (r,g,b), age_s, i0), ...] — реакции возле мухи (prepare_timeline; i0 — кадр появления)
  caption_pos = [(cx, cy, scale), ...] — где их рисовать (воркер: caption_positions по геометрии кадра i0)
  fly_box, screen_box           — (x0, y0, x1, y1) мухи и экрана на картинке сцены (если нет — по fly_xy)
  ticker = (text, age_s) | None, ticker_prev      — бегущая строка событий (prepare_timeline)
  cut_age                       — кадров с последней смены ракурса (короткая вспышка-переход)
  fly_xy                        — (x, y) мухи на картинке сцены (куда ставить реакции)
  frame                         — номер кадра (для зерна)
  screen_rgb, vis               — кадр ролика и признаки зрения — для врезки «как видит муха»
                                  (Composer.flyeye, FLYEYE_BOX; воркер кладёт, только если врезка включена)

prepare_timeline(huds, fps, stages) — ОДИН раз в главном процессе по всему логу: сглаживание шкал,
пики, реакции (только по состояниям, пришедшим из нейронов), бегущая строка, смены ракурса.
"""
import re
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE = HERE / "_cache"
PATH_ANN = ROOT / "data" / "neuron_annotations.tsv"
PATH_COMP = ROOT / "brain_model" / "Completeness_783.csv"      # порядок нейронов = FlyBrain.ids
FONT_PATH = "C:/Windows/Fonts/bahnschrift.ttf"                 # ВНИМАНИЕ: нет стрелок, ●, ★
MN9_ID = 720575940660219265

W, H = 1080, 1920
SCENE_H = 1150
PANEL_H = H - SCENE_H
SAFE_R = 930            # правее — кнопки TikTok (лайк/коммент) -> важное левее

HONESTY = "мозг: коннектом FlyWire, 139 тыс. нейронов · глаз: упрощённая сетчатка · тело: анимация по командам мозга"

# ключ, подпись, подподпись, цвет, максимум шкалы, вид, «горячо» с какой доли
GAUGES = [
    ("fear_hz", "Страх", "нейроны побега", (255, 78, 96), 120.0, "hz", 0.5),
    ("interest", "Интерес", "следит за объектом", (70, 215, 255), 40.0, "hz", 0.5),
    ("hearing_hz", "Слух", "орган Джонстона", (178, 136, 255), 14.0, "hz", 0.55),
    ("hunger", "Голод", "шкала игры", (255, 186, 64), 1.0, "pct", 0.85),
    ("steer", "Руль", "DNa01/02", (255, 150, 70), 1.0, "steer", 0.3),
    ("gf_hz", "Взлёт", "гигантское волокно", (110, 196, 255), 120.0, "hz", 0.3),
    ("mn9_hz", "Хоботок", "MN9", (255, 212, 90), 25.0, "hz", 0.3),
    ("groom_hz", "Чистка", "DNg84/35", (110, 255, 176), 150.0, "hz", 0.3),
]
GAUGE_MAX = {g[0]: g[4] for g in GAUGES}

BRAIN_GROUPS = [  # (подпись, цвет) — как в hud.GROUPS
    ("зрение", (90, 200, 255)),
    ("органы чувств", (255, 215, 90)),
    ("центр", (190, 160, 255)),
    ("команды телу", (255, 80, 150)),
    ("мотонейроны", (255, 150, 50)),
]

# метки на мозге: группа -> (подпись, шкала, куда сдвинуть подпись)
ANCHORS = {
    "lc10": ("LC10 · объект", "interest", (-150, -44)),
    "jo": ("орган Джонстона", "hearing_hz", (-190, 26)),
    "gf": ("DNp01 · взлёт", "gf_hz", (40, -58)),
    "dna": ("DNa01/02 · руль", "steer", (-200, -40)),
    "dng": ("DNg84/35 · чистка", "groom_hz", (-230, 34)),
    "mn9": ("MN9 · хоботок", "mn9_hz", (40, 30)),
}

# реакции: ключ -> (текст, подпись-нейроны, цвет)
REACT = {
    "takeoff": ("испугалась!", "взлёт · гигантское волокно DNp01", (255, 92, 92)),
    "proboscis": ("вкусно!", "хоботок · мотонейрон MN9", (255, 205, 80)),
    "groom": ("отряхивается", "чистка · DNg84 / DNg35", (120, 255, 180)),
    "track": ("следит", "LC10 · руль DNa01/02", (80, 215, 255)),
    "startle": ("вздрогнула", "гигантское волокно DNp01", (255, 140, 110)),
    "hear": ("слушает", "орган Джонстона", (185, 145, 255)),
}
CAPTION_DUR = 1.6

# приступ -> смерть -> новая муха (hud life / life_t / fly_no / n_active — из прохода 1)
LIFE_COL = {"seizure": (255, 70, 110), "dead": (170, 176, 200), "respawn": (90, 235, 170)}
LIFE_HONEST = "в модели нет торможения, которое есть у живой мухи — приступ сам не останавливается"


def life_texts(hud):
    """-> (фаза, секунды, крупно, мелко) или None для обычного кадра."""
    life, lt = str(hud.get("life") or "alive"), float(hud.get("life_t") or 0.0)
    if life == "alive":
        return None
    n = hud.get("fly_no")
    who = f"Муха №{n}" if n is not None else "Муха"
    if life == "seizure":
        k = max(1, int(round(float(hud.get("n_active") or 0) / 1000)))
        return life, lt, "приступ: мозг разогнался", f"горят ~{k} тыс. нейронов · защита отключена"
    if life == "dead":
        return life, lt, f"{who} выбыла", LIFE_HONEST
    return life, lt, f"{who} заступает", "новый мозг с нуля · защита на месте"

# врезка «как видит муха» (flyeye.py; включается в окне): слева сверху сцены, под меню TikTok.
# Подписи-реакции обходят прямоугольники EXTRA_AVOID (воркер кладёт туда FLYEYE_BOX, если врезка включена)
FLYEYE_BOX = (24, 150, 304, 430)
EXTRA_AVOID = []

# ---------------------------------------------------------------- шрифты / текст
_FONTS = {}


def font(size, weight="Regular"):
    k = (size, weight)
    if k not in _FONTS:
        f = ImageFont.truetype(FONT_PATH, size)
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
        _FONTS[k] = f
    return _FONTS[k]


class Sprite:
    """Картинка с альфой: rgb уже умножен на альфу (float32 0..255), a — 0..1."""
    __slots__ = ("rgb", "a", "w", "h")

    def __init__(self, rgba):
        rgba = np.asarray(rgba, np.float32)
        self.a = np.ascontiguousarray(rgba[..., 3] / 255.0)
        self.rgb = np.ascontiguousarray(rgba[..., :3] * self.a[..., None])
        self.h, self.w = self.a.shape

    def scaled(self, s):
        if abs(s - 1) < 1e-3:
            return self
        w, h = max(1, int(self.w * s)), max(1, int(self.h * s))
        sp = Sprite.__new__(Sprite)
        sp.rgb = cv2.resize(self.rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        sp.a = cv2.resize(self.a, (w, h), interpolation=cv2.INTER_LINEAR)
        sp.w, sp.h = w, h
        return sp


def blit(dst, sp, x, y, alpha=1.0):
    """Альфа-наложение спрайта на uint8-картинку (x, y — левый верхний угол)."""
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sp.w, dst.shape[1]), min(y + sp.h, dst.shape[0])
    if x1 <= x0 or y1 <= y0 or alpha <= 0.003:
        return
    sa = sp.a[y0 - y:y1 - y, x0 - x:x1 - x]
    sr = sp.rgb[y0 - y:y1 - y, x0 - x:x1 - x]
    reg = dst[y0:y1, x0:x1].astype(np.float32)
    if alpha >= 0.999:
        reg *= (1.0 - sa)[..., None]
        reg += sr
    else:
        reg *= (1.0 - sa * alpha)[..., None]
        reg += sr * alpha
    np.clip(reg, 0, 255, out=reg)
    dst[y0:y1, x0:x1] = reg


def add_glow(dst, glow_u8, x, y, k):
    """Аддитивное свечение (uint8 rgb) с множителем k."""
    h, w = glow_u8.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0 or k <= 0.01:
        return
    reg = dst[y0:y1, x0:x1]
    cv2.scaleAdd(glow_u8[y0 - y:y1 - y, x0 - x:x1 - x], float(k), reg, dst=reg)


class TextCache:
    """Текст рендерится PIL один раз -> спрайт. Числа меняются каждый кадр, поэтому LRU."""
    def __init__(self, limit=4000):
        self.d = OrderedDict()
        self.limit = limit

    def get(self, text, size, weight="Regular", color=(255, 255, 255), stroke=0, stroke_color=(0, 0, 0),
            glow=0, glow_color=None, glow_gain=1.6, pill=None):
        k = (text, size, weight, color, stroke, stroke_color, glow, glow_color, glow_gain, pill)
        sp = self.d.get(k)
        if sp is not None:
            self.d.move_to_end(k)
            return sp
        sp = Sprite(render_text(text, size, weight, color, stroke, stroke_color, glow, glow_color, glow_gain, pill))
        self.d[k] = sp
        if len(self.d) > self.limit:
            self.d.popitem(last=False)
        return sp


_CMAP = None
_SUBST = {"→": "—", "←": "—", "↔": "—", "⇒": "—", "↑": "", "↓": "", "●": "·", "•": "·", "★": "*", "✓": "+"}


def safe_text(s):
    """В Bahnschrift нет стрелок, ●, ★, эмодзи — вместо «тофу»-квадратиков заменяем/убираем
    символы, которых нет в шрифте (заголовки роликов TikTok часто с эмодзи)."""
    global _CMAP
    s = str(s)
    if _CMAP is None:
        try:
            from fontTools.ttLib import TTFont
            _CMAP = frozenset(TTFont(FONT_PATH, lazy=True).getBestCmap())
        except Exception:
            _CMAP = frozenset()
    if not _CMAP:
        return "".join(_SUBST.get(c, c) for c in s)
    out = "".join(c if ord(c) in _CMAP else _SUBST.get(c, "") for c in s)
    return re.sub(r"\s{2,}", " ", out).strip() if out != s else out


def render_text(text, size, weight="Regular", color=(255, 255, 255), stroke=0, stroke_color=(0, 0, 0),
                glow=0, glow_color=None, glow_gain=1.6, pill=None):
    """-> RGBA uint8. pill=(r,g,b,a) — подложка-«таблетка» под текстом."""
    text = safe_text(text)
    f = font(size, weight)
    pad = stroke + glow * 3 + (size // 2 if pill else 3)
    x0, y0, x1, y1 = f.getbbox(text, stroke_width=stroke)
    asc, desc = f.getmetrics()
    if pill:
        y0, y1 = min(y0, int(asc * 0.28)), max(y1, asc)          # одинаковая высота таблетки
    w, h = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if pill:
        pv = size // 3
        d.rounded_rectangle((pad - size // 2 + 2, pad - pv, w - pad + size // 2 - 2, h - pad + pv),
                            radius=(h - 2 * pad + 2 * pv) // 2, fill=tuple(pill))
    d.text((pad - x0, pad - y0), text, font=f, fill=(*color, 255), stroke_width=stroke,
           stroke_fill=(*stroke_color, 255))
    if not glow:
        return np.asarray(im)
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((pad - x0, pad - y0), text, font=f, fill=255, stroke_width=stroke + 2)
    m = np.asarray(m.filter(ImageFilter.GaussianBlur(glow)), np.float32) / 255 * glow_gain
    gc = np.array(glow_color or color, np.float32)
    base = np.asarray(im, np.float32)
    a = base[..., 3:] / 255
    ga = np.clip(m, 0, 1)[..., None]
    out_a = a + ga * (1 - a)
    out_rgb = (base[..., :3] * a + gc * ga * (1 - a)) / np.maximum(out_a, 1e-4)
    return np.concatenate([out_rgb, out_a * 255], -1).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------- мозг
def _brain_cache(w, h):
    """Точки всех 139k нейронов (вид спереди, координаты FlyWire pos_x/pos_y) -> пиксели.
    Кэш _cache/braincloud_v2_WxH.npz (воркеры не читают 100-мегабайтные аннотации)."""
    p = CACHE / f"braincloud_v2_{w}x{h}.npz"
    if p.exists():
        try:
            z = np.load(p)
            return {k: z[k] for k in z.files}
        except Exception:
            pass
    import pandas as pd
    ids = pd.read_csv(PATH_COMP, index_col=0).index.values
    a = pd.read_csv(PATH_ANN, sep="\t", usecols=["root_id", "pos_x", "pos_y", "super_class", "cell_type", "side"],
                    low_memory=False).drop_duplicates("root_id").set_index("root_id").reindex(ids)
    x, y = a.pos_x.values, a.pos_y.values
    x0, x1 = np.nanpercentile(x, [0.2, 99.8])
    y0, y1 = np.nanpercentile(y, [0.2, 99.8])
    s = min((w - 16) / (x1 - x0), (h - 16) / (y1 - y0))
    fx = (x - x0) * s + (w - (x1 - x0) * s) / 2
    fy = (y - y0) * s + (h - (y1 - y0) * s) / 2
    ok = np.isfinite(fx) & np.isfinite(fy) & (fx >= 1) & (fx < w - 1) & (fy >= 1) & (fy < h - 1)
    px = np.nan_to_num(fx).astype(np.int64).clip(1, w - 2)
    py = np.nan_to_num(fy).astype(np.int64).clip(1, h - 2)
    sc = np.array([str(s) for s in a.super_class.fillna("").tolist()], dtype=object)
    grp = np.full(len(ids), 2, np.uint8)                                     # центр
    grp[np.isin(sc, ["optic", "visual_projection", "visual_centrifugal"])] = 0
    grp[np.isin(sc, ["sensory", "sensory_ascending"])] = 1
    grp[sc == "descending"] = 3
    grp[sc == "motor"] = 4
    pix = np.where(ok, py * w + px, -1).astype(np.int32)
    pix2 = np.where(ok, (py // 2) * (w // 2) + px // 2, -1).astype(np.int32)
    # метки групп нейронов (медиана координат; двусторонние — по сторонам)
    ct = np.array([str(s) for s in a.cell_type.fillna("").tolist()], dtype=object)
    side = np.array([str(s) for s in a.side.fillna("").tolist()], dtype=object)
    sel = {"lc10": np.isin(ct, ["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"]),
           "jo": np.array([s.startswith("JO-") for s in ct]), "gf": ct == "DNp01",
           "dna": np.isin(ct, ["DNa01", "DNa02"]), "dng": np.isin(ct, ["DNg84", "DNg35"]), "mn9": ids == MN9_ID}
    anc = {}
    for k, m in sel.items():
        pts = []
        for sd in (("left", "right") if k in ("lc10", "jo") else (None,)):
            mm = m & ok & ((side == sd) if sd else True)
            if mm.any():
                pts.append((float(np.median(fx[mm])), float(np.median(fy[mm]))))
        anc["anc_" + k] = np.array(pts, np.float32)
    CACHE.mkdir(exist_ok=True)
    d = dict(pix=pix, pix2=pix2, grp=grp, **anc)
    np.savez_compressed(p, **d)
    return d


class BrainCloudSparse:
    """Как hud.BrainCloud, но на вход — разреженная УЖЕ затухшая активность (idx, act).
    Кадр полностью определяется (idx, act) -> можно рисовать кадры вразнобой в разных процессах."""
    def __init__(self, w=1040, h=380):
        c = _brain_cache(w, h)
        self.w, self.h = w, h
        self.pix, self.pix2, self.grp = c["pix"], c["pix2"], c["grp"]
        self.anchors = {k[4:]: c[k] for k in c if k.startswith("anc_")}
        self.gcol = np.array([col for _, col in BRAIN_GROUPS], np.float32)
        self.n = len(self.pix)
        ok = self.pix >= 0
        # фон: плотность всех нейронов тусклым цветом группы
        wts = self.gcol[self.grp[ok]] / 255 * 0.05
        base = np.zeros((w * h, 3))
        for ch in range(3):
            base[:, ch] = np.bincount(self.pix[ok], wts[:, ch], minlength=w * h)
        self.base = np.clip(base.reshape(h, w, 3) ** 0.6 * 1.0, 0, 0.5).astype(np.float32)
        # цвет ореола: средний цвет групп вокруг (полразрешения, размыто, нормировано)
        w2, h2 = w // 2, h // 2
        cm = np.zeros((w2 * h2, 3))
        wts = self.gcol[self.grp[ok]] / 255
        for ch in range(3):
            cm[:, ch] = np.bincount(self.pix2[ok], wts[:, ch], minlength=w2 * h2)
        cm = cv2.GaussianBlur(cm.reshape(h2, w2, 3).astype(np.float32), (0, 0), 4)
        self.halo_col = (cm / np.maximum(cm.max(-1, keepdims=True), 1e-6)).astype(np.float32)
        self.core = np.zeros((h * w, 3), np.uint8)                 # ядра точек (очищаем разреженно)
        self._prev = np.zeros(0, np.int64)
        self._halo_up = np.empty((h, w, 3), np.uint8)
        self._nb = np.array([-1, 1, -w, w], np.int64)

    def draw(self, dst, idx, act, gain=1.0):
        """Добавляет свечение в dst (uint8 h x w x 3, можно срез кадра). -> число горящих точек."""
        core = self.core
        if len(self._prev):
            core[self._prev] = 0
            self._prev = self._prev[:0]
        if idx is None or len(idx) == 0:
            return 0
        idx = np.asarray(idx, np.int64)
        act = np.asarray(act, np.float32)
        m = idx < self.n
        if not m.all():
            idx, act = idx[m], act[m]
        p = self.pix[idx]
        ok = p >= 0
        p = p[ok].astype(np.int64)
        a = np.minimum(act[ok], 2.0) * 0.5 * gain                  # 0..1
        g = self.grp[idx[ok]]
        # ядро: цвет группы, у самых активных — к белому; «звёздочка» 1 + 4 соседа
        v = a ** 0.6
        col = self.gcol[g]
        col = col + (255 - col) * (0.45 * v * v)[:, None]
        rgb = np.minimum(col * (0.25 + 1.1 * v)[:, None], 255).astype(np.uint8)
        nb = (p[:, None] + self._nb).ravel()
        core[nb] = (rgb >> 1).repeat(4, 0)
        core[p] = rgb
        self._prev = np.concatenate([nb, p])
        # ореол: суммарная активность на полразрешения -> размыть -> цвет -> вверх
        w2, h2 = self.w // 2, self.h // 2
        hl = np.bincount(self.pix2[idx[ok]], a, minlength=w2 * h2).astype(np.float32).reshape(h2, w2)
        hl = cv2.GaussianBlur(hl, (0, 0), 3.0)
        hl3 = cv2.convertScaleAbs(self.halo_col * hl[..., None], alpha=255 * 2.6)
        cv2.resize(hl3, (self.w, self.h), dst=self._halo_up, interpolation=cv2.INTER_LINEAR)
        cv2.add(dst, self._halo_up, dst=dst)
        cv2.add(dst, core.reshape(self.h, self.w, 3), dst=dst)
        return len(p)


# ---------------------------------------------------------------- вспомогательное рисование
def _rrect_mask(w, h, r, ss=3):
    im = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=r * ss, fill=255)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255


def _grad(w, cols):
    """Горизонтальный градиент по списку цветов -> (w, 3) float."""
    cols = np.array(cols, np.float32)
    xs = np.linspace(0, len(cols) - 1, w)
    i0 = np.floor(xs).astype(int).clip(0, len(cols) - 2)
    f = (xs - i0)[:, None]
    return cols[i0] * (1 - f) + cols[i0 + 1] * f


def _glow_rect(w, h, r, col, width=3, blur=7, gain=1.5, pad=22):
    """Размытая светящаяся рамка -> uint8 rgb (для cv2.scaleAdd)."""
    m = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    ImageDraw.Draw(m).rounded_rectangle((pad, pad, pad + w, pad + h), radius=r, outline=255, width=width)
    m = np.asarray(m.filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255 * gain
    return (np.clip(m, 0, 1)[..., None] * np.array(col, np.float32)).astype(np.uint8)


# ---------------------------------------------------------------- где ставить реакции
CAP_W, CAP_H = 560, 150            # габарит одной подписи (крупный текст + строка нейронов) при масштабе 1
CAP_SCALES = (1.0, 0.72)           # если рядом с мухой тесно — подпись поменьше, но не на муху


def _grid(s):
    hw = CAP_W * s / 2
    gx, gy = np.meshgrid(np.arange(hw + 16, W - hw - 15, 28, dtype=np.float32),
                         np.arange(250, SCENE_H - CAP_H * s / 2 - 28, 24, dtype=np.float32))
    return gx.ravel(), gy.ravel(), np.full(gx.size, s, np.float32)


_G = [np.concatenate(a) for a in zip(*[_grid(s) for s in CAP_SCALES])]
_GX, _GY, _GS = _G


def _overlap(x0, y0, x1, y1, box):
    """Площадь пересечения прямоугольников-кандидатов (массивы) с box."""
    bx0, by0, bx1, by1 = box
    return np.clip(np.minimum(x1, bx1) - np.maximum(x0, bx0), 0, None) * \
        np.clip(np.minimum(y1, by1) - np.maximum(y0, by0), 0, None)


def place_caption(geom, avoid=()):
    """-> (cx, cy, масштаб) подписи рядом с мухой: не на экране с роликом, не на самой мухе, не под
    меню TikTok (сверху) и не под кнопками справа; avoid — прямоугольники уже висящих подписей.
    geom: fly_box (x0,y0,x1,y1) или fly_xy, screen_box — в пикселях сцены 1080x1150.
    geom может быть списком таких dict (кадры за всё время жизни подписи): тогда наложение на
    экран/муху берётся худшее по кадрам (камера в переходе не «наезжает» экраном на подпись)."""
    geoms = [g for g in (geom if isinstance(geom, (list, tuple)) else [geom]) if g] or [{}]
    hw, hh = CAP_W * _GS / 2, CAP_H * _GS / 2
    x0, x1, y0, y1 = _GX - hw, _GX + hw, _GY - hh, _GY + hh
    area = CAP_W * CAP_H
    bad = np.zeros_like(_GX)                         # худшее наложение на муху/экран по кадрам
    near = np.zeros_like(_GX)                        # среднее «далеко от мухи»
    for g in geoms:
        fb = g.get("fly_box")
        if fb is None:
            fx, fy = g.get("fly_xy") or (W * 0.5, SCENE_H * 0.62)
            fb = (fx - 90, fy - 50, fx + 90, fy + 60)
        fb = tuple(float(v) for v in fb)
        c = 5.0 * _overlap(x0, y0, x1, y1, fb) / area
        sb = g.get("screen_box")
        if sb is not None:
            sb = (sb[0] - 12, sb[1] - 12, sb[2] + 12, sb[3] + 12)
            c += 12.0 * _overlap(x0, y0, x1, y1, sb) / area
        np.maximum(bad, c, out=bad)
        dx = np.maximum(0, np.maximum(fb[0] - x1, x0 - fb[2]))                      # зазор до мухи
        dy = np.maximum(0, np.maximum(fb[1] - y1, y0 - fb[3]))
        near += (np.hypot(dx, dy) / 160.0 + 0.15 * np.abs(_GX - (fb[0] + fb[2]) / 2) / W) / len(geoms)
    cost = bad + near
    for r in (*avoid, *EXTRA_AVOID):
        cost += 9.0 * _overlap(x0, y0, x1, y1, r) / area
    cost += 4.0 * _overlap(x0, y0, x1, y1, (SAFE_R, 720, W, SCENE_H)) / area      # кнопки TikTok (аватар, лайк...)
    cost += 0.3 * (_GS < 1)                                                          # крупная — лучше
    k = int(np.argmin(cost))
    return float(_GX[k]), float(_GY[k]), float(_GS[k])


def caption_rect(c):
    s = c[2] if len(c) > 2 else 1.0
    return (c[0] - CAP_W * s / 2, c[1] - CAP_H * s / 2, c[0] + CAP_W * s / 2, c[1] + CAP_H * s / 2)


CAP_SAMPLES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)     # с (возраст подписи): где проверяем геометрию


def caption_positions(caps, geom_at, caps_at, memo=None, fps=30):
    """Позиции реакций caps = [(text, sub, col, age, i0), ...] (как в prepare_timeline).
    Подпись ставится один раз — по геометрии кадров ВСЕЙ её жизни, начиная с кадра появления i0
    (geom_at(j) -> dict с fly_box/screen_box; j может быть за концом лога — geom_at сам ограничит),
    и потом не прыгает: если камера в это время едет, подпись не окажется на экране или мухе.
    Обходит подписи, которые уже висели в момент i0 (caps_at(i0) — список реакций кадра i0).
    Детерминировано -> одинаково в любом воркере. memo — dict на воркер (кэш по (i0, text))."""
    memo = {} if memo is None else memo

    def pos_of(i0, text, depth=0):
        key = (i0, text)
        if key not in memo:
            avoid = []
            if depth < 12:
                for c in (caps_at(i0) or []):
                    if (c[4], c[0]) == key:
                        break
                    avoid.append(caption_rect(pos_of(c[4], c[0], depth + 1)))
            geoms = [geom_at(i0 + int(round(a * fps))) for a in CAP_SAMPLES]
            memo[key] = place_caption(geoms, avoid)
        return memo[key]

    return [pos_of(c[4], c[0]) for c in caps]


# ---------------------------------------------------------------- композитор
class Composer:
    flyeye = None          # хук: объект с render(frame_rgb, feats_row, (w, h)) -> RGBA; None — без врезки

    def __init__(self, seed=0, fx=True):
        self.tc = TextCache()
        self.fx = fx
        self.rng = np.random.default_rng(seed)
        self._key = None
        # раскладка панели (y — от верха панели)
        self.title_y = 18
        self.row_y = [122, 236]
        self.card_w, self.card_h, self.card_gap = 214, 86, 12
        self.card_x = [40 + k * (self.card_w + self.card_gap) for k in range(4)]
        self.ticker_y = 340
        self.brain_y = 388
        self.brain_x = 20
        self.brain = BrainCloudSparse(1040, PANEL_H - self.brain_y - 2)
        self._build_fx()
        self._build_bars()

    # ---------- статичные слои ----------
    def _build_fx(self):
        h, w = SCENE_H, W
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w * 0.62)) ** 2 + ((yy - h * 0.48) / (h * 0.64)) ** 2)
        v = 1 - 0.45 * np.clip(r - 0.4, 0, 1) ** 1.5
        top = np.clip(yy / 190, 0, 1) ** 0.8 * 0.3 + 0.7           # сверху темнее: там меню TikTok
        bot = 1 - 0.6 * np.clip((yy - (h - 120)) / 120, 0, 1) ** 1.6   # снизу плавно в панель
        self.vign = np.repeat((np.clip(v * top * bot, 0, 1) * 255)[..., None], 3, -1).astype(np.uint8)
        # зерно: знаковое (+ и - отдельно), несколько тайлов по кругу
        self.grain = []
        for _ in range(6):
            g = self.rng.normal(0, 4.0, (h // 2, w // 2)).astype(np.float32)
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)
            gp = np.repeat(np.clip(g, 0, 255)[..., None], 3, -1).astype(np.uint8)
            gn = np.repeat(np.clip(-g, 0, 255)[..., None], 3, -1).astype(np.uint8)
            self.grain.append((gp, gn))
        self._bloom_sizes = [(w // 2, h // 2), (w // 4, h // 4), (w // 8, h // 8)]
        self._bloom_up = np.empty((h, w, 3), np.uint8)
        self._scene_buf = np.empty((h, w, 3), np.uint8)

    def _build_bars(self):
        bw = self.card_w - 28
        self.bar_w, self.bar_h = bw, 10
        self.bar_mask = _rrect_mask(bw, self.bar_h, 5)
        self.cap = _rrect_mask(12, self.bar_h, 5)[:, 6:]
        self.bars = {}
        for g in GAUGES:
            c = np.array(g[3], np.float32)
            gr = _grad(bw, [c * 0.5, c, np.minimum(c * 1.2 + 60, 255)])
            self.bars[g[0]] = np.broadcast_to(gr[None], (self.bar_h, bw, 3)).astype(np.float32)
        self.hot = {g[0]: _glow_rect(self.card_w, self.card_h, 14, g[3]) for g in GAUGES}

    def _build_panel(self, title, fly_no):
        pw, ph = W, PANEL_H
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        f = (yy / ph)[..., None]
        bg = np.array([15, 17, 29], np.float32) * (1 - f) + np.array([4, 5, 10], np.float32) * f
        # мягкое фиолетовое свечение у шва + под мозгом, сетка точек
        bg += np.exp(-(((xx - pw / 2) / 560) ** 2 + (yy / 140) ** 2))[..., None] * np.array([34, 24, 70], np.float32)
        bcy = self.brain_y + self.brain.h * 0.5
        bg += np.exp(-(((xx - pw / 2) / 470) ** 2 + ((yy - bcy) / 170) ** 2))[..., None] * np.array([10, 16, 34],
                                                                                                    np.float32)
        bg += ((xx % 24 < 1.5) & (yy % 24 < 1.5))[..., None] * 6.0
        pil = Image.fromarray(bg.clip(0, 255).astype(np.uint8)).convert("RGBA")
        ov = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        # шов между сценой и панелью: неоновая линия с мягким свечением вниз
        seam = _grad(pw, [(60, 200, 255), (170, 120, 255), (255, 90, 170)])
        seam_img = np.zeros((16, pw, 4), np.float32)
        for hh in range(16):
            seam_img[hh, :, :3] = seam
            seam_img[hh, :, 3] = 255 if hh < 2 else 170 * np.exp(-(hh - 1) / 2.6)
        ov.alpha_composite(Image.fromarray(seam_img.astype(np.uint8)), (0, 0))
        # --- заголовок ---
        ty = self.title_y
        head = f"МУХА №{fly_no}" if fly_no is not None else "МУХА"
        fh = font(32, "Bold")
        d.text((40, ty), head, font=fh, fill=(255, 255, 255, 255))
        x = 40 + d.textlength(head, font=fh) + 12
        fw = font(27, "SemiLight")
        d.text((x, ty + 4), "смотрит:", font=fw, fill=(150, 195, 255, 255))
        x += d.textlength("смотрит:", font=fw) + 12
        tt, size, lim = safe_text(title or ""), 29, SAFE_R
        while size > 18 and x + d.textlength(tt, font=font(size, "SemiBold")) > lim:
            size -= 1
        if x + d.textlength(tt, font=font(size, "SemiBold")) > lim:
            while tt and x + d.textlength(tt + "…", font=font(size, "SemiBold")) > lim:
                tt = tt[:-1]
            tt = tt.rstrip() + "…"
        d.text((x, ty + 3 + (29 - size) * 0.7), tt, font=font(size, "SemiBold"), fill=(255, 224, 140, 255))
        fs = font(15, "Regular")
        while d.textlength(HONESTY, font=fs) > SAFE_R - 40 and fs.size > 12:
            fs = font(fs.size - 1, "Regular")
        d.text((40, ty + 48), HONESTY, font=fs, fill=(140, 147, 170, 255))
        # --- ряды карточек ---
        for r, (a, b) in enumerate((("ЧТО ЧУВСТВУЕТ", "· считано с нейронов"), ("РЕШЕНИЯ МОЗГА", "· команды телу"))):
            yh = self.row_y[r] - 25
            d.text((40, yh), a, font=font(15, "SemiBold"), fill=(205, 210, 230, 255))
            d.text((40 + d.textlength(a, font=font(15, "SemiBold")) + 8, yh), b, font=font(15, "Light"),
                   fill=(125, 132, 158, 255))
        self.cards = []
        for n, (key, lab, sub, col, vmax, kind, hot) in enumerate(GAUGES):
            r, c = divmod(n, 4)
            x0, y0 = self.card_x[c], self.row_y[r]
            x1, y1 = x0 + self.card_w, y0 + self.card_h
            d.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=(255, 255, 255, 13),
                                outline=(255, 255, 255, 36), width=1)
            d.rounded_rectangle((x0 + 1, y0 + 14, x0 + 4, y1 - 14), radius=2, fill=(*col, 235))
            d.text((x0 + 15, y0 + 9), lab, font=font(23, "SemiBold"), fill=(246, 247, 252, 255))
            d.text((x0 + 15, y0 + 38), sub, font=font(14, "Regular"),
                   fill=(*[int(v * 0.55 + 95) for v in col], 255))
            bx, by = x0 + 15, y1 - 21
            d.rounded_rectangle((bx, by, bx + self.bar_w, by + self.bar_h), radius=5, fill=(255, 255, 255, 24))
            if kind == "steer":
                cx = bx + self.bar_w // 2
                d.line((cx, by - 3, cx, by + self.bar_h + 2), fill=(255, 255, 255, 160), width=2)
            self.cards.append(dict(key=key, col=col, vmax=vmax, kind=kind, hot=hot, x=x0, y=y0, bx=bx, by=by))
        # --- бегущая строка ---
        ty = self.ticker_y
        d.rounded_rectangle((40, ty, SAFE_R, ty + 40), radius=20, fill=(255, 255, 255, 13),
                            outline=(255, 255, 255, 30), width=1)
        d.rounded_rectangle((44, ty + 4, 44 + 118, ty + 36), radius=16, fill=(255, 64, 118, 235))
        fb = font(16, "Bold")
        d.text((44 + 59 - d.textlength("СОБЫТИЕ", font=fb) / 2, ty + 10), "СОБЫТИЕ", font=fb,
               fill=(255, 255, 255, 255))
        self.ticker_x = 44 + 118 + 14
        # --- мозг: подпись + легенда ---
        by = self.brain_y
        d.text((40, by + 4), "МОЗГ СЕЙЧАС", font=font(21, "Bold"), fill=(255, 255, 255, 255))
        self.count_x = 40 + d.textlength("МОЗГ СЕЙЧАС", font=font(21, "Bold")) + 12
        lx, fl = SAFE_R, font(14)
        for name, col in reversed(BRAIN_GROUPS):
            lx -= d.textlength(name, font=fl)
            d.text((lx, by + 8), name, font=fl, fill=(170, 176, 195, 255))
            lx -= 15
            d.ellipse((lx, by + 12, lx + 9, by + 21), fill=(*col, 255))
            lx -= 17
        pil.alpha_composite(ov)
        img = np.asarray(pil.convert("RGB")).copy()
        b = self.brain
        reg = img[by:by + b.h, self.brain_x:self.brain_x + b.w].astype(np.float32)
        img[by:by + b.h, self.brain_x:self.brain_x + b.w] = (reg + b.base * 255 * (1 - reg / 510)).clip(0, 255)
        self.panel = img

    def _ensure(self, hud):
        key = (hud.get("title"), hud.get("fly_no"))
        if key != self._key:
            self._build_panel(*key)
            self._key = key

    # ---------- сцена ----------
    def _scene_fx(self, scene, hud, fi, dst):
        s = scene
        if s.ndim == 3 and s.shape[2] == 4:
            s = s[..., :3]
        if s.shape[:2] != (SCENE_H, W):
            s = cv2.resize(np.ascontiguousarray(s), (W, SCENE_H), interpolation=cv2.INTER_AREA)
        if not self.fx:
            dst[:] = s
            return
        # блум: 3 раза пополам (дёшево и без мерцания), порог, размытие, обратно вверх
        sm = s
        for sz in self._bloom_sizes:
            sm = cv2.resize(sm, sz, interpolation=cv2.INTER_LINEAR)
        sm = cv2.subtract(sm, (140, 140, 140, 0))
        sm = cv2.GaussianBlur(sm, (0, 0), 2.6)
        cut = hud.get("cut_age")
        k = 0.9
        if cut is not None and 0 <= cut < 8:                    # смена ракурса: короткая вспышка
            k += 2.4 * (1 - cut / 8) ** 2
        sm = cv2.convertScaleAbs(sm, alpha=k)
        cv2.resize(sm, (W, SCENE_H), dst=self._bloom_up, interpolation=cv2.INTER_LINEAR)
        cv2.add(s, self._bloom_up, dst=dst)
        cv2.multiply(dst, self.vign, dst=dst, scale=1 / 255)
        gp, gn = self.grain[fi % len(self.grain)]
        cv2.add(dst, gp, dst=dst)
        cv2.subtract(dst, gn, dst=dst)

    # ---------- реакции возле мухи ----------
    def _captions(self, img, hud):
        caps = hud.get("captions") or []
        if not caps:
            return
        order = sorted(range(len(caps)), key=lambda k: -caps[k][3])  # старые первыми, свежая — поверх
        pos = hud.get("caption_pos")
        if not pos or len(pos) != len(caps):                       # нет позиций от воркера: по этому кадру
            pos, avoid = [None] * len(caps), []
            for k in order:
                pos[k] = place_caption(hud, avoid)
                avoid.append(caption_rect(pos[k]))
        for k in order:
            text, sub, col, age = caps[k][:4]
            if age > CAPTION_DUR or age < 0:
                continue
            cx, cy = pos[k][:2]
            sc = pos[k][2] if len(pos[k]) > 2 else 1.0
            if age < 0.14:                                   # «выпрыгивает» с перелётом
                s = 0.5 + 0.68 * (age / 0.14)
            elif age < 0.3:
                s = 1.18 - 0.18 * (age - 0.14) / 0.16
            else:
                s = 1.0
            a = min(1.0, (CAPTION_DUR - age) / 0.35, 0.35 + age / 0.06)
            y = cy - 22 * sc - 14 * age
            big = self.tc.get(text, 78, "Bold", (255, 255, 255), stroke=5, stroke_color=(18, 10, 28),
                              glow=11, glow_color=tuple(col), glow_gain=2.3).scaled(s * sc)
            blit(img, big, cx - big.w / 2, y - big.h / 2, a)
            if sub:
                sp = self.tc.get(sub, 23, "SemiBold", tuple(col), pill=(10, 10, 20, 170)).scaled(max(sc, 0.85))
                blit(img, sp, cx - sp.w / 2, y + big.h / 2 - 8 * sc, a * min(1.0, max(0.0, (age - 0.08) / 0.2)))

    # ---------- приступ / смерть / новая муха ----------
    def _life(self, img, hud, fi):
        lt_ = life_texts(hud)
        if lt_ is None:
            return
        life, lt, big, small = lt_
        col = LIFE_COL[life]
        if life == "seizure":                                    # тряска кадра и красная пульсация по краям
            k = min(1.0, 0.35 + lt / 2.6)
            r = np.random.default_rng(fi)
            dx, dy = (r.integers(-9, 10, 2) * k).astype(int)
            img[:] = np.roll(img, (int(dy), int(dx)), (0, 1))
            pul = (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * lt)) * 0.45 * k
            ed = (255 - self.vign[..., :1].astype(np.float32)) / 255
            img[:] = np.clip(img + ed * pul * np.array(col, np.float32), 0, 255).astype(np.uint8)
        else:                                                    # муха выбыла: сцена темнеет и выцветает
            k = min(1.0, lt / 1.2) if life == "dead" else 1 - min(1.0, max(0.0, lt - 1.9) / 0.8)
            k *= 0.5
            g = img.mean(2, keepdims=True)
            img[:] = (img * (1 - k * 0.9) + g * k * 0.5).clip(0, 255).astype(np.uint8)
        s = 0.6 + 0.4 * lt / 0.18 if lt < 0.18 else 1.0
        a = min(1.0, lt / 0.2)
        if life == "respawn":                                    # «заступает» — когда новую муху ставят
            a, s = (0.0 if lt < 1.9 else min(1.0, (lt - 1.9) / 0.2)), 1.0
        if a <= 0:
            return
        cy = 330
        sp = self.tc.get(big, 84 if life != "seizure" else 70, "Bold", (255, 255, 255), stroke=5,
                         stroke_color=(18, 10, 28), glow=12, glow_color=col, glow_gain=2.3).scaled(s)
        if sp.w > SAFE_R - 40:
            sp = sp.scaled((SAFE_R - 40) / sp.w)
        blit(img, sp, (SAFE_R + 20) / 2 - sp.w / 2, cy - sp.h / 2, a)
        sb = self.tc.get(small, 25 if life == "dead" else 28, "SemiBold",
                         tuple(int(min(255, c * 0.5 + 128)) for c in col), pill=(10, 10, 20, 190))
        if sb.w > SAFE_R - 40:
            sb = sb.scaled((SAFE_R - 40) / sb.w)
        blit(img, sb, (SAFE_R + 20) / 2 - sb.w / 2, cy + sp.h / 2 + 4,
             a if life == "respawn" else a * min(1.0, max(0.0, (lt - 0.15) / 0.3)))

    # ---------- панель ----------
    def _bar(self, pan, key, x, y, n, m, seg):
        reg = pan[y:y + self.bar_h, x:x + n].astype(np.float32)
        reg += (seg - reg) * m[..., None]
        pan[y:y + self.bar_h, x:x + n] = reg

    def _gauges(self, pan, hud):
        bw = self.bar_w
        for cd in self.cards:
            key, kind = cd["key"], cd["kind"]
            v = float(hud.get("g_" + key, hud.get(key, 0.0)) or 0.0)
            pk = hud.get("pk_" + key)
            bx, by = cd["bx"], cd["by"]
            if kind == "steer":
                f = float(np.clip(v, -1, 1))
                n = int(abs(f) * bw / 2)
                if n >= 2:
                    seg = self.bars[key][:, bw - n:]                  # яркий конец — снаружи
                    if f > 0:                                          # + = влево от центра
                        self._bar(pan, key, bx + bw // 2 - n, by, n, self.bar_mask[:, :n], seg[:, ::-1])
                    else:
                        self._bar(pan, key, bx + bw // 2, by, n, self.bar_mask[:, bw - n:], seg)
                txt = "влево" if f > 0.12 else "вправо" if f < -0.12 else "прямо"
                fr = abs(f)
            else:
                vmax = cd["vmax"]
                fr = float(np.clip(v / vmax, 0, 1))
                n = int(fr * bw)
                if n >= 3:
                    m = self.bar_mask[:, :n]
                    if 8 <= n < bw - 6:
                        m = m.copy()
                        m[:, -6:] *= self.cap
                    self._bar(pan, key, bx, by, n, m, self.bars[key][:, :n])
                if pk is not None:                                     # отметка недавнего пика
                    px = int(np.clip(pk / vmax, 0, 1) * bw)
                    if px > n + 4:
                        pan[by - 2:by + self.bar_h + 2, bx + px - 1:bx + px + 1] = \
                            np.minimum(np.array(cd["col"]) * 0.85 + 40, 255)
                txt = f"{v * 100:.0f}%" if kind == "pct" else f"{v:.0f} Гц"
            hot = min(1.0, max(0.0, (fr - cd["hot"]) / max(0.2, 1 - cd["hot"]) * 1.6))
            if hot > 0.02:
                add_glow(pan, self.hot[key], cd["x"] - 22, cd["y"] - 22, hot)
            sp = self.tc.get(txt, 21, "SemiBold", (255, 255, 255) if hot < 0.4 else
                             tuple(int(min(255, c * 0.6 + 110)) for c in cd["col"]))
            blit(pan, sp, cd["x"] + self.card_w - 8 - sp.w, cd["y"] + 6)

    def _ticker(self, pan, hud):
        # рисуем в окно строки (правее плашки «СОБЫТИЕ», левее кнопок TikTok): уезжающий старый текст
        # не залезает на плашку, длинный текст ужимается и обрезается, а не уходит под кнопки
        x0 = self.ticker_x - 12
        box = pan[:, x0:SAFE_R - 16]
        bw = box.shape[1]
        ty = self.ticker_y + 20
        tk = hud.get("ticker")
        prev = hud.get("ticker_prev")
        x = self.ticker_x - x0

        def fit(sp):
            return sp.scaled(max(0.7, (bw - x - 4) / sp.w)) if sp.w > bw - x - 4 else sp

        if tk:
            text, age = tk
            sp = fit(self.tc.get(text, 22, "SemiBold", (255, 255, 255)))
            slide = max(0.0, 1 - age / 0.25) ** 2
            if prev and age < 0.15:                         # старый текст быстро уезжает и гаснет
                pp = fit(self.tc.get(prev, 22, "SemiBold", (170, 170, 185)))
                blit(box, pp, x - 30 * age / 0.15, ty - pp.h / 2, (1 - age / 0.15) ** 2)
            flash = max(0.0, 1 - age / 0.7)
            if flash > 0:
                hl = self.tc.get(text, 22, "SemiBold", (255, 200, 220), glow=7, glow_color=(255, 60, 130))
                hl = hl.scaled(sp.w / max(1, self.tc.get(text, 22, "SemiBold", (255, 255, 255)).w))
                blit(box, hl, x + 40 * slide - (hl.w - sp.w) / 2, ty - hl.h / 2, flash)
            a = min(1.0, age / 0.2) if age < 6 else max(0.5, 1 - (age - 6) / 3)
            blit(box, sp, x + 40 * slide, ty - sp.h / 2, a * (1 - flash * 0.6))
        else:
            sp = self.tc.get("все реакции мухи — из активности её нейронов", 19, "Regular", (140, 146, 170))
            blit(box, sp, x, ty - sp.h / 2)

    def _brain(self, pan, hud):
        b = self.brain
        idx, act = hud.get("act_idx"), hud.get("act_val")
        y0, x0 = self.brain_y, self.brain_x
        reg = pan[y0:y0 + b.h, x0:x0 + b.w]
        n_pts = b.draw(reg, idx, act)
        # метки групп нейронов, которые сейчас решают (из тех же шкал)
        for k, (lab, key, (dx, dy)) in ANCHORS.items():
            pts = b.anchors.get(k)
            if pts is None or not len(pts):
                continue
            v = float(hud.get("g_" + key, hud.get(key, 0.0)) or 0.0)
            fr = abs(v) if key == "steer" else v / GAUGE_MAX[key]
            a = min(1.0, max(0.0, (fr - 0.35) / 0.25))
            if a <= 0.02:
                continue
            col = next(g[3] for g in GAUGES if g[0] == key)
            for (px, py) in pts:
                cv2.circle(reg, (int(px), int(py)), 11, tuple(int(c * a) for c in col), 2, cv2.LINE_AA)
            px, py = pts[0]
            lx, ly = px + dx, py + dy
            cv2.line(reg, (int(px + (6 if dx > 0 else -6)), int(py + (6 if dy > 0 else -6))),
                     (int(lx + (0 if dx > 0 else 150)), int(ly + 12)),
                     tuple(int(c * a * 0.8) for c in col), 1, cv2.LINE_AA)
            sp = self.tc.get(lab, 16, "SemiBold", col, pill=(8, 9, 16, 200))
            blit(reg, sp, lx if dx > 0 else lx + 150 - sp.w, ly + 12 - sp.h / 2, a)
        n = hud.get("n_active")
        if n is None:
            n = n_pts if idx is not None else 0
        txt = f"горят {int(n):,} нейронов".replace(",", " ")
        sp = self.tc.get(txt, 21, "Regular", (140, 215, 255))
        blit(pan, sp, self.count_x - 3, y0 + 4 - 3)

    # ---------- кадр ----------
    def compose(self, scene_rgb, hud=None, t=0.0, out=None):
        hud = hud or {}
        self._ensure(hud)
        fi = int(hud.get("frame", round(t * 30)))
        if out is None:
            out = np.empty((H, W, 3), np.uint8)
        top = out[:SCENE_H]
        self._scene_fx(scene_rgb, hud, fi, top)
        if self.flyeye is not None:                         # «как видит муха»: кадр ролика + признаки зрения
            x0, y0, x1, y1 = FLYEYE_BOX
            blit(top, Sprite(self.flyeye.render(hud.get("screen_rgb"), hud.get("vis"), (x1 - x0, y1 - y0))), x0, y0)
        self._captions(top, hud)
        self._life(top, hud, fi)
        pan = out[SCENE_H:]
        pan[:] = self.panel
        self._brain(pan, hud)
        self._gauges(pan, hud)
        self._ticker(pan, hud)
        return out


_COMPOSER = None


def compose(scene_rgb, hud_state, t=0.0, out=None):
    """-> uint8 (1920, 1080, 3). См. описание модуля."""
    global _COMPOSER
    if _COMPOSER is None:
        _COMPOSER = Composer()
    return _COMPOSER.compose(scene_rgb, hud_state, t, out)


# ---------------------------------------------------------------- таймлайн интерфейса (главный процесс)
def _is_cut(a, b):
    """Жёсткая смена ракурса? Имя/номер пресета сменился, или камера (Stage.cam: pos3, fwd3, ...)
    прыгнула за кадр дальше, чем бывает при плавном пролёте."""
    if a is None or b is None:
        return False
    if isinstance(a, (str, int, np.integer)) and isinstance(b, (str, int, np.integer)):
        return a != b
    try:
        a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
        if a.shape != b.shape or a.size < 3:
            return False
        jump = np.linalg.norm(b[:3] - a[:3])
        if a.size >= 6:
            jump += 5.0 * np.linalg.norm(b[3:6] - a[3:6])
        return bool(jump > 3.0)
    except (TypeError, ValueError):
        return False


def configure(gauge_max=None):
    """Поменять максимумы шкал (например из meta['gauge_max'] лога): {'hearing_hz': 14, ...}.
    Вызывать до Composer()/prepare_timeline — и в главном процессе, и в каждом воркере."""
    global GAUGES
    if not gauge_max:
        return
    GAUGES = [(k, lab, sub, col, float(gauge_max.get(k, vmax)), kind, hot)
              for (k, lab, sub, col, vmax, kind, hot) in GAUGES]
    GAUGE_MAX.update({g[0]: g[4] for g in GAUGES})


def prepare_timeline(huds, fps=30, stages=None):
    """По всему логу (список hud-словарей по кадрам) -> список добавок для каждого кадра:
    сглаженные шкалы g_*, пики pk_*, реакции captions, бегущая строка ticker, cut_age, frame.

    Реакции включаются ТОЛЬКО состояниями, которые пришли из нейронов (прогон 1, senses.Behaviour):
      takeoff   — hud['takeoff'] (событие кадра) или hud['flying'] стал True
                  (если ни того ни другого не пишут — gf_hz > 45, порог THRESH['takeoff'])
      startle   — hud['startle'] (вздрагивание: 25 < gf <= 45); если ключа нет — gf_hz поднялся
                  выше 25 (THRESH['startle']) не в полёте. Вздрагивание, за которым в 0.2 с
                  следует взлёт, выкидываем (GF растёт за 1-2 кадра — это начало того же взлёта)
      proboscis — hud['eat_start'] или hud['proboscis'] > 0.3 (Behaviour: 0.3 ~ mn9 8 Гц, начало
                  еды; иначе mn9_hz > 8)
      groom     — hud['groom_start'] или hud['grooming'] стал True (иначе groom_hz > 35)
      track     — interest / max > 0.55 держится 0.3 с
      hear      — hearing_hz / max > 0.7 держится 0.3 с (не чаще раза в 10 с)
      hud['react'] = ключ из REACT — принудительно (прогон 1 сам решил по нейронам).
    События бегущей строки: hud['event'] = 'бас-дроп! — капля сахара' (только в кадре события).
    Смена ракурса: hud['cut'] / stage['cut'] = True, либо смена stage['cam'|'shot'|'camera'|'cam_id']
    (у stage.Stage cam — массив; «прыжок» камеры за кадр = склейка)."""
    dt = 1.0 / fps
    out = [dict() for _ in range(len(huds))]
    g = {k: 0.0 for k in GAUGE_MAX}
    pk = {k: 0.0 for k in GAUGE_MAX}
    alpha = 1 - np.exp(-dt / 0.09)
    prev = {}
    hold = {"track": 0.0, "hear": 0.0}
    last = {k: -99.0 for k in REACT}
    cool = {"takeoff": 2.0, "startle": 1.5, "proboscis": 3.0, "groom": 3.0, "track": 5.0, "hear": 10.0}
    caps = []
    ev_cur, ev_prev = None, None
    cut_i, cam_prev = None, None

    def rise(name, cond):
        was = prev.get(name, False)
        prev[name] = bool(cond)
        return bool(cond) and not was

    def num(h, k):
        try:
            return float(h.get(k, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    trigs = []
    for i, h in enumerate(huds):
        h = h or {}
        d = out[i]
        for k in GAUGE_MAX:
            g[k] += alpha * (num(h, k) - g[k])
            pk[k] = max(abs(g[k]) if k == "steer" else g[k], pk[k] - dt * GAUGE_MAX[k] * 0.35)
            d["g_" + k] = g[k]
            d["pk_" + k] = pk[k]
        trig = []
        if str(h.get("life") or "alive") != "alive":             # приступ/смерть/смена мухи: реакций нет
            prev.clear()
            trigs.append(trig)
            continue
        if "takeoff" in h or "flying" in h:
            fly_now = rise("takeoff", h.get("flying", False))
            if bool(h.get("takeoff")) or fly_now:
                trig.append("takeoff")
        elif rise("takeoff", num(h, "gf_hz") > 45):
            trig.append("takeoff")
        if "startle" in h:
            if h.get("startle"):
                trig.append("startle")
        elif rise("startle", num(h, "gf_hz") > 25) and not h.get("flying"):
            trig.append("startle")
        if "eat_start" in h or "proboscis" in h:
            pro_now = rise("proboscis", num(h, "proboscis") > 0.3)
            if bool(h.get("eat_start")) or pro_now:
                trig.append("proboscis")
        elif rise("proboscis", num(h, "mn9_hz") > 8):
            trig.append("proboscis")
        if "groom_start" in h or "grooming" in h:
            gr_now = rise("groom", h.get("grooming", False))
            if bool(h.get("groom_start")) or gr_now:
                trig.append("groom")
        elif rise("groom", num(h, "groom_hz") > 35):
            trig.append("groom")
        for key, gk, thr in (("track", "interest", 0.55), ("hear", "hearing_hz", 0.7)):
            hold[key] = hold[key] + dt if g[gk] / GAUGE_MAX[gk] > thr else 0.0
            if rise(key, hold[key] >= 0.3):
                trig.append(key)
        if h.get("react") in REACT:
            trig.append(h["react"])
        trigs.append(trig)
    # вздрагивание прямо перед взлётом — это начало взлёта (GF растёт за 1-2 кадра): не дублируем
    look = max(1, int(round(0.2 * fps)))
    for i, trig in enumerate(trigs):
        if "startle" in trig and any("takeoff" in trigs[j] for j in range(i, min(len(trigs), i + look + 1))):
            trig[:] = [k for k in trig if k != "startle"]

    for i, h in enumerate(huds):
        h = h or {}
        t = i * dt
        d = out[i]
        for k in dict.fromkeys(trigs[i]):
            if t - last[k] >= cool.get(k, 2.0):
                last[k] = t
                caps.append((k, t, i))
        caps = [c for c in caps if t - c[1] < CAPTION_DUR][-2:]
        d["captions"] = [(*REACT[k], t - t0, i0) for k, t0, i0 in caps]
        ev = h.get("event")
        if isinstance(ev, (list, tuple)):
            ev = " · ".join(str(e) for e in ev if e) or None
        if ev:
            ev_prev = ev_cur[0] if ev_cur else None
            ev_cur = (str(ev), t)
        d["ticker"] = (ev_cur[0], t - ev_cur[1]) if ev_cur else None
        d["ticker_prev"] = ev_prev
        st = (stages[i] if stages is not None else None)
        st = st if isinstance(st, dict) else {}
        cam = next((st[k] for k in ("cam", "shot", "camera", "cam_id") if k in st), None)
        if h.get("cut") or st.get("cut"):
            cut_i = i
        elif _is_cut(cam_prev, cam):
            cut_i = i
        cam_prev = cam
        d["cut_age"] = None if cut_i is None else i - cut_i
        d["frame"] = i
    return out
