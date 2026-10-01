"""Формат «Стример»: вертикальный кадр 1080x1920 в стиле реакции стримера.

    import compose_streamer as CS
    CS.configure(gauge_max)                       # как compose.configure
    extra = CS.prepare_timeline(huds, fps, stages)   # главный процесс, один раз по логу
    comp = CS.Composer(fmt=fmt)                   # fmt — formats.fmt_cfg(...) (вариант, вотермарка)
    comp.flyeye = flyeye.FlyEye()                 # хук «как видит муха» (None — нет врезки)
    img = comp.compose(scene_rgb, hud, t, out=buf)   # scene_rgb — StreamerStage.render(SCENE_W, SCENE_H)

Зоны (пиксели кадра, x0, y0, x1, y1) — см. FORMATS.md, менять только вместе со спекой:
  TOP      (0, 0, 1080, 190)      плашка LIVE + «Муха №N» + «смотрит: <название>» + время эфира
  CLIP     (0, 190, 1080, 830)    ролик 1080x640 (альбом — по ширине, портрет — по высоте + размытые поля),
                                  вотермарка в правом нижнем углу ролика, «смотрит сюда» по повороту головы
  FACECAM  (0, 830, 1080, 1570)   рендер StreamerStage 1080x740 в окне «вебки» с неоновой рамкой
                                  (цвет и режим — подсветка комнаты style.led); «сейчас: …» и алерты поверх
  BRAIN    (0, 1570, 1080, 1920)  событие, компактный мозг 694x226, 5 мини-шкал, строка честности
  FLYEYE_BOX (24, 566, 274, 816)  врезка «как видит муха» 250x250 (в зоне ролика, слева снизу)

hud кадра = hud прохода 1 + добавки prepare_timeline + от воркера: screen_rgb (кадр ролика RGB),
pose (formats.pose_track: doing и т.д.), vis (признаки зрения, если включена врезка), act_idx/act_val,
title, fly_no, fly_box/fly_xy (если StreamerStage их даёт).

Правдивость: «сейчас: …» — только pose['doing'] (formats.DOING), алерты — только реакции compose
(captions, из нейронов; тексты — словами автора), «смотрит влево/вправо» — поворот головы куклы
pose['head'] (руль DNa01/02). Оформление (цвета, узор фона, режим подсветки рамки) — RoomStyle варианта.
Composer не хранит состояние между кадрами: анимации — по возрастам из hud (doing_age, age реакций,
now — плавная смена «сейчас», посчитанная prepare_timeline по порядку кадров).
"""
import colorsys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import compose as C
import formats as F

W, H = 1080, 1920
SCENE_W, SCENE_H = 1080, 740
TOP = (0, 0, 1080, 190)
CLIP = (0, 190, 1080, 830)
FACECAM = (0, 830, 1080, 1570)
BRAIN = (0, 1570, 1080, 1920)
FLYEYE_BOX = (24, 566, 274, 816)
SAFE_R = C.SAFE_R                  # правее — кнопки TikTok
TIKTOK_BTN_Y = 720                 # ниже — кнопки TikTok справа (x > SAFE_R): никакого текста
BRAIN_SIZE = (694, 226)            # размер из готового кэша _cache/braincloud_v2_694x226.npz
BRAIN_XY = (20, 1630)

# окно «вебки»: сцена обрезается по краям на отступ (лицо мухи ROOM держит в x 25–75 %)
CAM_IN_X, CAM_IN_Y, CAM_R = 14, 12, 28
CAM_WIN = (FACECAM[0] + CAM_IN_X, FACECAM[1] + CAM_IN_Y, FACECAM[2] - CAM_IN_X, FACECAM[3] - CAM_IN_Y)

NOW_XY = (40, 852)                 # «сейчас: …»: зона фейскама 0..620 x 0..130 свободна от лица
NOW_H, NOW_MIN_W, NOW_MAX_W = 124, 380, 600
ALERT_X, ALERT_Y1, ALERT_MAX_X = 40, 1550, 900
ALERT_H, ALERT_GAP = 106, 8        # два алерта: 1550 - 2 * 106 - 8 = 1330 (зона спеки y 1330..1560)
ALERT_LIFT = 0.12                  # с: старый алерт уезжает вверх, освобождая место свежему
ALERT_WAIT = 0.06                  # с: свежий (при старом над ним) проявляется, когда место уже почти свободно
ALERT_OUT = 0.14                   # с: вытесненный третьим алерт уезжает вверх и гаснет (не пропадает резко)
GAUGE_X0, GAUGE_X1, GAUGE_Y0, GAUGE_DY = 736, 920, 1638, 44
MINI = ("fear_hz", "interest", "hearing_hz", "hunger", "mn9_hz")        # Страх, Интерес, Слух, Голод, Хоботок
TICKER_Y = 1582

# цвет «сейчас: …» по ключу formats.DOING (как цвета реакций compose.REACT)
DOING_COL = {
    "takeoff": (255, 92, 92), "startle": (255, 140, 110), "drink": (255, 205, 80), "proboscis": (255, 205, 80),
    "groom": (120, 255, 180), "listen": (185, 145, 255), "track": (80, 215, 255), "watch": None,   # None — акцент
    "seizure": C.LIFE_COL["seizure"], "dead": C.LIFE_COL["dead"], "respawn": C.LIFE_COL["respawn"],
}
DOING_HZ = {"takeoff": "gf_hz", "startle": "gf_hz", "drink": "mn9_hz", "proboscis": "mn9_hz",
            "groom": "groom_hz", "listen": "hearing_hz", "track": "interest"}
FLASH = ("takeoff", "startle", "drink", "proboscis", "groom", "seizure", "dead", "respawn")          # смена на эти — вспышка плашки
PEAK = ("takeoff", "startle")      # события: в «сейчас» — пик частоты с начала события (сглаженная уже спала)
NOW_FADE = 0.16                    # с: смена «сейчас» (плавно, без мигания на коротких состояниях)
HEAD_ON = (0.08, 0.15)             # рад: «смотрит влево/вправо» проявляется между порогами поворота головы

# алерты: ключ реакции compose.REACT -> текст стримера (те же триггеры из нейронов, слова автора).
# Алерт — момент начала реакции, «сейчас» — состояние: тексты разные, чтобы не повторять одно слово дважды
ALERT = {"takeoff": "испугалась!", "startle": "встрепенулась!", "groom": "трёт лапками",
         "track": "заметила объект", "hear": "слушает"}
ALERT_PROB = ("пробует сахар", "выпустила хоботок")     # хоботок: есть капля сахара / нет
DROP_ON = 0.03                                           # как formats.pose_track: drink при drop > 0.03

# метки групп нейронов на компактном мозге: группа -> (подпись, шкала, куда подпись от точки:
# (dx, dy) — сдвиг левого/правого края подписи; dx < 0 — подпись левее точки)
ANCHORS = {
    "lc10": ("LC10 · объект", "interest", (-16, -9)),
    "jo": ("орган Джонстона", "hearing_hz", (-30, 26)),
    "gf": ("DNp01 · взлёт", "gf_hz", (24, -46)),
    "dna": ("DNa01/02 · руль", "steer", (-24, -52)),
    "dng": ("DNg84/35 · чистка", "groom_hz", (-20, 18)),
    "mn9": ("MN9 · хоботок", "mn9_hz", (20, 22)),
}
BRAIN_HEAD_H = 30                  # сверху мозга — «МОЗГ · горят N нейронов»: метки ниже

_GM = None                         # максимумы шкал лога (configure) — для formats.pose_track


def configure(gauge_max=None):
    global _GM
    _GM = dict(gauge_max) if gauge_max else None
    C.configure(gauge_max=gauge_max)


# ---------------------------------------------------------------- таймлайн (главный процесс)
def prepare_timeline(huds, fps=30, stages=None):
    """Как compose.prepare_timeline (шкалы g_*, пики, реакции captions, событие ticker) + своё:
      now       [(ключ DOING, альфа), ...] — плавная смена «сейчас: …» по pose['doing'] (последний —
                текущий, остальные гаснут за NOW_FADE); считается по порядку кадров, поэтому здесь
      now_pk    {ключ PEAK: Гц} — пик сырой частоты (DOING_HZ) с начала события, пока плашка видна
      captions  (текст, нейроны, цвет, age, i0, ключ) — тексты алертов стримера (ALERT), триггеры те же
      captions_out  [(... как captions ..., out_age)] — алерт, который вытеснил третий (compose держит
                максимум 2): ещё ALERT_OUT с уезжает вверх и гаснет, а не пропадает за один кадр"""
    out = C.prepare_timeline(huds, fps, stages)
    n = len(huds)
    sts = list(stages) if stages is not None else [{}] * n
    tr = F.pose_track(huds, sts, fps, _GM)
    rev = {v[0]: k for k, v in C.REACT.items()}
    dt = 1.0 / fps
    step = dt / NOW_FADE
    alpha = {}
    peak = {}
    prev_caps = {}                     # (i0, текст) -> алерт прошлого кадра
    pushed = {}                        # (i0, текст) -> (алерт, кадр, когда вытеснен)

    def raw(j, k):
        try:
            return float((huds[j] or {}).get(DOING_HZ[k], 0.0) or 0.0) if j >= 0 else 0.0
        except (TypeError, ValueError):
            return 0.0

    for i, d in enumerate(out):
        tgt = tr[i]["doing"]
        for k in list(alpha):
            if k != tgt:
                alpha[k] -= step
                if alpha[k] <= 1e-3:
                    del alpha[k]
        alpha[tgt] = min(1.0, alpha.get(tgt, 0.0) + step)
        d["now"] = [(k, round(a, 4)) for k, a in alpha.items() if k != tgt] + [(tgt, round(alpha[tgt], 4))]
        # пик частоты события: сброс на его первом кадре (и кадр до него — фронт GF), дальше максимум
        if tgt in PEAK:
            if tgt not in peak or (tr[i]["doing_age"] or 0.0) < 0.5 * dt:
                peak[tgt] = max(raw(i - 1, tgt), raw(i, tgt))
            else:
                peak[tgt] = max(peak[tgt], raw(i, tgt))
        for k in list(peak):
            if k not in alpha:
                del peak[k]
        d["now_pk"] = {k: round(v, 2) for k, v in peak.items()}
        caps = []
        for c in d.get("captions") or []:
            k = rev.get(c[0])
            txt = ALERT.get(k, c[0])
            if k == "proboscis":                           # капля сахара рядом с хоботком в момент реакции
                i0 = int(c[4])
                drop = max((abs(tr[j]["drop"]) for j in range(max(0, i0), min(n, i0 + 6))), default=0.0)
                txt = ALERT_PROB[0] if drop > DROP_ON else ALERT_PROB[1]
            caps.append((txt, c[1], c[2], c[3], c[4], k))
        d["captions"] = caps
        # вытесненные: были в прошлом кадре, ещё не истекли, а сейчас их нет
        cur = {(c[4], c[0]): c for c in caps}
        for key, c in prev_caps.items():
            if key not in cur and c[3] + dt < C.CAPTION_DUR - 1e-6:
                pushed[key] = (c, i)
        for key in [key for key, (c, i1) in pushed.items() if (i - i1) * dt >= ALERT_OUT or key in cur]:
            del pushed[key]
        d["captions_out"] = [(*c[:3], c[3] + (i - i1) * dt, *c[4:], (i - i1) * dt) for c, i1 in pushed.values()]
        prev_caps = cur
    return out


# ---------------------------------------------------------------- цвета
def _c255(col):
    return tuple(int(round(255 * float(np.clip(x, 0, 1)))) for x in col)


def _lum(c):
    c = np.asarray(c, np.float32) / 255
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def _vivid(col, v_min=0.9, s_max=None):
    """Цвет 0..1 -> яркий вариант (для неона и акцентов на тёмном фоне)."""
    h, s, v = colorsys.rgb_to_hsv(*[float(np.clip(x, 0, 1)) for x in col])
    if s_max is not None:
        s = min(s, s_max)
    return colorsys.hsv_to_rgb(h, s, max(v, v_min))


def _hue_dist(a, b):
    ha, sa, _ = colorsys.rgb_to_hsv(*a)
    hb, sb, _ = colorsys.rgb_to_hsv(*b)
    d = abs(ha - hb)
    return min(d, 1 - d) * min(sa, sb) + abs(sa - sb) * 0.5


def palette(style):
    """Палитра оформления из RoomStyle: фон (тёмный, в тон стены/overlay.bg), акцент (= подсветка
    комнаты led.color), второй акцент (overlay.accent2; если почти как первый — цвет кресла/наушников)."""
    acc = tuple(style.overlay["accent"])
    cands = [tuple(style.overlay["accent2"]), tuple(style.chair["accent"]), tuple(style.headphones["color"]),
             tuple(style.chair["primary"])]
    acc2 = cands[0]
    if _hue_dist(acc, acc2) < 0.12:
        acc2 = max(cands, key=lambda c: _hue_dist(acc, c) + 0.3 * colorsys.rgb_to_hsv(*c)[1])
    if _hue_dist(acc, acc2) < 0.12:                           # всё в один тон — сдвигаем оттенок
        h, s, v = colorsys.rgb_to_hsv(*acc)
        acc2 = colorsys.hsv_to_rgb((h + 0.5) % 1, max(s, 0.6), 1.0)
    bg = np.asarray(style.overlay["bg"], np.float32)
    bg = bg * min(1.0, 0.075 / max(1e-3, float(bg.max())))  # всегда тёмный: белый текст читается
    bg = bg * 0.8 + np.asarray(acc, np.float32) * 0.018
    a = _c255(_vivid(acc, 0.92))
    b = _c255(_vivid(acc2, 0.92))
    return dict(bg=np.asarray(_c255(bg), np.float32), acc=a, acc2=b, led=_c255(_vivid(style.led["color"], 0.95)),
                led_mode=style.led.get("mode", "solid"), led_speed=float(style.led.get("speed", 0.3)),
                pattern=style.wall.get("pattern", "plain"))


def _on(col):
    """Цвет текста на заливке col."""
    return (18, 16, 26) if _lum(col) > 0.6 else (255, 255, 255)


def _tint(col, k, base=(255, 255, 255)):
    return tuple(int(round(b + (c - b) * k)) for c, b in zip(col, base))


# ---------------------------------------------------------------- рисование
def _rrect(w, h, r, ss=3):
    im = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=r * ss, fill=255)
    return im.resize((w, h), Image.LANCZOS)


def _text_layer(text, size, weight, color, track=0, stroke=0, stroke_color=(0, 0, 0)):
    """-> PIL RGBA с текстом (track — разрядка, px)."""
    text = C.safe_text(text)
    f = C.font(size, weight)
    asc, desc = f.getmetrics()
    if track:
        ws = [f.getlength(ch) for ch in text]
        tw = int(sum(ws) + track * max(0, len(text) - 1)) + 2 * stroke + 2
    else:
        ws = None
        tw = int(f.getlength(text)) + 2 * stroke + 2
    im = Image.new("RGBA", (max(1, tw), asc + desc + 2 * stroke), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if ws is None:
        d.text((stroke, stroke), text, font=f, fill=(*color, 255), stroke_width=stroke,
               stroke_fill=(*stroke_color, 255))
    else:
        x = stroke
        for ch, w_ in zip(text, ws):
            d.text((x, stroke), ch, font=f, fill=(*color, 255), stroke_width=stroke, stroke_fill=(*stroke_color, 255))
            x += w_ + track
    return im


def _card(w, h, r, fill, border=None, bw=1):
    """Скруглённая подложка -> PIL RGBA."""
    ss = 3
    im = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=r * ss, fill=tuple(fill),
                        outline=tuple(border) if border else None, width=bw * ss)
    return im.resize((w, h), Image.LANCZOS)


def _glow_card(w, h, r, col, blur=9, gain=1.7, pad=24, width=3):
    return C._glow_rect(w, h, r, col, width=width, blur=blur, gain=gain, pad=pad)


def _ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def _ease_back(x):
    x = min(1.0, max(0.0, x))
    c = 1.9
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


def now_vis(now):
    """Видимость текстов «сейчас: …» по альфам hud['now'] (последний — текущий): уходящие видны первую
    половину своего угасания, текущий проявляется, когда самый яркий уходящий погас наполовину.
    При обычной смене это то же «перелистывание», при быстрой смене туда-обратно плашка не пустеет."""
    if not now:
        return []
    others = [float(a) for _, a in now[:-1]]
    vis = [min(1.0, max(0.0, 2 * a - 1)) for a in others]
    if others:
        vis.append(min(1.0, max(0.0, 1 - 2 * max(others))))
    else:                                                     # начало лога: плавно вместе с подложкой
        vis.append(min(1.0, 1.4 * float(now[-1][1])))
    return vis


def _pattern(kind, w, h, seed):
    """Узор стены комнаты (style.wall.pattern) — очень слабый, на фоне оформления. -> float 0..1 (h, w)."""
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    if kind == "panels":
        for x in range(0, w + 1, 120):
            d.line((x, 0, x, h), fill=255, width=2)
        for y in range(0, h + 1, 240):
            d.line((0, y, w, y), fill=150, width=2)
    elif kind == "foam":                                      # акустический поролон: клинья
        s = 44
        for j, y in enumerate(range(0, h + s, s)):
            for i, x in enumerate(range(0, w + s, s)):
                if (i + j) % 2:
                    d.polygon([(x, y), (x + s, y), (x + s // 2, y + s // 2)], fill=150)
                else:
                    d.polygon([(x, y), (x, y + s), (x + s // 2, y + s // 2)], fill=90)
    elif kind == "brick":
        bh, bw_ = 40, 110
        for j, y in enumerate(range(0, h + bh, bh)):
            d.line((0, y, w, y), fill=255, width=2)
            off = (bw_ // 2) * (j % 2)
            for x in range(-off, w + bw_, bw_):
                d.line((x, y, x, y + bh), fill=255, width=2)
    elif kind == "hex":
        r = 30
        dx, dy = r * 1.732, r * 1.5
        for j in range(int(h / dy) + 2):
            for i in range(int(w / dx) + 2):
                cx = i * dx + (dx / 2 if j % 2 else 0)
                cy = j * dy
                pts = [(cx + r * np.cos(np.pi / 6 + k * np.pi / 3), cy + r * np.sin(np.pi / 6 + k * np.pi / 3))
                       for k in range(7)]
                d.line(pts, fill=255, width=2)
    else:                                                     # plain: мягкий диагональный шум
        rng = np.random.default_rng(seed)
        z = rng.random((h // 24 + 2, w // 24 + 2)).astype(np.float32)
        z = cv2.resize(z, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.clip(z, 0, 1) * 0.5
    return np.asarray(im, np.float32) / 255


class Composer:
    flyeye = None          # хук: объект с render(frame_rgb, feats_row, (w, h)) -> RGBA; None — без врезки

    def __init__(self, fmt=None):
        from style import RoomStyle
        self.fmt = dict(fmt or {})
        self.variant = self.fmt.get("variant")
        self.style = RoomStyle(self.variant or 0)
        self.pal = palette(self.style)
        self.watermark = (self.fmt.get("watermark") or "").strip()
        self.tc = C.TextCache()
        self.brain = C.BrainCloudSparse(*BRAIN_SIZE)
        self._key = None
        self._bg = None
        self._cards = {}
        self._wm = {}
        self._build_cam()
        self._build_gauges()

    # ------------------------------------------------------------ статичные слои
    def _build_bg(self, title, fly_no):
        P = self.pal
        bg = np.empty((H, W, 3), np.float32)
        bg[:] = P["bg"]
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        bg *= 1.15 - 0.35 * yy
        pat = _pattern(P["pattern"], W, H, self.variant or 0)
        bg += pat[..., None] * (np.asarray(P["acc"], np.float32) * 0.035 + 5.0)
        # мягкое свечение подсветки по углам (как свет LED в комнате)
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        for (cx, cy, rx, ry, col, k) in ((0, 0, 620, 300, P["acc"], 0.16), (W, 190, 520, 260, P["acc2"], 0.10),
                                         (0, 1570, 700, 260, P["acc2"], 0.10), (W, H, 700, 340, P["acc"], 0.12)):
            g = np.exp(-(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2))
            bg += g[..., None] * np.asarray(col, np.float32) * k
        del ys, xs
        pil = Image.fromarray(bg.clip(0, 255).astype(np.uint8)).convert("RGBA")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        # --- верх: LIVE + «Муха №N» + «смотрит: …» ---
        ly = 134
        live = _card(128, 48, 12, (232, 36, 62, 255))
        ov.alpha_composite(live, (40, ly))
        tl = _text_layer("LIVE", 30, "Bold", (255, 255, 255), track=2)
        ov.alpha_composite(tl, (40 + 44, ly + (48 - tl.height) // 2 + 1))
        self.live_dot = (40 + 24, ly + 24)
        x = 40 + 128 + 18
        who = f"Муха №{fly_no}" if fly_no is not None else "Муха"
        wl = _text_layer(who, 38, "Bold", (255, 255, 255))
        ov.alpha_composite(wl, (x, ly + (48 - wl.height) // 2 + 1))
        x += wl.width + 18
        lim = SAFE_R
        sl = _text_layer("смотрит:", 25, "SemiLight", _tint(P["acc"], 0.45))
        ov.alpha_composite(sl, (x, ly + (48 - sl.height) // 2 + 3))
        x += sl.width + 9
        tt = C.safe_text(title or "").strip() or "ролик"
        size = 27
        f = C.font(size, "SemiBold")
        while size > 20 and x + f.getlength(tt) > lim:
            size -= 1
            f = C.font(size, "SemiBold")
        if x + f.getlength(tt) > lim:
            while tt and x + f.getlength(tt + "…") > lim:
                tt = tt[:-1]
            tt = tt.rstrip() + "…"
        ttl = _text_layer(tt, size, "SemiBold", _tint(P["acc2"], 0.3))
        ov.alpha_composite(ttl, (x, ly + (48 - ttl.height) // 2 + 3))
        # время эфира: справа (выше кнопок TikTok — там можно)
        self.clock_box = (944, ly, 1052, ly + 48)
        ov.alpha_composite(_card(108, 48, 12, (0, 0, 0, 120), border=(*P["acc"], 120), bw=2), (944, ly))
        # полоска под верхом
        seam = C._grad(W, [P["acc"], P["acc2"], P["acc"]])
        seam_a = np.zeros((3, W, 4), np.uint8)
        seam_a[..., :3] = seam.astype(np.uint8)
        seam_a[..., 3] = 180
        ov.alpha_composite(Image.fromarray(seam_a), (0, CLIP[1] - 3))
        # --- низ: событие, мозг, шкалы, честность ---
        by = BRAIN[1]
        seam_a[..., 3] = 150
        ov.alpha_composite(Image.fromarray(seam_a[:2]), (0, by))
        pill = _card(128, 36, 18, (*P["acc2"], 235))
        ov.alpha_composite(pill, (40, TICKER_Y + 2))
        tl = _text_layer("СОБЫТИЕ", 17, "Bold", _on(P["acc2"]), track=2)
        ov.alpha_composite(tl, (40 + (128 - tl.width) // 2, TICKER_Y + 2 + (36 - tl.height) // 2 + 1))
        self.ticker_x = 40 + 128 + 14
        bx, byy = BRAIN_XY
        ml = _text_layer("МОЗГ", 19, "Bold", (255, 255, 255), track=2)
        ov.alpha_composite(ml, (bx + 18, byy + 4))
        self.count_xy = (bx + 18 + ml.width + 10, byy + 5)
        # мини-шкалы
        for n, key in enumerate(MINI):
            g = self.gauge[key]
            y = GAUGE_Y0 + n * GAUGE_DY
            ov.alpha_composite(_text_layer(g["lab"], 18, "SemiBold", (236, 238, 246)), (GAUGE_X0, y))
            d.rounded_rectangle((GAUGE_X0, y + 27, GAUGE_X1, y + 27 + 7), radius=3, fill=(255, 255, 255, 26))
            d.rectangle((GAUGE_X0 - 10, y + 5, GAUGE_X0 - 7, y + 34), fill=(*g["col"], 220))
        # строка честности (мелко)
        hy = 1866
        for line, col in ((C.HONESTY, (150, 156, 178)),
                          (f"кресло, наушники и комната — декор (вариант {self.variant if self.variant is not None else 0})"
                           " · подписи действий — только из нейронов", (120, 126, 148))):
            size = 15
            while size > 11 and C.font(size).getlength(line) > SAFE_R - 40:
                size -= 1
            ov.alpha_composite(_text_layer(line, size, "Regular", col), (40, hy))
            hy += 22
        pil.alpha_composite(ov)
        img = np.asarray(pil.convert("RGB")).copy()
        # тусклое облако всех нейронов под мозгом
        b = self.brain
        reg = img[byy:byy + b.h, bx:bx + b.w].astype(np.float32)
        img[byy:byy + b.h, bx:bx + b.w] = (reg + b.base * 255 * (1 - reg / 510)).clip(0, 255)
        self._bg = img
        # углы окна «вебки» (вне скругления — фон)
        x0, y0, x1, y1 = CAM_WIN
        R = CAM_R
        m = np.asarray(_rrect(x1 - x0, y1 - y0, R), np.float32) / 255
        self.cam_corners = []
        for (cx, cy) in ((x0, y0), (x1 - R, y0), (x0, y1 - R), (x1 - R, y1 - R)):
            mm = m[cy - y0:cy - y0 + R, cx - x0:cx - x0 + R][..., None]
            self.cam_corners.append((cx, cy, mm, img[cy:cy + R, cx:cx + R].astype(np.float32) * (1 - mm)))

    def _build_cam(self):
        """Неоновая рамка окна «вебки»: полосы вокруг окна (яркость uint8) и угол по периметру (0..255),
        цвет на кадр — через LUT по режиму подсветки комнаты (solid / breath / wave / rainbow)."""
        x0, y0, x1, y1 = CAM_WIN
        pad = 26
        fw, fh = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
        m = Image.new("L", (fw, fh), 0)
        ImageDraw.Draw(m).rounded_rectangle((pad, pad, pad + x1 - x0 - 1, pad + y1 - y0 - 1), radius=CAM_R,
                                            outline=255, width=4)
        core = np.asarray(m, np.float32) / 255
        glow = np.asarray(m.filter(ImageFilter.GaussianBlur(8)), np.float32) / 255
        inten = np.clip(core * 0.95 + glow * 2.2, 0, 1)
        ys, xs = np.mgrid[0:fh, 0:fw].astype(np.float32)
        cx, cy = fw / 2, fh / 2
        ang = (np.arctan2((ys - cy) / (fh / 2), (xs - cx) / (fw / 2)) / (2 * np.pi)) % 1.0
        theta = (ang * 255.99).astype(np.uint8)
        # полосы: верх, низ, лево, право (внутренность окна не трогаем)
        b = pad + CAM_R + 6
        gx0, gy0 = x0 - pad, y0 - pad
        bands = [(0, 0, fw, b), (0, fh - b, fw, fh), (0, b, b, fh - b), (fw - b, b, fw, fh - b)]
        self.cam_bands = []
        for (a0, c0, a1, c1) in bands:
            X0, Y0 = gx0 + a0, gy0 + c0
            sx0, sy0 = max(0, -X0), max(0, -Y0)
            X1, Y1 = min(W, gx0 + a1), min(H, gy0 + c1)
            it = inten[c0 + sy0:c0 + sy0 + (Y1 - Y0 - sy0), a0 + sx0:a0 + sx0 + (X1 - X0 - sx0)]
            th = theta[c0 + sy0:c0 + sy0 + it.shape[0], a0 + sx0:a0 + sx0 + it.shape[1]]
            self.cam_bands.append((X0 + sx0, Y0 + sy0, np.repeat((it * 255).astype(np.uint8)[..., None], 3, -1),
                                   np.ascontiguousarray(np.repeat(th[..., None], 3, -1))))
        self._lut = np.zeros((1, 256, 3), np.uint8)
        self._lut_ang = np.linspace(0, 1, 256, endpoint=False, dtype=np.float32)

    def _build_gauges(self):
        self.gauge = {}
        for g in C.GAUGES:
            if g[0] in MINI:
                c = np.array(g[3], np.float32)
                bw = GAUGE_X1 - GAUGE_X0
                gr = C._grad(bw, [c * 0.55, c, np.minimum(c * 1.2 + 60, 255)])
                self.gauge[g[0]] = dict(lab=g[1], col=g[3], kind=g[5], hot=g[6],
                                        bar=np.broadcast_to(gr[None], (7, bw, 3)).astype(np.float32))
        self.bar_mask = np.asarray(_rrect(GAUGE_X1 - GAUGE_X0, 7, 3), np.float32) / 255

    def _ensure(self, hud):
        key = (hud.get("title"), hud.get("fly_no"))
        if key != self._key:
            self._key = key
            self._build_bg(*key)

    # ------------------------------------------------------------ ролик
    def _watermark_sprite(self, size, room):
        """Как stage._watermark: размер 8 % меньшей стороны ролика; длинный ник — мельче, чтобы влез."""
        k = (self.watermark, size, room)
        if k not in self._wm:
            f = C.font(size, "Regular")
            tl = f.getlength(self.watermark)
            if tl > room - int(size * 0.35) - 8:
                size = max(8, int(size * (room - int(size * 0.35) - 8) / tl))
                f = C.font(size, "Regular")
                tl = f.getlength(self.watermark)
            tw = int(tl) + 8
            th = int(size * 1.35)
            im = Image.new("L", (tw, th), 0)
            ImageDraw.Draw(im).text((4, 0), self.watermark, font=f, fill=255)
            a = np.asarray(im).astype(np.float32) / 255
            sh = cv2.GaussianBlur(a, (0, 0), max(1.0, size * 0.06))
            # тёмный контур — только на светлом фоне ролика (белое на белом иначе не видно)
            r = max(1, int(round(size * 0.05)))
            ol = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            ol = cv2.GaussianBlur(ol, (0, 0), max(0.8, r * 0.6))
            self._wm[k] = (a, sh, ol, int(size * 0.35))
        return self._wm[k]

    def _clip(self, out, rgb):
        """-> (ox, oy, nw, nh) видимой картинки ролика."""
        x0, y0, x1, y1 = CLIP
        cw, chh = x1 - x0, y1 - y0
        if rgb is None:                                             # нет кадра: заглушка (вотермарка всё равно)
            out[y0:y1, x0:x1] = (self._bg[y0:y1, x0:x1] * 0.6).astype(np.uint8)
            sp = self.tc.get("нет кадра ролика", 26, "Regular", (150, 150, 170))
            C.blit(out, sp, (W - sp.w) / 2, (y0 + y1 - sp.h) / 2)
            ox, oy, nw, nh = x0, y0, cw, chh
        else:
            rgb = rgb[..., :3]
            fh, fw = rgb.shape[:2]
            s = min(cw / fw, chh / fh)
            nw, nh = min(cw, max(2, int(round(fw * s)))), min(chh, max(2, int(round(fh * s))))
            if nw < cw - 1 or nh < chh - 1:                        # поля — размытый тот же кадр
                fill = cv2.resize(rgb, (cw // 12, chh // 12), interpolation=cv2.INTER_AREA)
                fill = cv2.GaussianBlur(fill, (0, 0), 2.2)
                cv2.resize(fill, (cw, chh), dst=out[y0:y1, x0:x1], interpolation=cv2.INTER_LINEAR)
                cv2.convertScaleAbs(out[y0:y1, x0:x1], dst=out[y0:y1, x0:x1], alpha=0.42)
            ox, oy = x0 + (cw - nw) // 2, y0 + (chh - nh) // 2
            cv2.resize(rgb, (nw, nh), dst=out[oy:oy + nh, ox:ox + nw],
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if self.watermark:                                          # как stage._compose_canvas
            right = min(ox + nw, SAFE_R - 6) if oy + nh > TIKTOK_BTN_Y else ox + nw   # не под кнопками TikTok
            a, sh, ol, pad = self._watermark_sprite(max(14, int(0.08 * min(nw, nh))), int(0.9 * (right - ox)))
            th, tw = a.shape
            wy = max(oy, oy + nh - th - pad // 2)
            wx = max(ox, (ox + nw - tw - pad) if right == ox + nw else right - tw)
            a, sh, ol = a[:oy + nh - wy, :W - wx], sh[:oy + nh - wy, :W - wx], ol[:oy + nh - wy, :W - wx]
            th, tw = a.shape
            roi = out[wy:wy + th, wx:wx + tw].astype(np.float32)
            # светлый фон под знаком (по пикселям, сглажено) -> сила контура 0..1
            lum = cv2.GaussianBlur(roi.mean(axis=2), (0, 0), 3.0) / 255
            kb = np.clip((lum - 0.55) / 0.3, 0, 1)
            roi *= 1 - np.minimum(0.82, 0.45 * sh + 0.62 * ol * kb * kb * (3 - 2 * kb))[..., None]
            roi = roi * (1 - 0.65 * a[..., None]) + 255 * 0.65 * a[..., None]
            out[wy:wy + th, wx:wx + tw] = np.clip(roi, 0, 255).astype(np.uint8)
        return ox, oy, nw, nh

    def _gaze(self, out, rect, head):
        """«смотрит сюда»: мягкая подсветка края ролика, куда повёрнута голова (pose['head'], + = влево)."""
        k = (abs(head) - HEAD_ON[0]) / (HEAD_ON[1] - HEAD_ON[0])
        if k <= 0.02:
            return
        k = min(1.0, k)
        ox, oy, nw, nh = rect
        bw = min(150, nw // 3)
        key = ("gaze", bw, nh)
        if key not in self._cards:
            ramp = (1 - np.linspace(0, 1, bw, dtype=np.float32)) ** 2.2
            g = np.broadcast_to(ramp[None, :, None], (nh, bw, 1)) * np.asarray(self.pal["acc"], np.float32) * 0.55
            self._cards[key] = np.ascontiguousarray(g.astype(np.uint8))
        g = self._cards[key]
        left = head > 0
        x = ox if left else ox + nw - bw
        gg = g if left else g[:, ::-1]
        reg = out[oy:oy + nh, x:x + bw]
        cv2.scaleAdd(gg, k, reg, dst=reg)
        sp = self.tc.get("смотрит сюда", 20, "SemiBold", _on(self.pal["acc"]), pill=(*self.pal["acc"], 225))
        ty = oy + 14
        tx = ox + 16 if left else ox + nw - sp.w - 16
        C.blit(out, sp, tx, ty, k)

    # ------------------------------------------------------------ фейскам
    def _facecam(self, out, scene):
        x0, y0, x1, y1 = CAM_WIN
        if scene is None:
            out[y0:y1, x0:x1] = 12
        else:
            s = scene[..., :3]
            if s.shape[:2] != (SCENE_H, SCENE_W):
                s = cv2.resize(np.ascontiguousarray(s), (SCENE_W, SCENE_H), interpolation=cv2.INTER_AREA)
            out[y0:y1, x0:x1] = s[y0 - FACECAM[1]:y1 - FACECAM[1], x0 - FACECAM[0]:x1 - FACECAM[0]]
        R = CAM_R
        for (cx, cy, mm, bgm) in self.cam_corners:
            reg = out[cy:cy + R, cx:cx + R]
            reg[:] = (reg * mm + bgm).astype(np.uint8)

    def _led_frame(self, out, t):
        """Неоновая рамка окна: цвет/режим — подсветка комнаты (style.led), анимация только по времени."""
        P = self.pal
        col = np.asarray(P["led"], np.float32)
        mode, sp = P["led_mode"], max(0.05, P["led_speed"])
        a = self._lut_ang
        if mode == "breath":
            g = np.full_like(a, 0.55 + 0.45 * np.sin(2 * np.pi * sp * t))
            lut = g[:, None] * col
        elif mode == "wave":                                        # бегущий блик по периметру
            ph = (sp * 0.5 * t) % 1.0
            d = np.abs(((a - ph + 0.5) % 1.0) - 0.5)
            d2 = np.abs(((a - ph) % 1.0) - 0.5)
            g = 0.42 + 0.9 * np.exp(-(d / 0.07) ** 2) + 0.45 * np.exp(-(d2 / 0.07) ** 2)
            lut = np.minimum(g[:, None] * col, 255)
        elif mode == "rainbow":
            ph = (sp * 0.35 * t) % 1.0
            h = (a + ph) % 1.0
            lut = np.array([colorsys.hsv_to_rgb(float(x), 0.75, 1.0) for x in h[::8]], np.float32).repeat(8, 0) * 255
        else:
            lut = np.broadcast_to(col, (256, 3))
        self._lut[0] = np.clip(lut, 0, 255).astype(np.uint8)
        for (X, Y, inten, th) in self.cam_bands:
            hh, ww = inten.shape[:2]
            cm = cv2.LUT(th, self._lut)
            cv2.multiply(cm, inten, dst=cm, scale=1 / 255)
            reg = out[Y:Y + hh, X:X + ww]
            cv2.add(reg, cm, dst=reg)

    # ------------------------------------------------------------ «сейчас: …»
    def _now_col(self, k):
        c = DOING_COL.get(k)
        return c if c is not None else self.pal["acc"]

    def _now_texts(self, k):
        key = ("now", k)
        if key not in self._cards:
            text, sub = F.DOING.get(k, F.DOING["watch"])
            big = C.Sprite(np.asarray(_text_layer(text, 44, "Bold", (255, 255, 255))))
            self._cards[key] = (big, sub)
        return self._cards[key]

    def _now_bg(self, w):
        w = int(np.clip(round(w / 4) * 4, NOW_MIN_W, NOW_MAX_W))
        key = ("nowbg", w)
        if key not in self._cards:
            im = _card(w, NOW_H, 20, (8, 8, 14, 168), border=(255, 255, 255, 30), bw=1)
            lab = _text_layer("СЕЙЧАС", 17, "SemiBold", self.pal["acc"], track=3)
            im.alpha_composite(lab, (26, 12))
            self._cards[key] = (C.Sprite(np.asarray(im)), w)
        return self._cards[key]

    def _now(self, out, hud):
        p = hud.get("pose") or {}
        now = hud.get("now") or [(p.get("doing", "watch"), 1.0)]
        items = []
        for k, a in now:
            big, sub = self._now_texts(k)
            hk = DOING_HZ.get(k)
            pk = (hud.get("now_pk") or {}).get(k) if k in PEAK else None
            if k == "seizure":                                        # сколько нейронов горит — из лога
                sub = C.life_texts(dict(hud, life="seizure"))[3]
            elif pk is not None:
                sub = f"{sub} · пик {pk:.0f} Гц"
            elif hk is not None:
                v = float(hud.get("g_" + hk, hud.get(hk, 0.0)) or 0.0)
                sub = f"{sub} · {v:.0f} Гц"
            sp = self.tc.get(sub, 21, "Regular", _tint(self._now_col(k), 0.35))
            items.append((k, float(a), big, sp))
        wsum = sum(a for _, a, _, _ in items) or 1.0
        wt = sum(a * max(b.w, s.w + 8) for _, a, b, s in items) / wsum + 26 + 26
        vis_all = now_vis([(k, a) for k, a, _, _ in items])
        bgsp, w = self._now_bg(wt)
        x, y = NOW_XY
        # вспышка при смене на событие (по pose['doing_age'])
        dk, dage = p.get("doing"), p.get("doing_age")
        if dk in FLASH and dage is not None and dage < 0.6:
            key = ("nowglow", w, dk)
            if key not in self._cards:
                self._cards[key] = _glow_card(w, NOW_H, 20, self._now_col(dk))
            C.add_glow(out, self._cards[key], x - 24, y - 24, (1 - dage / 0.6) ** 1.5)
        C.blit(out, bgsp, x, y, min(1.0, wsum * 1.4))
        col = np.zeros(3, np.float32)
        for k, a, _, _ in items:
            col += np.asarray(self._now_col(k), np.float32) * a / wsum
        out[y + 16:y + NOW_H - 16, x + 10:x + 15] = col.astype(np.uint8)
        # смена текста «перелистыванием»: старый уезжает вверх и гаснет, потом новый въезжает снизу
        tgt = items[-1][0]
        box = out[y + 30:y + NOW_H - 6, x + 16:x + w - 6]
        for (k, a, big, sp), vis in zip(items, vis_all):
            if vis <= 0.01:
                continue
            dy = (1 - _ease_out(vis)) * (26 if k == tgt else -26)
            C.blit(box, big, 10, 2 + dy, vis)
            C.blit(box, sp, 11, 56 + dy, vis)
        # «смотрит влево/вправо» — поворот головы куклы (руль DNa01/02), + = влево
        head = float(p.get("head", 0.0) or 0.0)
        k = (abs(head) - HEAD_ON[0]) / (HEAD_ON[1] - HEAD_ON[0])
        if k > 0.02:
            txt = "смотрит влево" if head > 0 else "смотрит вправо"
            hs = self.tc.get(txt, 19, "SemiBold", (255, 255, 255), pill=(*self.pal["acc"], 78))
            C.blit(out, hs, x + w - hs.w - 12, y + 6, min(1.0, k))

    # ------------------------------------------------------------ приступ / смерть / новая муха
    def _life(self, out, hud):
        lt_ = C.life_texts(hud)
        if lt_ is None:
            return
        life, lt, big, small = lt_
        col = C.LIFE_COL[life]
        if life == "seizure":                                         # текст — в «СЕЙЧАС»; рамка вебки пульсирует
            x0, y0, x1, y1 = CAM_WIN
            k = (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * lt)) * min(1.0, 0.35 + lt / 2.6)
            reg = out[y0:y1, x0:x1]
            e = 48
            ed = np.zeros(reg.shape[:2], np.float32)
            ramp = np.linspace(1, 0, e, dtype=np.float32)
            ed[:, :e] = ramp[None]
            ed[:, -e:] = np.maximum(ed[:, -e:], ramp[::-1][None])
            ed[:e] = np.maximum(ed[:e], ramp[:, None])
            ed[-e:] = np.maximum(ed[-e:], ramp[::-1][:, None])
            reg[:] = np.clip(reg + ed[..., None] * (k * 0.6) * np.array(col, np.float32), 0, 255).astype(np.uint8)
            return
        a = min(1.0, lt / 0.25)
        if life == "respawn":                                         # «заступает» — когда новую муху ставят
            a = 0.0 if lt < 1.9 else min(1.0, (lt - 1.9) / 0.2)
        if a <= 0:
            return
        cy = CAM_WIN[1] + 250
        sp = self.tc.get(big, 76, "Bold", (255, 255, 255), stroke=5, stroke_color=(18, 10, 28), glow=12,
                         glow_color=col, glow_gain=2.3)
        if sp.w > SAFE_R - 60:
            sp = sp.scaled((SAFE_R - 60) / sp.w)
        C.blit(out, sp, W / 2 - sp.w / 2, cy - sp.h / 2, a)
        sb = self.tc.get(small, 23, "SemiBold", _tint(col, 0.35), pill=(8, 8, 14, 200))
        if sb.w > W - 60:
            sb = sb.scaled((W - 60) / sb.w)
        C.blit(out, sb, W / 2 - sb.w / 2, cy + sp.h / 2 + 6, a)

    # ------------------------------------------------------------ алерты
    def _alert_sprite(self, text, sub, col):
        key = ("alert", text, sub, tuple(col))
        if key not in self._cards:
            big = _text_layer(text, 52, "Bold", (255, 255, 255))
            sl = _text_layer(sub, 19, "SemiBold", _tint(col, 0.2)) if sub else None
            w = max(big.width, sl.width if sl else 0) + 34 + 24
            h = ALERT_H if sl else 84                                 # два алерта с зазором — ровно в y 1330..1560
            im = _card(w, h, 18, (10, 10, 18, 205), border=(*col, 150), bw=2)
            d = ImageDraw.Draw(im)
            d.rounded_rectangle((12, 14, 18, h - 14), radius=3, fill=(*col, 255))
            im.alpha_composite(big, (32, 8 if sl else 10))
            if sl:
                im.alpha_composite(sl, (33, 69))
            self._cards[key] = (C.Sprite(np.asarray(im)), _glow_card(w, h, 18, col))
        return self._cards[key]

    def _face_rect(self, hud):
        """Лицо мухи на кадре — алерты его не закрывают. fly_xy (StreamerStage.fly_screen_xy — лицо) +
        размер по fly_box; нет fly_xy — верх fly_box."""
        fb, xy = hud.get("fly_box"), hud.get("fly_xy")
        if xy is not None:
            bh = float(fb[3] - fb[1]) if fb is not None else 360.0
            hw, up, dn = max(80.0, 0.2 * bh), max(90.0, 0.26 * bh), max(70.0, 0.18 * bh)
            fx0, fy0, fx1, fy1 = xy[0] - hw, xy[1] - up, xy[0] + hw, xy[1] + dn
        elif fb is not None:
            fx0, fy0, fx1, fy1 = [float(v) for v in fb]
            fy1 = fy0 + 0.45 * (fy1 - fy0)
        else:
            return None
        return fx0 + FACECAM[0], fy0 + FACECAM[1], fx1 + FACECAM[0], fy1 + FACECAM[1]

    def _alerts(self, out, hud):
        caps = [c for c in (hud.get("captions") or []) if 0 <= c[3] <= C.CAPTION_DUR]
        if not caps:
            return
        caps = sorted(caps, key=lambda c: c[3])[:2]                 # свежая — снизу, старая над ней
        gone = [c for c in (hud.get("captions_out") or []) if 0 <= c[-1] < ALERT_OUT][:1]
        face = self._face_rect(hud)
        two = len(caps) > 1 or bool(gone)
        lift = 0.0
        y = ALERT_Y1
        draw = []
        for n, c in enumerate(caps + gone):
            text, sub, col, age = c[0], c[1], tuple(int(v) for v in c[2]), float(c[3])
            sp, glow = self._alert_sprite(text, sub, col)
            if n == 0:                                                # старые плавно уезжают вверх
                y = ALERT_Y1 - sp.h
                lift = (sp.h + ALERT_GAP) * _ease_out(age / ALERT_LIFT)
            elif n == 1:
                y = ALERT_Y1 - sp.h - lift
            else:                                                     # вытесненная — над старой, едет с ней
                y = y - ALERT_GAP - sp.h
            x = ALERT_X
            if face is not None and x < face[2] and x + sp.w > face[0] and y < face[3] and y + sp.h > face[1]:
                x2 = face[2] + 16                                     # лицо: правее его, если влезает
                x = x2 if x2 + sp.w <= ALERT_MAX_X else max(24, face[0] - 16 - sp.w)
            # свежая при старой над ней — проявляется, когда место почти освободилось
            ae = age - ALERT_WAIT if (n == 0 and two) else age
            e = _ease_back(ae / 0.24)
            a = min(1.0, max(0.0, ae) / 0.1, (C.CAPTION_DUR - age) / 0.35)
            dx = -(1 - e) * (sp.w * 0.35) - max(0.0, 0.35 - (C.CAPTION_DUR - age)) * 60
            if n >= len(caps):                                        # вытесненная: гаснет, пока её поднимают
                a = min(a, 1 - _ease_out(float(c[-1]) / ALERT_OUT))
            draw.append((x + dx, y, sp, glow, a, max(0.0, ae)))
        for x, y, sp, glow, a, age in reversed(draw):                 # старая первой, свежая поверх
            if age < 0.6 and a > 0.003:
                C.add_glow(out, glow, int(x) - 24, int(y) - 24, (1 - age / 0.6) ** 1.5 * 1.2 * min(1.0, a * 2))
            C.blit(out, sp, x, y, a)

    # ------------------------------------------------------------ низ: событие, мозг, шкалы
    def _ticker(self, out, hud, t):
        x0 = self.ticker_x
        x1 = SAFE_R
        ty = TICKER_Y + 20
        tk = hud.get("ticker")
        if not tk:
            sp = self.tc.get("все реакции мухи — из активности её нейронов", 19, "Regular", (140, 146, 170))
            C.blit(out, sp, x0, ty - sp.h / 2)
            return
        text, age = tk
        te = max(0.0, float(t) - float(age))
        stamp = f"{int(te // 60)}:{int(te % 60):02d}"
        ss = self.tc.get(stamp, 20, "SemiBold", _tint(self.pal["acc"], 0.25))
        slide = max(0.0, 1 - age / 0.25) ** 2
        a = min(1.0, age / 0.2) if age < 6 else max(0.55, 1 - (age - 6) / 3)
        C.blit(out, ss, x0, ty - ss.h / 2, a)
        xt = x0 + ss.w + 10
        sp = self.tc.get(text, 22, "SemiBold", (255, 255, 255))
        if sp.w > x1 - xt:
            sp = sp.scaled(max(0.7, (x1 - xt) / sp.w))
        flash = max(0.0, 1 - age / 0.7)
        if flash > 0:
            hl = self.tc.get(text, 22, "SemiBold", (255, 255, 255), glow=7, glow_color=self.pal["acc2"])
            hl = hl.scaled(sp.w / max(1, self.tc.get(text, 22, "SemiBold", (255, 255, 255)).w))
            C.blit(out[:, :x1], hl, xt + 30 * slide - (hl.w - sp.w) / 2, ty - hl.h / 2, flash)
        C.blit(out[:, :x1], sp, xt + 30 * slide, ty - sp.h / 2, a * (1 - flash * 0.5))

    def _brain(self, out, hud):
        b = self.brain
        bx, by = BRAIN_XY
        reg = out[by:by + b.h, bx:bx + b.w]
        idx, act = hud.get("act_idx"), hud.get("act_val")
        n_pts = b.draw(reg, idx, act)
        placed = [(0, 0, 330, BRAIN_HEAD_H)]                          # заголовок «МОЗГ · горят …»
        for k, (lab, key, (dx, dy)) in ANCHORS.items():
            pts = b.anchors.get(k)
            if pts is None or not len(pts):
                continue
            v = float(hud.get("g_" + key, hud.get(key, 0.0)) or 0.0)
            fr = abs(v) if key == "steer" else v / C.GAUGE_MAX[key]
            a = min(1.0, max(0.0, (fr - 0.35) / 0.25))
            if a <= 0.02:
                continue
            col = next(g[3] for g in C.GAUGES if g[0] == key)
            for (px, py) in pts:
                cv2.circle(reg, (int(px), int(py)), 9, tuple(int(c * a) for c in col), 2, cv2.LINE_AA)
            px, py = pts[0]
            sp = self.tc.get(lab, 15, "SemiBold", col, pill=(8, 9, 16, 210))
            lx = px + dx if dx > 0 else px + dx - sp.w
            ly = py + dy - sp.h / 2
            for _ in range(6):                                        # не на заголовок и не на другие метки
                lx = float(np.clip(lx, 0, b.w - sp.w))
                ly = float(np.clip(ly, BRAIN_HEAD_H, b.h - sp.h))
                hit = [r for r in placed if lx < r[2] and lx + sp.w > r[0] and ly < r[3] and ly + sp.h > r[1]]
                if not hit:
                    break
                ly = hit[0][3] + 2 if dy >= 0 or hit[0][3] + 2 + sp.h <= b.h else hit[0][1] - sp.h - 2
            placed.append((lx, ly, lx + sp.w, ly + sp.h))
            ex = lx + sp.w if lx + sp.w < px else (lx if lx > px else lx + sp.w / 2)
            ey = ly + sp.h / 2
            cv2.line(reg, (int(px), int(py)), (int(ex), int(ey)), tuple(int(c * a * 0.7) for c in col), 1, cv2.LINE_AA)
            C.blit(reg, sp, lx, ly, a)
        n = hud.get("n_active")
        if n is None:
            n = n_pts if idx is not None else 0
        txt = f"горят {int(n):,} нейронов".replace(",", " ")
        sp = self.tc.get(txt, 18, "Regular", _tint(self.pal["acc"], 0.5))
        C.blit(out, sp, *self.count_xy)

    def _gauges(self, out, hud):
        bw = GAUGE_X1 - GAUGE_X0
        for n, key in enumerate(MINI):
            g = self.gauge[key]
            y = GAUGE_Y0 + n * GAUGE_DY
            v = float(hud.get("g_" + key, hud.get(key, 0.0)) or 0.0)
            vmax = C.GAUGE_MAX[key]
            fr = float(np.clip(v / vmax, 0, 1))
            m = int(fr * bw)
            by = y + 27
            if m >= 3:
                reg = out[by:by + 7, GAUGE_X0:GAUGE_X0 + m].astype(np.float32)
                reg += (g["bar"][:, :m] - reg) * self.bar_mask[:, :m, None]
                out[by:by + 7, GAUGE_X0:GAUGE_X0 + m] = reg
            pk = hud.get("pk_" + key)
            if pk is not None:
                px = int(np.clip(pk / vmax, 0, 1) * bw)
                if px > m + 4:
                    out[by - 2:by + 9, GAUGE_X0 + px - 1:GAUGE_X0 + px + 1] = np.minimum(np.array(g["col"]) * 0.85 + 40,
                                                                                          255)
            hot = min(1.0, max(0.0, (fr - g["hot"]) / max(0.2, 1 - g["hot"]) * 1.6))
            txt = f"{v * 100:.0f}%" if g["kind"] == "pct" else f"{v:.0f} Гц"
            col = (255, 255, 255) if hot < 0.4 else tuple(int(min(255, c * 0.6 + 110)) for c in g["col"])
            sp = self.tc.get(txt, 18, "SemiBold", col)
            C.blit(out, sp, GAUGE_X1 - sp.w + 3, y - 2)

    # ------------------------------------------------------------ верх: точка LIVE, время
    def _top(self, out, t):
        cx, cy = self.live_dot
        k = 0.55 + 0.45 * (0.5 + 0.5 * np.cos(2 * np.pi * t * 0.8))
        cv2.circle(out, (cx, cy), 7, (int(255 * k + 232 * (1 - k)), int(255 * k + 36 * (1 - k)),
                                      int(255 * k + 62 * (1 - k))), -1, cv2.LINE_AA)
        s = max(0, int(t))
        txt = f"{s // 60:02d}:{s % 60:02d}"
        x0, y0, x1, y1 = self.clock_box
        cw = 17
        x = (x0 + x1) / 2 - cw * (len(txt) - 0.5) / 2 - 2
        for ch in txt:
            sp = self.tc.get(ch, 28, "SemiBold", (255, 255, 255))
            step = cw * 0.5 if ch == ":" else cw
            C.blit(out, sp, x + (step - sp.w) / 2 + 2, (y0 + y1) / 2 - sp.h / 2)
            x += step

    # ------------------------------------------------------------ кадр
    def compose(self, scene_rgb, hud=None, t=0.0, out=None):
        hud = hud or {}
        self._ensure(hud)
        if out is None:
            out = np.empty((H, W, 3), np.uint8)
        t = float(t if t is not None else hud.get("frame", 0) / 30.0)
        out[:] = self._bg
        p = hud.get("pose") or {}
        # --- ролик + вотермарка + «смотрит сюда» ---
        rect = self._clip(out, hud.get("screen_rgb"))
        if hud.get("screen_rgb") is not None:
            self._gaze(out, rect, float(p.get("head", 0.0) or 0.0))
        # --- врезка «как видит муха» ---
        if self.flyeye is not None:
            fx0, fy0, fx1, fy1 = FLYEYE_BOX
            rgba = self.flyeye.render(hud.get("screen_rgb"), hud.get("vis"), (fx1 - fx0, fy1 - fy0))
            C.blit(out, C.Sprite(rgba), fx0, fy0)
        # --- фейскам в окне с неоновой рамкой ---
        self._facecam(out, scene_rgb)
        self._led_frame(out, t)
        self._now(out, hud)
        self._alerts(out, hud)
        self._life(out, hud)
        # --- верх и низ ---
        self._top(out, t)
        self._ticker(out, hud, t)
        self._brain(out, hud)
        self._gauges(out, hud)
        return out
