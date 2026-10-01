"""Варианты комнаты стримера: RoomStyle(seed) -> детерминированный набор косметики.

    import style
    st = style.RoomStyle(123456)        # то же число -> та же комната (в пределах одной версии кода)
    st.chair["model"], st.led["color"], st.camera["name"] ...
    st.to_dict()                         # JSON-совместимый словарь (в лог/метаданные)

Только косметика: ни одно поле не влияет на поведение мухи и подписи.
Цвета — (r, g, b) float 0..1 (как rgba MuJoCo); compose переводит в 0..255 сам.
Каждое поле тянется из своего генератора (seed + имя поля) — новое поле не перетасует старые.

Поля (chair, headphones, wall, led, props, light, camera, toon, overlay) + добавки ROOM
(chair.stripes/rgb_edges/pillow/wheels, headphones.band/size, led.layout, props[].slot/art/color2, extras) —
из отдельных генераторов, старые значения от них не меняются. Рисует всё это stage_streamer.py.
"""
import zlib

import numpy as np

CHAIRS = ("racer", "office", "egg", "throne")
HEADPHONES = ("over_ear", "cat_ears", "studio", "headset_mic")
WALLS = ("plain", "panels", "foam", "brick", "hex")
LED_MODES = ("solid", "breath", "wave", "rainbow")
PROPS = ("neon_sign", "poster", "shelf", "plant", "lamp", "figure", "speaker", "window")
MOODS = ("night_neon", "warm_cozy", "cool_studio", "sunset")
CAMERAS = ("front", "left34", "right34", "low", "high")
NEON_TEXTS = ("МУХА ТВ", "ЖЖЖ", "НЕ МЕШАТЬ", "В ЭФИРЕ", "СТРИМ", "МУХА")
WALL_PROPS = ("neon_sign", "poster", "shelf", "window")      # остальные стоят на полу
LED_LAYOUTS = ("strip", "bars", "frame")                     # лента на стене / две световые стойки / рамка верх+низ
FLOORS = ("wood", "carpet", "tiles")
FLOOR_COLORS = {"wood": (0.55, 0.36, 0.22), "carpet": (0.25, 0.22, 0.30), "tiles": (0.35, 0.36, 0.40)}
DESK_COLORS = ((0.10, 0.10, 0.12), (0.92, 0.92, 0.94), (0.50, 0.33, 0.20), (0.16, 0.14, 0.22))

# палитры: (основной, второй, акцент) — подобраны вручную, чтобы любые сочетания смотрелись
PALETTES = (
    ((0.93, 0.18, 0.32), (0.10, 0.10, 0.12), (1.00, 0.85, 0.20)),     # красный гоночный
    ((0.20, 0.55, 1.00), (0.08, 0.09, 0.14), (0.30, 1.00, 0.85)),     # синий
    ((0.62, 0.30, 1.00), (0.12, 0.08, 0.18), (1.00, 0.35, 0.80)),     # фиолетовый
    ((0.10, 0.85, 0.55), (0.06, 0.12, 0.10), (0.95, 1.00, 0.30)),     # мятный
    ((1.00, 0.55, 0.15), (0.14, 0.10, 0.08), (0.30, 0.80, 1.00)),     # оранжевый
    ((1.00, 0.45, 0.70), (0.96, 0.93, 0.95), (0.55, 0.40, 1.00)),     # розовый «кавай»
    ((0.95, 0.95, 0.97), (0.16, 0.17, 0.20), (0.20, 0.90, 1.00)),     # белый
)
MOOD_LIGHT = {   # key, fill, rim, ambient, exposure
    "night_neon": ((0.55, 0.50, 0.75), (0.15, 0.10, 0.30), (1.0, 0.30, 0.80), (0.05, 0.04, 0.09), 1.0),
    "warm_cozy": ((1.00, 0.80, 0.60), (0.30, 0.20, 0.12), (1.0, 0.60, 0.30), (0.10, 0.07, 0.05), 1.05),
    "cool_studio": ((0.85, 0.92, 1.00), (0.25, 0.30, 0.40), (0.4, 0.80, 1.00), (0.08, 0.09, 0.11), 1.1),
    "sunset": ((1.00, 0.60, 0.40), (0.35, 0.15, 0.35), (1.0, 0.40, 0.20), (0.09, 0.05, 0.07), 1.0),
}


def _rng(seed, field):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(field.encode("utf-8"))])


def _reddish(col):
    import colorsys
    h, sat, v = colorsys.rgb_to_hsv(*[float(x) for x in col])
    return sat > 0.45 and v > 0.35 and (h < 0.06 or h > 0.9)


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _jitter(rng, col, amt=0.06):
    c = np.asarray(col, float) + rng.uniform(-amt, amt, 3)
    return tuple(round(float(x), 3) for x in np.clip(c, 0, 1))


class RoomStyle:
    """Косметический вариант комнаты по номеру seed (int >= 0). Поля — простые dict/tuple (пиклятся)."""

    def __init__(self, seed):
        self.seed = int(seed)
        s = self.seed
        r = _rng(s, "palette")
        pal = PALETTES[int(r.integers(len(PALETTES)))]
        r = _rng(s, "chair")
        self.chair = dict(model=_pick(r, CHAIRS), primary=_jitter(r, pal[0]), secondary=_jitter(r, pal[1], 0.03),
                          accent=_jitter(r, pal[2]), tilt=round(float(r.uniform(-0.12, 0.05)), 3))
        r = _rng(s, "headphones")
        hp_pal = PALETTES[int(r.integers(len(PALETTES)))]
        self.headphones = dict(model=_pick(r, HEADPHONES), color=_jitter(r, hp_pal[0]),
                               cushion=_jitter(r, hp_pal[1], 0.03),
                               led=_jitter(r, pal[2]) if r.random() < 0.7 else None, mic=bool(r.random() < 0.5))
        r = _rng(s, "wall")
        base = np.array(pal[1]) * 0.6 + r.uniform(0.02, 0.16, 3)
        self.wall = dict(pattern=_pick(r, WALLS), color=_jitter(r, np.clip(base, 0, 1), 0.02),
                         color2=_jitter(r, np.clip(base * 1.3 + 0.03, 0, 1), 0.02))
        r = _rng(s, "led")
        self.led = dict(color=_jitter(r, pal[2] if r.random() < 0.6 else pal[0]), mode=_pick(r, LED_MODES),
                        speed=round(float(r.uniform(0.15, 0.6)), 3))
        r = _rng(s, "props")
        n = int(r.integers(3, 6))
        names = list(r.choice(PROPS, size=n, replace=False))
        self.props = []
        for nm in names:
            p = dict(name=str(nm), side=_pick(r, ("left", "right", "back")), color=_jitter(r, _pick(r, pal)),
                     scale=round(float(r.uniform(0.8, 1.2)), 3))
            if nm == "neon_sign":
                p["text"] = _pick(r, NEON_TEXTS)
            self.props.append(p)
        r = _rng(s, "light")
        mood = _pick(r, tuple(MOOD_LIGHT))
        key, fill, rim, amb, exp = MOOD_LIGHT[mood]
        self.light = dict(mood=mood, key=_jitter(r, key, 0.04), fill=_jitter(r, fill, 0.03), rim=_jitter(r, rim, 0.05),
                          ambient=tuple(amb), exposure=float(exp))
        r = _rng(s, "camera")
        self.camera = dict(name=_pick(r, CAMERAS), dist=round(float(r.uniform(0.9, 1.15)), 3),
                           yaw=round(float(r.uniform(-0.12, 0.12)), 3), pitch=round(float(r.uniform(-0.06, 0.08)), 3),
                           roll=round(float(r.uniform(-0.03, 0.03)), 3), drift=round(float(r.uniform(0.3, 1.0)), 3))
        r = _rng(s, "toon")
        self.toon = dict(levels=int(r.integers(3, 6)), outline_px=int(r.integers(2, 4)),
                         outline=(0.05, 0.04, 0.07), saturation=round(float(r.uniform(1.1, 1.4)), 3))
        r = _rng(s, "overlay")
        self.overlay = dict(bg=_jitter(r, np.clip(np.array(pal[1]) * 0.5, 0, 1), 0.02), accent=tuple(self.led["color"]),
                            accent2=_jitter(r, pal[0]))
        # ---- добавки ROOM (свои генераторы: старые поля выше не сдвигаются) ----
        r = _rng(s, "chair_detail")
        self.chair.update(stripes=int(r.integers(0, 3)), rgb_edges=bool(r.random() < 0.75),
                          pillow=_pick(r, ("accent", "primary", "secondary")), wheels=_pick(r, ("black", "chrome")))
        r = _rng(s, "headphones_detail")
        if _reddish(self.headphones["color"]):          # красные глаза мухи + красные чашки = каша
            alt = [p[0] for p in PALETTES if not _reddish(p[0])]
            self.headphones["color"] = _jitter(r, alt[int(r.integers(len(alt)))])
        self.headphones.update(band=_jitter(r, np.clip(np.array(self.headphones["cushion"]) * 0.8 + 0.05, 0, 1), 0.02),
                               size=round(float(r.uniform(0.95, 1.1)), 3))
        r = _rng(s, "led_layout")
        self.led.update(layout=_pick(r, LED_LAYOUTS))
        self._place_props(_rng(s, "props_detail"), pal)
        r = _rng(s, "extras")
        floor = _pick(r, FLOORS)
        self.extras = dict(desk=bool(r.random() < 0.75), mic_arm=bool(r.random() < 0.4), floor=floor,
                           floor_color=_jitter(r, FLOOR_COLORS[floor], 0.04), rug=bool(r.random() < 0.6),
                           rug_color=_jitter(r, _pick(r, pal)), desk_color=_jitter(r, _pick(r, DESK_COLORS), 0.03),
                           swivel=round(float(r.uniform(-0.18, 0.18)), 3))

    def _place_props(self, r, pal):
        """Место каждого пропса в комнате (slot) + вид (art): без наложений, детерминированно.
        Настенные: C_hi (над креслом), L_hi/R_hi, L_lo/R_lo; напольные: L_fl/R_fl (у кресла), L_bk/R_bk (у стены)."""
        used = set()
        wall_pref = dict(left=("L_hi", "L_lo"), right=("R_hi", "R_lo"), back=("C_hi",))
        floor_pref = dict(left=("L_fl", "L_bk"), right=("R_fl", "R_bk"), back=("L_bk", "R_bk"))
        for p in sorted(self.props, key=lambda q: q["name"] != "window"):     # окно первым: ему нужна целая сторона
            nm = p["name"]
            if nm in WALL_PROPS:
                if nm == "window":                           # большое окно занимает всю сторону (hi + lo)
                    order = [x for x in ("L_hi", "R_hi") if x[0] == {"left": "L", "right": "R"}.get(p["side"], "?")]
                    order += ["L_hi", "R_hi"]
                    order = [x for x in order if x not in used and x.replace("hi", "lo") not in used]
                    slot = order[0] if order else None
                    if slot:
                        used.add(slot.replace("hi", "lo"))
                else:
                    order = list(wall_pref[p["side"]]) + ["C_hi", "L_hi", "R_hi", "L_lo", "R_lo"]
                    order = [x for x in order if x not in used]
                    slot = order[0] if order else None
            else:
                order = list(floor_pref[p["side"]]) + ["L_fl", "R_fl", "L_bk", "R_bk"]
                order = [x for x in order if x not in used]
                slot = order[0] if order else None
            if slot:
                used.add(slot)
            p["slot"] = slot
        for p in self.props:
            p["art"] = int(r.integers(0, 6))
            p["color2"] = _jitter(r, _pick(r, pal))

    def label(self):
        return f"v{self.seed}"

    def to_dict(self):
        return dict(seed=self.seed, chair=self.chair, headphones=self.headphones, wall=self.wall, led=self.led,
                    props=self.props, light=self.light, camera=self.camera, toon=self.toon, overlay=self.overlay,
                    extras=self.extras)

    def __repr__(self):
        return (f"RoomStyle({self.seed}: кресло {self.chair['model']}, наушники {self.headphones['model']}, "
                f"стена {self.wall['pattern']}, свет {self.light['mood']}, камера {self.camera['name']}, "
                f"пропсы {[(p['name'], p['slot']) for p in self.props]})")
