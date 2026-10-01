"""Формат «Стример»: муха NeuroMechFly сидит в игровом кресле в наушниках (фейскам для compose_streamer).

    import stage_streamer as SS, style
    SS.prepare_assets(style.RoomStyle(123))                          # главный процесс, до воркеров: текстуры
    st = SS.StreamerStage(style.RoomStyle(123), watermark="@ник")    # в воркере рендера (StageFactory), ~5 с
    st.set_state(state)            # state = Stage.get_state() из лога прохода 1 + state['pose'] (formats.pose_track)
    st.set_screen_frame(rgb)       # кадр ролика: свет монитора на мухе
    img = st.render(1080, 740)     # uint8 (h, w, 3) RGB — размер = compose_streamer.SCENE_W x SCENE_H
    st.fly_box(w, h), st.fly_screen_xy(w, h)                        # где муха/лицо на картинке (для подписей)
    st.close()

Одна модель MuJoCo: та же муха flygym (Fly + scene.add_extra_joints: хоботок, крылья) + своя арена: кресло,
наушники (геомы на теле Head — поворачиваются с головой), комната, реквизит. Кукольный режим: физику не шагаем,
ставим qpos и mj_kinematics. Масштаб — мм (муха ~2.6 мм), кресло смотрит вдоль +x (на монитор/камеру),
+y — левая сторона мухи (на картинке справа), z — вверх, верх сиденья z = 0.

Правдивость: поза мухи — ТОЛЬКО из state['pose'] (formats.pose_track: голова, хоботок, капля, крылья,
чистка, взлёт, вздрагивание — из нейронов и куклы прохода 1). Сидячая поза (корпус, ноги) — постоянная;
ходьбу из pose['legs'] не показываем, берём оттуда только передние лапы во время чистки. Вздрогнула —
взмах крыльев из позы + маленький подскок в кресле; взлёт (DNp01) — приподнимается над сиденьем (<= 0.5 длины
тела) и садится обратно. Кресло, наушники, комната, свет, камера, мульт-обработка — RoomStyle (косметика):
на позу и подписи не влияют. Кадр = функция (pose, style) — без накопления состояния между кадрами.

Мульт-вид: плоские материалы без бликов, ступенчатый свет (HSV-квантование V, style.toon.levels), контур по
сегментации (группы объектов, style.toon.outline_px), насыщенность, неоновое свечение.
"""
import os
import sys
import zlib
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import stage as SG                                                       # noqa: E402  (текстуры, nchannel, блум)
from flygym import Fly                                                   # noqa: E402
from flygym.arena import BaseArena                                       # noqa: E402
from flygym.simulation import SingleFlySimulation                        # noqa: E402
from flygym.examples.locomotion import PreprogrammedSteps                # noqa: E402
from scene import FlyPoser, KitchenArena                                 # noqa: E402

SCENE_W, SCENE_H = 1080, 740          # по умолчанию; фактический размер приходит в render()
ASSETS = HERE / "assets" / "streamer"
TEX_VER = 1

# ---------------- геометрия (мм) ----------------
FLOOR_Z = -2.75                        # пол (кресло на колёсиках)
WALL_X = -7.5                          # лицевая плоскость задней стены
SIDE_Y = 10.5                          # боковые стены
CEIL_Z = 9.0
BACK_PIVOT = np.array([-1.15, 0.0, -0.25])     # низ спинки (шарнир наклона)
BACK_TILT = 0.14                       # наклон спинки назад, рад
# сидячая поза мухи (постоянная): корпус поднят носом вверх, голова наклонена к камере
SEAT_PITCH = 1.15                      # рад: корпус от горизонтали
HEAD_PITCH = 0.8                       # рад: голова вниз (joint_Head) — лицо к камере
FLY_ROOT = np.array([1.3, 0.0, 1.18])  # позиция корня мухи (брюшко на сиденье z = 0)
# сидячая поза ног (постоянная; подобрана IK по точкам кресла — _scratch/room/ik_rest.py), левая сторона,
# правая — зеркально (roll/yaw со знаком минус): передние — «руки на животе», средние — «ладони на коленях»,
# задние — «бёдра» вперёд по сиденью, голени и лапки свисают с переднего края
LEG_REST = {"FCoxa": 1.323, "FCoxa_roll": 0.8, "FCoxa_yaw": -0.006, "FFemur": -1.198, "FFemur_roll": 0.274,
            "FTibia": 1.894, "FTarsus1": -0.037,
            "MCoxa": 0.144, "MCoxa_roll": 1.72, "MCoxa_yaw": 0.643, "MFemur": -1.29, "MFemur_roll": 0.274,
            "MTibia": 2.0, "MTarsus1": 0.689,
            "HCoxa": -0.138, "HCoxa_roll": 1.297, "HCoxa_yaw": -0.088, "HFemur": -0.672, "HFemur_roll": -1.134,
            "HTibia": 1.202, "HTarsus1": 0.059}
GROOM_IN, GROOM_OUT = 0.25, 0.25       # с: плавный переход передних лап в чистку и обратно
# постоянная поправка передних лап на время чистки (левая; правая — зеркально): корпус сидящей мухи откинут,
# голова наклонена (HEAD_PITCH) — «трёт лапками» из pose['legs'] наводим на её настоящее лицо, локти в стороны
# (IK по целям в системе головы — _scratch/room/groom_ik.py). Сама анимация — из позы, без изменений.
GROOM_AIM = {"Coxa": -1.314, "Coxa_roll": -0.589, "Coxa_yaw": -2.022, "Femur": 1.134, "Femur_roll": 1.208,
             "Tibia": 1.47, "Tarsus1": -0.917}
# путь передней лапы (левая, 7 суставов; отклонение от прямой в суставах) из позы покоя в среднюю позу чистки:
# 7 точек u = 1/8..7/8 —
# лапа идёт вперёд перед грудью и вверх к лицу, локоть наружу (IK — _scratch/room/groom_path.py); без них
# линейная интерполяция суставов проводит лапу сквозь голову
GROOM_PATH = [[0.203, 0.144, 0.305, 0.096, -0.191, 0.125, -0.1], [0.421, 0.264, 0.607, 0.3, -0.335, 0.369, -0.57],
              [0.7, 0.434, 0.808, 0.486, -0.378, 0.539, -0.92], [0.955, 0.709, 0.908, 0.576, -0.339, 0.608, -1.02],
              [1.035, 1.116, 0.925, 0.528, -0.261, 0.594, -0.983], [0.859, 1.441, 0.977, 0.334, -0.168, 0.476, -0.895],
              [0.698, 1.499, 1.0, 0.106, -0.145, 0.173, -0.684]]
HOP_T, HOP_H = 0.35, 0.32              # вздрагивание: подскок в кресле (с, мм)
LIFT_H = 1.2                           # взлёт: подъём над сиденьем, мм (< 0.5 длины тела 2.6 мм)
FOVY = 36.0
GROOM_ZOOM, GROOM_LIFT = 0.16, 0.3     # наезд камеры во время чистки: доля расстояния, подъём цели к лицу (мм)
PIP_D, PIP_L, PIP_BULB = (0.45, 0.75, 0.55), 2.4, (0.2, 0.2, 0.42)   # пипетка: направление от капли, длина, груша (мм)

# группы геомов: 1/2 — муха (flygym), 3 — комната/кресло, 4 — наушники/реквизит, 5 — ореолы (без сегментации)
G_ROOM, G_PROP, G_HALO = 3, 4, 5
GLOW_OBJS = ("bgled", "bgneon", "bglamp", "bgspk", "bgeye", "hpled")      # светятся (источник блума)

# ракурсы «фейскама»: азимут/возвышение (рад) камеры вокруг цели, расстояние (мм), цель (x, z)
CAM_PRESETS = {
    "front":   dict(az=0.0, el=0.13, D=8.6, tx=0.45, tz=1.45),
    "left34":  dict(az=-0.42, el=0.15, D=8.6, tx=0.40, tz=1.45),
    "right34": dict(az=0.42, el=0.15, D=8.6, tx=0.40, tz=1.45),
    "low":     dict(az=0.12, el=-0.02, D=8.0, tx=0.45, tz=1.55),
    "high":    dict(az=-0.10, el=0.34, D=9.0, tx=0.45, tz=1.35),
}

# места пропсов (в системе комнаты): настенные — (y, z) на задней стене, напольные — (x, y) на полу
WALL_SLOTS = {"C_hi": (2.6, 5.0), "L_hi": (-4.9, 2.4), "R_hi": (4.4, 2.9), "L_lo": (-5.2, -1.0),
              "R_lo": (4.6, -0.6)}
FLOOR_SLOTS = {"L_fl": (-3.0, -4.6), "R_fl": (-3.0, 4.4), "L_bk": (-6.2, -2.8), "R_bk": (-6.2, 2.9)}

# мульт-цвета мухи (узнаваемая NeuroMechFly: рыже-коричневый хитин, красные глаза, прозрачные крылья)
FLY_MAT = {"head": (0.86, 0.58, 0.28), "thorax": (0.78, 0.52, 0.26), "a12345": (0.90, 0.66, 0.32),
           "a6": (0.36, 0.22, 0.12), "eye": (0.93, 0.10, 0.12), "antenna": (0.60, 0.32, 0.13),
           "arista": (0.30, 0.20, 0.12), "proboscis": (0.72, 0.46, 0.24), "coxa": (0.74, 0.48, 0.24),
           "femur": (0.66, 0.42, 0.21), "tibia": (0.66, 0.42, 0.21), "tarsus": (0.52, 0.33, 0.17),
           "wing": (0.82, 0.93, 1.00), "haltere": (0.85, 0.62, 0.35)}
FLY_ALPHA = {"wing": 0.36}
FLY_EMIT = 0.25                         # доля «своего света» хитина: сглаживает рельеф меша под мульт-заливку
FLAT_SIGMA = {1: 3.0, 2: 1.2, 3: 1.6}   # мульт-заливка частей мухи (px половинного кадра): тело, лапы, глаза
FLAT_MIX = 0.85                         # доля заливки (остальное — исходная светотень меша)
ISLAND_PX = 7                           # замкнутые кусочки контура меньше (px половинного кадра) не рисуем
ANT_GROUP = "head"                      # усики — в группе головы: без своих колечек-контуров («очки» на лице)
ABD_BANDS = {"A1A2": (0.90, 0.66, 0.32), "A3": (0.46, 0.28, 0.14), "A4": (0.90, 0.66, 0.32),
             "A5": (0.46, 0.28, 0.14), "A6": (0.36, 0.22, 0.12)}


# ======================= мелочи =======================
def _q(axis, ang):
    """Кватернион MuJoCo (w, x, y, z) поворота на ang (рад) вокруг оси."""
    q = Rot.from_rotvec(np.asarray(axis, float) / np.linalg.norm(axis) * ang).as_quat()
    return tuple(float(x) for x in q[[3, 0, 1, 2]])


def _qm(*qs):
    """Произведение кватернионов (w, x, y, z): сначала последний."""
    r = Rot.identity()
    for q in qs:
        r = r * Rot.from_quat([q[1], q[2], q[3], q[0]])
    q = r.as_quat()
    return tuple(float(x) for x in q[[3, 0, 1, 2]])


def _c(col, k=1.0, a=1.0):
    c = np.clip(np.asarray(col, float)[:3] * k, 0, 1)
    return (float(c[0]), float(c[1]), float(c[2]), float(a))


def _lum(col):
    c = np.asarray(col, float)[:3]
    return float(0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2])


def _bright(col, lo=0.45):
    """Цвет для свечения: тёмные цвета палитры поднимаем по яркости, оттенок сохраняем."""
    c = np.clip(np.asarray(col, float)[:3], 0, 1)
    m = c.max()
    if m < 1e-3:
        return np.array([0.9, 0.9, 1.0])
    c = c / m
    return np.clip(c * max(lo, 1.0), 0, 1) if _lum(c) >= lo else np.clip(c + (lo - _lum(c)), 0, 1)


def _smooth(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def _mix(a, b, k):
    return tuple(float(x) for x in (np.asarray(a, float)[:3] * (1 - k) + np.asarray(b, float)[:3] * k))


# ======================= текстуры (общие для всех вариантов; тонируются цветом геома) =======================
def _gray(v):
    v = np.clip(v, 0, 1)
    return (np.repeat(v[..., None], 3, -1) * 255).astype(np.uint8)


def _noise(n, res, seed):
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.random((res, res)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)


def _tx_wall(pattern, n=512):
    """Серый узор стены 0.55..1 (цвет задаёт rgba геома). Тайл ~4 мм."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    nz = 0.03 * (_noise(n, 32, 3) - 0.5)
    if pattern == "panels":                       # вертикальные рейки с пазами
        p = 64
        u = (x % p) / p
        v = 0.86 + 0.1 * np.sin(np.pi * u) ** 0.5
        v = np.where(u > 0.9, 0.52, v)
    elif pattern == "foam":                       # акустический поролон «клин»
        half = n // 4
        tile = ((x // half + y // half) % 2).astype(bool)
        sx, sy = (x % half) / half * 4, (y % half) / half * 4
        saw = np.where(tile, sx % 1, sy % 1)
        v = np.where(saw < 0.5, 0.62 + 0.7 * saw, 1.32 - 0.7 * saw)
        v = np.where(((x % half) < 2) | ((y % half) < 2), 0.5, v)
    elif pattern == "brick":                      # кирпич: ряды со сдвигом, швы
        bh, bw = 48, 112
        row = (y // bh).astype(int)
        xs = x + (row % 2) * bw / 2
        col = (xs // bw).astype(int)
        rng = np.random.default_rng(9)
        var = rng.uniform(-0.06, 0.06, (n // bh + 2, n // bw + 3))
        v = 0.88 + var[row, col]
        v = np.where(((y % bh) < 5) | ((xs % bw) < 5), 0.58, v)
    elif pattern == "hex":                        # шестигранная плитка
        r = 40.0
        w, h = np.sqrt(3) * r, 1.5 * r
        best = np.full((n, n), 1e9, np.float32)
        second = np.full((n, n), 1e9, np.float32)
        cid = np.zeros((n, n), np.int32)
        k = 0
        for j in range(-1, int(n / h) + 2):
            for i in range(-1, int(n / w) + 2):
                cx, cy = i * w + (j % 2) * w / 2, j * h
                d = np.hypot(x - cx, y - cy)
                upd = d < best
                second = np.where(upd, best, np.minimum(second, d))
                best = np.where(upd, d, best)
                cid = np.where(upd, k, cid)
                k += 1
        rng = np.random.default_rng(4)
        var = rng.uniform(-0.05, 0.05, k + 1)
        v = 0.88 + var[cid]
        v = np.where(second - best < 5, 0.55, v)
    else:                                         # plain: ровная краска
        v = np.full((n, n), 0.92, np.float32)
    return _gray(v + nz)


def _tx_floor(kind, n=512):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    if kind == "wood":
        ph = 64
        row = (y // ph).astype(int)
        rng = np.random.default_rng(12)
        off = rng.uniform(0, n, 64)
        var = rng.uniform(-0.07, 0.07, 64)
        v = 0.82 + var[row % 64] + 0.04 * np.sin((x + off[row % 64]) / 9 + _noise(n, 16, 2) * 6)
        v = np.where((y % ph) < 3, 0.5, v)
        v = np.where(((x + off[row % 64] * 3) % 256) < 3, 0.55, v)
    elif kind == "tiles":
        p = 128
        v = 0.86 + 0.04 * (_noise(n, 8, 5) - 0.5)
        v = np.where(((x % p) < 4) | ((y % p) < 4), 0.55, v)
    else:                                         # carpet: ворс
        v = 0.8 + 0.12 * (_noise(n, 128, 6) - 0.5) + 0.06 * (_noise(n, 16, 7) - 0.5)
    return _gray(v)


def _tx_poster(k, W=384, H=512):
    """Цветные мульт-постеры (6 штук) с белой рамкой."""
    im = Image.new("RGB", (W, H), (20, 16, 30))
    dr = ImageDraw.Draw(im)
    rng = np.random.default_rng(100 + k)
    f_big, f_small = SG._font(64), SG._font(26, bold=False)

    def title(t, y, col=(255, 245, 250), f=f_big):
        tw = dr.textlength(t, font=f)
        dr.text(((W - tw) / 2, y), t, font=f, fill=col, stroke_width=3, stroke_fill=(20, 10, 30))

    if k == 0:                                    # ретро-закат «МУХА»
        for yy in range(H):
            a = yy / H
            dr.line([(0, yy), (W, yy)], fill=(int(40 + 120 * a), int(10 + 20 * a), int(90 - 40 * a)))
        cx, cy, r = W // 2, int(H * 0.5), int(W * 0.3)
        dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 190, 60))
        for i in range(5):
            dr.rectangle((0, cy + 10 + i * 18, W, cy + 16 + i * 18 + i), fill=(int(160 + 20 * i), 30, 80))
        dr.rectangle((0, cy + r - 30, W, H), fill=(30, 10, 60))
        for i in range(-8, 9):
            dr.line([(W / 2 + i * 12, cy + r - 30), (W / 2 + i * 70, H)], fill=(80, 220, 255), width=2)
        title("МУХА", 24)
    elif k == 1:                                  # поп-арт банан (плодовая муха!)
        dr.rectangle((0, 0, W, H), fill=(255, 220, 40))
        for yy in range(0, H, 22):
            for xx in range(0, W, 22):
                dr.ellipse((xx + (yy // 22 % 2) * 11, yy, xx + 8 + (yy // 22 % 2) * 11, yy + 8), fill=(255, 150, 40))
        pts = [(W * 0.18 + W * 0.64 * t, H * 0.62 - np.sin(np.pi * t) * H * 0.3) for t in np.linspace(0, 1, 30)]
        dr.line(pts, fill=(40, 30, 20), width=64, joint="curve")
        dr.line(pts, fill=(255, 235, 90), width=52, joint="curve")
        title("БАНАН", H * 0.74, (255, 255, 255))
    elif k == 2:                                  # пиксельная муха 8 бит
        dr.rectangle((0, 0, W, H), fill=(18, 24, 60))
        spr = ["..W....W..", "...W..W...", "..RRBBRR..", ".RRRBBRRR.", ".RRBBBBRR.", "..BBBBBB..",
               "WWBBBBBBWW", "WWWBBBBWWW", "..BB..BB..", ".B......B."]
        cs = {"R": (230, 40, 50), "B": (190, 120, 50), "W": (180, 230, 255)}
        s = 28
        ox, oy = (W - s * 10) // 2, 90
        for j, rowc in enumerate(spr):
            for i, ch in enumerate(rowc):
                if ch in cs:
                    dr.rectangle((ox + i * s, oy + j * s, ox + i * s + s - 2, oy + j * s + s - 2), fill=cs[ch])
        title("УРОВЕНЬ 1", 24, (120, 255, 160), SG._font(44))
        title("ЖМИ СТАРТ", H - 70, (255, 230, 120), f_small)
    elif k == 3:                                  # планета и звёзды
        dr.rectangle((0, 0, W, H), fill=(10, 8, 30))
        for _ in range(120):
            x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
            rr = rng.uniform(0.8, 2.4)
            dr.ellipse((x0 - rr, y0 - rr, x0 + rr, y0 + rr), fill=(255, 255, 230))
        cx, cy, r = W * 0.5, H * 0.52, W * 0.28
        dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(120, 80, 255))
        dr.ellipse((cx - r * 0.7, cy - r * 0.8, cx + r * 0.2, cy - r * 0.1), fill=(170, 130, 255))
        dr.ellipse((cx - r * 1.7, cy - r * 0.25, cx + r * 1.7, cy + r * 0.25), outline=(255, 200, 90), width=10)
        title("КОСМОС", 30, (255, 220, 120))
    elif k == 4:                                  # граффити «ЖЖЖ»
        dr.rectangle((0, 0, W, H), fill=(30, 30, 36))
        for _ in range(14):
            c = tuple(int(x) for x in rng.integers(60, 255, 3))
            x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
            rr = rng.uniform(20, 70)
            dr.ellipse((x0 - rr, y0 - rr, x0 + rr, y0 + rr), fill=c)
        f = SG._font(120)
        t = "ЖЖЖ"
        tw = dr.textlength(t, font=f)
        dr.text(((W - tw) / 2, H * 0.32), t, font=f, fill=(255, 255, 255), stroke_width=8, stroke_fill=(10, 10, 10))
    else:                                         # яблоко поп-арт
        dr.rectangle((0, 0, W, H), fill=(60, 200, 190))
        cx, cy, r = W * 0.5, H * 0.52, W * 0.3
        dr.ellipse((cx - r, cy - r * 0.9, cx + r, cy + r), fill=(230, 40, 60), outline=(20, 10, 20), width=8)
        dr.ellipse((cx - r * 0.55, cy - r * 0.6, cx - r * 0.2, cy - r * 0.2), fill=(255, 150, 160))
        dr.line([(cx, cy - r * 0.9), (cx + 10, cy - r * 1.35)], fill=(80, 50, 20), width=12)
        dr.ellipse((cx + 10, cy - r * 1.45, cx + r * 0.7, cy - r * 1.1), fill=(90, 200, 80), outline=(20, 10, 20), width=5)
        title("ФРУКТ", 26, (255, 255, 255))
    a = np.asarray(im).copy()
    a[:10] = a[-10:] = a[:, :10] = a[:, -10:] = (245, 245, 248)
    return a


def _tx_neon(text, W=1024, H=320):
    """Неон: (свечение RGBA, трубка RGBA) белым — цвет задаёт rgba геома."""
    f = SG._font(170)
    meas = ImageDraw.Draw(Image.new("L", (1, 1)))
    tw = meas.textlength(text, font=f)
    if tw > W * 0.86:
        f = SG._font(int(170 * W * 0.86 / tw))
        tw = meas.textlength(text, font=f)
    bb = meas.textbbox((0, 0), text, font=f)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).text(((W - tw) / 2, (H - (bb[3] + bb[1])) / 2), text, font=f, fill=255)
    m = np.asarray(mask).astype(np.float32) / 255
    glow = np.clip(cv2.GaussianBlur(m, (0, 0), 16) * 1.6 + cv2.GaussianBlur(m, (0, 0), 5) * 0.9, 0, 1)
    tube = np.clip(m - cv2.erode(m, np.ones((3, 3), np.uint8), iterations=6) * 0.0, 0, 1)
    white = np.full((H, W, 3), 255, np.uint8)
    return (np.dstack([white, (glow * 255).astype(np.uint8)]),
            np.dstack([white, (tube * 255).astype(np.uint8)]))


def _tx_window(k, W=512, H=640):
    """Вид за окном: 0 — ночной город, 1 — закат, 2 — звёзды и луна."""
    rng = np.random.default_rng(30 + k)
    yy = np.linspace(0, 1, H)[:, None, None]
    if k == 1:
        sky = (1 - yy) * np.array([0.35, 0.25, 0.65]) + yy * np.array([1.0, 0.55, 0.30])
    elif k == 2:
        sky = (1 - yy) * np.array([0.02, 0.03, 0.10]) + yy * np.array([0.08, 0.10, 0.28])
    else:
        sky = (1 - yy) * np.array([0.10, 0.04, 0.22]) + yy * np.array([0.05, 0.05, 0.16])
    img = np.broadcast_to(sky, (H, W, 3)).astype(np.float32).copy()
    im = Image.fromarray((img * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    if k == 2:
        for _ in range(90):
            x0, y0 = rng.uniform(0, W), rng.uniform(0, H * 0.7)
            r = rng.uniform(0.8, 2.2)
            dr.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=(255, 255, 235))
        dr.ellipse((W * 0.55, H * 0.12, W * 0.8, H * 0.12 + W * 0.25), fill=(255, 245, 200))
        dr.ellipse((W * 0.6, H * 0.1, W * 0.85, H * 0.1 + W * 0.25), fill=tuple(int(255 * x) for x in sky[0, 0]))
    if k == 1:
        dr.ellipse((W * 0.35, H * 0.55, W * 0.65, H * 0.55 + W * 0.3), fill=(255, 220, 120))
    xs = 0
    while xs < W:                                  # силуэты домов / холмов
        bw = int(rng.uniform(40, 110))
        top = int(rng.uniform(H * 0.5, H * 0.8))
        c = (25, 15, 40) if k != 1 else (70, 30, 60)
        dr.rectangle((xs, top, xs + bw, H), fill=c)
        if k != 1:
            for wy in range(top + 8, H, 16):
                for wx in range(xs + 6, xs + bw - 8, 13):
                    if rng.random() < 0.35:
                        dr.rectangle((wx, wy, wx + 6, wy + 7), fill=(255, 200, 110) if rng.random() < 0.7 else (140, 210, 255))
        xs += bw + int(rng.uniform(0, 10))
    return np.asarray(im)


def _tx_keys(W=512, H=160):
    """Клавиатура: тёмные клавиши, светлые щели (светятся цветом подсветки)."""
    v = np.full((H, W), 1.0, np.float32)
    rows = 5
    kh = H / rows
    for r in range(rows):
        n = 14 if r < 4 else 8
        kw = W / n
        for i in range(n):
            x0, y0 = int(i * kw + 3), int(r * kh + 3)
            v[y0:int(y0 + kh - 6), x0:int(x0 + kw - 6)] = 0.16
    return _gray(v)


def _tx_halo(n=256):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(x - n / 2 + 0.5, y - n / 2 + 0.5) / (n / 2)
    v = np.clip(np.exp(-(r / 0.5) ** 2 * 1.4) - 0.015, 0, 1)
    v = v / v.max()
    return np.dstack([np.full((n, n, 3), 255, np.uint8), (v * 255).astype(np.uint8)])


def _tx_glowbar(W=256, H=64):
    """Мягкая полоса света (засветка стены от ленты): альфа спадает к краям."""
    y = np.linspace(-1, 1, H)[:, None]
    x = np.linspace(-1, 1, W)[None, :]
    v = np.exp(-(y / 0.45) ** 2 * 2.0) * np.clip((1 - np.abs(x)) / 0.15, 0, 1)
    return np.dstack([np.full((H, W, 3), 255, np.uint8), (np.clip(v, 0, 1) * 255).astype(np.uint8)])


def _textures():
    t = {f"wall_{p}": (lambda p=p: _tx_wall(p)) for p in ("plain", "panels", "foam", "brick", "hex")}
    t.update({f"floor_{k}": (lambda k=k: _tx_floor(k)) for k in ("wood", "carpet", "tiles")})
    t.update({f"poster_{k}": (lambda k=k: _tx_poster(k)) for k in range(6)})
    t.update({f"window_{k}": (lambda k=k: _tx_window(k)) for k in range(3)})
    t.update(keys=_tx_keys, halo=_tx_halo, glowbar=_tx_glowbar)
    return t


def _neon_name(text):
    return f"neon_{zlib.crc32(text.encode('utf-8')) & 0xFFFFFF:06x}"


def asset_path(name, style_text=None):
    """Путь к текстуре (генерируется при первом обращении; запись атомарная — stage._save)."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    p = ASSETS / f"{name}_s{TEX_VER}.png"
    if not p.exists():
        if name.startswith("neon_"):
            glow, tube = _tx_neon(style_text)
            SG._save(glow, ASSETS / f"{name}g_s{TEX_VER}.png")
            SG._save(tube, p)
        else:
            SG._save(_textures()[name](), p)
    return p


def prepare_assets(style):
    """Один раз в главном процессе до воркеров: все текстуры этого варианта (общие для всех вариантов,
    кроме неона — по тексту вывески). tiktok/assets/streamer/*.png, запись атомарная."""
    for n in _textures():
        asset_path(n)
    for p in style.props:
        if p["name"] == "neon_sign":
            asset_path(_neon_name(p["text"]), p["text"])
    return ASSETS


# ======================= арена =======================
# вид: 'real' — обычные материалы и свет (по умолчанию), 'toon' — мульт (FLY_STREAMER_LOOK=toon)
LOOK = __import__("os").environ.get("FLY_STREAMER_LOOK", "real")

class _Room(BaseArena):
    """Пустая арена flygym без штатного света (свет ставим сами)."""

    def init_lights(self):
        pass

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

    def _get_max_floor_height(self):
        return FLOOR_Z


def _panel_mesh(A, name, w, h, d=0.05, urep=1.0, vrep=1.0):
    """Плита w x h: лицевая грань в плоскости x=0 смотрит в +x (на камеру), тыл x=-d.
    u — слева направо для зрителя, смотрящего вдоль -x (вправо = +y), v — сверху вниз."""
    a, b = w / 2, h / 2
    v = np.array([[0, -a, b], [0, a, b], [0, a, -b], [0, -a, -b],
                  [-d, -a, b], [-d, a, b], [-d, a, -b], [-d, -a, -b]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]] * 2, float) * [urep, vrep]
    return A.add("mesh", name=name, vertex=v.ravel(), face=f.ravel(), texcoord=uv.ravel())


def _egg_mesh(A, name, a, b, c, t, al_c, nu=18, nv=40):
    """Скорлупа кресла-яйца: эллипсоид (a, b, c) с вырезом вокруг +x (угол al_c от полюса -x), толщина t.
    Замкнутая сетка: наружная + внутренняя поверхности + кромка."""
    V, F = [], []

    def ring(al, s):
        ph = np.linspace(0, 2 * np.pi, nv, endpoint=False)
        return np.stack([-(a - s) * np.cos(al) * np.ones(nv), (b - s) * np.sin(al) * np.cos(ph),
                         (c - s) * np.sin(al) * np.sin(ph)], 1)

    def surf(s, flip):
        base = len(V)
        V.append([-(a - s), 0, 0])
        for k in range(1, nu + 1):
            V.extend(ring(al_c * k / nu, s).tolist())
        idx = lambda k, j: base if k == 0 else base + 1 + (k - 1) * nv + (j % nv)
        for k in range(nu):
            for j in range(nv):
                tris = [(idx(k, j), idx(k, j + 1), idx(k + 1, j + 1)), (idx(k, j), idx(k + 1, j + 1), idx(k + 1, j))]
                for tr in tris:
                    if len(set(tr)) < 3:
                        continue
                    F.append(tr[::-1] if flip else tr)
        return lambda j: idx(nu, j)

    o = surf(0.0, False)
    i = surf(t, True)
    for j in range(nv):
        F.append((o(j), o(j + 1), i(j + 1)))
        F.append((o(j), i(j + 1), i(j)))
    return A.add("mesh", name=name, vertex=np.array(V, float).ravel(), face=np.array(F).ravel(), inertia="exact")        # convex роняет компилятор MuJoCo 3.2.7


def _base_materials(A):
    """Мульт-материалы: плоский, сатин (лёгкий блик), металл, свечение. В каждой модели MJCF свои имена."""
    A.add("material", name="flat", specular=0.0, shininess=0.0, reflectance=0.0)
    A.add("material", name="satin", specular=0.18, shininess=0.35, reflectance=0.0)
    A.add("material", name="metal", specular=0.55, shininess=0.8, reflectance=0.0)
    A.add("material", name="glow", emission=1.0, specular=0.0, shininess=0.0)
    A.add("material", name="soft", emission=0.45, specular=0.0, shininess=0.0)
    A.add("material", name="glass", specular=0.6, shininess=0.9)


class _Builder:
    """Добавляет декоративные геомы: имя '<объект>__<часть>_sensorfx' (объект — группа контура), без контактов."""

    def __init__(self, A):
        self.A = A
        self.n = 0

    def g(self, parent, obj, group=G_ROOM, **kw):
        self.n += 1
        kw.setdefault("contype", 0)
        kw.setdefault("conaffinity", 0)
        kw.setdefault("material", "flat")
        return parent.add("geom", name=f"{obj}__{self.n}_sensorfx", group=group, **kw)

    def box(self, parent, obj, pos, half, rgba, **kw):
        return self.g(parent, obj, type="box", pos=tuple(pos), size=tuple(half), rgba=rgba, **kw)

    def cap(self, parent, obj, p0, p1, r, rgba, **kw):
        return self.g(parent, obj, type="capsule", fromto=(*p0, *p1), size=(r,), rgba=rgba, **kw)

    def cyl(self, parent, obj, p0, p1, r, rgba, **kw):
        return self.g(parent, obj, type="cylinder", fromto=(*p0, *p1), size=(r,), rgba=rgba, **kw)

    def ell(self, parent, obj, pos, half, rgba, **kw):
        return self.g(parent, obj, type="ellipsoid", pos=tuple(pos), size=tuple(half), rgba=rgba, **kw)

    def sph(self, parent, obj, pos, r, rgba, **kw):
        return self.g(parent, obj, type="sphere", pos=tuple(pos), size=(r,), rgba=rgba, **kw)


# ======================= сцена =======================
class StreamerStage:
    """Та же «поверхность», что у stage.Stage для воркеров: set_state, set_screen_frame, render, close
    (+ fly_box / fly_screen_xy — где муха на картинке)."""

    def __init__(self, style=None, watermark="", tex_size=(960, 540)):
        if style is None:
            from style import RoomStyle
            style = RoomStyle(0)
        self.style = style
        self.watermark = watermark or ""
        self.tex_size = tex_size
        prepare_assets(style)                     # есть уже — мгновенно
        self.pose = {}
        self.glow = np.array([0.35, 0.40, 0.55])
        self._leds = []                           # (имена сегментов, фазы 0..1)
        self._led_halos = []                      # (имя ореола, индексы сегментов в общем списке, альфа)
        self._hp_leds, self._spk_leds = [], []
        arena = _Room()
        self._build(arena.root_element)
        self.fly = Fly(spawn_pos=(0, 0, 0), enable_adhesion=False)
        KitchenArena.add_extra_joints(self.fly)
        self._build_headphones(self.fly.model)
        v = self.fly.model.visual
        v.quality.shadowsize = 2048
        v.quality.offsamples = 8
        getattr(v, "global").offwidth = 1920
        getattr(v, "global").offheight = 1920
        v.map.haze = 0.0
        L = self.style.light
        v.headlight.ambient = tuple(float(x) for x in np.clip(np.asarray(L["ambient"]) * 2.0 + 0.07, 0, 1))
        v.headlight.diffuse = (0.0, 0.0, 0.0)
        v.headlight.specular = (0.0, 0.0, 0.0)
        self.sim = SingleFlySimulation(fly=self.fly, arena=arena, cameras=[], timestep=1e-4)
        self.sim.reset()
        self.p = self.sim.physics
        self.m, self.d = self.p.model.ptr, self.p.data.ptr
        self._init_ids()
        if LOOK == "toon":
            self._toon_fly()
        self.poser = FlyPoser(self.p, self.fly.name)
        self._measure()
        self._renderers = {}
        self._post_cache = {}
        self._segtab = None
        self.opt = mujoco.MjvOption()
        self.opt_seg = mujoco.MjvOption()
        for g in range(6):
            self.opt.geomgroup[g] = 1
            self.opt_seg.geomgroup[g] = 0 if g == G_HALO else 1

    # ------------------------------------------------------------------ сборка комнаты
    def _build(self, r):
        S = self.style
        A, wb = r.asset, r.worldbody
        B = self.B = _Builder(A)
        tex = lambda n, **kw: A.add("texture", name=f"t_{n}", type="2d", file=str(asset_path(n)), **kw)
        _base_materials(A)
        wp = S.wall["pattern"]
        A.add("material", name="wall", texture=tex(f"wall_{wp}"), specular=0.0, shininess=0.0)
        A.add("material", name="floor", texture=tex(f"floor_{S.extras['floor']}"), specular=0.05, shininess=0.1)
        A.add("material", name="halo", texture=tex("halo", nchannel=4), emission=1.0, specular=0.0)
        A.add("material", name="glowbar", texture=tex("glowbar", nchannel=4), emission=1.0, specular=0.0)
        A.add("material", name="keys", texture=tex("keys"), emission=0.55, specular=0.0)
        for k in range(6):
            A.add("material", name=f"poster{k}", texture=tex(f"poster_{k}"), emission=0.3, specular=0.0)
        for k in range(3):
            A.add("material", name=f"window{k}", texture=tex(f"window_{k}"), emission=0.9, specular=0.0)
        self._build_room(wb)
        self._build_chair(wb)
        if S.extras["desk"]:
            self._build_desk(wb)
        self._build_feeder(wb)
        self._build_lights(wb)
        claw = wb.add("body", name="claw", pos=(1.0, 0.0, 3.0), mocap=True)
        cols = dict(metal="metal", dark="satin", glow="glow")

        def add(kind, rgba, mat, **kw):
            return B.g(claw, "claw", group=G_PROP, type=kind, rgba=rgba, material=cols[mat], **kw).name
        self._claw_names = SG.claw_build(add)
        wb.add("camera", name="facecam", pos=(10, 0, 3), mode="fixed", fovy=FOVY)

    def _led_strip(self, parent, obj, pts, r, glow_every=0, glow_size=(2.0, 2.0), glow_off=(0.05, 0, 0),
                   glow_alpha=0.5, mat="glow", halo_q=None):
        """Лента из сегментов-капсул по ломаной pts (N, 3); сегменты красятся по style.led каждый кадр."""
        pts = np.asarray(pts, float)
        names = []
        for k in range(len(pts) - 1):
            g = self.B.cap(parent, obj, pts[k], pts[k + 1], r, (1, 1, 1, 1), material=mat)
            names.append(g.name)
        base = sum(len(n) for n, _ in self._leds)
        self._leds.append((names, np.linspace(0, 1, len(names), endpoint=False)))
        if glow_every:
            A = self.B.A
            mesh = f"ledglow{len(self._led_halos)}"
            _panel_mesh(A, mesh, glow_size[0], glow_size[1], 0.005)
            for k in range(0, len(names), glow_every):
                idx = list(range(base + k, base + min(len(names), k + glow_every)))
                c = pts[k:k + glow_every + 1].mean(0) + np.asarray(glow_off)
                g = self.B.g(parent, "bgglow", group=G_HALO, type="mesh", mesh=mesh, pos=tuple(c),
                             quat=halo_q or (1, 0, 0, 0), material="glowbar", rgba=(1, 1, 1, glow_alpha))
                self._led_halos.append((g.name, idx, glow_alpha))
        return names

    def _halo(self, parent, pos, size, rgba, quat=(1, 0, 0, 0), mat="halo"):
        mesh = f"halo_{self.B.n}"
        _panel_mesh(self.B.A, mesh, size[0], size[1], 0.005)
        return self.B.g(parent, "bghalo", group=G_HALO, type="mesh", mesh=mesh, pos=tuple(pos), quat=quat,
                        material=mat, rgba=rgba)

    def _build_room(self, wb):
        S, B, A = self.style, self.B, self.B.A
        E = S.extras
        room = wb.add("body", name="room", quat=_q((0, 0, 1), -E["swivel"]))
        self.room_quat = _q((0, 0, 1), -E["swivel"])
        wc = np.clip(np.asarray(S.wall["color"]) * 1.9 + 0.05, 0, 1)
        wc2 = np.clip(np.asarray(S.wall["color2"]) * 1.9 + 0.05, 0, 1)
        H = CEIL_Z - FLOOR_Z
        tile = 4.0
        _panel_mesh(A, "wall_back", 2 * SIDE_Y, H, 0.2, urep=2 * SIDE_Y / tile, vrep=H / tile)
        B.g(room, "bgwall", type="mesh", mesh="wall_back", pos=(WALL_X, 0, (CEIL_Z + FLOOR_Z) / 2), material="wall",
            rgba=_c(wc))
        LS = 24.0
        _panel_mesh(A, "wall_side", LS, H, 0.2, urep=LS / tile, vrep=H / tile)
        for s in (1, -1):
            B.g(room, "bgwall", type="mesh", mesh="wall_side", pos=(WALL_X + LS / 2, s * SIDE_Y, (CEIL_Z + FLOOR_Z) / 2),
                quat=_q((0, 0, 1), -s * np.pi / 2), material="wall", rgba=_c(wc, 0.85))
        # плинтус и потолочный карниз (второй цвет стены)
        B.box(room, "bgtrim", (WALL_X + 0.08, 0, FLOOR_Z + 0.25), (0.1, SIDE_Y, 0.25), _c(wc2, 0.8))
        # пол + ковёр
        _panel_mesh(A, "floor_m", 40, 40, 0.1, urep=10, vrep=10)
        B.g(room, "bgfloor", type="mesh", mesh="floor_m", pos=(2, 0, FLOOR_Z), quat=_q((0, 1, 0), -np.pi / 2),
            material="floor", rgba=_c(E["floor_color"]))
        if E["rug"]:
            B.cyl(room, "rug", (0.2, 0, FLOOR_Z), (0.2, 0, FLOOR_Z + 0.04), 3.6, _c(E["rug_color"], 0.9))
        # подсветка стены
        led = _bright(S.led["color"])
        lay = S.led["layout"]
        q_wall = (1, 0, 0, 0)
        if lay == "strip":
            ys = np.linspace(-SIDE_Y + 0.5, SIDE_Y - 0.5, 49)
            pts = [(WALL_X + 0.08, y, 3.3) for y in ys]
            self._led_strip(room, "bgled", pts, 0.06, glow_every=4, glow_size=(2.4, 1.8), glow_off=(0.03, 0, 0),
                            glow_alpha=0.35)
        elif lay == "bars":
            for s in (-1, 1):
                x0, y0 = WALL_X + 0.9, s * 6.4
                B.cyl(room, "lbar", (x0, y0, FLOOR_Z), (x0, y0, FLOOR_Z + 0.12), 0.45, (0.06, 0.06, 0.07, 1))
                zs = np.linspace(FLOOR_Z + 0.25, 6.2, 17)
                pts = [(x0, y0, z) for z in zs]
                self._led_strip(room, "bgled", pts, 0.13, glow_every=2, glow_size=(2.6, 1.6),
                                glow_off=(-0.8, 0, 0), glow_alpha=0.3)
        else:                                            # frame: линии под потолком и у пола
            for z in (6.6, FLOOR_Z + 0.65):
                ys = np.linspace(-SIDE_Y + 0.5, SIDE_Y - 0.5, 37)
                pts = [(WALL_X + 0.08, y, z) for y in ys]
                self._led_strip(room, "bgled", pts, 0.06, glow_every=4, glow_size=(2.4, 1.8), glow_off=(0.03, 0, 0),
                                glow_alpha=0.35)
        for p in S.props:
            if p.get("slot"):
                self._build_prop(room, p, led)
        self.led_col = led

    # ------------------------------------------------------------------ пропсы
    def _build_prop(self, room, p, led):
        B, S = self.B, self.style
        nm, slot = p["name"], p["slot"]
        sc = float(p.get("scale", 1.0))
        col = np.asarray(p["color"], float)
        col2 = np.asarray(p.get("color2", p["color"]), float)
        glowc = _bright(col) if _lum(col) > 0.25 else led
        art = int(p.get("art", 0))
        o = f"p{nm}{slot}"
        if slot in WALL_SLOTS:
            y, z = WALL_SLOTS[slot]
            x = WALL_X
            if nm == "neon_sign":
                w = 5.2 if slot == "C_hi" else 4.2
                w *= sc
                h = w * 320 / 1024
                name = _neon_name(p["text"])
                tex_t = self.B.A.add("texture", name=f"t_{name}", type="2d", file=str(asset_path(name, p["text"])),
                                     nchannel=4)
                tex_g = self.B.A.add("texture", name=f"t_{name}g", type="2d", file=str(ASSETS / f"{name}g_s{TEX_VER}.png"),
                                     nchannel=4)
                self.B.A.add("material", name=f"m_{name}", texture=tex_t, emission=1.0, specular=0.0)
                self.B.A.add("material", name=f"m_{name}g", texture=tex_g, emission=1.0, specular=0.0)
                _panel_mesh(self.B.A, f"mesh_{name}", w, h, 0.01)
                B.box(room, o, (x + 0.06, y, z), (0.06, w / 2 * 0.98, h / 2 * 0.92), (0.04, 0.04, 0.06, 0.55),
                      material="glass")
                B.g(room, "bgneon", group=G_HALO, type="mesh", mesh=f"mesh_{name}", pos=(x + 0.14, y, z),
                    material=f"m_{name}g", rgba=_c(glowc, 1.0, 0.9))
                # альфа < 1: иначе MuJoCo рисует геом непрозрачным и альфа текстуры теряется (белый прямоугольник)
                B.g(room, "bgneon", group=G_ROOM, type="mesh", mesh=f"mesh_{name}", pos=(x + 0.16, y, z),
                    material=f"m_{name}", rgba=_c(_mix(glowc, (1, 1, 1), 0.55), 1.0, 0.99))
                self._halo(room, (x + 0.02, y, z), (w * 1.6, h * 3.0), _c(glowc, 1.0, 0.45))
            elif nm == "poster":
                w, h = 2.4 * sc, 3.2 * sc
                _panel_mesh(self.B.A, f"mesh_{o}", w, h, 0.03)
                B.g(room, o, type="mesh", mesh=f"mesh_{o}", pos=(x + 0.08, y, z), material=f"poster{art % 6}",
                    rgba=(1, 1, 1, 1))
                fr = _c(col2 if _lum(col2) > 0.15 else col, 0.9)
                for dz in (-1, 1):
                    B.box(room, o + "f", (x + 0.1, y, z + dz * (h / 2 + 0.06)), (0.06, w / 2 + 0.12, 0.06), fr)
                for dy in (-1, 1):
                    B.box(room, o + "f", (x + 0.1, y + dy * (w / 2 + 0.06), z), (0.06, 0.06, h / 2 + 0.12), fr)
            elif nm == "window":
                w, h = 3.8 * sc, 5.0 * sc
                z = 1.2
                _panel_mesh(self.B.A, f"mesh_{o}", w, h, 0.03)
                B.g(room, "bgwin", type="mesh", mesh=f"mesh_{o}", pos=(x + 0.02, y, z), material=f"window{art % 3}",
                    rgba=(1, 1, 1, 1))
                frc = _c(col2 if _lum(col2) > 0.3 else (0.92, 0.92, 0.95))
                for dz in (-1, 0, 1):
                    B.box(room, o, (x + 0.12, y, z + dz * h / 2), (0.1, w / 2 + 0.15, 0.1), frc)
                for dy in (-1, 0, 1):
                    B.box(room, o, (x + 0.12, y + dy * w / 2, z), (0.1, 0.1, h / 2 + 0.15), frc)
                B.box(room, o, (x + 0.35, y, z - h / 2 - 0.12), (0.35, w / 2 + 0.35, 0.08), frc)       # подоконник
                for dy in (-1, 1):                                                                        # шторы
                    B.ell(room, o + "c", (x + 0.35, y + dy * (w / 2 + 0.35), z + 0.2), (0.25, 0.45, h / 2 + 0.4),
                          _c(col, 0.9))
                self._halo(room, (x + 0.4, y, z), (w * 1.8, h * 1.5), _c(_mix((0.6, 0.7, 1.0), col, 0.2), 1.0, 0.18))
            elif nm == "shelf":
                w = 3.4 * sc
                zz = z - 0.6
                B.box(room, o, (x + 0.55, y, zz), (0.55, w / 2, 0.08), _c(col2 if _lum(col2) > 0.12 else (0.35, 0.25, 0.2)))
                self._led_strip(room, "bgled", [(x + 1.08, y - w / 2 + 0.1, zz - 0.1), (x + 1.08, y + w / 2 - 0.1, zz - 0.1)],
                                0.03)
                # книги, кубок, мини-растение, фигурка
                rng = np.random.default_rng(zlib.crc32(o.encode()))
                yy = y - w / 2 + 0.3
                for k in range(4):
                    bh = rng.uniform(0.6, 0.95)
                    bc = _c(rng.choice([col, col2, led, (0.9, 0.9, 0.85)]) * rng.uniform(0.7, 1.0))
                    B.box(room, o + "b", (x + 0.5, yy, zz + 0.08 + bh / 2), (0.35, 0.1, bh / 2), bc)
                    yy += 0.22
                gold = (1.0, 0.78, 0.3, 1)
                ty = y + 0.2
                B.box(room, o + "t", (x + 0.5, ty, zz + 0.18), (0.18, 0.18, 0.1), (0.1, 0.08, 0.06, 1))
                B.cyl(room, o + "t", (x + 0.5, ty, zz + 0.28), (x + 0.5, ty, zz + 0.5), 0.05, gold, material="metal")
                B.ell(room, o + "t", (x + 0.5, ty, zz + 0.68), (0.22, 0.22, 0.22), gold, material="metal")
                B.cyl(room, o + "t", (x + 0.5, ty, zz + 0.68), (x + 0.5, ty, zz + 0.95), 0.22, gold, material="metal")
                py = y + w / 2 - 0.45
                B.cyl(room, o + "p", (x + 0.5, py, zz + 0.08), (x + 0.5, py, zz + 0.45), 0.2, _c(col, 0.9))
                for k in range(5):
                    a = k * 2 * np.pi / 5
                    B.ell(room, o + "l", (x + 0.5 + 0.12 * np.cos(a), py + 0.12 * np.sin(a), zz + 0.7),
                          (0.07, 0.07, 0.3), (0.25, 0.7, 0.3, 1), quat=_q((-np.sin(a), np.cos(a), 0), 0.4))
        else:
            fx, fy = FLOOR_SLOTS[slot]
            z0 = FLOOR_Z
            if nm == "plant":
                h = 1.6 * sc
                B.cyl(room, o, (fx, fy, z0), (fx, fy, z0 + h), 0.75 * sc, _c(col if _lum(col) > 0.15 else col2))
                B.cyl(room, o + "r", (fx, fy, z0 + h - 0.05), (fx, fy, z0 + h + 0.08), 0.8 * sc, _c(col2, 0.8))
                rng = np.random.default_rng(zlib.crc32(o.encode()))
                greens = [(0.18, 0.62, 0.28), (0.28, 0.75, 0.35), (0.12, 0.5, 0.25)]
                for k in range(9):
                    a = k * 2 * np.pi / 9 + rng.uniform(-0.2, 0.2)
                    tilt = rng.uniform(0.25, 0.7)
                    L = rng.uniform(1.4, 2.4) * sc
                    d = np.array([np.cos(a) * np.sin(tilt), np.sin(a) * np.sin(tilt), np.cos(tilt)])
                    c = np.array([fx, fy, z0 + h]) + d * L * 0.55
                    qq = Rot.align_vectors([d], [[0, 0, 1]])[0].as_quat()
                    B.ell(room, o + "l", c, (0.28 * sc, 0.08, L * 0.55), _c(greens[k % 3]),
                          quat=(float(qq[3]), float(qq[0]), float(qq[1]), float(qq[2])))
            elif nm == "lamp":
                h = 6.2 * sc
                B.cyl(room, o, (fx, fy, z0), (fx, fy, z0 + 0.1), 0.7, (0.12, 0.12, 0.14, 1), material="metal")
                B.cap(room, o, (fx, fy, z0), (fx, fy, z0 + h), 0.07, (0.15, 0.15, 0.17, 1), material="metal")
                B.sph(room, "bglamp", (fx, fy, z0 + h - 0.1), 0.35, _c(_mix(glowc, (1, 0.95, 0.8), 0.6)), material="glow")
                B.cyl(room, o + "s", (fx, fy, z0 + h - 0.35), (fx, fy, z0 + h + 0.55), 0.8 * sc, _c(col, 1.0),
                      material="soft")
                self._halo(room, (WALL_X + 0.03, fy * 1.02, z0 + h), (6.0, 6.0), _c(_mix(glowc, (1, 0.9, 0.7), 0.5), 1, 0.5))
            elif nm == "speaker":
                h = 3.6 * sc
                B.box(room, o, (fx, fy, z0 + h / 2), (0.75, 0.75, h / 2), (0.07, 0.07, 0.09, 1), material="satin")
                for k, (zz, rr) in enumerate(((z0 + h * 0.3, 0.55), (z0 + h * 0.72, 0.32))):
                    g = B.cyl(room, "bgspk", (fx + 0.75, fy, zz), (fx + 0.78, fy, zz), rr, (1, 1, 1, 1), material="glow")
                    self._spk_leds.append(g.name)
                    B.cyl(room, o + "c", (fx + 0.76, fy, zz), (fx + 0.82, fy, zz), rr * 0.8, (0.15, 0.15, 0.17, 1))
                    B.sph(room, o + "c", (fx + 0.82, fy, zz), rr * 0.3, (0.3, 0.3, 0.33, 1), material="metal")
            elif nm == "figure":
                # тумба + игрушка (банан-плюш или робот)
                B.box(room, o, (fx, fy, z0 + 1.1), (0.7, 0.7, 1.1), _c(col2 if _lum(col2) > 0.12 else (0.2, 0.2, 0.25)))
                zt = z0 + 2.2
                if art % 2 == 0:
                    pts = [np.array([0.0, 1.0 * np.sin(a), 0.7 * (1 - np.cos(a))]) for a in np.linspace(-1.2, 1.2, 9)]
                    for k in range(8):
                        rad = 0.2 + 0.14 * np.sin(np.pi * (k + 0.5) / 8) ** 0.6
                        p1, p2 = pts[k] + [fx, fy, zt + rad], pts[k + 1] + [fx, fy, zt + rad]
                        tip = k in (0, 7)
                        B.cap(room, o + "b", p1, p2, rad, (0.35, 0.25, 0.1, 1) if tip else (1.0, 0.86, 0.25, 1))
                    for s in (1, -1):
                        B.sph(room, o + "e", (fx + 0.33, fy + 0.18 * s, zt + 0.45), 0.06, (0.02, 0.02, 0.02, 1))
                else:
                    c = _c(col if _lum(col) > 0.15 else led)
                    B.box(room, o + "r", (fx, fy, zt + 0.55), (0.35, 0.4, 0.45), c, material="satin")
                    B.box(room, o + "h", (fx, fy, zt + 1.25), (0.3, 0.35, 0.25), (0.85, 0.85, 0.9, 1), material="satin")
                    for s in (1, -1):
                        B.sph(room, "bgeye", (fx + 0.3, fy + 0.14 * s, zt + 1.28), 0.07, _c(led), material="glow")
                        B.cap(room, o + "a", (fx, fy + s * 0.45, zt + 0.8), (fx + 0.2, fy + s * 0.55, zt + 0.35), 0.09, c)
                    B.cap(room, o + "n", (fx, fy, zt + 1.5), (fx, fy, zt + 1.75), 0.03, (0.3, 0.3, 0.3, 1))
                    B.sph(room, "bgeye", (fx, fy, zt + 1.78), 0.07, _c(led), material="glow")

    # ------------------------------------------------------------------ кресло
    def _build_chair(self, wb):
        S, B = self.style, self.B
        ch = S.chair
        P, Sd, Ac = np.asarray(ch["primary"]), np.asarray(ch["secondary"]), np.asarray(ch["accent"])
        model = ch["model"]
        chair = wb.add("body", name="chair")
        back = chair.add("body", name="chair_back", pos=tuple(BACK_PIVOT), quat=_q((0, 1, 0), -BACK_TILT))
        pil = {"accent": Ac, "primary": P, "secondary": Sd}[ch.get("pillow", "accent")]
        wheel = (0.75, 0.76, 0.8, 1) if ch.get("wheels") == "chrome" else (0.08, 0.08, 0.09, 1)
        edges = []                                         # ломаные RGB-подсветки кресла
        cP, cS, cA = _c(P), _c(Sd), _c(Ac)
        if model == "racer":
            B.box(chair, "seat", (0.25, 0, -0.2), (1.05, 1.15, 0.2), cS, material="satin")
            B.cap(chair, "seatf", (1.3, -1.1, -0.17), (1.3, 1.1, -0.17), 0.2, cP, material="satin")
            for s in (-1, 1):
                B.cap(chair, "seatb", (-0.75, s * 1.32, 0.0), (1.25, s * 1.32, 0.0), 0.28, cP, material="satin")
            # спинка (в системе спинки: z вдоль спинки, x — вперёд)
            B.box(back, "back", (-0.22, 0, 1.75), (0.22, 1.2, 1.75), cS, material="satin")
            B.box(back, "backtop", (-0.22, 0, 3.85), (0.22, 0.95, 0.4), cS, material="satin")
            B.cap(back, "backtop", (-0.22, -0.8, 4.25), (-0.22, 0.8, 4.25), 0.22, cS, material="satin")
            for s in (-1, 1):
                B.cap(back, "backw", (0.08, s * 1.3, 0.35), (0.08, s * 1.32, 3.0), 0.32, cP, material="satin")
                B.cap(back, "backw", (0.08, s * 1.32, 3.0), (-0.05, s * 1.0, 3.9), 0.28, cP, material="satin")
                B.ell(back, "slot", (0.01, s * 0.55, 3.75), (0.03, 0.13, 0.24), (0.02, 0.02, 0.03, 1))
                edges.append([(0.1, s * 1.64, 0.4), (0.1, s * 1.66, 3.0), (-0.02, s * 1.3, 3.95)])
            if ch.get("stripes", 0) == 2:
                for s in (-1, 1):
                    B.box(back, "stripe", (0.005, s * 0.72, 1.9), (0.01, 0.1, 1.6), cA)
            elif ch.get("stripes", 0) == 1:
                B.box(back, "stripe", (0.005, 0, 3.9), (0.01, 0.55, 0.06), cA)
                B.box(back, "stripe", (0.005, 0, 3.6), (0.01, 0.55, 0.06), cA)
            B.ell(back, "logo", (0.01, 0, 4.12), (0.03, 0.22, 0.14), cA, material="soft")
            B.ell(back, "pillow", (0.28, 0, 2.62), (0.3, 0.8, 0.32), _c(pil), material="satin")
            B.ell(back, "lumbar", (0.2, 0, 0.9), (0.18, 0.75, 0.3), _c(pil, 0.85), material="satin")
            self._armrests(chair, Sd, Ac)
            self._star_base(chair, wheel)
        elif model == "office":
            B.box(chair, "seat", (0.25, 0, -0.2), (1.05, 1.25, 0.2), cS, material="flat")
            B.cap(chair, "seatf", (1.28, -1.2, -0.15), (1.28, 1.2, -0.15), 0.22, cS, material="flat")
            # рама спинки (скруглённый прямоугольник) + сетка
            fr = [(0.0, -1.25, 0.5), (0.0, -1.35, 2.6), (0.0, -1.05, 3.9), (0.0, 1.05, 3.9), (0.0, 1.35, 2.6),
                  (0.0, 1.25, 0.5), (0.0, -1.25, 0.5)]
            for k in range(len(fr) - 1):
                B.cap(back, "frame", fr[k], fr[k + 1], 0.13, cP, material="satin")
            B.box(back, "mesh", (-0.05, 0, 2.2), (0.04, 1.2, 1.65), _c(Sd, 1.0, 0.92), material="flat")
            B.cap(back, "lumbar", (0.08, -1.1, 1.2), (0.08, 1.1, 1.2), 0.12, cA, material="satin")
            B.ell(back, "pillow", (0.12, 0, 4.35), (0.18, 0.75, 0.35), _c(pil), material="satin")
            B.cap(back, "hpost", (-0.1, 0, 3.9), (-0.1, 0, 4.1), 0.08, cP, material="metal")
            edges.append([(0.05, -1.4, 0.6), (0.05, -1.5, 2.6), (0.05, -1.15, 4.02), (0.05, 1.15, 4.02),
                          (0.05, 1.5, 2.6), (0.05, 1.4, 0.6)])
            self._armrests(chair, (0.1, 0.1, 0.12), Ac, t_shape=True)
            self._star_base(chair, wheel)
        elif model == "egg":
            _egg_mesh(self.B.A, "egg_out", 2.7, 2.45, 3.2, 0.18, 1.62)
            _egg_mesh(self.B.A, "egg_in", 2.52, 2.27, 3.02, 0.12, 1.58)
            eq = _q((0, 1, 0), -0.12)
            ec = (-0.55, 0, 1.55)
            B.g(chair, "eggshell", type="mesh", mesh="egg_out", pos=ec, quat=eq, rgba=cP, material="satin")
            B.g(chair, "egglining", type="mesh", mesh="egg_in", pos=ec, quat=eq, rgba=cS, material="flat")
            B.ell(chair, "seat", (0.15, 0, -0.2), (1.5, 1.75, 0.3), _c(pil), material="flat")
            B.ell(chair, "pillow", (-1.6, 0, 2.5), (0.35, 0.9, 0.4), cS, material="flat")
            B.cyl(chair, "base", (-0.4, 0, -1.6), (-0.4, 0, FLOOR_Z + 0.1), 0.28, (0.8, 0.8, 0.82, 1), material="metal")
            B.cyl(chair, "base", (-0.4, 0, FLOOR_Z), (-0.4, 0, FLOOR_Z + 0.12), 1.7, (0.8, 0.8, 0.82, 1), material="metal")
            # кромка яйца со светодиодами
            al = 1.6
            ph = np.linspace(0, 2 * np.pi, 33)
            rim = np.stack([-2.62 * np.cos(al) * np.ones_like(ph), 2.38 * np.sin(al) * np.cos(ph),
                            3.12 * np.sin(al) * np.sin(ph)], 1)
            R = Rot.from_quat([eq[1], eq[2], eq[3], eq[0]]).as_matrix()
            rim = rim @ R.T + np.asarray(ec)
            rim = rim[rim[:, 2] > -0.15]
            edges.append([tuple(p + np.array([0.12, 0, 0])) for p in rim])
            edges_body = chair
        else:                                               # throne
            gold = (1.0, 0.78, 0.3, 1)
            B.box(chair, "seat", (0.25, 0, -0.25), (1.1, 1.35, 0.25), cP, material="satin")
            B.cap(chair, "seatf", (1.35, -1.35, -0.25), (1.35, 1.35, -0.25), 0.12, gold, material="metal")
            B.box(chair, "plinth", (0.1, 0, (FLOOR_Z - 0.5) / 2), (1.3, 1.5, (-0.5 - FLOOR_Z) / 2), _c(Sd, 1.2),
                  material="satin")
            B.box(back, "back", (-0.28, 0, 2.5), (0.28, 1.35, 2.5), cP, material="satin")
            B.box(back, "backin", (0.01, 0, 2.3), (0.02, 0.95, 1.9), _c(P, 0.7), material="flat")
            for s in (-1, 1):
                B.cap(back, "gold", (0.05, s * 1.35, 0.2), (0.05, s * 1.35, 4.9), 0.12, gold, material="metal")
                B.sph(back, "gold", (0.05, s * 1.35, 5.05), 0.25, gold, material="metal")
                edges.append([(0.12, s * 1.1, 0.3), (0.12, s * 1.1, 4.5)])
            B.cap(back, "gold", (0.05, -1.35, 4.95), (0.05, 1.35, 4.95), 0.12, gold, material="metal")
            for k, (yy, hh) in enumerate(((-0.8, 0.7), (0.0, 1.1), (0.8, 0.7))):
                B.ell(back, "crown", (-0.1, yy, 5.0 + hh / 2), (0.14, 0.3, hh / 2), cP, material="satin")
                B.sph(back, "gold", (-0.1, yy, 5.05 + hh), 0.17, gold, material="metal")
            B.ell(back, "gem", (0.06, 0, 4.4), (0.05, 0.26, 0.32), cA, material="soft")
            B.ell(back, "pillow", (0.28, 0, 2.62), (0.3, 0.8, 0.32), _c(pil), material="satin")
            for s in (-1, 1):
                B.box(chair, "arm", (0.25, s * 1.72, 0.1), (1.0, 0.3, 0.55), _c(Sd, 1.2), material="satin")
                B.cap(chair, "armtop", (-0.7, s * 1.72, 0.7), (1.2, s * 1.72, 0.7), 0.3, cP, material="satin")
                B.sph(chair, "gold", (1.3, s * 1.72, 0.75), 0.34, gold, material="metal")
        if model != "egg":
            edges_body = back
        if ch.get("rgb_edges", True):
            for pts in edges:
                self._led_strip(edges_body, "bgled", self._resample(pts, 0.28), 0.045)

    @staticmethod
    def _resample(pts, step):
        """Ломаная -> точки через ~step мм (сегменты ленты одинаковой длины)."""
        pts = np.asarray(pts, float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = max(2, int(s[-1] / step) + 1)
        u = np.linspace(0, s[-1], n)
        return np.stack([np.interp(u, s, pts[:, k]) for k in range(3)], 1)

    def _armrests(self, chair, col, acc, t_shape=False):
        B = self.B
        for s in (-1, 1):
            y = s * 1.9
            B.cap(chair, "armpost", (0.15, y, -0.35), (0.15, y, 0.7), 0.1, (0.1, 0.1, 0.12, 1), material="satin")
            if t_shape:
                B.cap(chair, "armpad", (-0.4, y, 0.8), (0.9, y, 0.8), 0.14, _c(col), material="satin")
            else:
                B.box(chair, "armpad", (0.25, y, 0.8), (0.75, 0.18, 0.08), _c(col), material="satin")
                B.box(chair, "armpad", (0.25, y, 0.885), (0.7, 0.13, 0.01), _c(acc), material="flat")

    def _star_base(self, chair, wheel):
        B = self.B
        c0 = np.array([0.2, 0.0, -0.4])
        B.box(chair, "under", (0.2, 0, -0.5), (0.7, 0.7, 0.08), (0.08, 0.08, 0.09, 1))
        B.cyl(chair, "gas", (0.2, 0, -0.55), (0.2, 0, -1.9), 0.17, (0.7, 0.7, 0.74, 1), material="metal")
        B.cyl(chair, "gas", (0.2, 0, -1.9), (0.2, 0, -2.2), 0.26, (0.1, 0.1, 0.11, 1))
        hub = np.array([0.2, 0.0, -2.2])
        for k in range(5):
            a = k * 2 * np.pi / 5 + np.pi / 5
            end = hub + np.array([1.9 * np.cos(a), 1.9 * np.sin(a), -0.15])
            B.cap(chair, "star", hub, end, 0.13, (0.1, 0.1, 0.11, 1), material="satin")
            B.cyl(chair, "wheel", end + [0, -0.12, -0.25], end + [0, 0.12, -0.25], 0.2, wheel, material="metal")

    # ------------------------------------------------------------------ стол, клавиатура, микрофон
    def _build_desk(self, wb):
        S, B, A = self.style, self.B, self.B.A
        E = S.extras
        x0 = 2.6
        top = 0.3
        B.box(wb, "desk", (x0 + 5, 0, top - 0.14), (5, 8, 0.14), _c(E["desk_color"]), material="satin")
        self._led_strip(wb, "bgled", [(x0 - 0.02, -7.8, top - 0.3), (x0 - 0.02, 7.8, top - 0.3)], 0.04)
        _panel_mesh(A, "kbd", 3.2, 1.0, 0.1, 1, 1)
        kq = _q((0, 1, 0), -np.pi / 2)                     # лицом вверх, ряды клавиш вдоль x
        g = B.g(wb, "kbd", type="mesh", mesh="kbd", pos=(x0 + 0.75, 0, top + 0.12), quat=kq, material="keys",
                rgba=(1, 1, 1, 1))
        self._kbd = g.name
        B.box(wb, "kbdb", (x0 + 0.75, 0, top + 0.04), (0.55, 1.68, 0.04), (0.06, 0.06, 0.07, 1))
        B.box(wb, "pad", (x0 + 0.9, 2.9, top + 0.01), (0.7, 0.9, 0.01), (0.08, 0.08, 0.1, 1))
        B.ell(wb, "mouse", (x0 + 0.9, 2.9, top + 0.1), (0.32, 0.2, 0.1), (0.12, 0.12, 0.14, 1), material="satin")
        if E.get("mic_arm"):
            base = np.array([x0 + 0.4, -2.7, top])
            joint = np.array([x0 + 0.4, -2.7, 1.7])
            mic = np.array([1.95, -1.45, 1.75])
            aim = np.array([0.75, -0.1, 2.05])
            B.cyl(wb, "micarm", base, base + [0, 0, 0.2], 0.25, (0.1, 0.1, 0.11, 1))
            B.cap(wb, "micarm", base, joint, 0.07, (0.12, 0.12, 0.13, 1), material="satin")
            B.cap(wb, "micarm", joint, mic + [0.25, -0.2, 0.25], 0.07, (0.12, 0.12, 0.13, 1), material="satin")
            d = (aim - mic) / np.linalg.norm(aim - mic)
            B.cap(wb, "mic", mic - d * 0.25, mic + d * 0.35, 0.17, (0.08, 0.08, 0.09, 1), material="satin")
            B.sph(wb, "micg", mic + d * 0.42, 0.19, (0.3, 0.3, 0.32, 1), material="metal")
            B.cyl(wb, "micr", mic - d * 0.02, mic + d * 0.04, 0.2, _c(self.led_col))

    def _build_feeder(self, wb):
        """Капля сахара (стимул прохода 1) + стеклянный капилляр, который подносит её к хоботку."""
        B = self.B
        drop = wb.add("body", name="drop", pos=(0.9, 0, 1.7), mocap=True)
        g = B.sph(drop, "drop", (0, 0, 0), 0.2, (1.0, 0.72, 0.2, 0.0), material="soft", group=G_PROP)
        self._drop = g.name
        g = B.ell(drop, "dropshine", (0.08, 0.06, 0.08), (0.04, 0.04, 0.05), (1, 1, 1, 0.0), material="glow", group=G_PROP)
        self._dropshine = g.name
        pip = wb.add("body", name="pipette", pos=(0.9, 0, 30), mocap=True)
        d = np.array(PIP_D, float)
        d /= np.linalg.norm(d)
        self.PIP_DIR = d
        # мульт-пипетка: стеклянная трубочка с сиропом + резиновая груша на конце (читается как «пипетка»)
        L, rb = PIP_L, PIP_BULB
        B.cap(pip, "pipette", d * 0.1, d * L, 0.06, (0.80, 0.93, 1.0, 0.5), material="glass", group=G_PROP)
        B.cap(pip, "pipsyr", d * 0.2, d * (L * 0.55), 0.032, (1.0, 0.66, 0.18, 0.95), material="flat", group=G_PROP)
        B.cyl(pip, "pipring", d * (L - 0.02), d * (L + 0.07), 0.09, (0.85, 0.85, 0.9, 1), material="metal", group=G_PROP)
        qb = Rot.align_vectors([d], [[0, 0, 1]])[0].as_quat()
        B.ell(pip, "pipbulb", d * (L + rb[2]), rb, (0.92, 0.22, 0.28, 1), material="satin", group=G_PROP,
              quat=(float(qb[3]), float(qb[0]), float(qb[1]), float(qb[2])))

    def _build_lights(self, wb):
        L = self.style.light
        tgt = np.array([0.4, 0.0, 1.5])

        def spot(name, pos, col, cut=35, exp=2, shadow=False, body=wb, target=tgt):
            pos = np.asarray(pos, float)
            d = np.asarray(target, float) - pos
            body.add("light", name=name, pos=tuple(pos), dir=tuple(d / np.linalg.norm(d)), directional=False,
                     castshadow=shadow, cutoff=cut, exponent=exp, diffuse=tuple(float(x) for x in np.clip(col, 0, 1)),
                     specular=(0.05, 0.05, 0.05), attenuation=(1, 0, 0))

        key = np.asarray(_mix(L["key"], (1, 1, 1), 0.35))
        spot("key", (7.0, -4.0, 8.5), key * 1.0, cut=28, exp=3, shadow=True)
        spot("fill", (6.5, 5.0, 1.8), np.asarray(L["fill"]) * 1.3 + 0.08, cut=40, exp=1)
        spot("rim", (-4.5, 3.8, 6.0), np.asarray(L["rim"]) * 1.2, cut=30, exp=2, target=(0.3, 0, 2.0))
        spot("rim2", (-4.0, -4.2, 3.0), self.led_col * 0.75, cut=30, exp=2, target=(0.3, 0, 1.5))
        spot("screen", (9.5, 0.0, 2.6), (0.5, 0.5, 0.6), cut=45, exp=1)
        wall_c = np.asarray(L["fill"]) * 1.4 + np.asarray(L["key"]) * 0.25 + 0.06
        spot("wallwash", (3.0, 0.0, 8.0), wall_c, cut=70, exp=1, target=(WALL_X, 0, 1.0))

    # ------------------------------------------------------------------ наушники (на теле Head)
    def _build_headphones(self, fm):
        """Геомы наушников на теле Head модели мухи — поворачиваются вместе с головой.
        Система головы: x — вперёд (лицо), y — влево, z — вверх (спина); голова ~0.5 x 0.77 x 0.68 мм."""
        H = self.style.headphones
        _base_materials(fm.asset)
        head = fm.find("body", "Head")
        B = self.B
        k = float(H.get("size", 1.0))
        col, cush, band = _c(H["color"]), _c(H["cushion"]), _c(H.get("band", H["cushion"]))
        ledc = H.get("led")
        model = H["model"]
        xc = -0.02                                  # позади глаз (затылок)
        cy = 0.5 * k
        rc = 0.175 * k
        o = "hp"
        # глазные блики (мульт; постоянные)
        for s, nm in ((1, "eyeL"), (-1, "eyeR")):
            B.ell(head, f"{nm}shine", (0.33, s * 0.39, 0.1), (0.05, 0.02, 0.04), (1, 1, 1, 1), material="glow",
                  group=G_PROP)
        self._hp_leds = []
        for s in (-1, 1):
            if model == "studio":
                B.ell(head, o, (xc, s * cy, -0.03), (0.17 * k, 0.085 * k, 0.24 * k), col, material="satin", group=G_PROP)
                B.ell(head, o + "c", (xc, s * (cy - 0.07 * k), -0.03), (0.15 * k, 0.04, 0.21 * k), cush, group=G_PROP)
                if ledc is not None:
                    g = B.sph(head, "hpled", (xc - 0.05, s * (cy + 0.08 * k), -0.2 * k), 0.028, (1, 1, 1, 1),
                              material="glow", group=G_PROP)
                    self._hp_leds.append(g.name)
            else:
                B.cyl(head, o, (xc, s * (cy - 0.07 * k), -0.02), (xc, s * (cy + 0.07 * k), -0.02), rc, col,
                      material="satin", group=G_PROP)
                B.cyl(head, o + "c", (xc, s * (cy - 0.1 * k), -0.02), (xc, s * (cy - 0.06 * k), -0.02), rc * 0.9, cush,
                      group=G_PROP)
                if ledc is not None:
                    g = B.cyl(head, "hpled", (xc, s * (cy + 0.07 * k), -0.02), (xc, s * (cy + 0.085 * k), -0.02),
                              rc * 0.78, (1, 1, 1, 1), material="glow", group=G_PROP)
                    self._hp_leds.append(g.name)
                    B.cyl(head, o + "d", (xc, s * (cy + 0.08 * k), -0.02), (xc, s * (cy + 0.092 * k), -0.02),
                          rc * 0.5, col, material="satin", group=G_PROP)
            # скоба от чашки к дужке
            B.cap(head, o, (xc, s * cy, rc * 0.8), (xc, s * cy * 0.97, rc + 0.08), 0.035 * k, band, group=G_PROP)
        # дужка через макушку: дуга в плоскости x = xc
        R = 0.52 * k
        th = np.linspace(-1.1, 1.1, 11)
        arc = [(xc, R * np.sin(t), 0.02 + R * np.cos(t) * 0.92) for t in th]
        for i in range(len(arc) - 1):
            B.cap(head, o + "b", arc[i], arc[i + 1], 0.045 * k, band if model != "studio" else (0.7, 0.7, 0.74, 1),
                  material="satin" if model != "studio" else "metal", group=G_PROP)
        if model == "studio":
            arc2 = [(xc - 0.08, R * 0.98 * np.sin(t), 0.02 + R * np.cos(t) * 0.9) for t in th]
            for i in range(len(arc2) - 1):
                B.cap(head, o + "b", arc2[i], arc2[i + 1], 0.03 * k, col, material="satin", group=G_PROP)
        if model == "cat_ears":
            for s in (-1, 1):
                t = s * 0.5
                base = np.array([xc, R * np.sin(t), 0.02 + R * np.cos(t) * 0.92])
                qq = _q((1, 0, 0), -s * 0.45)
                B.ell(head, o + "ear", base + [0, s * 0.03, 0.1], (0.06, 0.1, 0.15), col, quat=qq, material="satin",
                      group=G_PROP)
                g = B.ell(head, "hpled", base + [0.045, s * 0.03, 0.1], (0.025, 0.06, 0.1), (1, 1, 1, 1), quat=qq,
                          material="glow", group=G_PROP)
                if ledc is not None:
                    self._hp_leds.append(g.name)
                else:
                    g.rgba = _c(cush)
                    g.material = "flat"
        if model == "headset_mic" or H.get("mic"):
            pts = [(xc + 0.05, -(cy + 0.04), -0.14), (0.3, -0.52, -0.3), (0.52, -0.42, -0.4), (0.66, -0.3, -0.42)]
            for i in range(len(pts) - 1):
                B.cap(head, o + "m", pts[i], pts[i + 1], 0.03, (0.1, 0.1, 0.11, 1), material="satin", group=G_PROP)
            B.ell(head, o + "mf", (0.7, -0.28, -0.42), (0.07, 0.06, 0.06), (0.12, 0.12, 0.13, 1), group=G_PROP)
            if ledc is not None:
                g = B.sph(head, "hpled", (0.74, -0.26, -0.39), 0.018, (1, 1, 1, 1), material="glow", group=G_PROP)
                self._hp_leds.append(g.name)

    # ------------------------------------------------------------------ id, мульт-материалы мухи
    def _init_ids(self):
        m = self.m
        name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        fp = f"{self.fly.name}/"
        full = lambda n: n if gid(n) >= 0 else fp + n
        self.led_gid = np.array([gid(full(n)) for names, _ in self._leds for n in names], int)
        self.led_phase = np.concatenate([ph for _, ph in self._leds]) if self._leds else np.zeros(0)
        self.led_halo = [(gid(full(n)), np.array(idx), a) for n, idx, a in self._led_halos]
        self.hp_led_gid = np.array([gid(full(n)) for n in self._hp_leds], int)
        self.spk_gid = np.array([gid(full(n)) for n in self._spk_leds], int)
        self.drop_gid = gid(full(self._drop))
        self.dropshine_gid = gid(full(self._dropshine))
        assert (self.led_gid >= 0).all() and self.drop_gid >= 0 and (self.hp_led_gid >= 0).all()
        jadr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{fp}{n}")]
        self.leg_adr = np.array([jadr(j) for j in self.fly.actuated_joints])
        self.front_idx = np.array([i for i, j in enumerate(self.fly.actuated_joints) if j[6:8] in ("LF", "RF")])
        self.head_roll_adr = jadr("joint_Head_roll")
        self.head_pitch_adr = jadr("joint_Head")
        self.wing_adr = {s: (jadr(f"joint_{s}Wing_yaw"), jadr(f"joint_{s}Wing_roll")) for s in "LR"}
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        assert len(free) == 1
        self.root_adr = m.jnt_qposadr[free[0]]
        self.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "facecam")
        body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.drop_mocap = m.body_mocapid[body("drop")]
        self.pip_mocap = m.body_mocapid[body("pipette")]
        self.light = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, n) for n in ("key", "fill", "rim", "screen")}
        # приступ/смерть/новая муха: захват, наушники (сползают), свет (гаснет)
        self.claw_gid = {k: [gid(full(n)) for n in v] for k, v in self._claw_names.items()}
        self.claw_mocap = m.body_mocapid[body("claw")]
        m.body_pos[body("claw")] = SG.CLAW_HIDE
        self.light_diff0 = {k: m.light_diffuse[v].copy() for k, v in self.light.items()}
        hb = body(f"{fp}Head")
        self.hp_gid = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == hb and name(g)[len(fp):].startswith("hp")], int)
        self.hp_pos0 = m.geom_pos[self.hp_gid].copy()
        self.hp_quat0 = m.geom_quat[self.hp_gid].copy()
        self.fly_geoms = np.array([g for g in range(m.ngeom) if name(g).startswith(fp)
                                   and "__" not in name(g)])
        # группы контура: объект = часть имени до '__' (декор) или часть тела мухи
        groups = {}
        self.group_of = np.zeros(m.ngeom + 2, np.int32)
        self.glow_of = np.zeros(m.ngeom + 2, np.uint8)            # светящиеся геомы — источник свечения (блум)
        self.kind_of = np.zeros(m.ngeom + 2, np.uint8)            # заливка мухи: 1 тело, 2 лапы, 3 глаза (0 — нет)
        for g in range(m.ngeom):
            n = name(g)
            if n.startswith(fp):
                n = n[len(fp):]
                if "__" not in n:
                    part = self._fly_part(n)
                    thin = part.startswith("leg") or n[1:] in ("Pedicel", "Funiculus", "Arista")   # свой цвет, мало размытия
                    self.kind_of[g + 1] = 2 if thin else 3 if part.startswith("eye") else \
                        0 if part.startswith("wing") else 1
            if "__" in n:
                key = n.split("__", 1)[0]
                if key in GLOW_OBJS:
                    self.glow_of[g + 1] = 1
                if key.startswith("bg") or key.startswith("hpled"):
                    key = "bg" if key.startswith("bg") else "hp"
                if key.endswith("shine"):
                    key = key[:-5]                    # блик глаза — без своего контура
            else:
                key = self._fly_part(n)
            if key == "bg":
                self.group_of[g + 1] = 0
            else:
                self.group_of[g + 1] = groups.setdefault(key, len(groups) + 1)
        self.group_of[-1] = len(groups) + 7           # смешанные id на краях (MSAA) — тоже граница
        m.vis.map.znear = 0.05 / m.stat.extent
        m.vis.map.zfar = 120.0 / m.stat.extent

    @staticmethod
    def _fly_part(n):
        if n in ("LEye", "REye"):
            return "eye" + n[0]
        if n.startswith("A") and n[1:2].isdigit():
            return "abd"
        # усики и жужжальца — до ног: 'LFuniculus' / 'LHaltere' начинаются как лапы LF / LH
        if n[1:] in ("Pedicel", "Funiculus", "Arista"):
            return ANT_GROUP
        if "Haltere" in n:
            return "halt"
        for leg in ("LF", "LM", "LH", "RF", "RM", "RH"):
            if n.startswith(leg) and n.endswith("Coxa"):
                return "thorax"
            if n.startswith(leg):
                return "leg" + leg
        if "Wing" in n:
            return "wing" + n[0]
        if n in ("Rostrum", "Haustellum") or n.startswith("prob_"):
            return "prob"
        return n.lower()

    def _toon_fly(self):
        """Мульт-материалы мухи (только в этой модели; файлы flygym не трогаем): без текстур и бликов."""
        m = self.m
        fp = f"{self.fly.name}/"
        for part, col in FLY_MAT.items():
            mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"{fp}{part}_material")
            if mid < 0:
                continue
            m.mat_texid[mid, :] = -1
            m.mat_rgba[mid] = (*col, FLY_ALPHA.get(part, 1.0))
            m.mat_specular[mid] = 0.25 if part == "eye" else 0.0
            m.mat_shininess[mid] = 0.4 if part == "eye" else 0.0
            m.mat_emission[mid] = 0.0 if part == "wing" else FLY_EMIT
            m.mat_reflectance[mid] = 0.0
        for seg, col in ABD_BANDS.items():
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{fp}{seg}")
            if g >= 0:
                m.geom_rgba[g] = (*col, 1.0)
        for n in ("prob_tube_sensorfx", "prob_lobe-1_sensorfx", "prob_lobe1_sensorfx"):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{fp}{n}")
            if g >= 0:
                m.geom_rgba[g, :3] = FLY_MAT["proboscis"]

    # ------------------------------------------------------------------ поза
    def _rest_legs(self):
        q = PreprogrammedSteps().default_pose.copy().ravel()
        names = list(self.fly.actuated_joints)
        for k, v in LEG_REST.items():
            for side in "LR":
                i = names.index(f"joint_{side}{k}")
                q[i] = v * (-1 if (side == "R" and ("roll" in k or "yaw" in k)) else 1)
        return q

    def _measure(self):
        """Один раз: поза покоя, точки мухи для рамки/кадра, где кончик хоботка (капля)."""
        m, d = self.m, self.d
        self.drop_pos = np.array([0.9, 0.0, 1.7])
        self.legs_rest = self._rest_legs()
        names = list(self.fly.actuated_joints)
        self.groom_aim = np.zeros(len(names))
        for k, v in GROOM_AIM.items():
            for side in "LR":
                sg = -1 if (side == "R" and ("roll" in k or "yaw" in k)) else 1       # правая — зеркально
                self.groom_aim[names.index(f"joint_{side}F{k}")] = v * sg
        # средняя поза чистки (Stage._groom_angles: между «лицо» и «макушка») уже с поправкой GROOM_AIM — из неё
        # плавный выход из чистки, когда последняя поза чистки кадру неизвестна
        base = PreprogrammedSteps().default_pose.copy().ravel()
        self.groom_goal = self.legs_rest.copy()
        for side in "LR":
            for k, v in (("Coxa", -0.75), ("Femur", -1.5), ("Tibia", 0.8), ("Coxa_roll", 0.0)):
                i = names.index(f"joint_{side}F{k}")
                self.groom_goal[i] = base[i] + v
        fi = self.front_idx
        self.groom_goal[fi] += self.groom_aim[fi]
        self._path_spline = None
        self._apply_pose({"prob": 1.0})
        tips = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{self.fly.name}/{n}")
                for n in ("prob_lobe-1_sensorfx", "prob_lobe1_sensorfx")]
        tip = d.geom_xpos[tips].mean(0)
        hau = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{self.fly.name}/Haustellum")]
        dirv = tip - hau
        dirv /= max(1e-6, np.linalg.norm(dirv))
        self.drop_pos = tip + dirv * 0.16
        # точки мухи (углы AABB геомов) в покое и с раскрытыми крыльями — для рамки и кадра
        pts = []
        for spread in (0.0, 1.2):
            self._apply_pose({"wing_yaw": (-spread, spread)})
            for g in self.fly_geoms:
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                loc = c + h * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
                pts.append(d.geom_xpos[g] + loc @ d.geom_xmat[g].reshape(3, 3).T)
        self._apply_pose({})
        head = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{self.fly.name}/Head")].copy()
        hgeoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                                                                         f"{self.fly.name}/Head")]
        hp = []
        for g in hgeoms:
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            loc = c + h * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            hp.append(d.geom_xpos[g] + loc @ d.geom_xmat[g].reshape(3, 3).T)
        self.fly_pts = np.vstack(pts)
        self.head_pts = np.vstack(hp)
        eye = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{self.fly.name}/{n}") for n in ("LEye", "REye")]
        self.face_pt = d.geom_xpos[eye].mean(0) + np.array([0.1, 0, 0])
        self.head_pos = head

    def _groom_w(self, p):
        """Вес передних лап «из позы» (чистка): 0 — поза покоя, 1 — pose['legs']."""
        if p.get("grooming") and p.get("legs") is not None:
            return _smooth((float(p.get("groom_t") or 0.0)) / GROOM_IN), True
        ea = p.get("groom_end_age")
        if ea is not None and ea < GROOM_OUT:
            return 1.0 - _smooth(ea / GROOM_OUT), False
        return 0.0, False

    def _groom_path(self, u, goal):
        """Ноги на пути покой -> goal (u 0..1): прямая в суставах + изгиб GROOM_PATH (сплайн по равным шагам u,
        на концах ноль), чтобы лапа шла перед грудью к лицу, а не сквозь голову."""
        u = float(np.clip(u, 0, 1))
        legs = self.legs_rest + u * (np.asarray(goal, float) - self.legs_rest)
        if GROOM_PATH is None:
            return legs
        if self._path_spline is None or self._path_spline[0] is not GROOM_PATH:
            from scipy.interpolate import CubicSpline
            names = list(self.fly.actuated_joints)
            J7 = ("Coxa", "Coxa_roll", "Coxa_yaw", "Femur", "Femur_roll", "Tibia", "Tarsus1")
            gp = np.asarray(GROOM_PATH, float)
            K = len(gp)
            off = np.zeros((K + 2, len(names)))
            for side in "LR":
                for j, nm in enumerate(J7):
                    sg = -1 if (side == "R" and ("roll" in nm or "yaw" in nm)) else 1
                    off[1:K + 1, names.index(f"joint_{side}F{nm}")] = gp[:, j] * sg
            self._path_spline = (GROOM_PATH, CubicSpline(np.linspace(0, 1, K + 2), off, bc_type="clamped"))
        return legs + self._path_spline[1](u)

    def _offsets(self, p):
        """Смещение корня мухи (мир): подскок при вздрагивании и подъём при взлёте — из нейронных событий."""
        off = np.zeros(3)
        sa = p.get("startle_age")
        if sa is not None and 0 <= sa < HOP_T:
            h = np.sin(np.pi * sa / HOP_T)
            off += np.array([0.12, 0.0, HOP_H]) * h
        if p.get("flying"):
            fa = float(p.get("flight_age") or 0.0)
            k = _smooth(fa / 0.22) * (1.0 - _smooth((fa - 0.62) / 0.24))
            off += np.array([0.35, 0.0, LIFT_H]) * k
        return off

    def _apply_pose(self, p):
        m, d = self.m, self.d
        q = d.qpos
        legs = self.legs_rest.copy()
        w, live = self._groom_w(p)
        if w > 0:
            fi = self.front_idx
            goal = self.groom_goal
            if live:                                   # «трёт лапками» из позы + наводка на лицо сидящей мухи
                goal = self.legs_rest.copy()
                goal[fi] = np.asarray(p["legs"], float)[fi] + self.groom_aim[fi]
            legs[fi] = self._groom_path(w, goal)[fi]
        q[self.leg_adr] = legs
        self.poser.apply(float(np.clip(p.get("prob", 0.0), 0, 1)), 0.0, 0.0)
        wy = p.get("wing_yaw") or (0.0, 0.0)
        wr = p.get("wing_roll") or (0.0, 0.0)
        for s, i in (("L", 0), ("R", 1)):
            q[self.wing_adr[s][0]] = float(wy[i])
            q[self.wing_adr[s][1]] = float(wr[i])
        q[self.head_roll_adr] = float(p.get("head", 0.0))
        q[self.head_pitch_adr] = HEAD_PITCH
        off = self._offsets(p)
        q[self.root_adr:self.root_adr + 3] = FLY_ROOT + off
        q[self.root_adr + 3:self.root_adr + 7] = _q((0, 1, 0), -SEAT_PITCH)
        self._life_pose(p)
        # капля и капилляр (стимул прохода 1)
        raw = float(p.get("drop", 0.0))
        dr = float(np.clip(abs(raw), 0, 1))
        m.geom_rgba[self.drop_gid, :3] = (0.35, 0.85, 0.3) if raw < 0 else (1.0, 0.72, 0.2)   # горькая — зелёная
        vis = dr > 0.03
        m.geom_rgba[self.drop_gid, 3] = 0.92 if vis else 0.0
        m.geom_size[self.drop_gid, 0] = 0.08 + 0.17 * dr
        m.geom_rgba[self.dropshine_gid, 3] = 0.9 if vis else 0.0
        d.mocap_pos[self.drop_mocap] = self.drop_pos + np.array([0, 0, -0.17 * dr * 0.4])
        slide = 5.0 * (1.0 - _smooth(dr / 0.2)) if dr > 0.005 else 40.0
        d.mocap_pos[self.pip_mocap] = self.drop_pos + self.PIP_DIR * (0.1 + 0.17 * dr + slide)
        mujoco.mj_kinematics(m, d)

    # ------------------------------------------------------------------ приступ / смерть / новая муха
    CURL = dict(Coxa=0.5, Femur=-0.9, Tibia=0.6, Tarsus1=-0.9)     # поджатые лапки (к позе покоя)

    def _slump(self, k):
        """Корпус и голова мёртвой мухи: сползает в кресле (k 0..1) -> (pos, quat корня, наклон, поворот головы)."""
        pos = FLY_ROOT + np.array([0.55, 0.0, -0.45]) * k
        quat = _qm(_q((0, 1, 0), -(SEAT_PITCH + 0.35 * k)), _q((1, 0, 0), 0.22 * k))
        return pos, quat, HEAD_PITCH + 0.45 * k, 0.45 * k

    def _hp_slip(self, k):
        """Наушники съезжают с головы (правка модели в этом процессе; k = 0 — на месте)."""
        m = self.m
        if not len(self.hp_gid):
            return
        R = Rot.from_rotvec([0.55 * k, 0.0, 0.0])
        m.geom_pos[self.hp_gid] = R.apply(self.hp_pos0) + np.array([0.05, 0.12, -0.3]) * k
        qr = R.as_quat()
        qr = np.array([qr[3], qr[0], qr[1], qr[2]])
        for i, g in enumerate(self.hp_gid):
            mujoco.mju_mulQuat(m.geom_quat[g], qr, self.hp_quat0[i])

    def _life_pose(self, p):
        m, d = self.m, self.d
        q = d.qpos
        life, lt = SG.life_of(p)
        d.mocap_pos[self.claw_mocap] = SG.CLAW_HIDE
        self._dim = 1.0
        if life == "alive":
            if getattr(self, "_hp_moved", False):
                self._hp_slip(0.0)
                self._hp_moved = False
            return
        names = list(self.fly.actuated_joints)
        curl = self.legs_rest.copy()
        for k, v in self.CURL.items():
            for side in "LR":
                for leg in "FMH":
                    curl[names.index(f"joint_{side}{leg}{k}")] += v
        if life == "seizure":
            k = SG.seizure_k(lt)
            j = SG.life_jitter(lt, 60, 1)
            q[self.leg_adr] = self.legs_rest + 0.5 * k * j[:42]
            q[self.head_roll_adr] += 0.45 * k * j[48]
            q[self.head_pitch_adr] = HEAD_PITCH + 0.3 * k * j[49]
            for s_, sg, jj in (("L", -1.0, j[50:52]), ("R", 1.0, j[52:54])):
                q[self.wing_adr[s_][0]] = sg * (0.35 + 0.35 * jj[0]) * k
                q[self.wing_adr[s_][1]] = -sg * 0.9 * k * jj[1]
            q[self.root_adr:self.root_adr + 3] += k * np.array([0.06 * j[45], 0.07 * j[46], 0.1 * abs(j[47])])
            q[self.root_adr + 3:self.root_adr + 7] = _qm(_q((0, 1, 0), -SEAT_PITCH + 0.12 * k * j[43]),
                                                        _q((1, 0, 0), 0.12 * k * j[44]))
            return
        self._hp_moved = True
        if life == "dead":
            z, op, carry = 0.0, 1.0, None
            sk = SG._smooth(lt / 0.9)
        else:
            z, op, carry = SG.claw_track(lt)
            sk = 1.0
        old = life == "dead" or carry == "old" or (carry is None and lt < 1.8)
        if old:
            u = SG._smooth((lt - 0.1) / 1.0) if life == "dead" else 1.0
            tw = np.exp(-lt / 1.4) * 0.3 * SG.life_jitter(lt, 42, 7) * (life == "dead" and lt > 0.5)
            q[self.leg_adr] = self.legs_rest + (curl - self.legs_rest) * u + tw
            pos, quat, hp_, hr = self._slump(sk)
            q[self.head_pitch_adr], q[self.head_roll_adr] = hp_, hr
            for s_, sg in (("L", -1.0), ("R", 1.0)):
                q[self.wing_adr[s_][0]] = sg * 0.35 * sk
                q[self.wing_adr[s_][1]] = sg * 0.4 * sk
            self.poser.apply(0.0, 0.0, 0.0)
            if carry == "old":
                quat = _qm(_q((1, 0, 0), 0.1 * np.sin(6 * lt)), quat)
            self._hp_slip(SG._smooth(lt / 1.6) if life == "dead" else 1.0)
            self._dim = 1.0 - 0.65 * sk if life == "dead" else 0.35
        else:                                                   # новая муха: свежая сидячая поза
            pos, quat = FLY_ROOT.copy(), _q((0, 1, 0), -SEAT_PITCH)
            q[self.leg_adr] = self.legs_rest
            q[self.head_roll_adr], q[self.head_pitch_adr] = 0.0, HEAD_PITCH
            self._hp_slip(0.0)
            self._dim = 0.35 + 0.65 * SG._smooth((lt - 2.3) / 0.7)
        q[self.root_adr:self.root_adr + 3] = pos + np.array([0.0, 0.0, z if carry else 0.0])
        q[self.root_adr + 3:self.root_adr + 7] = quat
        if life == "respawn":
            mujoco.mj_kinematics(m, d)
            th = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{self.fly.name}/Thorax")].copy()
            th[2] -= z if carry else 0.0
            d.mocap_pos[self.claw_mocap] = th + np.array([0.0, 0.0, SG.CLAW_GRIP + (0.2 if old else 0.0) + z])
            SG.claw_fingers(m, self.claw_gid, op)

    # ------------------------------------------------------------------ состояние кадра
    def set_state(self, s):
        """Лёгкая: только запоминает позу (render_parallel зовёт её и для соседних кадров ради рамок)."""
        s = s or {}
        self.pose = s.get("pose") or {}

    def set_screen_frame(self, rgb):
        if rgb is None:
            return
        small = cv2.resize(rgb, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32) / 255
        c = small.mean(0)
        c = 0.6 * c + 0.4 * small.max(0) * (c.sum() > 0.05)
        self.glow = np.clip(c * 1.1, 0, 1)

    # ------------------------------------------------------------------ свет и подсветка кадра
    def _led_rgb(self, phase, t, a):
        L = self.style.led
        base = np.asarray(self.led_col, float)
        sp = float(L.get("speed", 0.3))
        mode = L.get("mode", "solid")
        n = len(phase)
        if mode == "rainbow":
            col = SG._hsv(phase + sp * t * 0.5, 0.8, 1.0)
        elif mode == "wave":
            k = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * (phase * 2 - sp * t)))
            col = base[None] * k[:, None]
        elif mode == "breath":
            k = 0.55 + 0.45 * np.sin(2 * np.pi * sp * t)
            col = np.repeat(base[None] * k, n, 0)
        else:
            col = np.repeat(base[None], n, 0)
        return np.clip(col * (0.85 + 0.3 * a), 0, 1)

    def _apply_fx(self, p):
        m = self.m
        t = float(p.get("t", 0.0))
        a = float(np.clip(p.get("audio", 0.0), 0, 1))
        if len(self.led_gid):
            col = self._led_rgb(self.led_phase, t, a)
            m.geom_rgba[self.led_gid, :3] = col
            for g, idx, alpha in self.led_halo:
                c = col[idx].mean(0)
                m.geom_rgba[g, :3] = np.clip(c / max(0.35, c.max()), 0, 1)
                m.geom_rgba[g, 3] = alpha * (0.35 + 0.65 * c.max())
        H = self.style.headphones
        if len(self.hp_led_gid):
            hc = _bright(H["led"] if H.get("led") is not None else self.led_col)
            m.geom_rgba[self.hp_led_gid, :3] = np.clip(hc * (0.45 + 0.55 * a), 0, 1)
        if len(self.spk_gid):
            m.geom_rgba[self.spk_gid, :3] = np.clip(self.led_col * (0.3 + 0.7 * a), 0, 1)
        gcol = np.clip(self.glow, 0, 1)
        m.light_diffuse[self.light["screen"]] = 0.12 + 0.6 * gcol
        dim = getattr(self, "_dim", 1.0)                   # муха выбыла: свет гаснет
        for k in ("key", "fill", "rim"):
            m.light_diffuse[self.light[k]] = self.light_diff0[k] * dim
        if dim < 1.0 and len(self.led_gid):
            m.geom_rgba[self.led_gid, :3] *= 0.4 + 0.6 * dim

    # ------------------------------------------------------------------ камера
    @staticmethod
    def _groom_zoom(p):
        """0..1: наезд камеры на чистку — по времени с её начала и с её конца (кадр = функция позы)."""
        if p.get("grooming"):
            return _smooth(float(p.get("groom_t") or 0.0) / 0.5)
        ea = p.get("groom_end_age")
        if ea is not None and ea < 0.7:
            return 1.0 - _smooth(ea / 0.7)
        return 0.0

    def _cam_params(self, p, aspect):
        """Камера кадра: (pos, R (столбцы right, up, -fwd), fovy) — по style.camera и времени/событиям."""
        C = self.style.camera
        pr = CAM_PRESETS.get(C.get("name"), CAM_PRESETS["front"])
        t = float(p.get("t", 0.0))
        dr = float(C.get("drift", 0.5))
        ph = (self.style.seed % 997) * 0.37
        az = pr["az"] + float(C.get("yaw", 0.0)) + dr * 0.04 * np.sin(0.19 * t + ph)
        el = pr["el"] + float(C.get("pitch", 0.0)) + dr * 0.015 * np.sin(0.23 * t + 2 * ph)
        roll = float(C.get("roll", 0.0)) + dr * 0.006 * np.sin(0.27 * t + ph)
        D = pr["D"] * float(C.get("dist", 1.0)) * (1.0 + dr * 0.02 * np.sin(0.13 * t + 3 * ph))
        # короткий наезд на нейронные события (вздрогнула / начало взлёта)
        sa = p.get("startle_age")
        if sa is not None and 0 <= sa < 0.5:
            D *= 1.0 - 0.07 * np.sin(np.pi * sa / 0.5)
        if p.get("flying"):
            fa = float(p.get("flight_age") or 0.0)
            D *= 1.0 + 0.06 * _smooth(fa / 0.3)
        # чистка (нейронное состояние): плавный наезд на лицо, чтобы тонкие лапки было видно
        gz = self._groom_zoom(p)
        D *= 1.0 - GROOM_ZOOM * gz
        off = self._offsets(p)
        tgt = np.array([pr["tx"], 0.0, pr["tz"] + GROOM_LIFT * gz]) + off * np.array([0.5, 0, 0.6])
        pos = tgt + D * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        f = tgt - pos
        f /= np.linalg.norm(f)
        right = np.cross(f, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        c, s = np.cos(roll), np.sin(roll)
        right, up = c * right + s * up, -s * right + c * up
        return pos, np.stack([right, up, -f], 1), FOVY

    def _project(self, pts, w, h, p=None):
        pos, R, fovy = self._cam_params(self.pose if p is None else p, w / h)
        f = (h / 2) / np.tan(np.deg2rad(fovy) / 2)
        v = (np.atleast_2d(pts) - pos) @ R
        z = np.maximum(-v[:, 2], 1e-6)
        return np.stack([w / 2 + f * v[:, 0] / z, h / 2 - f * v[:, 1] / z], 1)

    def fly_screen_xy(self, w, h):
        """Лицо мухи на картинке (x, y) пикселей."""
        off = self._offsets(self.pose)
        xy = self._project(self.face_pt + off, w, h)[0]
        return float(xy[0]), float(xy[1])

    def fly_box(self, w, h):
        """Рамка мухи на картинке (x0, y0, x1, y1): тело, лапы, крылья, наушники."""
        off = self._offsets(self.pose)
        xy = self._project(np.vstack([self.fly_pts, self.head_pts]) + off, w, h)
        x0, y0 = xy.min(0)
        x1, y1 = xy.max(0)
        return (float(max(0, x0)), float(max(0, y0)), float(min(w, x1)), float(min(h, y1)))

    # ------------------------------------------------------------------ рендер
    def _renderer(self, width, height):
        key = (width, height)
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.m, height=height, width=width)
            self._renderers[key] = r
        return r

    def _seg_groups(self, r, w2, h2):
        """Сегментация уменьшенного кадра -> карта групп контура (h2, w2) int32. Ореолы в неё не попадают."""
        sc = r.scene
        r.update_scene(self.d, camera=self.cam_id, scene_option=self.opt_seg)
        flags = sc.flags.copy()
        sc.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
        sc.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
        sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        sc.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        rect = mujoco.MjrRect(0, 0, w2, h2)
        r._gl_context.make_current()
        mujoco.mjr_render(rect, sc, r._mjr_context)
        buf = np.empty((h2, w2, 3), np.uint8)
        mujoco.mjr_readPixels(buf, None, rect, r._mjr_context)
        np.copyto(sc.flags, flags)
        n = sc.ngeom
        key = (n, sc.geoms[0].objid if n else -1, sc.geoms[n - 1].objid if n else -1)
        if self._segtab is None or self._segtab[0] != key:
            tab = np.full(n + 2, self.group_of[-1], np.int32)
            gtab = np.zeros(n + 2, np.uint8)
            ktab = np.zeros(n + 2, np.uint8)
            tab[0] = 0                                        # фон (ничего не нарисовано)
            gt = int(mujoco.mjtObj.mjOBJ_GEOM)
            for i in range(n):
                g = sc.geoms[i]
                if g.segid >= 0 and g.objtype == gt:
                    tab[g.segid + 1] = self.group_of[g.objid + 1]
                    gtab[g.segid + 1] = self.glow_of[g.objid + 1]
                    ktab[g.segid + 1] = self.kind_of[g.objid + 1]
                elif g.segid >= 0:
                    tab[g.segid + 1] = 0
            self._segtab = (key, tab, gtab, ktab)
        _, tab, gtab, ktab = self._segtab
        b4 = np.zeros((h2, w2, 4), np.uint8)
        b4[..., :3] = buf[::-1]
        ids = np.minimum(b4.view(np.uint32)[..., 0], len(tab) - 1)
        return tab[ids], gtab[ids], ktab[ids]

    def _post(self, w, h):
        """Кэш по размеру кадра: LUT квантования/насыщенности, виньетка."""
        c = self._post_cache.get((w, h))
        if c is None:
            T = self.style.toon
            Lv = int(T.get("levels", 4))
            v = np.arange(256) / 255.0
            x = v * (Lv - 1)
            fr = x - np.floor(x)
            st = np.clip((fr - 0.4) / 0.2, 0, 1)
            st = st * st * (3 - 2 * st)
            stair = (np.floor(x) + st) / (Lv - 1)
            vq = np.maximum(stair, v * 0.6)                    # тени не проваливаются в чёрное
            lut_v = np.clip(vq * 255, 0, 255).astype(np.uint8)
            sat = float(T.get("saturation", 1.2))
            lut_s = np.clip(np.arange(256) * sat, 0, 255).astype(np.uint8)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            rr = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
            ex = float(self.style.light.get("exposure", 1.0))
            vig = np.clip(ex * (1.06 - 0.22 * np.clip(rr - 0.35, 0, None) ** 1.5), 0, 1.3)
            vig = (np.repeat(vig[..., None], 3, -1) * 128).astype(np.uint8)
            ol = np.array(T.get("outline", (0.05, 0.04, 0.07))) * 255
            olimg = np.empty((h, w, 3), np.uint8)
            olimg[:] = np.clip(ol, 0, 255).astype(np.uint8)
            c = dict(lut_v=lut_v, lut_s=lut_s, vig=vig, olimg=olimg, px=int(T.get("outline_px", 2)))
            self._post_cache[(w, h)] = c
        return c

    def _flat_fly(self, img, kinds, w, h):
        """Мульт-заливка мухи: внутри каждой части (тело / лапы / глаза) цвет усредняется размытием с маской —
        мелкий рельеф меша (щетинки, бугорки, самотени) не дробит заливку; форма и свет частей остаются."""
        h2, w2 = kinds.shape
        mask = (kinds > 0).astype(np.uint8)
        bx, by, bw, bh = cv2.boundingRect(mask)
        if bw == 0 or bh == 0:
            return img
        # считаем только в рамке мухи (+ запас на размытие): муха — ~5-15 % кадра
        pad = int(3 * max(FLAT_SIGMA.values())) + 2
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(w2, bx + bw + pad), min(h2, by + bh + pad)
        X0, Y0, X1, Y1 = x0 * w // w2, y0 * h // h2, x1 * w // w2, y1 * h // h2
        kc = kinds[y0:y1, x0:x1]
        small = cv2.resize(img[Y0:Y1, X0:X1], (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32)
        out = small.copy()
        for k, sig in FLAT_SIGMA.items():
            m = (kc == k).astype(np.float32)
            if not m.any():
                continue
            num = cv2.GaussianBlur(small * m[..., None], (0, 0), sig)
            den = cv2.GaussianBlur(m, (0, 0), sig)
            sm = num / np.maximum(den, 1e-3)[..., None]
            np.copyto(out, sm, where=(m > 0)[..., None])
        size = (X1 - X0, Y1 - Y0)
        flat = cv2.resize(np.clip(out, 0, 255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(mask[y0:y1, x0:x1] * int(255 * FLAT_MIX), size, interpolation=cv2.INTER_LINEAR)
        a3 = cv2.merge([a, a, a])
        img = img.copy()
        roi = img[Y0:Y1, X0:X1]
        img[Y0:Y1, X0:X1] = cv2.add(cv2.multiply(roi, cv2.bitwise_not(a3), scale=1 / 255),
                                    cv2.multiply(flat, a3, scale=1 / 255))
        return img

    def _toon(self, img, seg, w, h, bloom):
        P = self._post(w, h)
        groups, glow, kinds = seg
        if LOOK != "toon":                                # реалистичный вид: без заливки, ступеней и обводки
            img = cv2.multiply(np.ascontiguousarray(img), P["vig"], scale=1 / 128)
            return self._bloom(img, glow, w, h, bloom)
        img = self._flat_fly(img, kinds, w, h)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hh, ss, vv = cv2.split(hsv)
        # ступенчатый свет по сглаженной яркости (мелкий рельеф меша не дробит заливку), немного деталей обратно
        vq = cv2.LUT(cv2.GaussianBlur(vv, (0, 0), 2.4), P["lut_v"])
        vv = cv2.addWeighted(vq, 0.72, vv, 0.28, 0)
        img = cv2.cvtColor(cv2.merge([hh, cv2.LUT(ss, P["lut_s"]), vv]), cv2.COLOR_HSV2RGB)
        # контур по сменам группы (уменьшенная сегментация -> полный размер)
        g = groups
        e = np.zeros(g.shape, np.uint8)
        e[:, 1:] |= (g[:, 1:] != g[:, :-1])
        e[1:, :] |= (g[1:, :] != g[:-1, :])
        # мелкие замкнутые «островки» контура (лапа чуть проходит сквозь тело, бугорки меша) — убрать
        n, lab, stt, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
        tiny = (stt[:, cv2.CC_STAT_WIDTH] < ISLAND_PX) & (stt[:, cv2.CC_STAT_HEIGHT] < ISLAND_PX)
        tiny[0] = False
        if tiny.any():
            e[tiny[lab]] = 0
        e = cv2.resize(e * 255, (w, h), interpolation=cv2.INTER_LINEAR)
        if P["px"] >= 3:
            e = cv2.dilate(e, np.ones((2, 2), np.uint8))
        # смешение с цветом контура целиком в uint8 (float-вариант стоил ~28 мс на кадр)
        a = cv2.convertScaleAbs(cv2.GaussianBlur(e, (3, 3), 0.6), alpha=0.92)
        a3 = cv2.merge([a, a, a])
        img = cv2.add(cv2.multiply(img, cv2.bitwise_not(a3), scale=1 / 255),
                      cv2.multiply(P["olimg"], a3, scale=1 / 255))
        img = cv2.multiply(img, P["vig"], scale=1 / 128)
        return self._bloom(img, glow, w, h, bloom)

    def _bloom(self, img, glow, w, h, bloom):
        # свечение: только от светящихся геомов (лента, неон, лампа, LED наушников)
        h2, w2 = glow.shape
        src = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
        src = cv2.multiply(src, cv2.merge([glow, glow, glow]))
        g1 = cv2.GaussianBlur(src, (0, 0), 3)
        g2 = cv2.GaussianBlur(cv2.resize(src, (w2 // 4, h2 // 4), interpolation=cv2.INTER_AREA), (0, 0), 3)
        k = 0.5 + float(bloom)
        b = cv2.addWeighted(g1, 0.8 * k, cv2.resize(g2, (w2, h2), interpolation=cv2.INTER_LINEAR), 1.1 * k, 0)
        img = cv2.add(img, cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR))
        return img

    def render(self, width=SCENE_W, height=SCENE_H, bloom=0.0):
        p = self.pose
        self._apply_pose(p)
        self._apply_fx(p)
        pos, R, fovy = self._cam_params(p, width / height)
        m = self.m
        m.cam_pos[self.cam_id] = pos
        q = Rot.from_matrix(R).as_quat()
        m.cam_quat[self.cam_id] = (q[3], q[0], q[1], q[2])
        m.cam_fovy[self.cam_id] = fovy
        mujoco.mj_camlight(m, self.d)
        r = self._renderer(width, height)
        r.update_scene(self.d, camera=self.cam_id, scene_option=self.opt)
        sc = r.scene
        sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        sc.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        sc.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        sc.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
        img = r.render()
        seg = self._seg_groups(r, width // 2, height // 2)
        return self._toon(img, seg, width, height, bloom)

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
