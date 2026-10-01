"""3D-сцена для вертикального TikTok: муха на шаре в крошечной игровой комнате
смотрит ролик на мониторе. ОДНА модель MuJoCo (муха + шар + весь интерьер),
один вызов рендера на кадр.

Масштаб — миллиметры (муха ~2.5 мм, шар R=5.39 мм). Муха в начале координат,
смотрит вдоль +x (на монитор), +y — её левая сторона, z — вверх.
Кукольный режим: физику не шагаем. Ноги ставит генератор шага (CPG) как в
обычной ходьбе flygym, шар крутим кинематически, хоботок — FlyPoser, крылья — _pose_apply.

Использование (два прохода):
    import stage as S
    S.prepare_assets()                      # один раз в главном процессе (текстуры в tiktok/assets)
    st = S.Stage()                          # ~1-8 с (первый раз дольше: генерация текстур)

    # проход 1 (без рендера), каждый кадр:
    st.set_orientation("portrait", t_anim=1.2)     # при смене ролика; монитор плавно поворачивается
    st.set_audio_level(bass_0_1)                   # колонки/подсветка качают от баса
    st.set_camera("ots_left", "low_hero", blend, shake=1.0)   # имя пресета или dict(pos, look, roll, margin)
    st.step_puppet(dt, drive_L, drive_R, head_turn, proboscis, wing_spread, wing_flap, groom_t=None, drop=0.0)
    state = st.get_state()                  # dict маленьких numpy-массивов: qpos, mocap_pos, mocap_quat, fx, cam

    # проход 2 (в любом процессе, у каждого свой контекст OpenGL):
    st.set_state(state)
    st.set_screen_frame(rgb_uint8)          # кадр исходника (любая ориентация/пропорции)
    img = st.render(1080, 1150)             # uint8 (h, w, 3), до 1920x1920

    st.check_visibility()                   # доля видимого экрана/знака/мухи по сегментации

Пресеты камеры: Stage.PRESETS, порядок смены — Stage.PRESET_ORDER. Кадр подбирается
автоматически (направление + угол обзора), чтобы и муха, и весь экран всегда были в кадре.
"""
import os
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from flygym import Fly                                                   # noqa: E402
from flygym.arena.tethered import Ball                                   # noqa: E402
from flygym.simulation import SingleFlySimulation                        # noqa: E402
from flygym.examples.locomotion import CPGNetwork, PreprogrammedSteps    # noqa: E402
from flygym.examples.locomotion.turning_controller import (              # noqa: E402
    _tripod_coupling_weights, _tripod_phase_biases)
from scene import FlyPoser, KitchenArena                                 # noqa: E402
from dm_control.mjcf import schema as _mjcf_schema                       # noqa: E402

# dm_control не знает атрибут texture/nchannel (MuJoCo 3.2 его понимает) — регистрируем в процессе,
# чтобы мягкие ореолы были RGBA-текстурами с прозрачностью
_tex_spec = _mjcf_schema.MUJOCO.children["asset"].children["texture"]
if "nchannel" not in _tex_spec.attributes:
    _tex_spec.attributes["nchannel"] = _tex_spec.attributes["width"]._replace(name="nchannel")

ASSETS = HERE / "assets"
FONT = "C:/Windows/Fonts/bahnschrift.ttf"
WATERMARK = "@xyz_web3"

# ---------------- геометрия (мм) ----------------
BALL_C = np.array([-0.09867235483, -0.05435809692, -5.20309506806])   # как в flygym Ball
BALL_R = 5.390852782067457
DESK_Z = -11.5                          # столешница (шар стоит в чашке на столе)
CUP_TOP = -8.3                          # верх чашки с подсветкой
MON_C = np.array([13.0, 0.0, 1.8])      # центр экрана (ось поворота монитора)
SCR_W, SCR_H = 16.0, 9.0                # активная область экрана 16:9
BEZEL = 0.28
WALL_X = 19.0                           # задняя стена
SIDE_Y = 24.0                           # боковые стены y = ±24
DROP_POS = np.array([1.02, 0.02, 0.27])       # центр капли сахара — сюда дотягивается хоботок
PIPETTE_TIP = np.array([1.12, 0.03, 0.40])     # кончик капилляра (капля висит под ним)

# группы геомов: 0 — шар, 1/2 — муха (flygym), 3 — интерьер, 4 — кормушка/привязь
G_SET, G_PROP = 3, 4
FLY_GROUPS = (1, 2)


# ======================= процедурные текстуры =======================
def _save(img, path):
    """Атомарная запись PNG (параллельные процессы не портят файл; путь с кириллицей)."""
    tmp = path.with_suffix(f".{os.getpid()}.tmp.png")
    Image.fromarray(img).save(tmp)
    try:
        os.replace(tmp, path)
    except OSError:                       # файл уже создал и держит другой процесс — он такой же
        tmp.unlink(missing_ok=True)
        if not path.exists():
            raise


def _font(size, bold=True):
    f = ImageFont.truetype(FONT, size)
    try:
        f.set_variation_by_name("Bold" if bold else "Regular")
    except Exception:
        pass
    return f


def _tex_foam(path, n=256):
    """Акустический поролон «клин»: плитки с рёбрами, чередуя направление."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    half = n // 2
    tile = ((x // half + y // half) % 2).astype(bool)
    k = 5                                            # рёбер на плитку
    sx = (x % half) / half * k
    sy = (y % half) / half * k
    saw = np.where(tile, sx % 1, sy % 1)
    shade = np.where(saw < 0.5, 0.55 + 0.9 * saw, 1.45 - 0.9 * saw)   # свет сбоку на рёбрах
    seam = ((x % half) < 1.5) | ((y % half) < 1.5)
    rng = np.random.default_rng(7)
    grain = cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 0.8)
    v = shade * (0.85 + 0.3 * grain)
    v = np.where(seam, 0.35, v)
    base = np.array([0.105, 0.10, 0.14], np.float32)
    img = np.clip(v[..., None] * base * 1.6, 0, 1)
    _save((img * 255).astype(np.uint8), path)


def _tex_desk(path, n=512):
    """Тёмная глянцевая столешница: мелкая «шлифовка» + едва заметные волокна."""
    rng = np.random.default_rng(11)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    fib = cv2.resize(rng.random((8, 256)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
    v = 0.06 + 0.025 * fib + 0.012 * rng.random((n, n)).astype(np.float32) + 0.01 * np.sin(y * 60 + fib * 4)
    img = np.clip(np.stack([v * 1.0, v * 0.97, v * 1.08], -1), 0, 1)
    _save((img * 255).astype(np.uint8), path)


def _tex_ball(path, w=1024, h=512):
    """Шар из пенопласта: тёплый светлый с мелкими точками (как для трекинга в лаборатории)."""
    rng = np.random.default_rng(5)
    base = 0.74 + 0.06 * cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 6)
    base += 0.05 * rng.random((h, w)).astype(np.float32)
    img = np.stack([base * 1.0, base * 0.97, base * 0.92], -1)
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for _ in range(260):
        cx, cy = rng.random() * w, rng.random() * h
        r = rng.uniform(0.9, 2.2)
        sq = 1.0 / max(0.25, np.sin(np.pi * cy / h))      # у полюсов точки растянуты по u
        c = int(rng.uniform(55, 95))
        dr.ellipse((cx - r * sq, cy - r, cx + r * sq, cy + r), fill=(c, c - 5, c - 10))
    im = im.filter(ImageFilter.GaussianBlur(0.7))
    _save(np.asarray(im), path)


def _glow_text_layer(size, text, font, pos, color, core=(255, 255, 255), stroke=0, blur=(28, 10, 3)):
    """Неоновая надпись: широкое свечение + ближнее свечение + светлая «трубка»."""
    W, H = size
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).text(pos, text, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
    m = np.asarray(mask).astype(np.float32) / 255
    out = np.zeros((H, W, 3), np.float32)
    col = np.array(color, np.float32) / 255
    for s, g in zip(blur, (1.25, 0.9, 0.55)):
        out += cv2.GaussianBlur(m, (0, 0), s)[..., None] * col * g
    thin = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, stroke // 2 + 1))
    out = out * (1 - m[..., None]) + m[..., None] * (col * 0.75 + 0.25)         # цветная трубка
    out = out * (1 - thin[..., None]) + thin[..., None] * (np.array(core, np.float32) / 255)  # белая сердцевина
    return out


def _tex_neon(path, W=1024, H=384):
    """Неоновая вывеска «МУХА» на чёрном акриле (свечение запечено в текстуру)."""
    img = np.zeros((H, W, 3), np.float32)
    f = _font(230)
    tw = ImageDraw.Draw(Image.new("L", (1, 1))).textlength("МУХА", font=f)
    img += _glow_text_layer((W, H), "МУХА", f, ((W - tw) / 2, 12), (255, 40, 170), core=(255, 225, 245),
                            blur=(34, 12, 4))
    # подчёркивание голубой неоновой трубкой
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rounded_rectangle((W * 0.18, H - 62, W * 0.82, H - 50), radius=6, fill=255)
    mm = np.asarray(m).astype(np.float32) / 255
    cyan = np.array([40, 220, 255], np.float32) / 255
    for s, g in ((26, 1.1), (8, 0.8)):
        img += cv2.GaussianBlur(mm, (0, 0), s)[..., None] * cyan * g
    img = img * (1 - mm[..., None]) + mm[..., None] * np.array([0.8, 0.97, 1.0])
    _save((np.clip(img, 0, 1) * 255).astype(np.uint8), path)


def _tex_city(path, W=512, H=640):
    """Ночной город за окном, сильно размыт (боке)."""
    rng = np.random.default_rng(21)
    y = np.linspace(0, 1, H)[:, None, None]
    sky = (1 - y) * np.array([0.10, 0.03, 0.20]) + y * np.array([0.02, 0.03, 0.10])
    img = np.broadcast_to(sky, (H, W, 3)).astype(np.float32).copy()
    # силуэты домов с окнами
    xs = 0
    while xs < W:
        bw = int(rng.uniform(40, 110)); top = int(rng.uniform(H * 0.35, H * 0.75))
        img[top:, xs:xs + bw] = np.array([0.03, 0.03, 0.06])
        for wy in range(top + 8, H, 14):
            for wx in range(xs + 6, xs + bw - 6, 12):
                if rng.random() < 0.35:
                    c = np.array([1.0, 0.75, 0.4]) if rng.random() < 0.7 else np.array([0.5, 0.8, 1.0])
                    img[wy:wy + 6, wx:wx + 5] = c * rng.uniform(0.35, 0.8)
        xs += bw + int(rng.uniform(0, 12))
    img = cv2.GaussianBlur(img, (0, 0), 3.0)
    # боке: крупные мягкие диски
    bok = np.zeros_like(img)
    pal = [(1.0, 0.35, 0.75), (0.3, 0.8, 1.0), (1.0, 0.7, 0.3), (0.6, 0.4, 1.0), (1.0, 0.95, 0.8)]
    for _ in range(46):
        cx, cy = rng.uniform(0, W), rng.uniform(H * 0.1, H * 0.95)
        r = rng.uniform(10, 34)
        c = np.array(pal[rng.integers(len(pal))]) * rng.uniform(0.25, 0.7)
        disk = np.zeros((H, W), np.float32)
        cv2.circle(disk, (int(cx), int(cy)), int(r), 1.0, -1, lineType=cv2.LINE_AA)
        disk = cv2.GaussianBlur(disk, (0, 0), 2.0)
        ring = disk - cv2.GaussianBlur(cv2.erode(disk, np.ones((5, 5))), (0, 0), 3) * 0.35
        bok = np.maximum(bok, ring[..., None] * c)
    img = np.clip(img + bok, 0, 1)
    _save((img * 255).astype(np.uint8), path)


def _tex_halo(path, n=256):
    """Мягкое круглое свечение: белый RGBA, альфа спадает к краю (цвет задаёт rgba геома)."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(x - n / 2 + 0.5, y - n / 2 + 0.5) / (n / 2)
    v = np.clip(np.exp(-(r / 0.5) ** 2 * 1.4) - 0.015, 0, 1)
    v = v / v.max()
    img = np.dstack([np.full((n, n, 3), 255, np.uint8), (v * 255).astype(np.uint8)])
    _save(img, path)


def _tex_poster(path, W=384, H=512):
    """Постер «игры про муху»: неоновая сетка-горизонт и солнце (ретровейв)."""
    img = np.zeros((H, W, 3), np.float32)
    yy = np.linspace(0, 1, H)[:, None]
    img[:] = ((1 - yy) * np.array([0.12, 0.02, 0.22]) + yy * np.array([0.02, 0.01, 0.06]))[:, None, :].reshape(H, 1, 3)
    # солнце с полосками
    sun = np.zeros((H, W), np.float32)
    cv2.circle(sun, (W // 2, int(H * 0.42)), int(W * 0.28), 1.0, -1, lineType=cv2.LINE_AA)
    for k in range(6):
        yb = int(H * 0.42 + k * 14 - 6)
        sun[yb:yb + 3 + k] = 0
    sun[int(H * 0.56):] = 0
    grad = np.clip((np.arange(H) / H - 0.14) / 0.42, 0, 1)[:, None]
    scol = (1 - grad)[..., None] * np.array([1.0, 0.85, 0.25]) + grad[..., None] * np.array([1.0, 0.2, 0.6])
    img = img * (1 - sun[..., None]) + sun[..., None] * scol
    img += cv2.GaussianBlur(sun, (0, 0), 18)[..., None] * np.array([1.0, 0.3, 0.6]) * 0.5
    # сетка
    hz = int(H * 0.56)
    grid = np.zeros((H, W), np.float32)
    for k in range(1, 14):
        yl = hz + int((H - hz) * (k / 13) ** 1.8)
        cv2.line(grid, (0, yl), (W, yl), 1.0, 1, cv2.LINE_AA)
    for k in range(-12, 13):
        cv2.line(grid, (W // 2 + k * 8, hz), (W // 2 + k * 70, H), 1.0, 1, cv2.LINE_AA)
    img += cv2.GaussianBlur(grid, (0, 0), 1.2)[..., None] * np.array([0.2, 0.9, 1.0]) * 1.4
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    f = _font(58)
    t = "МУХА"
    tw = dr.textlength(t, font=f)
    dr.text(((W - tw) / 2, H * 0.06), t, font=f, fill=(255, 240, 250))
    f2 = _font(22, bold=False)
    t2 = "139 000 НЕЙРОНОВ"
    dr.text(((W - dr.textlength(t2, font=f2)) / 2, H * 0.06 + 66), t2, font=f2, fill=(120, 230, 255))
    arr = np.asarray(im).copy()
    arr[:4] = arr[-4:] = arr[:, :4] = arr[:, -4:] = (230, 230, 235)       # белая рамка
    _save(arr, path)


TEXTURES = dict(foam=_tex_foam, desk=_tex_desk, ball=_tex_ball, neon=_tex_neon, city=_tex_city,
                halo=_tex_halo, poster=_tex_poster)
TEX_VER = 5
TEX_FLIP_V = False       # строка 0 PNG/tex_data <-> v=0 (проверено рендером)


def prepare_assets():
    """Сгенерировать все текстуры заранее (вызвать в главном процессе до запуска воркеров)."""
    for n in TEXTURES:
        asset_path(n)


def asset_path(name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    p = ASSETS / f"{name}_v{TEX_VER}.png"
    if not p.exists():
        TEXTURES[name](p)
    return p


# ======================= мелочи =======================
def _hsv(h, s, v):
    """Векторный HSV -> RGB (все аргументы broadcast)."""
    h, s, v = np.broadcast_arrays(np.asarray(h, float) % 1.0, np.asarray(s, float), np.asarray(v, float))
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1)


def _smooth(x):
    x = float(np.clip(x, 0, 1))
    return x * x * x * (x * (6 * x - 15) + 10)


def _look_quat(fwd, up):
    """Кватернион MuJoCo (w,x,y,z) камеры, смотрящей вдоль fwd (камера смотрит по -z)."""
    f = fwd / np.linalg.norm(fwd)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    q = Rot.from_matrix(np.stack([r, u, -f], axis=1)).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def slab_vertices(w, h, d):
    """Тонкая плита: лицевая грань в плоскости x=-d смотрит в -x.
    u идёт слева направо ДЛЯ ЗРИТЕЛЯ (он смотрит вдоль +x, значит вправо = -y), v — сверху вниз."""
    v = np.array([[-d, w / 2, h / 2], [-d, -w / 2, h / 2], [-d, -w / 2, -h / 2], [-d, w / 2, -h / 2],
                  [0, w / 2, h / 2], [0, -w / 2, h / 2], [0, -w / 2, -h / 2], [0, w / 2, -h / 2]])
    f = np.array([[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    return v, f


# ======================= сцена =======================
DROP_SUGAR = (1.0, 0.7, 0.18)                          # капля сахара — янтарная
DROP_BITTER = (0.35, 0.85, 0.3)                        # горькая — зелёная

# ---------------- приступ -> смерть -> новая муха (hud life / life_t; оба формата) ----------------
CLAW_HIDE = np.array([0.0, 0.0, -400.0])     # робо-захват прячется далеко под столом, пока не нужен
CLAW_GRIP = 1.2                              # мм: центр хвата (грудь мухи) ниже ступицы захвата
CLAW_UP = 10.0                               # мм: насколько захват уходит вверх, за кадр
CLAW_COL = (0.70, 0.72, 0.76, 1.0)           # металл
CLAW_DARK = (0.16, 0.16, 0.18, 1.0)          # резиновые подушечки, сочленения
CLAW_LED = (1.0, 0.45, 0.1, 1.0)             # предупреждающее кольцо
DEATH_T = 5.0                                # с: фаза «dead» (react.DEATH_S)


def life_of(s):
    """(фаза, секунды в фазе) из state/позы кадра; нет полей — ('alive', 0)."""
    s = s or {}
    return str(s.get("life") or "alive"), float(s.get("life_t") or 0.0)


def life_jitter(t, n, seed=0):
    """Детерминированная дрожь кадра (n чисел -1..1): смесь случайного по номеру кадра и ~7 Гц."""
    k = int(round(t * 30))
    r = np.random.default_rng(1000003 * seed + k).uniform(-1, 1, n)
    ph = np.arange(n) * 1.7
    return 0.65 * r + 0.35 * np.sin(2 * np.pi * 7.3 * t + ph)


def seizure_k(lt):
    """Сила приступа 0..1: быстро нарастает за 4 с."""
    return 0.35 + 0.65 * _smooth(lt / 2.6)


def claw_track(lt):
    """Сценарий захвата за 3 с новой мухи: -> (подъём над точкой хвата мм, раскрытие 0..1, какую муху несёт:
    'old' | 'new' | None)."""
    ez = lambda a, b: _smooth((lt - a) / (b - a))
    if lt < 0.8:                                             # спускается раскрытый к старой мухе
        return CLAW_UP * (1 - ez(0.0, 0.8)), 1.0, None
    if lt < 1.05:                                            # сжимает пальцы
        return 0.0, 1.0 - ez(0.8, 1.05), None
    if lt < 1.8:                                             # уносит старую вверх
        return CLAW_UP * ez(1.1, 1.75), 0.0, "old"
    if lt < 2.45:                                            # опускает новую
        return CLAW_UP * (1 - ez(1.8, 2.45)), 0.0, "new"
    if lt < 2.65:                                            # раскрывает
        return 0.0, ez(2.45, 2.65), None
    return CLAW_UP * ez(2.65, 3.0), 1.0, None                # уходит вверх


def claw_segments(open_k):
    """Пальцы захвата в его системе (ступица в 0, z вверх): 3 пальца x 2 фаланги [(p0, p1, r)].
    open_k 0 — сжат (обхватывает грудь мухи), 1 — раскрыт."""
    phi = 0.38 + 0.5 * open_k
    b = 0.5 - 0.45 * open_k
    out = []
    for i in range(3):
        a = 2 * np.pi * i / 3 + np.pi / 2
        u = np.array([np.cos(a), np.sin(a), 0.0])
        k0 = u * 0.3 + np.array([0.0, 0.0, -0.1])
        k1 = k0 + 0.8 * (u * np.sin(phi) - np.array([0.0, 0.0, np.cos(phi)]))
        k2 = k1 + 0.62 * (-u * np.sin(b) - np.array([0.0, 0.0, np.cos(b)]))
        out += [(k0, k1, 0.075), (k1, k2, 0.06)]
    return out


def claw_build(add):
    """Геомы захвата. add(kind, rgba, mat, **geom) -> имя; mat: 'metal' | 'dark' | 'glow'.
    -> dict(seg=[имена 6 фаланг], mid=[3 сустава], tip=[3 подушечки]) — их двигает claw_fingers."""
    add("cylinder", CLAW_COL, "metal", fromto=(0, 0, 0.3, 0, 0, 16.0), size=(0.12,))          # штанга
    add("cylinder", CLAW_DARK, "dark", fromto=(0, 0, 0.12, 0, 0, 0.55), size=(0.24,))          # запястье
    add("cylinder", CLAW_COL, "metal", fromto=(0, 0, -0.12, 0, 0, 0.14), size=(0.45,))          # ступица
    add("cylinder", CLAW_LED, "glow", fromto=(0, 0, 0.14, 0, 0, 0.2), size=(0.4,))              # кольцо
    ids = dict(seg=[], mid=[], tip=[])
    segs = claw_segments(1.0)
    for i in range(3):
        add("sphere", CLAW_DARK, "dark", pos=tuple(segs[2 * i][0]), size=(0.11,))
    for p0, p1, r in segs:
        ids["seg"].append(add("capsule", CLAW_COL, "metal", fromto=(*p0, *p1), size=(r,)))
    for i in range(3):
        ids["mid"].append(add("sphere", CLAW_DARK, "dark", pos=tuple(segs[2 * i][1]), size=(0.09,)))
        ids["tip"].append(add("sphere", CLAW_DARK, "dark", pos=tuple(segs[2 * i + 1][1]), size=(0.075,)))
    return ids


def claw_fingers(m, gid, open_k):
    """Сгибает пальцы захвата (правка модели в этом процессе; gid — id геомов из claw_build)."""
    segs = claw_segments(open_k)
    for g, (p0, p1, r) in zip(gid["seg"], segs):
        v = p1 - p0
        L = float(np.linalg.norm(v))
        q = Rot.align_vectors([v / L], [[0, 0, 1.0]])[0].as_quat()
        m.geom_pos[g] = (p0 + p1) / 2
        m.geom_quat[g] = (q[3], q[0], q[1], q[2])
        m.geom_size[g, 1] = L / 2
    for i in range(3):
        m.geom_pos[gid["mid"][i]] = segs[2 * i][1]
        m.geom_pos[gid["tip"][i]] = segs[2 * i + 1][1]


class Stage:
    """Муха на шаре + игровая комнатка + монитор с роликом. См. docstring модуля."""

    # ракурсы: pos — где камера, look — куда примерно смотрит (кадр потом
    # автоматически центрируется и подбирается угол обзора так, чтобы в кадр
    # целиком влезли и муха, и экран), roll — завал горизонта (рад), margin — поля.
    PRESETS = {
        "ots_left":  dict(pos=(-4.6, 2.3, 4.4), look=(8.0, -0.5, 1.0), roll=0.0, margin=0.04,
                          pos_p=(-4.6, 2.6, 5.6)),
        "ots_right": dict(pos=(-4.6, -2.3, 4.4), look=(8.0, 0.5, 1.0), roll=0.0, margin=0.04),
        "behind":    dict(pos=(-4.6, 0.5, 4.8), look=(9.0, 0.0, 1.0), roll=0.0, margin=0.04,
                          pos_p=(-4.4, 1.3, 6.6)),        # в портрете выше и левее: голова не закрывает знак
        "low_hero":  dict(pos=(-3.9, -4.4, 1.1), look=(6.0, 0.0, 1.3), roll=0.03, margin=0.04),
        "high_left": dict(pos=(-3.6, 3.4, 7.0), look=(6.0, 0.0, 0.0), roll=0.0, margin=0.04),
        "high_34":   dict(pos=(-3.4, -3.6, 7.4), look=(6.0, 0.0, 0.0), roll=0.0, margin=0.04),
        # POV: кадр по экрану + голова/спинка мухи крупно снизу слева (муха видна частично)
        "pov":       dict(pos=(-3.0, -1.0, 2.9), look=(13.0, 0.0, 1.8), roll=-0.02, margin=0.02, fit="pov"),
    }
    PRESET_ORDER = ["ots_left", "low_hero", "behind", "ots_right", "pov", "high_left", "high_34"]

    def __init__(self, tex_size=(960, 540), tether=False):
        self.tex_w, self.tex_h = tex_size
        arena = Ball()
        r = arena.root_element
        for ln in ("light_top", "light_bottom"):             # штатный свет flygym засвечивает всё
            el = r.find("light", ln)
            if el is not None:
                el.remove()
        self._dyn = {}                                         # имена динамических геомов
        self._build_set(r, tether)
        self.fly = Fly(spawn_pos=(0, 0, 0), enable_adhesion=False)
        KitchenArena.add_extra_joints(self.fly)               # хоботок + крылья
        self._set_quality(self.fly.model)
        self.sim = SingleFlySimulation(fly=self.fly, arena=arena, cameras=[], timestep=1e-4)
        self.sim.reset()
        self.p = self.sim.physics
        self.m, self.d = self.p.model.ptr, self.p.data.ptr
        self._init_ids()
        # --- ноги: CPG как в HybridTurningController ---
        self.steps = PreprogrammedSteps()
        self.cpg = CPGNetwork(timestep=1e-3, intrinsic_freqs=np.ones(6) * 12, intrinsic_amps=np.zeros(6),
                              coupling_weights=_tripod_coupling_weights, phase_biases=_tripod_phase_biases,
                              convergence_coefs=np.ones(6) * 20)
        self.base_pose = self.steps.default_pose.copy().ravel()
        self.dof = {n: i for i, n in enumerate(self.steps.dofs_per_leg)}
        self.pose = FlyPoser(self.p, self.fly.name)
        self.fly_pts = self._fly_keypoints()
        # --- состояние куклы и эффектов ---
        self.t = 0.0
        self.ball_q = Rot.identity()
        self.head_angle = 0.0
        self.flap_phase = 0.0
        self.prob = 0.0
        self.drop = 0.0
        self.audio = 0.0
        self.orient = 0.0                 # угол поворота монитора, 0 = альбом, pi/2 = портрет
        self._or_from, self._or_to, self._or_t0, self._or_T = 0.0, 0.0, 0.0, 0.0
        self.content_portrait = False     # как раскладывать кадр на панели
        self.glow = np.array([0.35, 0.4, 0.55])
        self.cam = None                   # [pos3, fwd3, up3, tan_x, tan_y]
        self._cam_req = None              # последний запрос set_camera (для пересчёта в step_puppet)
        self._frame = None
        self._tex_dirty = True
        self._tex_key = None
        self._renderers = {}
        self._uploaded = {}               # renderer key -> версия текстуры
        self._tex_ver = 0
        self._wm = None
        self.opt = mujoco.MjvOption()
        for g in range(6):
            self.opt.geomgroup[g] = 1 if g <= G_PROP else 0
        self.d.qpos[self.leg_adr] = np.concatenate([self.steps.get_joint_angles(leg, 0.0, 0.0)
                                                    for leg in self.steps.legs])      # спокойная стойка
        self.set_camera("ots_left")
        self._apply()

    # ------------------------------------------------------------------ build
    @staticmethod
    def _set_quality(model):
        v = model.visual
        v.quality.shadowsize = 4096
        v.quality.offsamples = 8
        getattr(v, "global").offwidth = 1920
        getattr(v, "global").offheight = 1920
        getattr(v, "global").fovy = 40
        v.map.haze = 0.0
        v.map.znear = 0.001
        v.map.zfar = 4.0
        v.headlight.ambient = (0.05, 0.05, 0.07)
        v.headlight.diffuse = (0.10, 0.10, 0.12)
        v.headlight.specular = (0.02, 0.02, 0.02)

    def _g(self, parent, name, group=G_SET, **kw):
        """Декоративный геом: без контактов, в имени '_sensorfx' (иначе flygym считает его полом)."""
        kw.setdefault("contype", 0)
        kw.setdefault("conaffinity", 0)
        return parent.add("geom", name=f"{name}_sensorfx", group=group, **kw)

    def _slab(self, A, name, w, h, d=0.03, urep=1.0, vrep=1.0):
        v, f = slab_vertices(w, h, d)
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]] * 2, float) * [urep, vrep]
        if TEX_FLIP_V:
            uv[:, 1] = vrep - uv[:, 1]
        return A.add("mesh", name=name, vertex=v.ravel(), face=f.ravel(), texcoord=uv.ravel())

    def _build_set(self, r, tether):
        A, wb = r.asset, r.worldbody
        tex = lambda n, **kw: A.add("texture", name=f"t_{n}", type="2d", file=str(asset_path(n)), **kw)
        # --- материалы ---
        m_glow = A.add("material", name="glow", emission=1.0, specular=0.0, shininess=0.0)
        m_soft = A.add("material", name="softglow", emission=0.6, specular=0.1, shininess=0.1)
        m_black = A.add("material", name="black", specular=0.6, shininess=0.85, reflectance=0.0)
        m_matte = A.add("material", name="matte", specular=0.08, shininess=0.1)
        m_metal = A.add("material", name="metal", specular=1.0, shininess=0.95)
        m_gold = A.add("material", name="gold", specular=1.0, shininess=0.9, rgba=(1.0, 0.78, 0.3, 1))
        m_glass = A.add("material", name="glass", specular=1.0, shininess=1.0, rgba=(0.75, 0.9, 1.0, 0.22))
        m_foam = A.add("material", name="foam", texture=tex("foam"), specular=0.02, shininess=0.0)
        m_desk = A.add("material", name="desk", texture=tex("desk"), texrepeat=(0.08, 0.08), texuniform=True,
                       reflectance=0.28, specular=0.9, shininess=0.95)
        m_ball = A.add("material", name="ballfoam", texture=tex("ball"), specular=0.12, shininess=0.2)
        m_neon = A.add("material", name="neon", texture=tex("neon"), emission=1.0, specular=0.0)
        m_city = A.add("material", name="city", texture=tex("city"), emission=1.0, specular=0.0)
        m_halo = A.add("material", name="halo", texture=tex("halo", nchannel=4), emission=1.0, specular=0.0)
        m_poster = A.add("material", name="poster", texture=tex("poster"), emission=0.35, specular=0.3,
                         shininess=0.6)
        scr = self._screen_png()
        t_scr = A.add("texture", name="screen_tex", type="2d", file=str(scr))
        m_scr = A.add("material", name="screen", texture=t_scr, emission=1.0, specular=0.0, shininess=0.0,
                      rgba=(0.97, 0.97, 0.97, 1))

        # --- шар: пенопласт ---
        ball = r.find("geom", "treadmill")
        ball.material = m_ball
        ball.rgba = (0.62, 0.61, 0.6, 1)

        # --- стол, стены ---
        self._g(wb, "desk", type="plane", size=(60, 60, 1), pos=(0, 0, DESK_Z), material=m_desk)
        H_WALL = 50.0
        self._slab(A, "wall_back", 2 * SIDE_Y + 2, H_WALL, 0.2, urep=(2 * SIDE_Y + 2) / 5, vrep=H_WALL / 5)
        self._g(wb, "wall_back", type="mesh", mesh="wall_back", pos=(WALL_X + 0.2, 0, DESK_Z + H_WALL / 2),
                material=m_foam)
        L_SIDE = WALL_X + 25
        self._slab(A, "wall_side", L_SIDE, H_WALL, 0.2, urep=L_SIDE / 5, vrep=H_WALL / 5)
        for s in (1, -1):
            self._g(wb, f"wall_side{s}", type="mesh", mesh="wall_side",
                    pos=(WALL_X - L_SIDE / 2, s * (SIDE_Y + 0.2), DESK_Z + H_WALL / 2),
                    euler=(0, 0, s * 90), material=m_foam)

        # --- чашка под шаром + кольцо подсветки ---
        cup_h = (CUP_TOP - DESK_Z) / 2
        self._g(wb, "cup", type="cylinder", size=(4.95, cup_h), pos=(BALL_C[0], BALL_C[1], DESK_Z + cup_h),
                material=m_black, rgba=(0.05, 0.05, 0.06, 1))
        self._g(wb, "cup_base", type="cylinder", size=(5.6, 0.18), pos=(BALL_C[0], BALL_C[1], DESK_Z + 0.18),
                material=m_metal, rgba=(0.18, 0.18, 0.2, 1))
        self._dyn["ring"] = []
        NR = 56
        for k in range(NR):
            a = 2 * np.pi * k / NR
            rr = 4.68
            self._g(wb, f"ring{k}", type="box", size=(0.1, 0.27, 0.05),
                    pos=(BALL_C[0] + rr * np.cos(a), BALL_C[1] + rr * np.sin(a), CUP_TOP + 0.03),
                    euler=(0, 0, np.rad2deg(a)), material=m_glow, rgba=(0.2, 0.9, 1, 1))
            self._dyn["ring"].append(f"ring{k}")
        # --- монитор (mocap: поворот «альбом/портрет» вокруг оси взгляда) ---
        mon = wb.add("body", name="monitor", pos=tuple(MON_C), mocap=True)
        self._slab(A, "screen_mesh", SCR_W, SCR_H, 0.03)
        self._g(mon, "screen", type="mesh", mesh="screen_mesh", pos=(-0.26, 0, 0), material=m_scr)
        hw, hh, b = SCR_W / 2, SCR_H / 2, BEZEL
        bz = dict(material=m_black, rgba=(0.025, 0.025, 0.03, 1))
        self._g(mon, "bez_t", type="box", size=(0.1, hw + b, b / 2), pos=(-0.22, 0, hh + b / 2), **bz)
        self._g(mon, "bez_b", type="box", size=(0.1, hw + b, b / 2), pos=(-0.22, 0, -hh - b / 2), **bz)
        self._g(mon, "bez_l", type="box", size=(0.1, b / 2, hh), pos=(-0.22, hw + b / 2, 0), **bz)
        self._g(mon, "bez_r", type="box", size=(0.1, b / 2, hh), pos=(-0.22, -hw - b / 2, 0), **bz)
        self._g(mon, "housing", type="box", size=(0.2, hw + b, hh + b), pos=(0.05, 0, 0), material=m_black,
                rgba=(0.04, 0.04, 0.05, 1))
        self._g(mon, "back", type="box", size=(0.35, hw * 0.55, hh * 0.6), pos=(0.5, 0, 0), material=m_black,
                rgba=(0.05, 0.05, 0.06, 1))
        self._g(mon, "power_led", type="sphere", size=(0.06,), pos=(-0.33, -hw + 0.3, -hh - b / 2),
                material=m_glow, rgba=(0.3, 0.8, 1.0, 1))
        # стойка (не вращается)
        sx = MON_C[0] + 1.6
        self._g(wb, "hinge", type="cylinder", size=(0.55, 0.45), pos=(MON_C[0] + 1.1, 0, MON_C[2]),
                euler=(0, 90, 0), material=m_metal, rgba=(0.22, 0.22, 0.25, 1))
        self._g(wb, "column", type="box", size=(0.35, 0.7, (MON_C[2] - DESK_Z) / 2),
                pos=(sx, 0, (MON_C[2] + DESK_Z) / 2), material=m_metal, rgba=(0.2, 0.2, 0.23, 1))
        self._g(wb, "stand_base", type="box", size=(2.2, 3.2, 0.14), pos=(sx - 0.8, 0, DESK_Z + 0.14),
                material=m_metal, rgba=(0.16, 0.16, 0.18, 1))
        # свечение стены за монитором (цвет экрана)
        HALO = 30.0
        self._slab(A, "halo_mesh", HALO, HALO, 0.02)
        self._g(wb, "halo", type="mesh", mesh="halo_mesh", pos=(WALL_X - 0.02, 0, MON_C[2]), material=m_halo,
                rgba=(0.4, 0.4, 0.6, 0.9))

        # --- колонки по бокам монитора (пульсируют от баса) ---
        self._dyn["spk_ring"], self._dyn["spk_cone"] = [], []
        for s in (1, -1):
            sp = wb.add("body", name=f"spk{s}", pos=(MON_C[0] + 2.2, s * 11.6, DESK_Z + 3.6), euler=(0, 0, s * 18))
            self._g(sp, f"spk{s}_box", type="box", size=(1.6, 1.8, 3.6), material=m_black, rgba=(0.035, 0.035, 0.045, 1))
            self._g(sp, f"spk{s}_ring", type="cylinder", size=(1.42, 0.02), pos=(-1.61, 0, -1.1), euler=(0, 90, 0),
                    material=m_glow, rgba=(1, 0.2, 0.8, 1))
            self._g(sp, f"spk{s}_surround", type="cylinder", size=(1.25, 0.03), pos=(-1.62, 0, -1.1),
                    euler=(0, 90, 0), material=m_matte, rgba=(0.02, 0.02, 0.025, 1))
            self._g(sp, f"spk{s}_cone", type="cylinder", size=(1.0, 0.03), pos=(-1.64, 0, -1.1), euler=(0, 90, 0),
                    material=m_black, rgba=(0.09, 0.09, 0.1, 1))
            self._g(sp, f"spk{s}_cap", type="ellipsoid", size=(0.16, 0.36, 0.36), pos=(-1.66, 0, -1.1),
                    material=m_black, rgba=(0.12, 0.12, 0.13, 1))
            self._g(sp, f"spk{s}_tw_ring", type="cylinder", size=(0.55, 0.02), pos=(-1.61, 0, 1.75),
                    euler=(0, 90, 0), material=m_glow, rgba=(0.3, 0.9, 1, 1))
            self._g(sp, f"spk{s}_tw", type="cylinder", size=(0.45, 0.03), pos=(-1.63, 0, 1.75), euler=(0, 90, 0),
                    material=m_metal, rgba=(0.25, 0.25, 0.28, 1))
            self._dyn["spk_ring"] += [f"spk{s}_ring", f"spk{s}_tw_ring"]
            self._dyn["spk_cone"] += [f"spk{s}_cone", f"spk{s}_cap"]

        # --- неоновая вывеска «МУХА» ---
        NW, NH = 10.5, 10.5 * 384 / 1024
        self._slab(A, "neon_mesh", NW, NH, 0.08)
        self._g(wb, "neon", type="mesh", mesh="neon_mesh", pos=(WALL_X - 0.05, 9.2, 9.6), material=m_neon,
                rgba=(1, 1, 1, 1))
        # --- соты-светильники справа ---
        rr = 1.25
        hv = []
        for k in range(6):
            a = np.pi / 6 + k * np.pi / 3
            hv += [[-0.12, rr * np.cos(a), rr * np.sin(a)], [0.0, rr * np.cos(a), rr * np.sin(a)]]
        A.add("mesh", name="hex", vertex=np.array(hv).ravel())
        self._dyn["hex"] = []
        dx, dz = rr * np.sqrt(3), rr * 1.5
        cells = [(0, 0), (1, 0), (0.5, 1), (1.5, 1), (-0.5, 1), (0, 2), (1, -1)]
        for k, (cx, cz) in enumerate(cells):
            self._g(wb, f"hex{k}", type="mesh", mesh="hex",
                    pos=(WALL_X - 0.02, -9.0 - cx * dx * 1.06, 4.0 + cz * dz * 1.06), material=m_glow,
                    rgba=(0.2, 0.3, 1, 1))
            self._dyn["hex"].append(f"hex{k}")
        # ореолы-«засветки» на стене: эмиссионный геом не светит сам, поэтому рисуем свет текстурой
        self._slab(A, "neon_glow_mesh", 19.0, 11.0, 0.01)
        self._g(wb, "neon_glow", type="mesh", mesh="neon_glow_mesh", pos=(WALL_X - 0.01, 9.2, 9.6), material=m_halo,
                rgba=(1.0, 0.15, 0.65, 0.55))
        self._slab(A, "hex_glow_mesh", 6.5, 6.5, 0.01)
        self._dyn["hex_glow"] = []
        for k, (cx, cz) in enumerate(cells):
            self._g(wb, f"hexglow{k}", type="mesh", mesh="hex_glow_mesh",
                    pos=(WALL_X - 0.012 + 0.001 * k, -9.0 - cx * dx * 1.06, 4.0 + cz * dz * 1.06), material=m_halo,
                    rgba=(0.3, 0.3, 1.0, 0.35))
            self._dyn["hex_glow"].append(f"hexglow{k}")
        # --- постер слева-снизу ---
        self._slab(A, "poster_mesh", 5.4, 7.2, 0.05)
        self._g(wb, "poster", type="mesh", mesh="poster_mesh", pos=(WALL_X - 0.02, 17.5, 1.0), material=m_poster,
                rgba=(1, 1, 1, 1))
        # --- светодиодная лента вдоль стены у стола и над монитором ---
        self._dyn["wall_led"] = []
        NL = 64
        for k in range(NL):
            y = -SIDE_Y + (k + 0.5) * 2 * SIDE_Y / NL
            self._g(wb, f"wled{k}", type="box", size=(0.06, SIDE_Y / NL * 1.0, 0.1), pos=(WALL_X - 0.08, y, DESK_Z + 0.5),
                    material=m_glow, rgba=(1, 0, 0, 1))
            self._dyn["wall_led"].append(f"wled{k}")
        # --- окно с ночным городом на левой стене ---
        WW, WH = 13.0, 16.0
        self._slab(A, "city_mesh", WW, WH, 0.05)
        wx, wz = 4.5, 4.0
        self._g(wb, "city", type="mesh", mesh="city_mesh", pos=(wx, SIDE_Y - 0.05, wz), euler=(0, 0, 90),
                material=m_city, rgba=(1, 1, 1, 1))
        fr = dict(material=m_black, rgba=(0.02, 0.02, 0.025, 1))
        for zz in (wz - WH / 2, wz + WH / 2, wz):
            self._g(wb, f"win_h{zz:.0f}", type="box", size=(WW / 2 + 0.25, 0.25, 0.22), pos=(wx, SIDE_Y - 0.3, zz), **fr)
        for xx in (wx - WW / 2, wx + WW / 2, wx):
            self._g(wb, f"win_v{xx:.0f}", type="box", size=(0.22, 0.25, WH / 2 + 0.25), pos=(xx, SIDE_Y - 0.3, wz), **fr)
        # --- полки с призами ---
        for s in (1, -1):
            self._g(wb, f"shelf{s}", type="box", size=(1.6, 3.4, 0.12), pos=(WALL_X - 1.6, s * 12.2, 3.6),
                    material=m_black, rgba=(0.06, 0.05, 0.07, 1))
            self._g(wb, f"shelf{s}_led", type="box", size=(0.05, 3.3, 0.04), pos=(WALL_X - 3.18, s * 12.2, 3.46),
                    material=m_glow, rgba=(1.0, 0.55, 0.9, 1))
        # кубок с кубиком сахара
        tx, ty, tz = WALL_X - 1.7, 12.6, 3.72
        self._g(wb, "tro_base", type="box", size=(0.55, 0.55, 0.22), pos=(tx, ty, tz + 0.22), material=m_black,
                rgba=(0.08, 0.07, 0.06, 1))
        self._g(wb, "tro_stem", type="cylinder", size=(0.14, 0.35), pos=(tx, ty, tz + 0.8), material=m_gold)
        self._g(wb, "tro_cup", type="cylinder", size=(0.62, 0.42), pos=(tx, ty, tz + 1.55), material=m_gold)
        self._g(wb, "tro_cup_b", type="sphere", size=(0.62,), pos=(tx, ty, tz + 1.15), material=m_gold)
        for s in (1, -1):
            self._g(wb, f"tro_h{s}", type="capsule", fromto=(tx, ty + s * 0.6, tz + 1.8, tx, ty + s * 0.95, tz + 1.3),
                    size=(0.07,), material=m_gold)
        self._g(wb, "sugar_cube", type="box", size=(0.42, 0.42, 0.42), pos=(tx, ty, tz + 2.35),
                euler=(0, 0, 20), material=m_soft, rgba=(1, 0.98, 0.95, 1))
        # банан-плюш
        bx, by, bz = WALL_X - 1.8, -12.2, 3.72
        pts = [np.array([0.0, 1.9 * np.sin(a), 1.2 * (1 - np.cos(a))]) for a in np.linspace(-1.2, 1.2, 11)]
        for k in range(10):
            rad = 0.28 + 0.2 * np.sin(np.pi * (k + 0.5) / 10) ** 0.6
            p1, p2 = pts[k] + [bx, by, bz + rad], pts[k + 1] + [bx, by, bz + rad]
            tip = k == 0 or k == 9
            self._g(wb, f"ban{k}", type="capsule", fromto=(*p1, *p2), size=(rad,), material=m_matte,
                    rgba=(0.35, 0.25, 0.1, 1) if tip else (1.0, 0.86, 0.25, 1))
        for s in (1, -1):                                     # глазки плюша
            self._g(wb, f"ban_eye{s}", type="sphere", size=(0.09,), pos=(bx - 0.46, by + 0.28 * s, bz + 0.55),
                    material=m_black, rgba=(0.02, 0.02, 0.02, 1))
        # мини-геймпад на правой полке
        gx, gy, gz = WALL_X - 1.9, -9.8, 3.72
        self._g(wb, "pad_body", type="capsule", fromto=(gx, gy - 0.5, gz + 0.3, gx, gy + 0.5, gz + 0.3), size=(0.3,),
                material=m_black, rgba=(0.9, 0.9, 0.92, 1))
        self._g(wb, "pad_btn", type="sphere", size=(0.08,), pos=(gx - 0.28, gy - 0.3, gz + 0.42), material=m_glow,
                rgba=(1, 0.2, 0.5, 1))

        # --- кормушка: стеклянный капилляр с сиропом, кончик у хоботка ---
        tip = PIPETTE_TIP
        dirv = np.array([0.80, 0.22, -0.32]); dirv /= np.linalg.norm(dirv)
        end = tip + dirv * 5.2
        self._g(wb, "pip_glass", group=G_PROP, type="capsule", fromto=(*tip, *end), size=(0.075,), material=m_glass)
        self._g(wb, "pip_syrup", group=G_PROP, type="capsule", fromto=(*(tip + dirv * 0.25), *end), size=(0.042,),
                material=m_soft, rgba=(1.0, 0.62, 0.12, 1))
        hold = end + dirv * 0.8
        self._g(wb, "pip_holder", group=G_PROP, type="capsule", fromto=(*end, *hold), size=(0.2,), material=m_metal,
                rgba=(0.3, 0.3, 0.33, 1))
        self._g(wb, "pip_rod", group=G_PROP, type="capsule", fromto=(*hold, hold[0], hold[1], DESK_Z + 0.2),
                size=(0.12,), material=m_metal, rgba=(0.25, 0.25, 0.28, 1))
        self._g(wb, "pip_foot", group=G_PROP, type="cylinder", size=(0.9, 0.12), pos=(hold[0], hold[1], DESK_Z + 0.12),
                material=m_metal, rgba=(0.2, 0.2, 0.22, 1))
        drop = wb.add("body", name="drop", pos=tuple(DROP_POS), mocap=True)
        self._g(drop, "drop", group=G_PROP, type="sphere", size=(0.2,), material=m_soft, rgba=(1.0, 0.7, 0.18, 0.0))
        # --- робо-захват (смена мухи после приступа); прячется под столом, пока не нужен ---
        claw = wb.add("body", name="claw", pos=(0.5, 0.0, 3.0), mocap=True)
        mats = dict(metal=m_metal, dark=m_metal, glow=m_glow)
        nc = [0]

        def add(kind, rgba, mat, **kw):
            nc[0] += 1
            self._g(claw, f"claw{nc[0]}", group=G_PROP, type=kind, rgba=rgba, material=mats[mat], **kw)
            return f"claw{nc[0]}"
        self._claw_names = claw_build(add)
        # --- привязь: тонкая проволока к спинке (муха не улетает — честно показываем) ---
        if tether:
            top = np.array([-0.2, 0.0, 9.0])
            self._g(wb, "tether", group=G_PROP, type="capsule", fromto=(0.25, 0.0, 1.55, *top), size=(0.022,),
                    material=m_metal, rgba=(0.55, 0.55, 0.6, 1))
            self._g(wb, "tether_arm", group=G_PROP, type="capsule", fromto=(*top, -9.0, 0.0, 9.4), size=(0.12,),
                    material=m_metal, rgba=(0.2, 0.2, 0.22, 1))

        # --- свет (всего до 8 с фарой камеры) ---
        L = lambda name, **kw: wb.add("light", name=name, directional=False, **kw)
        # ключевой — со стороны камер (сзади-сверху): тени от лап и крыльев падают вперёд, под муху
        L("key", pos=(-6.0, 1.5, 12.0), dir=(0.45, -0.1, -0.9), castshadow=True, cutoff=24, exponent=5,
          diffuse=(0.78, 0.74, 0.72), specular=(0.45, 0.45, 0.5))
        # свет экрана на муху (цвет = средний цвет кадра), без теней: прозрачные крылья дали бы тёмное пятно
        L("screen_light", pos=(MON_C[0] - 1.5, 0, MON_C[2]), dir=(-1, 0, -0.1), castshadow=False, cutoff=50,
          exponent=1, diffuse=(0.5, 0.5, 0.6), specular=(0.2, 0.2, 0.25))
        # контровые — спереди-сбоку (для камер сзади это «обводка» силуэта), не слишком насыщенные
        L("rim_pink", pos=(5.5, 7.0, 6.0), dir=(-0.55, -0.7, -0.45), castshadow=False, cutoff=20, exponent=6,
          diffuse=(0.85, 0.3, 0.65), specular=(0.9, 0.45, 0.8))
        L("rim_cyan", pos=(5.5, -7.0, 5.0), dir=(-0.55, 0.7, -0.38), castshadow=False, cutoff=20, exponent=6,
          diffuse=(0.42, 0.38, 0.95), specular=(0.5, 0.55, 1.0))
        L("ring_up", pos=(BALL_C[0], BALL_C[1], CUP_TOP - 1.0), dir=(0, 0, 1), castshadow=False, cutoff=75,
          exponent=2, diffuse=(0.1, 0.5, 0.7), specular=(0.1, 0.4, 0.5))
        L("neon_light", pos=(WALL_X - 5.0, 9.2, 8.5), dir=(1, 0, 0.05), castshadow=False, cutoff=55, exponent=2,
          diffuse=(0.6, 0.1, 0.4), specular=(0.0, 0.0, 0.0))
        wb.add("light", name="room_fill", directional=True, pos=(0, 0, 30), dir=(0.3, 0.1, -1), castshadow=False,
               diffuse=(0.13, 0.10, 0.19), specular=(0.0, 0.0, 0.0), ambient=(0.035, 0.03, 0.055))
        wb.add("camera", name="tk", pos=(-8, 0, 4), mode="fixed", fovy=40)

    def _screen_png(self):
        p = ASSETS / f"screen_{self.tex_w}x{self.tex_h}.png"
        if not p.exists():
            ASSETS.mkdir(parents=True, exist_ok=True)
            _save(np.zeros((self.tex_h, self.tex_w, 3), np.uint8), p)
        return p

    def _init_ids(self):
        m = self.m
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{n}_sensorfx")
        self.gid = {k: np.array([gid(n) for n in v]) for k, v in self._dyn.items()}
        for n in ("screen", "halo", "neon", "drop", "city", "neon_glow"):
            self.gid[n] = gid(n)
        assert (self.gid["ring"] >= 0).all() and self.gid["screen"] >= 0
        self.cone_pos0 = m.geom_pos[self.gid["spk_cone"]].copy()
        self.cone_size0 = m.geom_size[self.gid["spk_cone"]].copy()
        jadr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        self.leg_adr = np.array([jadr(f"{self.fly.name}/{j}") for j in self.fly.actuated_joints])
        # в NeuroMechFly имена осей головы перепутаны: joint_Head_yaw крутит вокруг x (наклон),
        # а поворот влево/вправо (вокруг вертикали z) — это joint_Head_roll
        self.head_yaw_adr = jadr(f"{self.fly.name}/joint_Head_roll")
        self.ball_adr = jadr("treadmill_joint")
        self.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "tk")
        self.mon_mocap = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "monitor")]
        self.light = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, n)
                      for n in ("key", "screen_light", "ring_up", "neon_light", "rim_pink", "rim_cyan")}
        self.tex_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "screen_tex")
        assert m.tex_width[self.tex_id] == self.tex_w and m.tex_height[self.tex_id] == self.tex_h
        assert m.tex_nchannel[self.tex_id] == 3
        self.tex_adr = m.tex_adr[self.tex_id]
        fly_prefix = f"{self.fly.name}/"
        self.fly_geoms = np.array([g for g in range(m.ngeom)
                                   if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(fly_prefix)])
        # «кинематографичные» материалы мухи (только в этой модели; файлы flygym не трогаем):
        # блеск хитина, блик в глазах, непрозрачные лапки, тонкие радужные крылья
        look = {"head": (0.35, 0.55, None), "thorax": (0.4, 0.6, None), "a12345": (0.35, 0.55, None),
                "a6": (0.35, 0.55, None), "eye": (0.7, 0.85, None), "antenna": (0.2, 0.4, 1.0),
                "proboscis": (0.2, 0.4, 1.0), "coxa": (0.25, 0.5, 1.0), "femur": (0.25, 0.5, 1.0),
                "tibia": (0.25, 0.5, 1.0), "tarsus": (0.25, 0.5, 1.0), "wing": (0.9, 0.95, 0.2),
                "haltere": (0.3, 0.5, 0.9)}
        for part, (spec, shin, alpha) in look.items():
            mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"{self.fly.name}/{part}_material")
            if mid >= 0:
                m.mat_specular[mid], m.mat_shininess[mid] = spec, shin
                if alpha is not None:
                    m.mat_rgba[mid, 3] = alpha
        # приступ/смерть/новая муха: захват, корень мухи (двигаем body_pos/quat модели), суставы головы
        cg = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{n}_sensorfx")
        self.claw_gid = {k: [cg(n) for n in v] for k, v in self._claw_names.items()}
        self.claw_mocap = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "claw")]
        self.fly_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{self.fly.name}/")
        self.fly_root_pos0 = m.body_pos[self.fly_root].copy()
        self.fly_root_quat0 = m.body_quat[self.fly_root].copy()
        self.head_tilt_adr = jadr(f"{self.fly.name}/joint_Head_yaw")
        self._life_on = False
        self.d.mocap_pos[self.claw_mocap] = CLAW_HIDE
        # клиппинг: znear/zfar в долях extent
        m.vis.map.znear = 0.03 / m.stat.extent
        m.vis.map.zfar = 150.0 / m.stat.extent

    # ------------------------------------------------------------------ puppet
    def _fly_keypoints(self):
        """Углы AABB всех геомов мухи в стойке и с раскрытыми крыльями — точный габарит для кадра."""
        m, d = self.m, self.d
        qpos0 = d.qpos.copy()
        angles = np.concatenate([self.steps.get_joint_angles(leg, 0.0, 0.0) for leg in self.steps.legs])
        pts = []
        for spread in (0.0, 1.0):
            d.qpos[self.leg_adr] = angles
            self._pose_apply(0.0, spread, 0.0)
            mujoco.mj_kinematics(m, d)
            for g in self.fly_geoms:
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                loc = c + h * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
                pts.append(d.geom_xpos[g] + loc @ d.geom_xmat[g].reshape(3, 3).T)
        d.qpos[:] = qpos0
        self._pose_apply(0.0, 0.0, 0.0)
        mujoco.mj_kinematics(m, d)
        pts = np.vstack(pts)
        # грубое прореживание: крайние точки по 26 направлениям хватает для рамки
        dirs = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)])
        c = pts.mean(0)
        idx = np.unique(np.argmax((pts - c) @ dirs.T, axis=0))
        return pts[idx]

    def _pose_apply(self, prob, spread, flap):
        """Хоботок — через FlyPoser; крылья ставим сами. У крыльев NMF локальная ось x смотрит вбок,
        поэтому yaw>0 (против часовой, если смотреть сверху) заводит ЛЕВОЕ крыло на спину, за середину
        тела (так делает FlyPoser — крылья скрещивались). Наружу: левое yaw<0, правое yaw>0.
        Взмах (roll) — знаки как в FlyPoser: оба крыла одновременно вверх."""
        self.pose.apply(prob, 0.0, 0.0)
        for s, k in (("L", -1.0), ("R", 1.0)):
            (yq, yv), (rq, rv) = self.pose.wings[s]
            self.d.qpos[yq] = k * 1.2 * spread
            self.d.qpos[rq] = -k * flap
            self.d.qvel[yv] = self.d.qvel[rv] = 0.0

    def _groom_angles(self, tg):
        """Чистка передними лапами (как groom_angles в show.py): лапы по голове, потом потирание."""
        j = self.base_pose.copy()
        ramp = min(1.0, tg / 0.2)
        w = 2 * np.pi * 4.0
        head_phase = (tg % 1.2) < 0.6
        P_FACE = dict(Femur=-1.6, Tibia=1.0, Coxa=-0.6, Coxa_roll=0.0)
        P_TOP = dict(Femur=-1.4, Tibia=0.6, Coxa=-0.9, Coxa_roll=0.0)
        for li, leg in enumerate(self.steps.legs):
            if leg[1] != "F":
                continue
            sgn = 1 if leg[0] == "L" else -1
            anti = 0 if leg[0] == "L" else np.pi
            if head_phase:
                a = 0.5 + 0.5 * np.sin(w * tg)
                dq = {k: P_FACE[k] + (P_TOP[k] - P_FACE[k]) * a for k in P_FACE}
            else:
                o = np.sin(w * 1.5 * tg + anti)
                dq = dict(Femur=-1.5 + 0.1 * o, Tibia=1.1 + 0.35 * o, Coxa=-0.5, Coxa_roll=(-0.6 + 0.35 * o))
            for dof, v in dq.items():
                j[li * 7 + self.dof[dof]] += ramp * v * (sgn if dof in ("Coxa_roll", "Coxa_yaw") else 1)
        return j

    def step_puppet(self, dt, drive_L=0.0, drive_R=0.0, head_turn=0.0, proboscis=0.0, wing_spread=0.0,
                    wing_flap=0.0, groom_t=None, drop=0.0):
        """Один кадр куклы (без рендера).
        drive_L/R — амплитуды шага левых/правых ног 0..1 (как у контроллера ходьбы flygym);
        head_turn -1..1 (+ = влево); proboscis 0..1; wing_spread 0..1; wing_flap 0..1 — амплитуда
        взмахов (сами взмахи генерируются здесь); groom_t — секунды с начала чистки или None;
        drop — размер капли сахара 0..1."""
        self.t += dt
        # --- ноги ---
        self.cpg.intrinsic_amps = np.repeat(np.abs([drive_L, drive_R]), 3)
        for _ in range(max(1, int(round(dt / 1e-3)))):
            self.cpg.step()
        if groom_t is not None:
            angles = self._groom_angles(groom_t)
        else:
            angles = np.concatenate([self.steps.get_joint_angles(leg, self.cpg.curr_phases[i],
                                                                 self.cpg.curr_magnitudes[i])
                                     for i, leg in enumerate(self.steps.legs)])
        self.d.qpos[self.leg_adr] = angles
        # --- шар под ногами: вперёд + поворот (правые сильнее -> муха поворачивает влево -> шар вправо) ---
        fwd = 9.0 * (drive_L + drive_R) / 2
        yaw = 2.2 * (drive_L - drive_R)
        dq = Rot.from_rotvec(np.array([0.0, -fwd / BALL_R, yaw]) * dt)
        self.ball_q = dq * self.ball_q
        q = self.ball_q.as_quat()
        self.d.qpos[self.ball_adr:self.ball_adr + 4] = q[[3, 0, 1, 2]]
        # --- голова, хоботок, крылья ---
        self.head_angle += (np.clip(head_turn, -1, 1) * 0.35 - self.head_angle) * min(1.0, dt / 0.08)
        self.d.qpos[self.head_yaw_adr] = self.head_angle
        self.flap_phase = (self.flap_phase + dt * 30 * 0.4142) % 1.0      # «стробоскоп» 200-Гц взмахов при 30 fps
        flap = float(np.clip(wing_flap, 0, 1)) * 0.9 * np.sin(2 * np.pi * self.flap_phase)
        self.prob = float(np.clip(proboscis, 0, 1))
        self._pose_apply(self.prob, float(np.clip(wing_spread, 0, 1)), flap)
        self.drop = float(np.clip(drop, -1, 1))              # < 0 — горькая капля
        # --- поворот монитора ---
        if self._or_T > 0:
            k = _smooth((self.t - self._or_t0) / self._or_T)
            self.orient = self._or_from + (self._or_to - self._or_from) * k
        if self._cam_req is not None:                 # кадр следует за поворотом монитора и дрожанием
            self.set_camera(*self._cam_req)
        self._apply()

    def set_orientation(self, mode, t_anim=1.2):
        """'landscape' | 'portrait'. Монитор плавно поворачивается на 90° вокруг оси взгляда за t_anim с.
        Картинка на панели сразу раскладывается под новую ориентацию."""
        target = np.pi / 2 if mode == "portrait" else 0.0
        self.content_portrait = mode == "portrait"
        self._tex_dirty = True
        if t_anim <= 0:
            self.orient, self._or_T = target, 0.0
            self._or_from = self._or_to = target
        else:
            self._or_from, self._or_to, self._or_t0, self._or_T = self.orient, target, self.t, t_anim
        if self._cam_req is not None:                 # мгновенная смена: кадр сразу под новый монитор
            self.set_camera(*self._cam_req)
        self._apply()

    def set_audio_level(self, x):
        """Громкость баса 0..1: колонки «качают», подсветка колонок ярче."""
        self.audio = float(np.clip(x, 0, 1))

    # ------------------------------------------------------------------ effects
    def _apply(self):
        """Всё, что зависит от скаляров состояния (t, audio, drop, orient...), + кинематика."""
        m, d, t = self.m, self.d, self.t
        # монитор
        qm = Rot.from_rotvec([self.orient, 0, 0]).as_quat()
        d.mocap_quat[self.mon_mocap] = qm[[3, 0, 1, 2]]
        d.mocap_pos[self.mon_mocap] = MON_C
        # капля сахара
        g = self.gid["drop"]
        m.geom_rgba[g, :3] = DROP_BITTER if self.drop < 0 else DROP_SUGAR
        m.geom_rgba[g, 3] = 0.88 if abs(self.drop) > 0.03 else 0.0
        m.geom_size[g, 0] = 0.07 + 0.17 * abs(self.drop)
        # хоботок (альфа удлинителя — как в FlyPoser)
        for pg in self.pose.prob_geoms:
            m.geom_rgba[pg, 3] = float(np.clip((self.prob - 0.25) / 0.5, 0, 1))
        a = self.audio
        # RGB: кольцо под шаром, коврик, лента на стене, соты
        n = len(self.gid["ring"])
        m.geom_rgba[self.gid["ring"], :3] = _hsv(0.5 + 0.12 * np.sin(2 * np.pi * (np.arange(n) / n * 2 - t * 0.25)),
                                                 0.75, 0.85 + 0.15 * a)
        n = len(self.gid["wall_led"])
        m.geom_rgba[self.gid["wall_led"], :3] = _hsv(0.83 + 0.25 * np.sin(np.arange(n) / n * 6 - t * 1.1), 0.85,
                                                     0.6 + 0.4 * a)
        n = len(self.gid["hex"])
        ph = np.arange(n) * 0.37
        hue = 0.62 + 0.18 * np.sin(t * 0.4 + ph)
        br = 0.5 + 0.5 * np.sin(t * 1.3 + ph * 2)
        m.geom_rgba[self.gid["hex"], :3] = _hsv(hue, 0.8, 0.35 + 0.25 * br + 0.3 * a)
        m.geom_rgba[self.gid["hex_glow"], :3] = _hsv(hue, 0.75, 1.0)
        m.geom_rgba[self.gid["hex_glow"], 3] = 0.12 + 0.18 * br + 0.2 * a
        # колонки: подсветка и «качающий» диффузор
        m.geom_rgba[self.gid["spk_ring"], :3] = _hsv(0.9 - 0.35 * a, 0.8, 0.35 + 0.65 * a)
        push = 0.12 * a
        m.geom_pos[self.gid["spk_cone"]] = self.cone_pos0 - np.array([push, 0, 0])
        m.geom_size[self.gid["spk_cone"]] = self.cone_size0 * np.array([1 + 0.05 * a, 1, 1])
        # неон: редкое мерцание (детерминировано по времени)
        fl = 1.0
        ph = (t % 23.0)
        if 11.0 < ph < 11.5:
            fl = 0.35 + 0.65 * (np.sin(ph * 90) > -0.2)
        m.geom_rgba[self.gid["neon"], :3] = fl
        m.geom_rgba[self.gid["neon_glow"], 3] = 0.55 * fl
        m.light_diffuse[self.light["neon_light"]] = np.array([0.6, 0.1, 0.4]) * fl
        # свет экрана и свечение стены
        gcol = np.clip(self.glow, 0, 1)
        m.light_diffuse[self.light["screen_light"]] = 0.25 + 0.85 * gcol
        m.light_specular[self.light["screen_light"]] = 0.25 * gcol
        m.geom_rgba[self.gid["halo"], :3] = 0.12 + 0.75 * gcol
        m.light_diffuse[self.light["ring_up"]] = _hsv(0.5 + 0.12 * np.sin(-2 * np.pi * t * 0.25), 0.8, 0.55)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        if self.cam is not None:
            self._place_camera()

    # ------------------------------------------------------------------ screen
    def _watermark(self, W, H):
        text = (WATERMARK or "").strip()                   # пусто / пробелы — без знака
        key = (W, H, text)                                 # текст в ключе: смена WATERMARK на живом Stage применяется
        if self._wm is None or self._wm[0] != key:
            size = max(14, int(0.08 * min(W, H)))
            f = _font(size)
            meas = ImageDraw.Draw(Image.new("L", (1, 1)))
            tl = meas.textlength(text, font=f)
            room = W * 0.9 - int(size * 0.35) - 8
            if text and tl > room:                         # длинный ник — мельче шрифт, чтобы влез в экран
                size = max(8, int(size * room / tl))
                f = _font(size)
                tl = meas.textlength(text, font=f)
            tw = int(tl) + 8
            th = int(size * 1.35)
            im = Image.new("L", (tw, th), 0)
            if text:
                ImageDraw.Draw(im).text((4, 0), text, font=f, fill=255)
            a = np.asarray(im).astype(np.float32) / 255
            sh = cv2.GaussianBlur(a, (0, 0), max(1.0, size * 0.06))           # мягкая тень для читаемости
            self._wm = (key, a, sh, int(size * 0.35))
        return self._wm

    def _compose_canvas(self, frame):
        """Кадр ролика в родных пропорциях, вписанный в панель целевой ориентации, + водяной знак
        справа снизу (полупрозрачный белый с мягкой тенью). Поля/поворот добавит _update_texture."""
        W, H = (self.tex_h, self.tex_w) if self.content_portrait else (self.tex_w, self.tex_h)
        if frame is None:
            canvas = np.zeros((H, W, 3), np.uint8)
        else:
            fh, fw = frame.shape[:2]
            sc = min(W / fw, H / fh)
            nw, nh = min(W, max(8, int(round(fw * sc)))), min(H, max(8, int(round(fh * sc))))
            canvas = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        H, W = canvas.shape[:2]
        _, a, sh, pad = self._watermark(W, H)
        th, tw = a.shape
        y0, x0 = max(0, H - th - pad // 2), max(0, W - tw - pad)
        a, sh = a[:H - y0, :W - x0], sh[:H - y0, :W - x0]
        roi = canvas[y0:y0 + a.shape[0], x0:x0 + a.shape[1]].astype(np.float32)
        roi = roi * (1 - 0.45 * sh[..., None])                                  # тень
        roi = roi * (1 - 0.65 * a[..., None]) + 255 * 0.65 * a[..., None]       # полупрозрачный белый
        canvas = canvas.copy() if frame is not None and canvas is frame else canvas
        canvas[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = np.clip(roi, 0, 255).astype(np.uint8)
        return canvas

    def set_screen_frame(self, rgb):
        """Новый кадр ролика (uint8 HxWx3 RGB, любая ориентация). Текстура обновится при рендере."""
        self._frame = rgb
        self._tex_dirty = True
        small = cv2.resize(rgb, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32) / 255
        c = small.mean(0)
        c = 0.6 * c + 0.4 * small.max(0) * (c.sum() > 0.05)      # чуть насыщеннее среднего
        self.glow = np.clip(c * 1.1, 0, 1)
        self._apply_glow()

    def _apply_glow(self):
        gcol = np.clip(self.glow, 0, 1)
        self.m.light_diffuse[self.light["screen_light"]] = 0.25 + 0.85 * gcol
        self.m.light_specular[self.light["screen_light"]] = 0.25 * gcol
        self.m.geom_rgba[self.gid["halo"], :3] = 0.12 + 0.75 * gcol

    def _update_texture(self):
        key = (self.content_portrait, round(self.orient, 5))
        if not self._tex_dirty and key == self._tex_key:
            return
        canvas = self._compose_canvas(self._frame)          # прямой кадр в целевой раскладке + знак
        th = self.orient
        TW, TH = self.tex_w, self.tex_h
        if abs(th) < 1e-4 and canvas.shape[:2] == (TH, TW):
            tex = canvas
        elif abs(th - np.pi / 2) < 1e-4 and canvas.shape[:2] == (TW, TH):
            tex = np.rot90(canvas, k=1)                        # панель повёрнута на 90° -> кладём на бок
        else:
            # промежуточный угол (или кадр «не той» ориентации): поворачиваем против панели на th и
            # вписываем — зритель видит картинку ровно, пока монитор крутится
            ch, cw = canvas.shape[:2]
            c, sn = abs(np.cos(th)), abs(np.sin(th))
            sc = min(TW / (cw * c + ch * sn), TH / (cw * sn + ch * c))
            M = cv2.getRotationMatrix2D((cw / 2, ch / 2), np.rad2deg(th), sc)
            M[:, 2] += (TW / 2 - cw / 2, TH / 2 - ch / 2)
            tex = cv2.warpAffine(canvas, M, (TW, TH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        n = TW * TH * 3
        self.m.tex_data[self.tex_adr:self.tex_adr + n] = np.ascontiguousarray(tex).reshape(-1)
        self._tex_dirty = False
        self._tex_key = key
        self._tex_ver += 1

    # ------------------------------------------------------------------ camera
    def _screen_corners(self, angle=None, pad=0.0):
        ang = self.orient if angle is None else angle
        R = Rot.from_rotvec([ang, 0, 0]).as_matrix()
        hw, hh = SCR_W / 2 + pad, SCR_H / 2 + pad
        loc = np.array([[-0.3, y, z] for y, z in ((hw, hh), (-hw, hh), (-hw, -hh), (hw, -hh))])
        return MON_C + loc @ R.T

    def _keypoints(self, fit="both"):
        scr = self._screen_corners(pad=BEZEL)
        if fit == "screen":
            return scr
        if fit == "pov":                           # POV: экран + голова мухи передним планом
            head = np.array([[x, y, z] for x in (0.45, 1.05) for y in (-0.38, 0.38) for z in (0.95, 1.65)])
            return np.vstack([scr, head])
        return np.vstack([self.fly_pts, scr])

    def _fit(self, pos, look, roll, margin, fit="both"):
        """Навести камеру на центр «муха+экран» и подобрать угол, чтобы всё влезло с полями.
        Возвращает pos, fwd, up и половины кадра в тангенсах (tx, ty) — угол обзора потом
        считается под соотношение сторон конкретного рендера."""
        pos = np.asarray(pos, float)
        fwd = np.asarray(look, float) - pos
        fwd /= np.linalg.norm(fwd)
        pts = self._keypoints(fit)
        c, s = np.cos(roll), np.sin(roll)

        def basis(f):
            right = np.cross(f, [0, 0, 1.0]); right /= np.linalg.norm(right)
            up = np.cross(right, f)
            return c * right + s * up, -s * right + c * up

        for _ in range(12):                           # центрируем в угловом пространстве (сходится)
            right, up = basis(fwd)
            v = pts - pos
            z = np.maximum(v @ fwd, 0.3)              # защита от точек за камерой
            thx, thy = np.arctan((v @ right) / z), np.arctan((v @ up) / z)
            cx, cy = (thx.max() + thx.min()) / 2, (thy.max() + thy.min()) / 2
            fwd = np.cos(cx) * fwd + np.sin(cx) * right
            fwd = np.cos(cy) * fwd + np.sin(cy) * up
            fwd /= np.linalg.norm(fwd)
            if abs(cx) + abs(cy) < 1e-5:
                break
        right, up = basis(fwd)
        v = pts - pos
        z = np.maximum(v @ fwd, 0.3)
        ax, ay = (v @ right) / z, (v @ up) / z
        k = 1.0 / (1 - 2 * margin)
        return pos, fwd, up, np.abs(ax).max() * k, np.abs(ay).max() * k

    def _preset(self, p):
        """Пресет с учётом ориентации монитора: pos_p/look_p (если заданы) подмешиваются по углу поворота."""
        P = dict(self.PRESETS[p] if isinstance(p, str) else p)
        w = float(np.clip(self.orient / (np.pi / 2), 0, 1))
        w = w * w * (3 - 2 * w)
        for k in ("pos", "look"):
            if k + "_p" in P:
                P[k] = (1 - w) * np.asarray(P[k], float) + w * np.asarray(P[k + "_p"], float)
        return P

    def set_camera(self, preset_a, preset_b=None, blend=0.0, shake=0.0, zoom=1.0):
        """Ракурс: имя пресета (или dict pos/look/roll/margin). preset_b + blend 0..1 — плавный
        переход (blend сглаживается внутри). shake 0..1 — «ручная камера» (медленный дрейф, по self.t).
        zoom < 1 — чуть ближе (поля урезаются, но муха и экран остаются в кадре)."""
        self._cam_req = (preset_a, preset_b, blend, shake, zoom)
        A = self._preset(preset_a)
        if preset_b is not None and blend > 0:
            B = self._preset(preset_b)
            k = _smooth(blend)
            lerp = lambda key, dflt=0.0: (1 - k) * np.asarray(A.get(key, dflt), float) + k * np.asarray(B.get(key, dflt), float)
            pa, pb = np.asarray(A["pos"], float), np.asarray(B["pos"], float)
            mid = 0.5 * (pa + pb)
            # дуга: чуть отодвигаемся от мухи в середине перехода, чтобы не пролетать сквозь неё
            out = mid / max(np.linalg.norm(mid), 1e-6)
            pos = (1 - k) * pa + k * pb + out * 1.5 * np.sin(np.pi * k) * (np.linalg.norm(pa - pb) > 4)
            P = dict(pos=pos, look=lerp("look"), roll=float(lerp("roll")), margin=float(lerp("margin", 0.06)),
                     fit=A.get("fit", "both") if k < 0.5 else B.get("fit", "both"))
        else:
            P = A
        pos = np.asarray(P["pos"], float).copy()
        look = np.asarray(P["look"], float).copy()
        roll = float(P.get("roll", 0.0))
        if shake > 0:
            t = self.t
            pos += shake * np.array([0.10 * np.sin(0.37 * t + 1.0) + 0.04 * np.sin(1.3 * t),
                                     0.14 * np.sin(0.29 * t + 2.0) + 0.05 * np.sin(1.1 * t + 0.5),
                                     0.09 * np.sin(0.41 * t + 4.0) + 0.03 * np.sin(1.7 * t)])
            look += shake * np.array([0.0, 0.12 * np.sin(0.23 * t), 0.08 * np.sin(0.31 * t + 1.0)])
            roll += shake * 0.012 * np.sin(0.27 * t + 0.3)
        margin = float(P.get("margin", 0.06)) - (1 - zoom) * 0.5
        margin = max(margin, 0.015)
        pos, fwd, up, tx, ty = self._fit(pos, look, roll, margin, P.get("fit", "both"))
        self.cam = np.concatenate([pos, fwd, up, [tx, ty]])
        self._place_camera()

    def _place_camera(self, aspect=None):
        c = self.cam
        pos, fwd, up, tx, ty = c[:3], c[3:6], c[6:9], c[9], c[10]
        if aspect is None:
            aspect = getattr(self, "_aspect", 1080 / 1150)
        tan_y = max(ty, tx / aspect)
        m = self.m
        m.cam_pos[self.cam_id] = pos
        m.cam_quat[self.cam_id] = _look_quat(fwd, up)
        m.cam_fovy[self.cam_id] = np.rad2deg(2 * np.arctan(tan_y))
        mujoco.mj_camlight(m, self.d)

    # ------------------------------------------------------------------ state
    def get_state(self):
        """Компактное состояние кадра (numpy) — для прохода 2 в другом процессе."""
        return dict(qpos=self.d.qpos.copy(), mocap_pos=self.d.mocap_pos.copy(), mocap_quat=self.d.mocap_quat.copy(),
                    fx=np.array([self.t, self.audio, self.drop, self.prob, self.orient, float(self.content_portrait)]),
                    cam=self.cam.copy())

    def set_state(self, s):
        self.d.qpos[:] = s["qpos"]
        mp, mq = np.asarray(s["mocap_pos"]), np.asarray(s["mocap_quat"])
        self.d.mocap_pos[:len(mp)] = mp                   # старые логи — без захвата
        self.d.mocap_quat[:len(mq)] = mq
        t, audio, drop, prob, orient, portrait = [float(x) for x in s["fx"]]
        self.t, self.audio, self.drop, self.prob, self.orient = t, audio, drop, prob, orient
        if bool(portrait) != self.content_portrait:
            self.content_portrait = bool(portrait)
            self._tex_dirty = True
        self._or_T = 0.0
        self.cam = np.asarray(s["cam"], float).copy()
        self._life_apply(s)
        self._apply()

    # ------------------------------------------------------------------ приступ / смерть / новая муха
    CURL = dict(Coxa=0.55, Coxa_roll=0.0, Femur=-1.0, Tibia=1.9, Tarsus1=-0.9)    # поджатые лапки мёртвой мухи

    def _life_prep(self):
        """Один раз: грудь мухи (точка поворота), лапы стойки и поджатые, высота «на спине» над шаром."""
        if getattr(self, "_life_c", None) is not None:
            return
        m, d = self.m, self.d
        q0 = d.qpos.copy()
        th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{self.fly.name}/Thorax")
        m.body_pos[self.fly_root], m.body_quat[self.fly_root] = self.fly_root_pos0, self.fly_root_quat0
        mujoco.mj_kinematics(m, d)
        self._life_c = d.xpos[th].copy()
        self._stance = np.concatenate([self.steps.get_joint_angles(leg, 0.0, 0.0) for leg in self.steps.legs])
        curl = self.base_pose.copy()
        for li in range(6):
            for k, v in self.CURL.items():
                curl[li * 7 + self.dof[k]] += v
        self._curl = curl
        # на спине: самая нижняя точка корпуса ложится на шар
        d.qpos[self.leg_adr] = curl
        self._fly_place(Rot.from_rotvec([np.pi, 0, 0]), self._life_c)
        mujoco.mj_kinematics(m, d)
        names = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") for g in self.fly_geoms]
        core = [g for g, n in zip(self.fly_geoms, names)
                if any(k in n for k in ("Thorax", "/A", "Head", "Eye"))]
        low = min(float((d.geom_xpos[g] - np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_aabb[g, 3:])[2])
                  for g in core)
        c = self._life_c
        top = BALL_C[2] + np.sqrt(max(0.0, BALL_R ** 2 - (c[0] - BALL_C[0]) ** 2 - (c[1] - BALL_C[1]) ** 2))
        self._life_cf = c + np.array([0.0, 0.0, top - low - 0.04])
        d.qpos[:] = q0
        m.body_pos[self.fly_root], m.body_quat[self.fly_root] = self.fly_root_pos0, self.fly_root_quat0
        mujoco.mj_kinematics(m, d)

    def _fly_place(self, R, c):
        """Вся муха как твёрдое тело: поворот R вокруг груди, грудь -> точка c (правка модели в этом процессе)."""
        q = R.as_quat()
        self.m.body_quat[self.fly_root] = (q[3], q[0], q[1], q[2])
        self.m.body_pos[self.fly_root] = np.asarray(c, float) - R.apply(self._life_c)

    def _dead_pose(self, lt, legs_from):
        """Поза мёртвой мухи через lt с после приступа: падает на спину, лапки поджимаются вверх."""
        k = _smooth(lt / 0.7)
        bounce = 0.1 * np.sin(np.pi * np.clip((lt - 0.7) / 0.35, 0, 1)) * (lt > 0.7)
        R = Rot.from_rotvec([(np.pi - 0.1) * k, 0, 0]) * Rot.from_rotvec([0, 0.06 * k, 0])
        c = self._life_c + (self._life_cf - self._life_c) * k + np.array([0, -0.35 * np.sin(np.pi * k), 0.3 * np.sin(np.pi * k) + bounce])
        u = _smooth((lt - 0.15) / 0.9)
        legs = legs_from + (self._curl - legs_from) * u
        # предсмертные подёргивания лапок: затухают
        tw = np.exp(-lt / 1.4) * 0.35 * life_jitter(lt, len(legs), 7) * (lt > 0.6)
        return R, c, legs + tw

    def _life_apply(self, s):
        """Кадр приступа / смерти / смены мухи (life, life_t из hud кадра; воркер кладёт их в state)."""
        life, lt = life_of(s)
        m, d = self.m, self.d
        if life == "alive":
            if self._life_on:                                  # обычный кадр: всё как было
                m.body_pos[self.fly_root], m.body_quat[self.fly_root] = self.fly_root_pos0, self.fly_root_quat0
                d.mocap_pos[self.claw_mocap] = CLAW_HIDE
                self._life_on = False
            return
        self._life_prep()
        self._life_on = True
        d.mocap_pos[self.claw_mocap] = CLAW_HIDE
        d.mocap_quat[self.claw_mocap] = (1, 0, 0, 0)
        legs = d.qpos[self.leg_adr].copy()
        if life == "seizure":
            k = seizure_k(lt)
            j = life_jitter(lt, 60, 1)
            d.qpos[self.leg_adr] = legs + 0.55 * k * j[:42]
            R = Rot.from_rotvec([0.14 * k * j[42], 0.1 * k * j[43], 0.12 * k * j[44]])
            c = self._life_c + k * np.array([0.05 * j[45], 0.06 * j[46], 0.07 * abs(j[47])])
            self._fly_place(R, c)
            d.qpos[self.head_yaw_adr] = 0.5 * k * j[48]
            d.qpos[self.head_tilt_adr] = 0.3 * k * j[49]
            for sd, sg, jj in (("L", -1.0, j[50:52]), ("R", 1.0, j[52:54])):
                (yq, _), (rq, _) = self.pose.wings[sd]
                d.qpos[yq] = sg * (0.35 + 0.35 * jj[0]) * k
                d.qpos[rq] = -sg * 0.9 * k * jj[1]
            return
        # смерть и смена мухи: капли нет, шар стоит (ball_q — шар в момент смерти)
        self.drop = 0.0
        self.prob = 0.0
        bq = s.get("ball_q")
        if bq is not None:
            d.qpos[self.ball_adr:self.ball_adr + 4] = np.asarray(bq, float)[self.ball_adr:self.ball_adr + 4]
            if life == "respawn" and lt > 2.45:                # новая муха пошла: шар догоняет лог
                w = _smooth((lt - 2.45) / 0.55)
                q1 = np.asarray(s["qpos"], float)[self.ball_adr:self.ball_adr + 4]
                qa, qb = np.asarray(bq, float)[self.ball_adr:self.ball_adr + 4], q1
                if np.dot(qa, qb) < 0:
                    qb = -qb
                qq = qa + (qb - qa) * w
                d.qpos[self.ball_adr:self.ball_adr + 4] = qq / np.linalg.norm(qq)
        z, op, carry = (0.0, 1.0, None) if life == "dead" else claw_track(lt)
        lift = np.array([0.0, 0.0, z if carry is not None else 0.0])
        if life == "dead" or carry == "old" or (carry is None and lt < 1.8):
            R, c0, lg = self._dead_pose(lt if life == "dead" else DEATH_T, legs)
            if carry == "old":                                  # несут: чуть раскачивается
                R = Rot.from_rotvec([0.08 * np.sin(6 * lt), 0, 0]) * R
            d.qpos[self.leg_adr] = lg
            d.qpos[self.head_yaw_adr] = 0.25
            d.qpos[self.head_tilt_adr] = 0.0
            self._pose_apply(0.0, 0.05, -0.2)
            self._fly_place(R, c0 + lift)
            hub = c0 + np.array([0.0, 0.0, CLAW_GRIP + 0.25])
        else:                                                   # новая муха: свежая поза (лапы — из лога)
            c0 = self._life_c
            if carry == "new":
                d.qpos[self.leg_adr] = self._stance
            self._fly_place(Rot.identity(), c0 + lift)
            hub = c0 + np.array([0.0, 0.0, CLAW_GRIP])
        if life == "respawn":
            d.mocap_pos[self.claw_mocap] = hub + np.array([0.0, 0.0, z])
            claw_fingers(m, self.claw_gid, op)

    # ------------------------------------------------------------------ render
    def _renderer(self, width, height):
        key = (width, height)
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.m, height=height, width=width)
            self._renderers[key] = r
            self._uploaded[key] = -1
        return r

    def render(self, width=1080, height=1150, bloom=0.3, segmentation=False, scene_option=None):
        """uint8 RGB (height, width, 3). segmentation=True -> int32 (h, w, 2) [id, тип объекта]."""
        self._aspect = width / height
        self._place_camera(self._aspect)
        self._update_texture()
        r = self._renderer(width, height)
        key = (width, height)
        if self._uploaded[key] != self._tex_ver:
            r._gl_context.make_current()
            mujoco.mjr_uploadTexture(self.m, r._mjr_context, self.tex_id)
            self._uploaded[key] = self._tex_ver
        r.update_scene(self.d, camera=self.cam_id, scene_option=scene_option or self.opt)
        sc = r.scene
        sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        sc.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        sc.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        sc.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
        if segmentation:
            return self._render_seg(r)
        img = r.render()
        if bloom > 0:
            img = self._bloom(img, bloom)
        return img

    @staticmethod
    def _render_seg(r):
        """Сегментация (h, w, 2) = [id объекта, тип]; -1 — фон. Свой декодер: при MSAA цвета-ID
        на краях смешиваются, и штатный Renderer падает на «несуществующих» id."""
        sc = r.scene
        flags = sc.flags.copy()
        sc.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
        sc.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
        sc.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        r._gl_context.make_current()
        mujoco.mjr_render(r._rect, sc, r._mjr_context)
        img = np.empty((r.height, r.width, 3), np.uint8)
        mujoco.mjr_readPixels(img, None, r._rect, r._mjr_context)
        np.copyto(sc.flags, flags)
        img = img[::-1].astype(np.int64)
        seg = img[..., 0] + img[..., 1] * 256 + img[..., 2] * 65536
        n = sc.ngeom
        table = np.full((n + 2, 2), -1, np.int32)
        for g in sc.geoms[:n]:
            if g.segid != -1:
                table[g.segid + 1] = (g.objid, g.objtype)
        return table[np.where(seg <= n, seg, n + 1)]

    @staticmethod
    def _bloom(img, strength):
        """Лёгкое свечение ярких мест (неон, экран): всё на cv2 в uint8, 1/4 и 1/8 разрешения (~2 мс)."""
        h, w = img.shape[:2]
        q = cv2.resize(cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR),
                       (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)
        c0, c1, c2 = cv2.split(q)
        k = cv2.convertScaleAbs(cv2.subtract(cv2.max(cv2.max(c0, c1), c2), 165), alpha=3)
        hi = cv2.multiply(q, cv2.merge([k, k, k]), scale=1 / 255)
        wide = cv2.GaussianBlur(cv2.resize(hi, (w // 8, h // 8), interpolation=cv2.INTER_AREA), (0, 0), 6)
        b = cv2.addWeighted(cv2.GaussianBlur(hi, (0, 0), 3), 0.6 * strength,
                            cv2.resize(wide, (hi.shape[1], hi.shape[0]), interpolation=cv2.INTER_LINEAR),
                            0.9 * strength, 0)
        return cv2.add(img, cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR))

    # ------------------------------------------------------------------ checks
    def project(self, p3, width, height):
        """3D -> пиксели текущей камеры (x, y, глубина)."""
        self._place_camera(width / height)
        pos, mat = self.d.cam_xpos[self.cam_id], self.d.cam_xmat[self.cam_id].reshape(3, 3)
        f = (height / 2) / np.tan(np.deg2rad(self.m.cam_fovy[self.cam_id]) / 2)
        v = (np.atleast_2d(p3) - pos) @ mat
        return np.stack([width / 2 + f * v[:, 0] / -v[:, 2], height / 2 - f * v[:, 1] / -v[:, 2], -v[:, 2]], 1)

    def _watermark_points(self, n=5):
        """Сетка точек по области водяного знака на экране (мировые координаты)."""
        canvas_hw = ((self.tex_w, self.tex_h) if self.content_portrait else (self.tex_h, self.tex_w))
        if self._frame is not None:
            fh, fw = self._frame.shape[:2]
            W, H = canvas_hw[1], canvas_hw[0]
            sc = min(W / fw, H / fh)
            canvas_hw = (min(H, int(round(fh * sc))), min(W, int(round(fw * sc))))
        ch, cw = canvas_hw
        _, a, _, pad = self._watermark(cw, ch)
        th_, tw_ = a.shape
        c, sn = abs(np.cos(self.orient)), abs(np.sin(self.orient))
        fit = min(self.tex_w / (cw * c + ch * sn), self.tex_h / (cw * sn + ch * c))
        mm = fit * SCR_W / self.tex_w                          # мм на пиксель холста
        u0, u1 = cw - tw_ - pad + tw_ * 0.1, cw - pad - tw_ * 0.1
        v0, v1 = ch - th_ - pad / 2 + th_ * 0.25, ch - pad / 2 - th_ * 0.2
        pts = []
        for u in np.linspace(u0, u1, n):
            for v in np.linspace(v0, v1, 3):
                pts.append([MON_C[0] - 0.3, MON_C[1] - (u - cw / 2) * mm, MON_C[2] - (v - ch / 2) * mm])
        return np.array(pts)

    def check_visibility(self, width=270, height=288):
        """Проверка кадра: экран и муха в кадре и не заслонены (по сегментации)."""
        seg = self.render(width, height, segmentation=True)
        ids = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
        corners = self.project(self._screen_corners(), width, height)
        poly = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(poly, np.round(corners[:, :2] * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
        scr_vis = (ids == self.gid["screen"]).sum() / max(1, poly.sum())
        inside = bool(((corners[:, 0] >= 0) & (corners[:, 0] < width) & (corners[:, 1] >= 0) &
                       (corners[:, 1] < height)).all())
        opt = mujoco.MjvOption()
        for g in range(6):
            opt.geomgroup[g] = 1 if g in FLY_GROUPS else 0
        seg_f = self.render(width, height, segmentation=True, scene_option=opt)
        ids_f = np.where(seg_f[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg_f[..., 0], -1)
        fly_only = np.isin(ids_f, self.fly_geoms).sum()
        fly_vis = np.isin(ids, self.fly_geoms).sum() / max(1, fly_only)
        fb = self.project(self.fly_pts, width, height)
        fly_in = bool(((fb[:, 0] >= 0) & (fb[:, 0] < width) & (fb[:, 1] >= 0) & (fb[:, 1] < height)).all())
        wm = self.project(self._watermark_points(), width, height)
        wx, wy = np.round(wm[:, 0]).astype(int), np.round(wm[:, 1]).astype(int)
        ok = (wx >= 0) & (wx < width) & (wy >= 0) & (wy < height)
        wm_vis = float((ids[wy[ok], wx[ok]] == self.gid["screen"]).sum() / len(wm))
        return dict(screen_visible=float(scr_vis), screen_inside=inside, watermark_visible=wm_vis,
                    fly_visible=float(fly_vis),
                    fly_inside=fly_in, fly_px_frac=float(fly_only / (width * height)),
                    screen_px_frac=float(poly.sum() / (width * height)))

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


# ======================= самопроверка =======================
SOURCE_VIDEO = ROOT / "видео на реакцию" / "VILLIAN,_madk1d_ДИНАСТИЯ_2_ft_Андрей_Гайдулян_Клип_2026.mp4"
SCRATCH = HERE / "_scratch"


def _grab_frame(path, t):
    """Один кадр ролика на момент t (PyAV)."""
    import av
    c = av.open(str(path))
    vs = c.streams.video[0]
    c.seek(int(t / vs.time_base), stream=vs)
    for f in c.decode(vs):
        if f.time is not None and f.time >= t - 1e-3:
            img = f.to_ndarray(format="rgb24")
            c.close()
            return img
    c.close()
    raise RuntimeError("кадр не найден")


def _label(img, lines):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    f = _font(max(14, img.shape[0] // 26))
    y = 8
    for ln in lines:
        dr.text((10, y), ln, font=f, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        y += int(f.size * 1.25)
    return np.asarray(im)


def _selftest():
    import time
    SCRATCH.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    st = Stage()
    t_build = time.perf_counter() - t0
    land = _grab_frame(SOURCE_VIDEO, 42.0)
    # вертикального исходника пока нет — имитируем: вырез 9:16 из клипа
    src = _grab_frame(SOURCE_VIDEO, 70.0)
    h, w = src.shape[:2]
    cw = int(h * 9 / 16)
    port = np.ascontiguousarray(src[:, (w - cw) // 2:(w - cw) // 2 + cw])
    for i in range(20):
        st.step_puppet(1 / 30, 0.55, 0.35, head_turn=0.4, proboscis=0.9, drop=0.8)
    st.set_audio_level(0.7)
    rows, report = [], []
    for orient, fr in (("landscape", land), ("portrait", port)):
        st.set_orientation(orient, 0)
        st.set_screen_frame(fr)
        row = []
        for name in Stage.PRESET_ORDER:
            st.set_camera(name)
            img = st.render(1080, 1150)
            chk = st.check_visibility()
            report.append((orient, name, chk, float(st.m.cam_fovy[st.cam_id])))
            small = cv2.resize(img, (432, 460), interpolation=cv2.INTER_AREA)
            row.append(_label(small, [f"{orient} · {name}",
                                      f"экран {chk['screen_visible']:.0%}  знак {chk['watermark_visible']:.0%}",
                                      f"муха {chk['fly_visible']:.0%} ({chk['fly_px_frac']:.1%} кадра)"]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    Image.fromarray(sheet).save(SCRATCH / "stage_presets.png")
    for orient, name, chk, fov in report:
        print(f"{orient:9s} {name:9s} fovy {fov:5.1f}  screen {chk['screen_visible']:.3f} in={chk['screen_inside']}  "
              f"wm {chk['watermark_visible']:.2f}  fly {chk['fly_visible']:.3f} in={chk['fly_inside']}  "
              f"fly {chk['fly_px_frac']:.3f}  scr {chk['screen_px_frac']:.3f}")
    # --- скорость ---
    st.set_orientation("landscape", 0)
    frames = [_grab_frame(SOURCE_VIDEO, 30.0 + k) for k in range(3)]
    states = []
    for i in range(60):
        st.step_puppet(1 / 30, 0.5, 0.5, head_turn=np.sin(i / 9), proboscis=0.5, drop=0.5)
        st.set_camera(Stage.PRESET_ORDER[(i // 20) % 3], Stage.PRESET_ORDER[(i // 20 + 1) % 3], (i % 20) / 20,
                      shake=1.0)
        states.append(st.get_state())
    tt = dict(render=[], render_nobloom=[], full=[], step=[])
    for i in range(40):
        a = time.perf_counter()
        st.render(1080, 1150)
        tt["render"].append(time.perf_counter() - a)
        a = time.perf_counter()
        st.render(1080, 1150, bloom=0)
        tt["render_nobloom"].append(time.perf_counter() - a)
        a = time.perf_counter()
        st.set_state(states[i])
        st.set_screen_frame(frames[i % 3])
        st.render(1080, 1150)
        tt["full"].append(time.perf_counter() - a)
        a = time.perf_counter()
        st.step_puppet(1 / 30, 0.5, 0.5)
        st.get_state()
        tt["step"].append(time.perf_counter() - a)
    for k, v in tt.items():
        v = np.array(v[5:]) * 1000
        print(f"время {k:15s} среднее {v.mean():6.1f} мс  медиана {np.median(v):6.1f}  макс {v.max():6.1f}")
    print(f"сборка сцены {t_build:.1f} с")
    # --- состояние: туда-обратно в этом процессе ---
    s0 = states[33]
    st.set_state(s0); st.set_screen_frame(frames[0])
    img0 = st.render(1080, 1150)
    for i in range(15):
        st.step_puppet(1 / 30, 1.0, 0.2, head_turn=-1, proboscis=0.0, wing_spread=1, wing_flap=1)
    st.set_orientation("portrait", 0.5)
    st.set_camera("high_34")
    st.render(1080, 1150)
    st.set_state(s0); st.set_screen_frame(frames[0])
    img1 = st.render(1080, 1150)
    d_in = np.abs(img0.astype(int) - img1).max()
    print("round trip (тот же процесс): макс. разница пикселей =", d_in)
    # --- состояние: в другом процессе ---
    np.savez(SCRATCH / "rt_state.npz", frame=frames[0], img=img0, **s0)
    import subprocess
    r = subprocess.run([sys.executable, __file__, "--roundtrip", str(SCRATCH / "rt_state.npz")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-2000:])
    st.close()


def _roundtrip(path):
    z = np.load(path)
    st = Stage()
    st.set_state({k: z[k] for k in ("qpos", "mocap_pos", "mocap_quat", "fx", "cam")})
    st.set_screen_frame(z["frame"])
    img = st.render(1080, 1150)
    diff = np.abs(img.astype(int) - z["img"])
    print(f"round trip (другой процесс): макс. разница = {diff.max()}, пикселей с разницей > 2: {(diff.max(-1) > 2).sum()}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--roundtrip":
        _roundtrip(sys.argv[2])
    else:
        _selftest()
