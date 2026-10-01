"""Врезка «как видит муха»: кадр ролика глазами мухи (фасеточная мозаика flygym) + лампы детекторов.

    import flyeye
    fe = flyeye.FlyEye()                                   # один раз на воркер (~0.15 с, +25 МБ)
    rgba = fe.render(frame_rgb, feats_row, (280, 280))     # uint8 (h, w, 4), альфа НЕ умножена
    # frame_rgb — кадр ролика (как на мониторе, родная ориентация), None — нет кадра (экран тёмный)
    # feats_row — dict признаков этого кадра, которые получил мозг (formats.VIS_KEYS:
    #             obj_L, obj_R, loom_L, loom_R, flow_x) или None — лампы серые, «нет данных»

Вызывают compose (Базовый) и compose_streamer через хук Composer.flyeye, если в окне включено
«как видит муха». Бюджет: < 15 мс на врезку (на деле ~2–3 мс при 280x280; таблицы экрана для
16:9, 9:16, 1:1 и раскладки 280/250 строятся в __init__, ~0.15 с; редкие пропорции — +10 мс один раз).

Как считается (всё — по flygym 1.2.1, flygym.vision.Retina + config.yaml):
  * глаза — камеры LEye_cam / REye_cam NeuroMechFly: fovy 157°, сырое изображение 512x450,
    позиция/поворот из config['vision']['sensor_positions'] (euler XYZ — оси мира, как в MJCF NMF);
    потом Retina.correct_fisheye (zoom 2.72, k 3.8) и 721 фасетка на глаз по ommatidia_id_map;
  * вместо рендера MuJoCo — трассировка лучей: для каждого пикселя фасетки считаем луч (с той же
    формулой рыбьего глаза, что Retina._correct_fisheye) и где он попадает в экран. Экран — как в
    stage.Stage: 16x9 мм (портретный ролик — монитор повёрнут, 9x16), центр в 12 мм перед глазами.
    Так ролик занимает в поле зрения ровно столько, сколько монитор: ~67°x41° (альбом), ~41°x67°
    (портрет). Для Стримера принято то же расстояние и тот же экран (своей геометрии монитора у
    комнаты нет); другое — FlyEye(screen_dist=..., panel=...);
  * фасетка = среднее по её пикселям, как Retina.raw_image_to_hex_pxls: «жёлтые» фасетки берут
    зелёный канал, «бледные» (pale_type_mask) — синий; красный канал flygym не использует вовсе
    (у мух R1-6 почти не чувствуют красный — в кадре красное выглядит тёмным). Показываем серым
    с лёгким холодным оттенком: про цветовое зрение ничего не утверждаем;
  * показ: гамма 0.75 и подстройка к средней яркости ролика (фоторецепторы адаптируются к фону):
    тёмный ролик светлее не больше чем в ADAPT_MAX раз, светлый — как есть. ommatidia() — без этого;
  * ориентация проверена рендером настоящих камер flygym (_scratch/flyeye/verify_flygym.py):
    обе камеры дают прямое (не зеркальное) изображение, верх — вверху; у левого глаза перёд мухи —
    у ПРАВОГО края картинки, у правого — у ЛЕВОГО. Поэтому врезка [левый | правый] — панорама с
    «носом» мухи посередине, ничего отражать не надо; середину экрана (±13°) видят оба глаза;
  * показываем переднюю часть поля зрения (рыскание от внутреннего края глаза до ~52° вбок,
    тангаж ±32°, для портретного ролика ±42°) — там весь монитор; масштаб углов настоящий
    (обрезка, не растяжение), грани фасеток видны;
  * комната вокруг экрана тёмная (немного подсвечена самим экраном) — вместо белых углов flygym.
Лампы — вход в мозг из лога (senses.inputs_from_frame): LC10 Л/П — мелкий движущийся объект,
LPLC2/LC4 Л/П — надвигание (ниже LOOM_MIN — не горит), поток T4/T5 влево/вправо (мёртвая зона
FLOW_DEADZONE); яркость = частота / max_hz. Ничего не выдумываем: нет данных — серые + «нет данных».
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "C:/Windows/Fonts/bahnschrift.ttf"
CAPTION = "как видит муха"
SUBCAPTION = "721 фасетка на глаз · упрощённо"

# --- геометрия, мм; система мухи stage.Stage: x — вперёд, y — влево, z — вверх, муха в (0, 0, 0) ---
SCREEN_DIST = 12.7 - 0.726      # плоскость экрана (stage: MON_C.x 13 - 0.3) минус x камер глаз
SCREEN_DZ = 1.8 - 1.314         # центр экрана (MON_C.z 1.8) выше глаз на ~0.5 мм
PANEL = (16.0, 9.0)             # активная область экрана (stage.SCR_W, SCR_H); портрет — 9x16
BEZEL = 0.28                    # рамка монитора (чёрная)
EYE_BODY = (0.756, 0.0, 1.314)  # тела LEye/REye в модели NMF (голова прямо), к ним — rel_pos камер
# что показываем, град: левый глаз — от внутреннего края (~-13°) до VIEW_YAW влево, тангаж ±VIEW_PITCH
VIEW_YAW = 52.0
VIEW_PITCH = {False: 32.0, True: 42.0}     # альбом/квадрат (экран ~41° в высоту) | портрет (~67°)
GRID = 64                       # кадр ролика -> GRIDxGRID (по UV экрана), ~5x5 отсчётов на фасетку
ROOM = 0.03                     # яркость тёмной комнаты вокруг экрана (0..1)
ROOM_GLOW = 0.10                # + подсветка комнаты экраном (вблизи экрана, x средняя яркость)
TINT = np.array([0.90, 1.0, 0.94], np.float32)       # лёгкий холодный оттенок серого
DISPLAY_GAMMA = 0.75
# подстройка к средней яркости экрана (фоторецепторы адаптируются): тёмный ролик показываем светлее,
# но не больше чем в ADAPT_MAX раз; светлые ролики не трогаем. Только показ — ommatidia() как у flygym
ADAPT_TARGET = 0.36
ADAPT_MAX = 2.4
PREBUILD_ASPECTS = ((16, 9), (9, 16), (1, 1))           # таблицы экрана — заранее, в __init__
PREBUILD_SIZES = ((280, 280), (250, 250))               # раскладки врезки Базового и Стримера

# цвета (RGB)
C_BG = (9, 11, 13)
C_TEXT = (238, 242, 240)
C_DIM = (150, 160, 158)
C_OBJ = (80, 205, 255)          # LC10 — мелкий объект
C_LOOM = (255, 128, 48)         # LPLC2/LC4 — надвигание
C_FLOW = (120, 232, 140)        # T4/T5 — общий поток (картинка едет влево/вправо)
C_OFF = (58, 63, 68)


def _size(size):
    if isinstance(size, (int, np.integer)):
        return int(size), int(size)
    return int(size[0]), int(size[1])


def _font(px, bold=False):
    f = ImageFont.truetype(FONT_PATH, max(8, int(round(px))))
    try:
        f.set_variation_by_name("Bold" if bold else "Regular")
    except Exception:
        pass
    return f


def _rot_xyz(a, b, c):
    """euler XYZ (заглавные = оси мира, как compiler eulerseq в MJCF NMF): R = Rz(c) Ry(b) Rx(a)."""
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rz @ ry @ rx


def eye_cameras():
    """Камеры глаз flygym в системе мухи: {'L'|'R': (pos, R 3x3 — столбцы: вправо, вверх, назад), fovy}."""
    from flygym.util import load_config
    cfg = load_config()["vision"]
    out = {}
    for s in "LR":
        sp = cfg["sensor_positions"][f"{s}Eye_cam"]
        out[s] = (np.array(EYE_BODY) + np.array(sp["rel_pos"], float), _rot_xyz(*sp["orientation"]))
    return out, float(cfg["fovy_per_eye"])


class FlyEye:
    """Врезка «как видит муха». Всё тяжёлое — в __init__ и в кэшах по размеру/пропорциям кадра;
    render() — функция кадра, без состояния между кадрами."""

    def __init__(self, screen_dist=SCREEN_DIST, screen_dz=SCREEN_DZ, panel=PANEL):
        from flygym.vision import Retina
        self.ret = ret = Retina()
        self.n = int(ret.num_ommatidia_per_eye)                          # 721
        self.idm = ret.ommatidia_id_map.astype(np.int32)                 # (512, 450), 0 — вне глаза
        self.npx = ret.num_pixels_per_ommatidia.astype(np.float64)
        self.pale = np.asarray(ret.pale_type_mask, np.int64)             # 1 — бледная (синий), 0 — жёлтая (зелёный)
        self.screen_dist, self.screen_dz, self.panel = float(screen_dist), float(screen_dz), tuple(panel)
        nr, nc = ret.nrows, ret.ncols
        cams, fovy = eye_cameras()
        # пиксели фасеток после рыбьего глаза -> пиксель сырого кадра (ровно как Retina._correct_fisheye)
        rows, cols = np.nonzero(self.idm)
        rn = ((2.0 * rows - nr) / nr) / ret.zoom
        cn = ((2.0 * cols - nc) / nc) / ret.zoom
        den = 1.0 - ret.distortion_coefficient * (rn * rn + cn * cn) + 1e-6
        sr = np.trunc((rn / den + 1.0) * nr / 2.0)                      # int() в numba — к нулю
        sc = np.trunc((cn / den + 1.0) * nc / 2.0)
        ok = (sr >= 0) & (sr < nr) & (sc >= 0) & (sc < nc)               # вне сырого кадра — чёрный (как flygym)
        f = (nr / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        ray = np.stack([(sc + 0.5 - nc / 2.0) / f, (nr / 2.0 - sr - 0.5) / f, -np.ones_like(sc)], -1)
        self._omm = self.idm[rows, cols] - 1                             # 0..720
        self._ok = ok
        self._dir, self._pos = {}, {}
        for s in "LR":
            pos, R = cams[s]
            d = ray @ R.T
            self._dir[s] = (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(np.float32)
            self._pos[s] = pos
        # подсветка комнаты экраном: чем ближе средний луч фасетки к центру экрана, тем светлее
        glow = []
        for s in "LR":
            m = np.zeros((self.n, 3))
            np.add.at(m, self._omm, self._dir[s])
            m /= np.linalg.norm(m, axis=1, keepdims=True)
            to_scr = np.array([self.screen_dist, -self._pos[s][1], self.screen_dz])
            ang = np.degrees(np.arccos(np.clip(m @ (to_scr / np.linalg.norm(to_scr)), -1, 1)))
            glow.append(np.exp(-(np.maximum(ang - 25.0, 0) / 30.0) ** 2))
        self._glow = np.concatenate(glow).astype(np.float32)
        # какую часть сетки показываем (правый глаз — симметричный диапазон столбцов, картинка не отражается)
        self._crop = {pt: self._view_crop(VIEW_PITCH[pt]) for pt in (False, True)}
        self._geo = {}
        self._lay = {}
        self._lut = self._make_lut()
        # частые пропорции кадра и размеры врезки — сразу, чтобы первый кадр ролика не был медленнее
        for fw, fh in PREBUILD_ASPECTS:
            self._geo[self._geo_key(fw, fh)] = self._screen_table(fw, fh)
        for sz in PREBUILD_SIZES:
            for pt in (False, True):
                self._layout(*sz, portrait=pt)

    @staticmethod
    def _geo_key(fw, fh):
        return fh > fw, round(fw / fh, 3)

    # ------------------------------------------------------------------ геометрия экрана
    def _view_crop(self, pitch):
        """Прямоугольник сетки левого глаза (r0, r1, c0, c1): от внутреннего края до VIEW_YAW вбок,
        тангаж ±pitch. Считается по лучам центральной строки/столбца."""
        nr, nc = self.idm.shape
        rows, cols = np.nonzero(self.idm)
        d = self._dir["L"]
        yaw = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        pit = np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1)))
        mid_r = np.abs(rows - nr // 2) <= 2
        c0 = int(cols[mid_r & (yaw <= VIEW_YAW)].min())
        mid_c = np.abs(cols - (c0 + nc) // 2) <= 2
        r0 = int(rows[mid_c & (pit <= pitch)].min())
        r1 = int(rows[mid_c & (pit >= -pitch)].max()) + 1
        return r0, r1, c0, nc

    def _screen_table(self, fw, fh):
        """Для пропорций кадра: пары (фасетка, ячейка сетки·канал, вес) и доля «комнаты» у фасетки.
        Ролик вписан в панель монитора как в stage._compose_canvas (поля — чёрные)."""
        portrait = fh > fw
        pw, ph = (self.panel[1], self.panel[0]) if portrait else self.panel
        sc = min(pw / fw, ph / fh)
        cw, ch = fw * sc, fh * sc
        keys, room = [], np.zeros(2 * self.n)
        for e, s in enumerate("LR"):
            d, p = self._dir[s], self._pos[s]
            dx = np.where(d[:, 0] > 1e-6, d[:, 0], np.nan)
            t = self.screen_dist / dx                                    # плоскость экрана: x = x_глаз + dist
            hy = p[1] + t * d[:, 1]                                      # от центра экрана, + = влево
            hz = t * d[:, 2] - self.screen_dz                            # от центра экрана, + = вверх
            hit = np.isfinite(t) & self._ok
            on_mon = hit & (np.abs(hy) <= pw / 2 + BEZEL) & (np.abs(hz) <= ph / 2 + BEZEL)
            in_clip = hit & (np.abs(hy) <= cw / 2) & (np.abs(hz) <= ch / 2)
            u = np.clip((cw / 2 - hy[in_clip]) / cw, 0, 1 - 1e-9)       # левый край ролика — слева у мухи
            v = np.clip((ch / 2 - hz[in_clip]) / ch, 0, 1 - 1e-9)
            omm = self._omm[in_clip]
            cell = (v * GRID).astype(np.int64) * GRID + (u * GRID).astype(np.int64)
            chan = 1 + self.pale[omm]                                     # жёлтая -> G, бледная -> B (flygym)
            keys.append((omm + e * self.n) * (GRID * GRID * 3) + cell * 3 + chan)
            rm = self._ok & ~on_mon
            room[e * self.n:(e + 1) * self.n] = np.bincount(self._omm[rm], minlength=self.n) / self.npx
        k, cnt = np.unique(np.concatenate(keys), return_counts=True)
        omm_g = k // (GRID * GRID * 3)
        pix = k % (GRID * GRID * 3)
        w = cnt / self.npx[omm_g % self.n]
        clip_frac = (cw * ch) / (pw * ph)
        return omm_g.astype(np.intp), pix.astype(np.intp), w.astype(np.float32), room.astype(np.float32), clip_frac

    def ommatidia(self, frame_rgb):
        """Яркость 2x721 фасеток (0..1): [левый глаз, правый глаз] — то же, что Retina.raw_image_to_hex_pxls
        по кадру камеры глаза (max по двум каналам). None -> тёмный экран."""
        return self._ommatidia(frame_rgb)[0]

    def _ommatidia(self, frame_rgb):
        """(яркость фасеток, средняя яркость ролика по зелёному/синему 0..1 или None — нет кадра)"""
        if frame_rgb is not None:
            if frame_rgb.ndim == 2:                                      # серый кадр -> RGB
                frame_rgb = np.repeat(frame_rgb[..., None], 3, axis=2)
            elif frame_rgb.shape[2] != 3:                                # RGBA -> RGB
                frame_rgb = frame_rgb[..., :3]
        if frame_rgb is None or frame_rgb.shape[0] < 1 or frame_rgb.shape[1] < 1:
            frame_rgb, fh, fw = None, 9, 16
        else:
            fh, fw = frame_rgb.shape[:2]
        key = self._geo_key(fw, fh)
        g = self._geo.get(key)
        if g is None:
            g = self._geo[key] = self._screen_table(fw, fh)
        omm, pix, w, room, clip_frac = g
        if frame_rgb is None:
            vals = np.zeros(2 * self.n, np.float64)
            mean = None
        else:
            small = cv2.resize(np.ascontiguousarray(frame_rgb), (GRID, GRID), interpolation=cv2.INTER_AREA)
            flat = small.reshape(-1).astype(np.float32)
            vals = np.bincount(omm, weights=flat[pix] * w, minlength=2 * self.n) / 255.0
            mean = float(small[..., 1:].mean()) / 255.0
        vals += room * (ROOM + ROOM_GLOW * (mean or 0.0) * clip_frac * self._glow)
        return vals, mean

    # ------------------------------------------------------------------ раскладка врезки
    def _make_lut(self):
        # показ: гамма 0.75 (фоторецепторы отвечают сжато, примерно как логарифм) — тёмные ролики читаемее
        v = np.linspace(0, 1, 256, dtype=np.float32) ** DISPLAY_GAMMA
        return np.clip(v[:, None] * TINT[None] * 255 + 0.5, 0, 255).astype(np.uint8)

    def _layout(self, w, h, portrait=False):
        k = (w, h, portrait)
        if k in self._lay:
            return self._lay[k]
        s = min(w, h) / 280.0
        pad = max(4, round(8 * s))
        fs1, fs2, fs3 = 19 * s, max(9.0, 11 * s), max(9.0, 11.5 * s)
        f1, f2, f3, f3b = _font(fs1, True), _font(fs2), _font(fs3), _font(fs3, True)
        top = pad + round(fs1 * 1.05) + round(fs2 * 1.25) + round(4 * s)
        row_h = round(fs3 * 1.55)
        lamps_h = 3 * row_h + round(2 * s)
        gap = max(3, round(4 * s))
        # глаза: пропорции обрезки сетки
        r0, r1, c0, c1 = self._crop[portrait]
        asp = (c1 - c0) / (r1 - r0)
        aw, ah = (w - 2 * pad - gap) / 2, h - top - pad - lamps_h - round(3 * s)
        ew, eh = (int(ah * asp), int(ah)) if aw / ah > asp else (int(aw), int(aw / asp))
        ex0 = (w - (2 * ew + gap)) // 2
        ey0 = top + max(0, (int(ah) - eh) // 2)
        # индекс фасетки для каждого пикселя врезки: левый глаз — столбцы c0..nc (перёд справа),
        # правый — столбцы 0..nc-c0 (перёд слева); порядок столбцов сохраняется — без отражения
        yy = r0 + (np.arange(eh) + 0.5) * (r1 - r0) / eh
        xx = c0 + (np.arange(ew) + 0.5) * (c1 - c0) / ew
        ri = np.clip(yy.astype(int), 0, self.idm.shape[0] - 1)
        ci = np.clip(xx.astype(int), 0, self.idm.shape[1] - 1)
        nc = self.idm.shape[1]
        id_l = self.idm[ri[:, None], ci[None, :]] - 1
        id_r = self.idm[ri[:, None], (nc - 1 - ci)[None, ::-1]] - 1
        n2 = 2 * self.n
        idx = np.full((eh, 2 * ew + gap), n2, np.int64)                   # n2 — фон (вне глаза / щель)
        idx[:, :ew] = np.where(id_l >= 0, id_l, n2)
        idx[:, ew + gap:] = np.where(id_r >= 0, id_r + self.n, n2)
        # грани фасеток: сосед справа/снизу — другая фасетка -> темнее
        edge = np.zeros(idx.shape, bool)
        edge[:, :-1] |= idx[:, :-1] != idx[:, 1:]
        edge[:-1, :] |= idx[:-1, :] != idx[1:, :]
        # мягкое затемнение обрезанных краёв (внешний, верх, низ); внутренний край — настоящий край глаза
        fx = np.minimum(1.0, (np.arange(ew) + 0.5) / (0.22 * ew))
        fy = np.minimum(1.0, np.minimum(np.arange(eh) + 0.5, eh - 0.5 - np.arange(eh)) / (0.16 * eh))
        vig_l = np.sqrt(fy[:, None] * fx[None, :])
        vig = np.ones(idx.shape, np.float32)
        vig[:, :ew] = vig_l
        vig[:, ew + gap:] = vig_l[:, ::-1]
        NV = 8
        lev = np.clip(np.round(vig * (NV - 1)), 0, NV - 1).astype(np.int64)
        lev = np.where(edge & (idx < n2), lev + NV, lev)                 # грань — второй набор уровней
        comb = lev * (n2 + 1) + idx
        mult = np.concatenate([np.linspace(0.18, 1.0, NV), np.linspace(0.18, 1.0, NV) * 0.5]).astype(np.float32)
        # подложка: скруглённая плашка (альфа), рамка; над мозаикой — непрозрачно
        rad = max(6, round(14 * s))
        pm = Image.new("L", (w, h), 0)
        ImageDraw.Draw(pm).rounded_rectangle((0, 0, w - 1, h - 1), rad, fill=236)
        alpha = np.asarray(pm).copy()
        alpha[ey0:ey0 + eh, ex0:ex0 + 2 * ew + gap] = 255
        panel = Image.new("RGB", (w, h), C_BG)
        ImageDraw.Draw(panel).rounded_rectangle((0, 0, w - 1, h - 1), rad, outline=(78, 86, 90),
                                                width=max(1, round(1.2 * s)))
        panel = np.asarray(panel).copy()
        # подписи (отдельный слой поверх мозаики и ламп)
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dr = ImageDraw.Draw(im)
        cx = w // 2
        dr.text((cx, pad + fs1 * 0.52), CAPTION, font=f1, fill=(*C_TEXT, 255), anchor="mm")
        dr.text((cx, pad + round(fs1 * 1.05) + fs2 * 0.6), SUBCAPTION, font=f2, fill=(*C_DIM, 255), anchor="mm")
        fl = _font(max(9.0, 12 * s), True)
        m_in = max(3, round(5 * s))
        for x, anc, t in ((ex0 + m_in, "la", "Л"), (ex0 + 2 * ew + gap - m_in, "ra", "П")):
            dr.text((x, ey0 + m_in), t, font=fl, fill=(*C_TEXT, 235), anchor=anc, stroke_width=max(1, round(2 * s)),
                    stroke_fill=(0, 0, 0, 255))
        ly0 = ey0 + eh + round(5 * s)
        rl = max(3, round(5.5 * s))
        R = rl * 3
        # лампа: диск + ореол (маски 0..1, цвет — по яркости кадра)
        yy2, xx2 = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
        dist = np.hypot(xx2, yy2)
        disk = np.clip(rl + 0.5 - dist, 0, 1)
        halo = np.exp(-(np.maximum(dist - rl, 0) / (rl * 0.9)) ** 2) * (1 - disk)
        ring = np.clip(1 - np.abs(dist - rl), 0, 1) * 0.5
        lamps = []
        # лампы: (ключ, подпись слева, справа, пояснение посередине, цвет)
        rows_ = (("obj", "LC10 Л", "LC10 П", "мелкий объект", C_OBJ),
                 ("loom", "LPLC2/LC4 Л", "LPLC2/LC4 П", "надвигается", C_LOOM),
                 ("flow", "влево", "вправо", "поток T4/T5", C_FLOW))
        xl, xr = pad + rl + 1, w - pad - rl - 1
        for j, (key, name_l, name_r, what, col) in enumerate(rows_):
            cy = ly0 + j * row_h + row_h // 2
            tl, tr = xl + rl + round(5 * s), xr - rl - round(5 * s)
            dr.text((tl, cy), name_l, font=f3b, fill=(*C_TEXT, 255), anchor="lm")
            dr.text((tr, cy), name_r, font=f3b, fill=(*C_TEXT, 255), anchor="rm")
            room_c = (tr - dr.textlength(name_r, font=f3b)) - (tl + dr.textlength(name_l, font=f3b))
            if dr.textlength(what, font=f3) + 8 * s < room_c:              # узко — без пояснения посередине
                dr.text((cx, cy), what, font=f3, fill=(*C_DIM, 255), anchor="mm")
            for side, lx in (("_L", xl), ("_R", xr)):
                x0_, y0_ = max(0, lx - R), max(0, cy - R)
                x1_, y1_ = min(w, lx + R + 1), min(h, cy + R + 1)
                sl = (slice(y0_ - (cy - R), y1_ - (cy - R)), slice(x0_ - (lx - R), x1_ - (lx - R)))
                lamps.append((key + side, (slice(y0_, y1_), slice(x0_, x1_)), np.array(col, np.float32),
                              disk[sl][..., None], halo[sl][..., None], ring[sl][..., None]))
        txt = np.asarray(im)
        tm = txt[..., 3] > 0
        # «нет данных» / «поток …» — готовые надписи
        fn = _font(max(9.0, 11 * s), True)

        def label(text, col, y):
            t = Image.new("RGBA", (w, round(fs3 * 2.2)), (0, 0, 0, 0))
            ImageDraw.Draw(t).text((w // 2, t.height // 2), text, font=fn, fill=(*col, 255), anchor="mm",
                                   stroke_width=max(1, round(2 * s)), stroke_fill=(0, 0, 0, 255))
            a = np.asarray(t)
            hh = max(0, min(a.shape[0], h - y))
            return y, a[:hh, :, :3].astype(np.float32), a[:hh, :, 3:4].astype(np.float32) / 255.0

        none_y = ey0 + eh - round(fs3 * 2.4)                           # над низом мозаики
        labels = {"none": label("нет данных", (200, 205, 205), none_y)}
        lay = dict(ex0=ex0, ey0=ey0, ew=ew, eh=eh, gap=gap, comb=comb, mult=mult, n2=n2, panel=panel,
                   alpha=alpha, txt_m=tm, txt_rgb=txt[..., :3][tm].astype(np.float32),
                   txt_a=(txt[..., 3][tm].astype(np.float32) / 255.0)[:, None], lamps=lamps, labels=labels)
        self._lay[k] = lay
        return lay

    # ------------------------------------------------------------------ кадр
    @staticmethod
    def rates(feats_row):
        """Яркость ламп 0..1 = частота входа в мозг / max_hz (senses.inputs_from_frame); None — нет данных.
        Поток: картинка едет влево -> левый глаз «спереди назад» (T4a/T5a), правый — «сзади вперёд»
        (T4b/T5b); лампа «влево» = ftb_L / max_hz, «вправо» = ftb_R / max_hz (мёртвая зона — как в senses)."""
        if feats_row is None:
            return None
        import senses
        r = senses.inputs_from_frame(feats_row)
        S = senses.SENSES
        v = {k: float(np.clip(r[k] / S[k]["max_hz"], 0, 1)) for k in ("obj_L", "obj_R", "loom_L", "loom_R")}
        v["flow_L"] = float(np.clip(r["ftb_L"] / S["ftb_L"]["max_hz"], 0, 1))
        v["flow_R"] = float(np.clip(r["ftb_R"] / S["ftb_R"]["max_hz"], 0, 1))
        return v

    def render(self, frame_rgb, feats_row, size):
        w, h = _size(size)
        lay = self._layout(w, h, frame_rgb is not None and frame_rgb.shape[0] > frame_rgb.shape[1])
        vals, mean = self._ommatidia(frame_rgb)
        if mean is not None:                                              # адаптация к яркости ролика (только показ)
            vals = vals * float(np.clip(ADAPT_TARGET / max(mean, 1e-3), 1.0, ADAPT_MAX))
        # мозаика: яркость фасетки -> цвет, уровни затемнения (края, грани) -> одна выборка по индексу
        q = np.clip(vals * 255 + 0.5, 0, 255).astype(np.uint8)
        col = np.empty((lay["n2"] + 1, 3), np.float32)
        col[:-1] = self._lut[q]
        col[-1] = C_BG
        table = (col[None] * lay["mult"][:, None, None]).astype(np.uint8)
        table[:, -1] = C_BG                                               # фон вне глаза не затемняем
        eye = table.reshape(-1, 3)[lay["comb"]]
        out = np.empty((h, w, 4), np.uint8)
        base = out[..., :3]
        base[...] = lay["panel"]
        y0, x0 = lay["ey0"], lay["ex0"]
        base[y0:y0 + eye.shape[0], x0:x0 + eye.shape[1]] = eye
        # лампы: яркость = частота входа в мозг / max_hz
        v = self.rates(feats_row)
        c_off = np.array(C_OFF, np.float32)
        for key, sl, con, disk, halo, ring in lay["lamps"]:
            a = 0.0 if v is None else v[key]
            reg = base[sl].astype(np.float32)
            if a > 0:
                reg += (con - reg) * (halo * (0.85 * a))
            reg += (c_off * (1 - a) + con * a - reg) * disk
            reg += (con * 0.5 + 60 - reg) * ring
            base[sl] = np.clip(reg, 0, 255).astype(np.uint8)
        # подписи поверх
        m = lay["txt_m"]
        bm = base[m].astype(np.float32)
        base[m] = (bm + (lay["txt_rgb"] - bm) * lay["txt_a"]).astype(np.uint8)
        # нет признаков зрения в логе — так и пишем
        lab = lay["labels"]["none"] if v is None else None
        if lab is not None and lab[1].shape[0] > 0:
            ly, rgb, a = lab
            reg = base[ly:ly + rgb.shape[0]].astype(np.float32)
            base[ly:ly + rgb.shape[0]] = (reg + (rgb - reg) * a).astype(np.uint8)
        out[..., 3] = lay["alpha"]
        return out
