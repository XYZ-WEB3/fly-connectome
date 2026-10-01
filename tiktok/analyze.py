"""«Движок интереса», часть 1: разбор ролика -> покадровая шкала (30 к/с) для прохода симуляции.

    import analyze as A
    tl = A.analyze(r"...\\ролик.mp4")          # кэш: tiktok/_cache/<имя>.json
    tl["meta"]["orientation"]                   # 'landscape' | 'portrait'
    tl["loom_L"][i]                             # numpy float32, длина meta["n_frames_out"]

Ключи результата (все покадровые массивы — numpy float32 длиной n_frames_out, кадр i <-> время i / fps_out):
  meta    : w, h (как показывается, с учётом поворота), fps_src, fps_out, duration, orientation, n_frames_out,
            rotation, content_box (область без чёрных полос, доли [x0, y0, x1, y1]),
            content_orientation / content_aspect (ориентация и w/h картинки внутри content_box: альбомный клип
            в вертикальном кадре с полосами -> 'landscape'), has_audio, tempo_bpm, ...
  t       : время кадра вывода, с
  зрение (упрощённая сетчатка; муха смотрит в экран -> левая половина кадра = левый глаз):
    obj_L, obj_R   0..1  мелкие объекты, которые движутся сами (поток минус движение камеры)  -> LC10
    obj_x         -1..1  где самый заметный мелкий объект (0 — нет)
    loom_L, loom_R 0..1  надвигание: изотропное расширение поля (наезд/приближение) или растущее
                         тёмное пятно; на панорамы, полосы, склейки не срабатывает            -> LPLC2 / LC4
    zoom           0..1  доля надвигания от расширения всего кадра (наезд камеры)
    flow_x        -1..1  общее горизонтальное движение поля (>0 — картинка едет вправо)
    cut            0..1  смена кадра (импульс в первом кадре нового плана)
    flash          0..1  резкое повышение яркости (импульс)
    motion         0..1  энергия движения
    light          0..1  средняя яркость кадра
  звук:
    loud  0..1 громкость (RMS, дБ, нормировка по ролику)
    bass  0..1 энергия 40-160 Гц
    hear  0..1 энергия 100-500 Гц с упором на 150-250 Гц (орган Джонстона настроен на песню ухаживания)
    onset 0..1 сила атаки (спектральный поток)
    beat  0/1  доли такта
    drop  0..1 «дроп»: скачок энергии/баса после более тихого куска
  excite 0..1 — сводная «острота момента» (звук + картинка)
  events : [{t, frame, kind, strength, caption, side?}], kind in drop | peak | cut | flash | zoom

Также: iter_frames(path, fps_out=30, size=None, start=0, stop=None) — кадры ролика (RGB, с учётом поворота)
в том же пересчёте на fps_out, что и анализ (кадр i = последний кадр источника с временем <= i / fps_out).
"""
import json
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "_cache"
VERSION = 3
PIX = 160 * 90          # пикселей в кадре анализа (альбом 160x90, портрет 90x160)
SR = 22050              # частота звука для анализа
FRAME_KEYS = ["t", "obj_L", "obj_R", "obj_x", "loom_L", "loom_R", "zoom", "flow_x", "cut", "flash", "motion",
              "light", "loud", "bass", "hear", "onset", "beat", "drop", "excite"]
EVENT_KINDS = ["drop", "peak", "cut", "flash", "zoom"]


# ---------------------------------------------------------------- чтение видео
def _small_size(w, h, pix=PIX):
    s = (pix / float(w * h)) ** 0.5
    return max(16, int(round(w * s / 2)) * 2), max(16, int(round(h * s / 2)) * 2)


def _yuv_planes(f):
    """Y, U, V кадра (uint8). U, V — половинного размера."""
    h, w = f.height, f.width
    if h % 2 == 0 and w % 2 == 0:
        try:
            a = f.to_ndarray(format="yuv420p").ravel()
            n = h * w
            if a.size == n * 3 // 2:
                return (a[:n].reshape(h, w), a[n:n + n // 4].reshape(h // 2, w // 2),
                        a[n + n // 4:].reshape(h // 2, w // 2))
        except Exception:
            pass
    yuv = cv2.cvtColor(f.to_ndarray(format="rgb24"), cv2.COLOR_RGB2YUV)
    return yuv[..., 0], np.ascontiguousarray(yuv[::2, ::2, 1]), np.ascontiguousarray(yuv[::2, ::2, 2])


def _rot_k(frame):
    """Сколько раз повернуть массив np.rot90, чтобы получить картинку как в плеере."""
    return int(round((getattr(frame, "rotation", 0) or 0) / 90.0)) % 4


def _read_video(path, fps_out=30):
    """Кадры в низком разрешении (серые) + цветовые гистограммы + времена (от начала контейнера).
    Берём только кадры, которые реально попадут на сетку fps_out (как в iter_frames): у 60 к/с — каждый
    второй. Зрение видит ровно то, что покажет монитор, а пороги потока не зависят от частоты источника."""
    c = av.open(str(path))
    vs = c.streams.video[0]
    vs.thread_type = "AUTO"
    fps = float(vs.average_rate or vs.guessed_rate or 30)
    t_base = (c.start_time or 0) / 1e6
    times, grays, hists, all_t = [], [], [], []
    st = dict(k_rot=0, csz=None, disp=None)

    def take(t, f):
        if st["csz"] is None:
            st["k_rot"] = k = _rot_k(f)
            cw, ch = f.width, f.height
            st["disp"] = (ch, cw) if k % 2 else (cw, ch)
            sw, sh = _small_size(*st["disp"])
            st["csz"] = (sh, sw) if k % 2 else (sw, sh)       # (w, h) в кодированной ориентации
        csz, k_rot = st["csz"], st["k_rot"]
        y, u, v = _yuv_planes(f)
        ys = cv2.resize(y, csz, interpolation=cv2.INTER_AREA)
        us = cv2.resize(u, (csz[0] // 2, csz[1] // 2), interpolation=cv2.INTER_AREA)
        vv = cv2.resize(v, (csz[0] // 2, csz[1] // 2), interpolation=cv2.INTER_AREA)
        if k_rot:
            ys = np.ascontiguousarray(np.rot90(ys, k_rot))
        hy = np.bincount((ys >> 4).ravel(), minlength=16).astype(np.float32)
        q = ((np.clip(us.astype(np.int16) - 64, 0, 127) >> 4) * 8 + (np.clip(vv.astype(np.int16) - 64, 0, 127) >> 4))
        huv = np.bincount(q.ravel(), minlength=64).astype(np.float32)
        hists.append(np.concatenate([hy / hy.sum(), huv / huv.sum()]))
        times.append(t)
        grays.append(ys)

    pend = None
    for n, f in enumerate(c.decode(vs)):
        t = f.time if f.time is not None else n / fps + t_base
        t -= t_base
        if all_t and t <= all_t[-1] + 1e-6:          # дубль/кривой pts
            continue
        all_t.append(t)
        # кадр показывается, только если до следующего кадра успевает наступить тик сетки fps_out
        if pend is not None and np.ceil(pend[0] * fps_out - 1e-6) < np.ceil(t * fps_out - 1e-6):
            take(*pend)
        pend = (t, f)
    if pend is not None:
        take(*pend)
    c.close()
    if not times:
        raise RuntimeError(f"в файле нет кадров: {path}")
    dt = np.diff(all_t)
    step = float(np.median(dt)) if len(dt) else 1.0 / fps
    disp = st["disp"]
    info = dict(w=int(disp[0]), h=int(disp[1]), fps_src=round(1.0 / step, 3) if step > 0 else fps,
                duration=round(all_t[-1] + step, 3), rotation=int(st["k_rot"] * 90), n_src=len(all_t),
                n_used=len(times))
    return np.array(times), grays, np.array(hists), info


def _content_box(grays):
    """Обрезка чёрных полос (леттербокс/пилларбокс). Возвращает y0, y1, x0, x1 в пикселях малого кадра."""
    sub = np.stack(grays[::max(1, len(grays) // 240)]).astype(np.float32)
    mx = np.percentile(sub, 99, axis=0)
    H, W = mx.shape
    row, col = np.percentile(mx, 95, axis=1), np.percentile(mx, 95, axis=0)

    def run(a):
        k = 0
        while k < len(a) and a[k] < 36:
            k += 1
        return k
    t, b, l, r = run(row), run(row[::-1]), run(col), run(col[::-1])
    t, b = [x if x >= 0.03 * H else 0 for x in (t, b)]
    l, r = [x if x >= 0.03 * W else 0 for x in (l, r)]
    t, b = min(t, int(0.35 * H)), min(b, int(0.35 * H))
    l, r = min(l, int(0.35 * W)), min(r, int(0.35 * W))
    return t, H - b, l, W - r


# ---------------------------------------------------------------- зрение
def _zncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else 1.0


def _iso_div(a1, b2):
    """Дивергенция, только если расширение изотропное (оба масштаба > 0 и похожи): наезд, а не параллакс панорамы."""
    if a1 <= 0 or b2 <= 0:
        return 0.0
    ratio = min(a1, b2) / max(a1, b2)
    return float((a1 + b2) * np.clip((ratio - 0.3) / 0.4, 0, 1))


def _loom_map(d):
    """1/с (скорость расширения площади) -> 0..1. Медленный наезд ~0.1/с — ноль, резкий ~3/с — максимум."""
    return float(np.clip((d - 0.45) / 1.6, 0, 1))


def _wls(X, w, u, v):
    A = X * w[:, None]
    M = X.T @ A + np.eye(3, dtype=np.float32) * 1e-6
    return np.linalg.solve(M, A.T @ u), np.linalg.solve(M, A.T @ v)


def _robust_affine(X, wt, u, v, iters=3):
    w = wt
    for _ in range(iters):
        pu, pv = _wls(X, w, u, v)
        r = np.hypot(u - X @ pu, v - X @ pv)
        s = 1.4826 * np.median(r) + 0.01
        w = wt / (1 + (r / (2 * s)) ** 2)
    return pu, pv


def _visual(times, grays, box, debug=None):
    """Признаки на каждый кадр источника (без сглаживания)."""
    y0, y1, x0, x1 = box
    n = len(grays)
    H, W = y1 - y0, x1 - x0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn, yn = (xx - (W - 1) / 2) / W, (yy - (H - 1) / 2) / W        # доли ширины, центр = 0
    m = 2
    sl = (slice(m, H - m, 2), slice(m, W - m, 2))
    Xg = np.stack([np.ones(xn[sl].size, np.float32), xn[sl].ravel(), yn[sl].ravel()], 1)
    gL = Xg[:, 1] < 0
    gR = ~gL
    border = np.zeros((H, W), bool)
    border[m:H - m, m:W - m] = True
    k3 = np.ones((3, 3), np.uint8)
    F = {k: np.zeros(n, np.float32) for k in
         ("light", "mc", "d_hist", "p2", "motion", "flowx", "divG", "divL", "divR", "locL", "locR",
          "objL", "objR", "objx", "blobL", "blobR")}
    blob_hist = []
    prev_u8 = prev_f = prev_z = None
    z32 = []
    for k in range(n):
        g8 = np.ascontiguousarray(grays[k][y0:y1, x0:x1])
        gf = np.clip((g8.astype(np.float32) - 16) / 219, 0, 1)
        F["light"][k] = gf.mean()
        z = cv2.resize(cv2.GaussianBlur(gf, (0, 0), 1.2), (32, max(8, int(round(32 * H / W)))),
                       interpolation=cv2.INTER_AREA)
        z32.append(z)
        if k == 0:
            prev_u8, prev_f = g8, gf
            continue
        dt = float(np.clip(times[k] - times[k - 1], 1 / 240, 0.25))
        fl = cv2.calcOpticalFlowFarneback(prev_u8, g8, None, 0.5, 3, 9, 3, 5, 1.1, 0)
        # сходство после компенсации движения: склейка — низкое, панорама/вспышка — высокое
        wp = cv2.remap(gf, xx + fl[..., 0], yy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ok = border & (xx + fl[..., 0] >= 0) & (xx + fl[..., 0] <= W - 1) & (yy + fl[..., 1] >= 0) & \
            (yy + fl[..., 1] <= H - 1)
        F["mc"][k] = 1 - _zncc(prev_f[ok], wp[ok]) if ok.sum() > 50 else 1.0
        u = fl[..., 0] * (1 / dt) / W                               # доли ширины в секунду
        v = fl[..., 1] * (1 / dt) / W
        mag = np.hypot(u, v)
        F["motion"][k] = float(mag[border].mean())
        # текстура: в гладких местах поток ненадёжен
        gx = cv2.Sobel(prev_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(prev_f, cv2.CV_32F, 0, 1, ksize=3)
        wt = np.clip(np.hypot(gx, gy)[sl].ravel() / 0.3, 0.02, 1)
        ug, vg = u[sl].ravel(), v[sl].ravel()
        pu, pv = _robust_affine(Xg, wt, ug, vg)
        F["flowx"][k] = pu[0]
        F["divG"][k] = _iso_div(pu[1], pv[2])
        for side, gm in (("divL", gL), ("divR", gR)):
            qu, qv = _robust_affine(Xg[gm], wt[gm], ug[gm], vg[gm], iters=2)
            F[side][k] = _iso_div(qu[1], qv[2])
        # остаток после вычета движения камеры -> самостоятельно движущиеся объекты
        mu, mv = pu[0] + pu[1] * xn + pu[2] * yn, pv[0] + pv[1] * xn + pv[2] * yn
        res = np.hypot(u - mu, v - mv)
        thr = np.maximum(max(0.8 / dt / W, 3.0 * float(np.median(res))), 0.35 * np.hypot(mu, mv))
        # кадр, предсказанный по движению камеры: где не совпало — там что-то движется само
        pred = cv2.remap(prev_f, xx - mu * W * dt, yy - mv * W * dt, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        dif = cv2.GaussianBlur(np.abs(gf - pred), (0, 0), 1.0)
        mov = ((res > thr) & (dif > 0.05) & border).astype(np.uint8)
        # расширение внутри движущейся области (человек идёт на камеру)
        rg = mov[sl].ravel().astype(bool)
        for side, gm in (("locL", gL), ("locR", gR)):
            sel = rg & gm
            frac = sel.sum() / max(1, gm.sum())
            if frac > 0.06:
                qu, qv = _wls(Xg[sel], wt[sel], ug[sel], vg[sel])
                F[side][k] = _iso_div(qu[1], qv[2]) * float(np.clip(frac / 0.25, 0, 1))
        nl, lab, st, cen = cv2.connectedComponentsWithStats(mov, connectivity=8)
        if debug is not None and k in debug:
            debug[k] = dict(mov=mov.copy(), res=res.copy(), dif=dif.copy(), thr=thr)
        if nl > 1:
            area = st[1:, cv2.CC_STAT_AREA].astype(np.float32) / (H * W)
            spd = np.bincount(lab.ravel(), weights=res.ravel(), minlength=nl)[1:] / np.maximum(
                st[1:, cv2.CC_STAT_AREA], 1)
            small = (area >= 0.0006) & (area <= 0.05)
            sal = np.where(small, np.sqrt(np.minimum(1, area / 0.01)) * np.clip(spd / 1.0, 0, 1), 0)
            cxs = cen[1:, 0] / W
            for side, sm in (("objL", cxs < 0.5), ("objR", cxs >= 0.5)):
                half = (xx < W / 2) if side == "objL" else (xx >= W / 2)
                clutter = float(mov[half].mean())
                F[side][k] = float(np.clip(sal[sm].sum(), 0, 1)) * float(np.clip((0.30 - clutter) / 0.20, 0, 1))
            if sal.max() > 0.05:
                F["objx"][k] = float(cxs[int(np.argmax(sal))] * 2 - 1)
        # надвигание-2: растёт контрастное тёмное пятно, края уходят наружу
        dcut = 0.5 * np.abs(prev_z - z).mean() if prev_z is not None else 0
        if F["mc"][k] > 0.5 or dcut > 0.12:
            blob_hist = []                                              # склейка — история пятна с нуля
        thr_d = min(0.25, max(0.05, float(np.median(gf)) - 0.12))
        dm = cv2.morphologyEx((gf < thr_d).astype(np.uint8), cv2.MORPH_OPEN, k3)
        nb, labb, stb, cenb = cv2.connectedComponentsWithStats(dm, connectivity=8)
        rec = None
        if nb > 1:
            ab = stb[1:, cv2.CC_STAT_AREA] / float(H * W)
            full = (stb[1:, cv2.CC_STAT_WIDTH] >= 0.95 * W) & (stb[1:, cv2.CC_STAT_HEIGHT] >= 0.95 * H)
            cand = np.where((ab >= 0.01) & (ab <= 0.6) & ~full)[0]
            if len(cand):
                j = cand[np.argmax(ab[cand])]
                rec = (times[k], float(ab[j]), cenb[j + 1][0] / W, cenb[j + 1][1] / W, j + 1)
        blob_hist.append(rec)
        blob_hist = [r for r in blob_hist if r is not None and times[k] - r[0] <= 0.3] if rec else []
        if rec and len(blob_hist) >= 4 and times[k] - blob_hist[0][0] >= 0.1:
            a_seq = [r[1] for r in blob_hist]
            drift = np.hypot(rec[2] - blob_hist[0][2], rec[3] - blob_hist[0][3])
            steady = all(b >= a * 0.97 for a, b in zip(a_seq, a_seq[1:]))
            rate = np.log(a_seq[-1] / a_seq[0]) / (times[k] - blob_hist[0][0])
            if steady and drift < 0.15 and rate > 0.3:
                msk = (labb == rec[4]).astype(np.uint8)
                ring = (cv2.dilate(msk, k3) - msk).astype(bool) & border
                if ring.sum() > 5:
                    cxp, cyp = rec[2] * W, rec[3] * W
                    rx, ry = xx[ring] - cxp, yy[ring] - cyp
                    rn = np.hypot(rx, ry) + 1e-6
                    outward = float(((u[ring] * rx + v[ring] * ry) / rn).mean())
                    expect = rate / 2 * np.sqrt(rec[1] * H / W / np.pi)
                    outside = ~msk.astype(bool) & border
                    fade = float(np.abs(gf[outside] - prev_f[outside]).mean()) if outside.any() else 1
                    if outward > 0.3 * expect and fade < 0.03:
                        val = _loom_map(rate)
                        sd = float(np.clip((rec[2] - 0.5) * 3 + 0.5, 0, 1))      # 0 — слева, 1 — справа
                        F["blobL"][k] = val * (1 - sd * 0.8)
                        F["blobR"][k] = val * (0.2 + sd * 0.8)
        prev_u8, prev_f, prev_z = g8, gf, z
    # склейки и «одно-кадровость»: похож ли кадр k+1 на k-1
    for k in range(1, n):
        a, b = z32[k - 1], z32[min(k + 1, n - 1)]
        F["p2"][k] = 1 - _zncc(a, b)
    return F


def _visual_job(job):
    times, grays, box, s0, s1 = job
    F = _visual(times, grays, box)
    return {k: v[s0:s1] for k, v in F.items()}


class _hide_main:
    """Дочерние процессы (spawn на Windows) не должны заново выполнять скрипт, который вызвал analyze()
    (там может грузиться мозг на GPU и т.п.). Прячем __main__.__file__/__spec__ на время запуска пула."""

    def __enter__(self):
        self.m = sys.modules.get("__main__")
        self.saved = {k: getattr(self.m, k) for k in ("__file__", "__spec__") if hasattr(self.m, k)}
        for k in self.saved:
            if k == "__spec__":
                self.m.__spec__ = None
            else:
                delattr(self.m, k)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.m, k, v)
        return False


def _visual_par(times, grays, box, workers):
    """То же, что _visual, но кусками в нескольких процессах (куски перекрываются на ~0.5 с)."""
    n = len(grays)
    if workers <= 1 or n < 400:
        return _visual(times, grays, box)
    fps = 1.0 / max(1e-3, float(np.median(np.diff(times))))
    ov = int(0.5 * fps) + 4
    bounds = np.linspace(0, n, workers + 1).astype(int)
    jobs = []
    for a, b in zip(bounds, bounds[1:]):
        lo, hi = max(0, a - ov), min(n, b + 1)
        jobs.append((times[lo:hi], grays[lo:hi], box, a - lo, b - lo))
    try:
        from concurrent.futures import ProcessPoolExecutor
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import analyze as me                                  # функция по имени модуля, а не __main__
        with _hide_main(), ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(me._visual_job, jobs))
    except Exception as e:                                   # нет процессов — считаем в одном
        print("[analyze] параллельно не вышло, считаю в одном процессе:", e)
        return _visual(times, grays, box)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _mfilt(x, w):
    if w <= 1:
        return x
    from scipy.ndimage import median_filter
    return median_filter(x, size=w, mode="nearest")


def _ema(x, times, tau):
    y = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        if i:
            acc += min(1.0, (times[i] - times[i - 1]) / tau) * (x[i] - acc)
        y[i] = acc
    return y


def _first_max(s, val):
    """Оставить val только в локальных максимумах s (у плато — первый кадр): импульс в один кадр."""
    lm = np.r_[True, s[1:] > s[:-1]] & np.r_[s[:-1] >= s[1:], True]
    return np.where(lm, val, 0).astype(np.float32)


def _visual_final(times, F, hists):
    """Склейки, вспышки, надвигание, объекты — финальные значения на кадр источника."""
    n = len(times)
    fps = 1.0 / max(1e-3, float(np.median(np.diff(times)))) if n > 1 else 30.0
    from scipy.ndimage import median_filter
    dh = np.zeros(n, np.float32)
    dh[1:] = 0.5 * np.abs(hists[1:] - hists[:-1]).sum(1) / 2         # 0..1
    F["d_hist"] = dh
    s = 0.5 * np.clip(F["mc"] / 0.8, 0, 1) + 0.5 * np.clip(dh / 0.4, 0, 1)
    wbase = max(5, int(fps * 1.0) | 1)
    base = median_filter(s, size=wbase, mode="nearest")
    cut = np.clip((s - base - 0.2) / 0.35, 0, 1) * np.clip((F["p2"] - 0.15) / 0.35, 0, 1)
    cut = _first_max(s, cut)
    cut[0] = 0
    # вспышки: скачок яркости вверх (относительно тёмного фона — сильнее)
    L = F["light"]
    prevL = np.minimum(np.r_[L[0], L[:-1]], np.r_[L[0], L[0], L[:-2]])
    jump = L - prevL
    rel = jump / (prevL + 0.12)
    flash = np.clip((rel - 0.25) / 0.75, 0, 1) * (jump > 0.05)
    lag = max(2, int(0.5 * fps))
    later = np.r_[L[lag:], np.repeat(L[-1], lag)]
    stays = later >= L - 0.5 * np.maximum(jump, 0)
    flash = _first_max(flash, flash)
    cut_near = np.maximum.reduce([cut, np.r_[0, cut[:-1]], np.r_[0, 0, cut[:-2]]])
    flash = np.where((cut_near > 0.3) & stays, flash * 0.2, flash).astype(np.float32)   # склейка на светлый план
    # ненадёжный поток: склейка, вспышка, «каша»
    bad = (cut > 0.15) | (flash > 0.15) | (F["mc"] > 0.6)
    bad = bad | np.r_[False, bad[:-1]]
    w = 5 if fps <= 35 else 9                                       # надвигание должно длиться >= ~0.15 с
    out = {}
    divG = np.where(bad, 0, _mfilt(np.where(bad, 0, F["divG"]), w))
    zoom = np.array([_loom_map(d) for d in divG], np.float32)
    side = {}
    for S in ("L", "R"):
        d_half = np.array([_loom_map(d) for d in _mfilt(np.where(bad, 0, F["div" + S]), w)])
        d_loc = np.array([_loom_map(d) for d in _mfilt(np.where(bad, 0, F["loc" + S]), w)])
        blob = _mfilt(np.where(bad, 0, F["blob" + S]), w)
        side[S] = np.maximum.reduce([zoom, 0.8 * d_half, 0.6 * d_loc, blob]) * ~bad
        out["loom_" + S] = _ema(side[S].astype(np.float32), times, 0.06)
        ob = np.where(bad, 0, F["obj" + S])
        ob = np.minimum(ob, np.maximum(np.r_[0, ob[:-1]], np.r_[0, 0, ob[:-2]]))   # объект держится >= 2 кадров
        out["obj_" + S] = _ema(ob.astype(np.float32), times, 0.06)
    out["zoom"] = _ema(zoom * ~bad, times, 0.06)
    out["obj_x"] = np.where(bad, 0, F["objx"]).astype(np.float32)
    # непрерывные — держим прошлое значение на плохих кадрах
    mot, fx = F["motion"].copy(), F["flowx"].copy()
    for k in range(1, n):
        if bad[k]:
            mot[k], fx[k] = mot[k - 1], fx[k - 1]
    out["flow_x"] = np.tanh(_ema(fx, times, 0.1) / 0.35).astype(np.float32)
    mot = _ema(mot, times, 0.1)
    out["motion"] = np.clip(mot / max(0.35, float(np.percentile(mot, 98))), 0, 1).astype(np.float32)
    out["cut"], out["flash"] = cut.astype(np.float32), flash.astype(np.float32)
    out["light"] = F["light"].astype(np.float32)
    return out


def _resample(times, vals, n_out, fps_out, impulse=False):
    """Кадр источника k виден с кадра вывода ceil(t_k * fps_out). Импульсы — max, непрерывные — среднее + удержание."""
    ib = np.ceil(times * fps_out - 1e-6).astype(int)
    keep = (ib >= 0) & (ib < n_out)
    ib, v = ib[keep], vals[keep].astype(np.float64)
    if impulse:
        out = np.zeros(n_out)
        np.maximum.at(out, ib, v)
        return out.astype(np.float32)
    s, c = np.zeros(n_out), np.zeros(n_out)
    np.add.at(s, ib, v)
    np.add.at(c, ib, 1)
    out = np.full(n_out, np.nan)
    out[c > 0] = s[c > 0] / c[c > 0]
    last = v[0] if len(v) else 0.0
    for i in range(n_out):
        if np.isnan(out[i]):
            out[i] = last
        last = out[i]
    return out.astype(np.float32)


# ---------------------------------------------------------------- звук
def _read_audio(path):
    c = av.open(str(path))
    if not c.streams.audio:
        c.close()
        return None, 0.0
    ast = c.streams.audio[0]
    t_base = (c.start_time or 0) / 1e6
    res = av.AudioResampler(format="fltp", layout="mono", rate=SR)
    chunks, a0 = [], None
    for fr in c.decode(ast):
        if a0 is None and fr.time is not None:
            a0 = fr.time - t_base
        for o in res.resample(fr):
            chunks.append(o.to_ndarray()[0])
    for o in res.resample(None):
        chunks.append(o.to_ndarray()[0])
    c.close()
    if not chunks:
        return None, 0.0
    return np.concatenate(chunks).astype(np.float32), float(a0 or 0.0)


def _frames_at(x, centers, nfft):
    xp = np.pad(x, (nfft, nfft))
    idx = np.clip(centers[:, None] + nfft - nfft // 2 + np.arange(nfft)[None, :], 0, len(xp) - 1)
    return xp[idx]


def _db_norm(e_db, lo_pct=5, hi_pct=99.5, span=36):
    hi = np.percentile(e_db, hi_pct)
    lo = max(np.percentile(e_db, lo_pct), hi - span)
    return np.clip((e_db - lo) / max(1e-6, hi - lo), 0, 1).astype(np.float32)


def _band_filters(freqs, n=48, fmin=40.0, fmax=10000.0):
    edges = np.geomspace(fmin, fmax, n + 2)
    fb = np.zeros((n, len(freqs)), np.float32)
    for i in range(n):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb


def _beat_track(env, fr, tightness=100.0):
    """Трекинг долей (динамическое программирование, Ellis 2007). env — сила атак с частотой fr."""
    n = len(env)
    if n < fr * 4 or env.std() < 1e-9:
        return np.array([], int), 0.0
    e = env / env.std()
    maxlag = int(fr * 60 / 50) + 2
    f = np.fft.rfft(e - e.mean(), 2 * n)
    ac = np.fft.irfft(np.abs(f) ** 2)[:maxlag]
    lags = np.arange(maxlag, dtype=float)
    lags[0] = 1e-9
    bpm = 60 * fr / lags
    prior = np.exp(-0.5 * (np.log2(bpm / 110.0) / 0.9) ** 2)
    sc = np.where((bpm >= 60) & (bpm <= 180), ac * prior, -np.inf)
    L = int(np.argmax(sc))
    if 1 <= L < maxlag - 1 and np.isfinite(sc[L - 1]) and np.isfinite(sc[L + 1]):
        a, b, c = ac[L - 1], ac[L], ac[L + 1]
        den = a - 2 * b + c
        period = L + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)
    else:
        period = float(L)
    tempo = 60 * fr / period
    sig = period / 32
    k = np.arange(-int(4 * sig) - 1, int(4 * sig) + 2)
    local = np.convolve(e, np.exp(-0.5 * (k / sig) ** 2), "same")
    win = np.arange(-int(round(2 * period)), -int(round(period / 2)) + 1)
    txwt = -tightness * np.log(-win / period) ** 2
    cum = np.zeros(n)
    back = -np.ones(n, int)
    for i in range(n):
        z = i + win
        ok = z >= 0
        if ok.any():
            cand = cum[z[ok]] + txwt[ok]
            j = int(np.argmax(cand))
            cum[i] = local[i] + cand[j]
            back[i] = z[ok][j]
        else:
            cum[i] = local[i]
    mx = np.r_[False, (cum[1:-1] > cum[:-2]) & (cum[1:-1] >= cum[2:]), False]
    if not mx.any():
        return np.array([], int), tempo
    med = np.median(cum[mx])
    last = int(np.nonzero(mx & (cum >= 0.5 * med))[0][-1])
    beats = [last]
    while back[beats[-1]] >= 0:
        beats.append(int(back[beats[-1]]))
    beats = np.array(beats[::-1])
    thr = 0.5 * np.sqrt(np.mean(local[beats] ** 2))
    good = np.nonzero(local[beats] >= thr)[0]
    if len(good):
        beats = beats[good[0]:good[-1] + 1]
    return beats, tempo


def _audio(path, n_out, fps_out):
    x, a0 = _read_audio(path)
    z = {k: np.zeros(n_out, np.float32) for k in ("loud", "bass", "hear", "onset", "beat", "drop")}
    if x is None or len(x) < SR:
        return z, dict(has_audio=False, tempo_bpm=0.0, audio_start=0.0)
    t_out = np.arange(n_out) / fps_out
    nfft = 2048
    centers = np.round((t_out - a0) * SR).astype(int)
    fr = _frames_at(x, centers, nfft)
    rms_db = 10 * np.log10((fr ** 2).mean(1) + 1e-10)
    S = np.abs(np.fft.rfft(fr * np.hanning(nfft).astype(np.float32), axis=1)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    bass_db = 10 * np.log10(S[:, (freqs >= 40) & (freqs <= 160)].sum(1) + 1e-8)
    # слух мухи: 100-500 Гц, вес-«колокол» с плато 150-250 Гц
    wh = np.clip(np.minimum((freqs - 100) / 50, (500 - freqs) / 250), 0, 1)
    hear_db = 10 * np.log10((S * wh[None, :]).sum(1) + 1e-8)
    z["loud"], z["bass"], z["hear"] = _db_norm(rms_db), _db_norm(bass_db), _db_norm(hear_db)
    # атаки: спектральный поток на мелкой сетке (hop 512 ~ 23 мс)
    hop, nf2 = 512, 1024
    nfine = int(len(x) / hop) + 1
    fr2 = _frames_at(x, np.arange(nfine) * hop, nf2)
    S2 = np.abs(np.fft.rfft(fr2 * np.hanning(nf2).astype(np.float32), axis=1))
    fb = _band_filters(np.fft.rfftfreq(nf2, 1 / SR))
    Lg = np.log1p(100 * (S2 @ fb.T))
    flux = np.r_[0, np.maximum(0, np.diff(Lg, axis=0)).sum(1)]
    frate = SR / hop
    from scipy.ndimage import uniform_filter1d
    env = np.maximum(0, flux - uniform_filter1d(flux, int(frate * 0.5) | 1, mode="nearest"))
    t_fine = a0 + np.arange(nfine) * hop / SR
    beats, tempo = _beat_track(env, frate)
    idx = np.clip(np.round(t_fine * fps_out).astype(int), 0, n_out - 1)
    on = np.zeros(n_out)
    np.maximum.at(on, idx, env)
    z["onset"] = np.clip(on / (np.percentile(env[env > 0], 99) if (env > 0).any() else 1), 0, 1).astype(np.float32)
    for b in beats:
        i = int(round(t_fine[b] * fps_out))
        if 0 <= i < n_out and z["loud"][i] > 0.15:
            z["beat"][i] = 1.0
    # дроп: энергия (бас + громкость) за 0.3 с после минус энергия за 3 с до
    e = 0.5 * rms_db + 0.5 * (bass_db - 10)
    cs = np.r_[0, np.cumsum(e)]

    def win_mean(i0, i1):
        i0, i1 = np.clip(i0, 0, n_out), np.clip(i1, 0, n_out)
        return np.where(i1 > i0, (cs[i1] - cs[i0]) / np.maximum(1, i1 - i0), np.nan)
    i = np.arange(n_out)
    after = win_mean(i, i + int(0.3 * fps_out))
    before = win_mean(i - int(3 * fps_out), i - int(0.25 * fps_out))
    enough = (i - int(0.25 * fps_out)) >= int(0.6 * fps_out)
    jump = np.where(enough, np.nan_to_num(after - before), 0)
    z["drop"] = np.clip((jump - 4) / 8, 0, 1).astype(np.float32)
    return z, dict(has_audio=True, tempo_bpm=round(float(tempo), 1), audio_start=round(a0, 4))


# ---------------------------------------------------------------- события
def _peaks(x, thr, min_dist):
    """Локальные максимумы >= thr с подавлением соседей ближе min_dist кадров (сильные — первыми)."""
    x = np.asarray(x, float)
    if len(x) < 3:
        return []
    lm = np.r_[x[0] > x[1], (x[1:-1] >= x[:-2]) & (x[1:-1] >= x[2:]), x[-1] > x[-2]] & (x >= thr)
    idx = np.nonzero(lm)[0]
    idx = idx[np.argsort(-x[idx], kind="stable")]
    keep = []
    for i in idx:
        if all(abs(i - j) >= min_dist for j in keep):
            keep.append(int(i))
    return sorted(keep)


def _caption(kind, strength, side=None, info=None):
    if kind == "drop":
        return "бас-дроп!" if strength >= 0.55 else "музыка набирает"
    if kind == "peak":
        if info == "bass":
            return "мощный удар баса"
        if info == "visual":
            return "резкое движение"
        return "острый момент"
    if kind == "cut":
        return "смена кадра"
    if kind == "flash":
        return "вспышка!" if strength >= 0.5 else "стало светлее"
    if kind == "zoom":
        if side == "both":
            return "резкий наезд камеры"
        return "надвигается слева" if side == "left" else "надвигается справа"
    return kind


def _events(tl, fps_out):
    ev = []
    n = len(tl["t"])
    onset = tl["onset"]

    def snap(i, rad):                      # к сильнейшей атаке рядом
        a, b = max(0, i - rad), min(n, i + rad + 1)
        return a + int(np.argmax(onset[a:b])) if b > a else i
    for i in _peaks(tl["drop"], 0.25, int(6 * fps_out)):
        j = snap(i, int(0.3 * fps_out))
        ev.append(dict(frame=j, kind="drop", strength=float(tl["drop"][i])))
    ex, loc = tl["excite"], tl["_excite_local"]
    pk = np.clip(0.5 * ex + 1.5 * (ex - loc), 0, 1)
    for i in _peaks(pk, 0.4, int(2 * fps_out)):
        aud = 0.45 * onset[i] + 0.3 * tl["loud"][i] + 0.25 * tl["bass"][i]
        vis = max(tl["motion"][i] * 0.7, tl["flash"][i], max(tl["loom_L"][i], tl["loom_R"][i]))
        info = "bass" if tl["bass"][i] > 0.85 and onset[i] > 0.5 else ("visual" if vis > aud else None)
        ev.append(dict(frame=i, kind="peak", strength=float(pk[i]), info=info))
    for i in _peaks(tl["cut"], 0.4, int(0.25 * fps_out)):
        ev.append(dict(frame=i, kind="cut", strength=float(tl["cut"][i])))
    for i in _peaks(tl["flash"], 0.3, int(0.4 * fps_out)):
        ev.append(dict(frame=i, kind="flash", strength=float(tl["flash"][i])))
    lm = np.maximum(tl["loom_L"], tl["loom_R"])
    for i in _peaks(lm, 0.3, int(1.5 * fps_out)):
        L, R, Z = tl["loom_L"][i], tl["loom_R"][i], tl["zoom"][i]
        side = "both" if Z >= 0.7 * lm[i] or min(L, R) >= 0.7 * max(L, R) else ("left" if L > R else "right")
        ev.append(dict(frame=i, kind="zoom", strength=float(lm[i]), side=side))
    for e in ev:
        e["t"] = round(e["frame"] / fps_out, 3)
        e["strength"] = round(min(1.0, e["strength"]), 3)
        e["caption"] = _caption(e["kind"], e["strength"], e.get("side"), e.pop("info", None))
    ev.sort(key=lambda e: (e["t"], EVENT_KINDS.index(e["kind"])))
    return ev


# ---------------------------------------------------------------- главное
def _cache_path(video_path):
    return CACHE_DIR / (Path(video_path).stem + ".json")


def _to_numpy(tl):
    for k in FRAME_KEYS:
        tl[k] = np.asarray(tl[k], np.float32)
    return tl


def analyze(video_path, fps_out=30, cache=True, force=False, verbose=True, workers=None):
    """Разбор ролика -> dict (см. описание модуля). Кэш — tiktok/_cache/<имя>.json.
    cache=False — не читать и не писать кэш; force=True — пересчитать и перезаписать кэш;
    workers — процессов для зрения (по умолчанию половина ядер, не больше 8; 1 — без процессов)."""
    video_path = Path(video_path)
    st = video_path.stat()
    import hashlib
    code = hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:10]     # поменялся анализ — кэш пересчитается
    sig = dict(size=st.st_size, mtime_ns=st.st_mtime_ns, fps_out=fps_out, version=VERSION, code=code)
    cp = _cache_path(video_path)
    if cache and not force and cp.exists():
        try:
            tl = json.loads(cp.read_text(encoding="utf-8"))
            if tl.get("meta", {}).get("sig") == sig:
                return _to_numpy(tl)
        except Exception:
            pass
    T0 = time.time()
    # звук параллельно с видео (декодеры отпускают GIL)
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(1)
    times, grays, hists, info = _read_video(video_path, fps_out)
    t_read = time.time() - T0
    n_out = max(1, int(round(info["duration"] * fps_out)))
    fut = ex.submit(_audio, video_path, n_out, fps_out)
    box = _content_box(grays)
    if workers is None:
        import os
        workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    F = _visual_par(times, grays, box, workers)
    V = _visual_final(times, F, hists)
    t_vis = time.time() - T0 - t_read
    A, ainfo = fut.result()
    ex.shutdown()
    tl = {"t": (np.arange(n_out) / fps_out).astype(np.float32)}
    for k, v in V.items():
        tl[k] = _resample(times, v, n_out, fps_out, impulse=k in ("cut", "flash"))
    tl.update(A)
    # «острота момента»: звук + картинка
    aud = 0.45 * tl["onset"] + 0.30 * tl["loud"] + 0.25 * tl["bass"]
    vis = np.maximum.reduce([0.7 * tl["motion"], tl["flash"], np.maximum(tl["loom_L"], tl["loom_R"]),
                             0.5 * tl["cut"], tl["drop"]])
    near_beat = np.convolve(tl["beat"], np.ones(3), "same") > 0
    exc = np.clip(0.6 * aud + 0.4 * vis + 0.1 * near_beat, 0, 1).astype(np.float32)
    from scipy.ndimage import uniform_filter1d
    tl["excite"] = exc
    tl["_excite_local"] = uniform_filter1d(exc, int(6 * fps_out) | 1, mode="nearest")
    events = _events(tl, fps_out)
    del tl["_excite_local"]
    H, W = grays[0].shape
    y0, y1, x0, x1 = box
    # ориентация самой картинки без чёрных полос (альбомный клип, вставленный в вертикальный кадр, — 'landscape')
    cw, ch = (x1 - x0) / W * info["w"], (y1 - y0) / H * info["h"]
    meta = dict(source=str(video_path), w=info["w"], h=info["h"], fps_src=info["fps_src"], fps_out=fps_out,
                duration=info["duration"], orientation="portrait" if info["h"] > info["w"] else "landscape",
                content_orientation="portrait" if ch > cw else "landscape", content_aspect=round(cw / ch, 4),
                n_frames_out=n_out, rotation=info["rotation"], n_frames_src=info["n_src"],
                n_frames_used=info["n_used"],
                content_box=[round(x0 / W, 4), round(y0 / H, 4), round(x1 / W, 4), round(y1 / H, 4)],
                analysis_px=[W, H], **ainfo,
                counts={k: sum(e["kind"] == k for e in events) for k in EVENT_KINDS} | {
                    "beat": int(tl["beat"].sum())},
                runtime_s=round(time.time() - T0, 2), sig=sig, keys=FRAME_KEYS)
    # округляем так же, как в кэше: свежий результат и прочитанный из кэша совпадают бит в бит
    # (иначе план стимулов в проходе 1 и в проходе 2 мог бы разойтись на пороговых значениях)
    lists = {k: [round(float(x), 3) for x in tl[k]] for k in FRAME_KEYS}
    out = {"meta": meta}
    out.update({k: np.asarray(lists[k], np.float32) for k in FRAME_KEYS})
    out["events"] = events
    if verbose:
        print(f"[analyze] {video_path.name}: {info['w']}x{info['h']} {meta['orientation']}, "
              f"{info['duration']:.1f} с, кадров вывода {n_out}; чтение {t_read:.1f} с, зрение {t_vis:.1f} с, "
              f"всего {meta['runtime_s']:.1f} с")
    if cache:
        import os
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        js = {"meta": meta, **lists, "events": events}
        tmp = cp.with_name(cp.name + f".{os.getpid()}.tmp")         # атомарно: параллельный читатель не увидит полфайла
        tmp.write_text(json.dumps(js, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        try:
            os.replace(tmp, cp)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
    return out


def load(video_path):
    """Прочитать кэш без пересчёта (None — если нет или файл битый). Подпись (размер/время файла) не проверяется."""
    cp = _cache_path(video_path)
    if not cp.exists():
        return None
    try:
        return _to_numpy(json.loads(cp.read_text(encoding="utf-8")))
    except (ValueError, KeyError, OSError):
        return None


# ---------------------------------------------------------------- кадры для рендера
def iter_frames(video_path, fps_out=30, size=None, start=0, stop=None):
    """Кадры ролика на сетке fps_out: yield (i, rgb uint8 HxWx3), с учётом поворота.
    Кадр i = последний кадр источника с временем <= i / fps_out (как в analyze()).
    size=(w, h) — размер после поворота (None — как есть). start/stop — номера кадров вывода
    (для параллельного рендера кусками; stop=meta['n_frames_out'] — ровно по шкале, хвост добивается
    последним кадром). Один и тот же numpy-массив может прийти несколько раз подряд (повтор кадра)."""
    c = av.open(str(video_path))
    try:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        t_base = (c.start_time or 0) / 1e6
        if stop is None:
            dur = float(vs.duration * vs.time_base) if vs.duration else (c.duration or 0) / 1e6
            stop = int(round(dur * fps_out)) if dur > 0 else None       # None — до конца ролика
        if start > 0:
            c.seek(int(max(0.0, start / fps_out - 1.0) / vs.time_base) + (vs.start_time or 0), stream=vs,
                   backward=True)

        def conv(f):
            k = _rot_k(f)
            if size is not None:
                w, h = (size[1], size[0]) if k % 2 else size
                im = f.reformat(width=int(w), height=int(h), format="rgb24", interpolation="AREA").to_ndarray()
            else:
                im = f.to_ndarray(format="rgb24")
            return np.ascontiguousarray(np.rot90(im, k)) if k else im

        i, cur, img = start, None, None
        for f in c.decode(vs):
            if f.time is None:
                continue
            t = f.time - t_base
            while (stop is None or i < stop) and t > i / fps_out + 1e-6 and cur is not None:
                if img is None:
                    img = conv(cur)                 # переводим в RGB только то, что реально показываем
                yield i, img
                i += 1
            if stop is not None and i >= stop:
                return
            cur, img = f, None
        if cur is not None:
            img = conv(cur) if img is None else img
            end = stop if stop is not None else i + 1
            while i < end:
                yield i, img
                i += 1
    finally:
        c.close()


# ---------------------------------------------------------------- отладочный график
def plot_debug(tl, plan=None, out_png=None, t_range=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = tl["t"]
    sel = slice(None) if t_range is None else (t >= t_range[0]) & (t <= t_range[1])
    fig, ax = plt.subplots(6, 1, figsize=(26, 16), dpi=72, sharex=True)
    ink, grid = "#333", "#ddd"
    rows = [
        ("звук", [("loud", "#1f6fd1"), ("bass", "#d1561f"), ("hear", "#2a9d57")]),
        ("атаки / доли / дроп", [("onset", "#7a4fd1"), ("drop", "#d11f5b")]),
        ("зрение: склейки, вспышки, движение", [("cut", "#555"), ("flash", "#e0b000"), ("motion", "#1f9dd1")]),
        ("надвигание (LPLC2/LC4)", [("loom_L", "#d1561f"), ("loom_R", "#1f6fd1"), ("zoom", "#999")]),
        ("мелкие объекты (LC10) и поток", [("obj_L", "#d1561f"), ("obj_R", "#1f6fd1"), ("flow_x", "#2a9d57")]),
        ("острота момента + события + стимулы", [("excite", "#444")]),
    ]
    for a, (title, series) in zip(ax, rows):
        for key, col in series:
            a.plot(t[sel], tl[key][sel], color=col, lw=1.0, label=key)
        a.set_title(title, loc="left", fontsize=12, color=ink)
        a.grid(True, color=grid, lw=0.5)
        a.legend(loc="upper right", fontsize=9, ncol=len(series), frameon=False)
        a.spines[["top", "right"]].set_visible(False)
    bt = t[tl["beat"] > 0]
    if t_range is not None:
        bt = bt[(bt >= t_range[0]) & (bt <= t_range[1])]
    ax[1].vlines(bt, 0, 0.15, color="#222", lw=0.8)
    cols = dict(drop="#d11f5b", peak="#7a4fd1", cut="#999", flash="#e0b000", zoom="#1f6fd1")
    ys = dict(drop=0.95, peak=0.85, cut=0.1, flash=0.75, zoom=0.65)
    for e in tl["events"]:
        if t_range is None or t_range[0] <= e["t"] <= t_range[1]:
            ax[5].plot([e["t"]], [ys[e["kind"]]], "v", color=cols[e["kind"]], ms=6 if e["kind"] != "cut" else 3)
    for p in plan or []:
        if t_range is None or t_range[0] <= p["t"] <= t_range[1]:
            c = "#f08c00" if p["type"] == "sugar" else "#00a3a3"
            ax[5].axvspan(p["t"], p["t"] + p["dur"], color=c, alpha=0.3)
            ax[5].text(p["t"], 1.02, "сахар" if p["type"] == "sugar" else "обдув", color=c, fontsize=9)
    ax[5].set_ylim(0, 1.12)
    ax[-1].set_xlabel("время, с")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    if out_png:
        import imageio.v3 as iio
        iio.imwrite(str(out_png), img)
    return img


def summary(tl, plan=None):
    m = tl["meta"]
    lines = [f"{Path(m['source']).name}: {m['w']}x{m['h']} {m['orientation']}, {m['duration']:.1f} с, "
             f"fps {m['fps_src']} -> {m['fps_out']}, кадров {m['n_frames_out']}, темп {m.get('tempo_bpm', 0)} уд/мин",
             "события: " + ", ".join(f"{k} {v}" for k, v in m["counts"].items())]
    big = [e for e in tl["events"] if e["kind"] in ("drop", "zoom", "flash") or
           (e["kind"] == "peak" and e["strength"] >= 0.6)]
    lines.append("главные: " + "; ".join(f"{e['t']:.1f}с {e['caption']} ({e['strength']:.2f})" for e in big[:40]))
    if plan is not None:
        lines.append(f"стимулы ({len(plan)}):")
        for p in plan:
            lines.append(f"  {p['t']:6.2f} с  {p['type']:5s} {p['dur']:.1f} с  сила {p['strength']:.2f}  — {p['reason']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--force", action="store_true", help="пересчитать и перезаписать кэш")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    tl = analyze(args.video, force=args.force)
    sys.path.insert(0, str(HERE))
    from stimuli import plan_stimuli
    plan = plan_stimuli(tl)
    print(summary(tl, plan))
    if args.plot:
        plot_debug(tl, plan, args.plot)
        print("график:", args.plot)
