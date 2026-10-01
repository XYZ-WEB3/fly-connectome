"""Проход 2: параллельный рендер вертикального ролика из лога состояний.

    from render_parallel import run_pass2
    from stage import Stage
    stats = run_pass2("tiktok/_cache/run1_log", "out.mp4", SRC_MP4, Stage, n_workers=8)
    # форматы и превью окна (FORMATS.md): fmt=formats.fmt_cfg("streamer", 123, flyeye=True), start=кадр, frames=N

Как устроено (и почему так):
  * главный процесс ОДИН раз последовательно декодирует исходный ролик (PyAV, без перемоток:
    у клипа ключевые кадры раз в 6 с, перемотка в каждом воркере стоила бы ~150 кадров декода),
    уменьшает кадр до 960 по длинной стороне (= текстура экрана Stage) и кладёт в кольцо общей
    памяти; воркер читает кадр экрана по номеру слота — без копий через pipe;
  * N воркеров (spawn): у каждого свой Stage (свой контекст MuJoCo/OpenGL) и свой Composer;
    берут номера кадров из очереди, рендерят 3D (1080x1150), собирают кадр 1080x1920 прямо
    в слот выходного кольца общей памяти;
  * главный процесс отдаёт готовые кадры строго по порядку в ОДИН ffmpeg (stdin rawvideo rgb24),
    ffmpeg кодирует (h264_nvenc, если есть, иначе libx264) и сразу кладёт ОРИГИНАЛЬНЫЙ звук
    (обрезан по длине видео) — отдельного шага склейки нет.
  Окно отправки = размер кольца: кадр i+R отправляется только после записи кадра i.

Лог состояний (пишет проход 1 через StateLogWriter) — папка:
  meta.json      {"fps":30, "n_frames":N, "title":..., "fly_no":..., "source": путь к ролику,
                  "src_t0": 0.0 (с какой секунды ролика начинается видео),
                  "gauge_max": {"hearing_hz": 14, ...} (необязательно: максимумы шкал), ...}
  frames.pkl     список по кадрам: {"stage": Stage.get_state(), "hud": dict для compose —
                  ключи см. compose.py (fear_hz, interest, hearing_hz, hunger, steer, gf_hz, mn9_hz,
                  groom_hz, flying/takeoff/startle/proboscis/eat_start/grooming/groom_start, event, ...)}
  act_idx.npy    int32  — индексы горящих нейронов, все кадры подряд
  act_val.npy    float16 — их затухшая активность (act > 0.05)
  act_off.npy    int64 (N+1) — кадр k = [off[k], off[k+1])
"""
import inspect
import json
import multiprocessing as mp
import os
import pickle
import queue
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRATCH = HERE / "_scratch"
DEFAULT_SRC = ROOT / "видео на реакцию" / "VILLIAN,_madk1d_ДИНАСТИЯ_2_ft_Андрей_Гайдулян_Клип_2026.mp4"
ACT_THR = 0.05
ACT_DECAY = 0.55


# ====================================================================== лог состояний
class StateLogWriter:
    """Проход 1 пишет кадр за кадром:
        w = StateLogWriter(path, meta=dict(fps=30, title=..., fly_no=..., source=SRC, src_t0=0.0))
        w.add(stage=stage.get_state(), hud=dict(fear_hz=..., ...), hz=hz)   # hz — Гц на нейрон (numpy N)
        ...
        w.close()
    Активность: либо hz= (писатель сам ведёт затухание act = act*0.55 + clip(hz/60, 0, 3), как
    hud.BrainCloud), либо act= (готовый плотный вектор), либо act_idx=/act_val= (уже разреженно).
    В лог попадают только act > 0.05."""

    def __init__(self, path, meta=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta or {})
        self.frames = []
        self.off = [0]
        self._fi = open(self.path / "act_idx.raw", "wb")
        self._fv = open(self.path / "act_val.raw", "wb")
        self._act = None

    def add(self, stage=None, hud=None, hz=None, act=None, act_idx=None, act_val=None):
        if hz is not None:
            hz = np.asarray(hz, np.float32)
            if self._act is None:
                self._act = np.zeros_like(hz)
            self._act *= ACT_DECAY
            self._act += np.clip(hz / 60.0, 0, 3)
            act = self._act
        if act is not None:
            act = np.asarray(act, np.float32)
            act_idx = np.nonzero(act > ACT_THR)[0]
            act_val = act[act_idx]
        if act_idx is None:
            act_idx, act_val = np.zeros(0, np.int32), np.zeros(0, np.float16)
        ai = np.asarray(act_idx, np.int32)
        av = np.asarray(act_val, np.float16)
        self._fi.write(ai.tobytes())
        self._fv.write(av.tobytes())
        self.off.append(self.off[-1] + len(ai))
        self.frames.append(dict(stage=stage, hud=dict(hud or {})))

    def close(self):
        self._fi.close()
        self._fv.close()
        n = self.off[-1]
        for name, dt in (("act_idx", np.int32), ("act_val", np.float16)):
            raw = self.path / f"{name}.raw"
            a = np.fromfile(raw, dtype=dt, count=n) if n else np.zeros(0, dt)
            np.save(self.path / f"{name}.npy", a)
            raw.unlink()
        np.save(self.path / "act_off.npy", np.array(self.off, np.int64))
        with open(self.path / "frames.pkl", "wb") as f:
            pickle.dump(self.frames, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.meta.setdefault("fps", 30)
        self.meta["n_frames"] = len(self.frames)
        (self.path / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=1, default=str),
                                             encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class StateLog:
    """Чтение лога (активность — через memmap, воркеры не держат копии)."""

    def __init__(self, path):
        p = Path(path)
        if p.is_file():
            p = p.parent
        self.path = p
        self.meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        with open(p / "frames.pkl", "rb") as f:
            self.frames = pickle.load(f)
        self.off = np.load(p / "act_off.npy")
        n = int(self.off[-1])
        self.idx = np.load(p / "act_idx.npy", mmap_mode="r") if n else np.zeros(0, np.int32)
        self.val = np.load(p / "act_val.npy", mmap_mode="r") if n else np.zeros(0, np.float16)
        self.n = len(self.frames)
        self.fps = float(self.meta.get("fps", 30))

    def act(self, i):
        a, b = int(self.off[i]), int(self.off[i + 1])
        return np.asarray(self.idx[a:b]), np.asarray(self.val[a:b], np.float32)


def load_state_log(path):
    return StateLog(path)


# ====================================================================== кадры экрана
class ScreenDecoder:
    """Последовательный декод исходного ролика в главном процессе: frame_at(t) -> кадр yuv420p
    (массив (h*3/2, w) uint8), показанный в момент t (секунды от src_t0). После конца — последний
    кадр. В RGB (BT.709/601, уменьшение до 960) переводят воркеры — параллельно (yuv420_to_rgb):
    так главный процесс тратит ~0.6 мс на кадр вместо ~4.5 мс (swscale в RGB)."""

    def __init__(self, path, max_side=960, t0=0.0):
        import av
        self.c = av.open(str(path))
        self.vs = self.c.streams.video[0]
        self.vs.thread_type = "AUTO"
        self.tb = float(self.vs.time_base)
        self.start = float(self.vs.start_time * self.tb) if self.vs.start_time is not None else 0.0
        w, h = self.vs.codec_context.width, self.vs.codec_context.height
        s = min(1.0, 2 * max_side / max(w, h))            # 4K и т.п. уменьшаем ещё при декоде
        self.w, self.h = max(4, int(w * s) // 4 * 4), max(4, int(h * s) // 4 * 4)   # кратно 4: у 706 плоскость цвета 353 — не собирается
        self.native = (self.w, self.h) == (w, h)
        self.shape = (self.h * 3 // 2, self.w)
        self.max_side = max_side
        self.t0 = float(t0)
        if self.t0 > 0:
            self.c.seek(int((self.t0 + self.start) / self.tb), stream=self.vs, backward=True)
        self.it = self.c.decode(self.vs)
        self.cur = None          # (время, av.Frame)
        self.nxt = self._next()
        f0 = self.nxt[1] if self.nxt else None
        self.colorspace = int(getattr(f0, "colorspace", 0) or 0)     # 1 = BT.709, 5/6 = BT.601
        self.color_range = int(getattr(f0, "color_range", 0) or 0)   # 2 = полный (JPEG)
        try:
            self.rotation = int(round(float(getattr(f0, "rotation", 0) or 0)))
        except (TypeError, ValueError):
            self.rotation = 0
        self.portrait = (h > w) != (abs(self.rotation) % 180 == 90)
        self.cur_yuv = None
        self.cur_key = None

    def info(self):
        """Что нужно воркеру, чтобы перевести кадр в RGB."""
        return dict(w=self.w, h=self.h, max_side=self.max_side, colorspace=self.colorspace,
                    color_range=self.color_range, rotation=self.rotation)

    def _next(self):
        try:
            f = next(self.it)
        except StopIteration:
            return None
        tt = (f.pts * self.tb if f.pts is not None else 0.0) - self.start - self.t0
        return (tt, f)

    def frame_at(self, t):
        while self.nxt is not None and (self.cur is None or self.nxt[0] <= t + 1e-6):
            self.cur = self.nxt
            self.nxt = self._next()
        if self.cur is None:
            z = np.full(self.shape, 128, np.uint8)
            z[:self.h] = 16
            return z
        if self.cur_key is not self.cur[1]:
            f = self.cur[1]
            if self.native and f.format.name == "yuv420p":
                self.cur_yuv = f.to_ndarray()
            else:
                self.cur_yuv = f.to_ndarray(format="yuv420p", width=self.w, height=self.h)
            self.cur_key = f
        return self.cur_yuv

    def close(self):
        self.c.close()


def _yuv_matrix(colorspace, color_range):
    """3x4: [R,G,B] = M @ [Y, U, V, 1]."""
    if colorspace in (5, 6):                       # BT.601
        kr, kb = 0.299, 0.114
    else:                                          # BT.709 (и «не указано» у HD)
        kr, kb = 0.2126, 0.0722
    kg = 1 - kr - kb
    if color_range == 2:
        ys, cs, y0 = 1.0, 1.0, 0.0
    else:
        ys, cs, y0 = 255 / 219, 255 / 224, 16.0
    rv = 2 * (1 - kr) * cs
    bu = 2 * (1 - kb) * cs
    gu = -2 * kb * (1 - kb) / kg * cs
    gv = -2 * kr * (1 - kr) / kg * cs
    M = np.array([[ys, 0, rv], [ys, gu, gv], [ys, bu, 0]], np.float32)
    off = -M @ np.array([y0, 128, 128], np.float32)
    return np.concatenate([M, off[:, None]], 1).astype(np.float32)


def yuv420_to_rgb(yuv, info):
    """yuv420p (h*3/2, w) -> RGB uint8, длинная сторона <= info['max_side'], с учётом поворота."""
    import cv2
    w, h = info["w"], info["h"]
    Y = yuv[:h]
    U = yuv[h:h + h // 4].reshape(h // 2, w // 2)
    V = yuv[h + h // 4:h + h // 2].reshape(h // 2, w // 2)
    s = min(1.0, info["max_side"] / max(w, h))
    ow, oh = max(2, int(round(w * s))), max(2, int(round(h * s)))
    if (ow, oh) != (w, h):
        Y = cv2.resize(Y, (ow, oh), interpolation=cv2.INTER_AREA)
    U = cv2.resize(U, (ow, oh), interpolation=cv2.INTER_LINEAR)
    V = cv2.resize(V, (ow, oh), interpolation=cv2.INTER_LINEAR)
    M = info.get("_M")
    if M is None:
        M = info["_M"] = _yuv_matrix(info["colorspace"], info["color_range"])
    rgb = cv2.transform(cv2.merge([Y, U, V]), M)
    r = info.get("rotation", 0) % 360
    if r == 90:
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif r == 270:
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
    elif r == 180:
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    return rgb


class _Prefetch(threading.Thread):
    """Декод кадров экрана в фоновом потоке главного процесса (PyAV отпускает GIL)."""

    def __init__(self, dec, fps, n, depth):
        super().__init__(daemon=True)
        self.dec, self.fps, self.n = dec, fps, n
        self.q = queue.Queue(maxsize=max(2, depth))
        self.err = None
        self.start()

    def run(self):
        try:
            for j in range(self.n):
                self.q.put(self.dec.frame_at(j / self.fps))
        except Exception:
            self.err = traceback.format_exc()
            self.q.put(None)

    def get(self):
        f = self.q.get()
        if f is None:
            raise RuntimeError("декод ролика: " + str(self.err))
        return f


# ====================================================================== ffmpeg
def _ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


_ENC_OK = {}


def encoder_available(enc):
    if enc not in _ENC_OK:
        r = subprocess.run([_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=1080x1920:r=30:d=0.2", "-c:v", enc, "-pix_fmt", "yuv420p",
                            "-f", "null", "-"], capture_output=True)
        _ENC_OK[enc] = r.returncode == 0
    return _ENC_OK[enc]


def _ffmpeg_cmd(out_mp4, width, height, fps, audio, audio_t0, duration, encoder, crf, comment=None):
    cmd = [_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-"]
    if audio:
        if audio_t0 > 0:
            cmd += ["-ss", f"{audio_t0:.3f}"]
        cmd += ["-t", f"{duration:.3f}", "-i", str(audio)]
    cmd += ["-map", "0:v:0"]
    if audio:
        # apad: если звук короче видео, добиваем тишиной (иначе -shortest обрезал бы видео)
        cmd += ["-map", "1:a:0?", "-af", f"apad=whole_dur={duration:.3f}", "-c:a", "aac", "-b:a", "192k",
                "-ar", "44100"]
    # RGB -> YUV по BT.709 (как помечаем), ограниченный диапазон — как ждут телефоны/TikTok
    cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv:threads=4,format=yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    if encoder == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(crf), "-b:v", "0",
                "-maxrate", "30M", "-bufsize", "60M", "-spatial-aq", "1", "-profile:v", "high", "-g", str(int(fps * 2))]
    else:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-profile:v", "high",
                "-g", str(int(fps * 2))]
    cmd += ["-pix_fmt", "yuv420p", "-r", f"{fps}", "-t", f"{duration:.3f}", "-movflags", "+faststart"]
    if comment:                                     # формат/вариант — в метаданные файла
        cmd += ["-metadata", f"comment={comment}"]
    cmd += [str(out_mp4)]
    return cmd


# ====================================================================== воркер
_OFFS6 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)


def _fly_body_pts(stage):
    """3D-точки ТЕЛА мухи (голова, грудь, брюшко — без размаха лап и крыльев) в текущей позе:
    центры геомов мухи рядом с осью тела ± их радиус. Нужен stage.m / stage.d (MuJoCo) и
    stage.fly_geoms (id геомов мухи). Иначе None."""
    m, d, ids = getattr(stage, "m", None), getattr(stage, "d", None), getattr(stage, "fly_geoms", None)
    if m is None or d is None or ids is None:
        return None
    ids = np.fromiter(ids, dtype=np.int64)
    if not len(ids):
        return None
    xp = np.asarray(d.geom_xpos)[ids]
    r = np.minimum(np.asarray(m.geom_rbound)[ids], 0.6)
    keep = (np.abs(xp[:, 1]) < 0.9) & (xp[:, 2] > 0.1)
    if keep.sum() < 3:
        return None
    return (xp[keep][:, None, :] + r[keep][:, None, None] * _OFFS6).reshape(-1, 3)


def _bbox(p):
    p = p[p[:, 2] > 1e-6]
    if not len(p):
        return None
    return (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))


def _stage_geometry(stage, w, h):
    """Где муха и экран на картинке (для подписей-реакций). Всё необязательно. Порядок поиска:
    stage.fly_box(w, h) / fly_screen_xy(w, h) / screen_box(w, h) — если Stage их даёт;
    иначе stage.project(p3, w, h) + тело мухи (_fly_body_pts) или FLY_BOX / stage.fly_pts,
    экран — stage._screen_corners()."""
    g = {}
    try:
        proj = getattr(stage, "project", None)
        if hasattr(stage, "fly_box"):
            g["fly_box"] = tuple(stage.fly_box(w, h))
        elif proj is not None:
            pts = _fly_body_pts(stage)
            if pts is None:
                mod = sys.modules.get(type(stage).__module__)
                b = getattr(stage, "FLY_BOX", None)
                b = getattr(mod, "FLY_BOX", None) if b is None else b
                if b is not None:
                    b = np.asarray(b, float)
                    pts = np.array([[x, y, z] for x in b[:, 0] for y in b[:, 1] for z in b[:, 2]])
                else:
                    pts = getattr(stage, "fly_pts", None)
            if pts is not None:
                bb = _bbox(proj(np.asarray(pts, float), w, h))
                if bb is not None:
                    g["fly_box"] = bb
        if hasattr(stage, "fly_screen_xy"):
            g["fly_xy"] = tuple(stage.fly_screen_xy(w, h))
        elif proj is not None:
            p = proj(np.array([[0.0, 0.0, 1.0]]), w, h)[0]
            g["fly_xy"] = (float(p[0]), float(p[1]))
        if hasattr(stage, "screen_box"):
            g["screen_box"] = tuple(stage.screen_box(w, h))
        elif proj is not None and hasattr(stage, "_screen_corners"):
            bb = _bbox(proj(stage._screen_corners(), w, h))
            if bb is not None:
                g["screen_box"] = bb
    except Exception:
        pass
    return g


def _worker(wid, cfg, task_q, done_q):
    try:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        import cv2
        cv2.setNumThreads(cfg["cv_threads"])
        sys.path.insert(0, str(HERE))
        import importlib
        C = importlib.import_module(cfg.get("compose_module", "compose"))     # compose | compose_streamer
        fmt = cfg.get("fmt") or {}
        log = StateLog(cfg["log"])
        extra = cfg["extra"]                           # по кадрам ЛОГА [0, start + N)
        k0 = int(cfg.get("start", 0))                  # окно превью: задача i -> кадр лога k0 + i
        shm_out = shared_memory.SharedMemory(name=cfg["shm_out"])
        shm_scr = shared_memory.SharedMemory(name=cfg["shm_scr"])
        R = cfg["ring"]
        W, H, SH = cfg["width"], cfg["height"], cfg["scene_h"]
        SW = cfg.get("scene_w", W)
        out_ring = np.ndarray((R, H, W, 3), np.uint8, buffer=shm_out.buf)
        scr_ring = np.ndarray((R, *cfg["scr_shape"]), np.uint8, buffer=shm_scr.buf)
        scr_info = cfg["scr_info"]
        stage = cfg["stage_factory"]()
        set_screen = getattr(stage, "set_screen", None) or getattr(stage, "set_screen_frame")
        rkw = {}
        try:
            if "bloom" in inspect.signature(stage.render).parameters:
                rkw["bloom"] = 0.0                     # свой блум делает compose
        except (TypeError, ValueError):
            pass
        C.configure(gauge_max=log.meta.get("gauge_max"))
        comp = C.Composer() if fmt.get("format", "base") == "base" else C.Composer(fmt=fmt)
        if fmt.get("flyeye"):                          # «как видит муха»: хук композитора + подписи обходят врезку
            import flyeye as FE
            comp.flyeye = FE.FlyEye()
            if hasattr(C, "EXTRA_AVOID") and getattr(C, "FLYEYE_BOX", None) is not None:
                C.EXTRA_AVOID.append(tuple(C.FLYEYE_BOX))
        need_rgb = bool(fmt.get("flyeye")) or fmt.get("format", "base") != "base"
        base_hud = dict(title=log.meta.get("title"), fly_no=log.meta.get("fly_no"))
        black = np.zeros((540, 960, 3), np.uint8)
        gcache, memo = {}, {}
        place = getattr(C, "caption_positions", None)

        # приступ -> смерть -> новая муха (hud life): первый кадр смерти — шар там замирает
        k_dead = next((j for j, f in enumerate(log.frames) if (f.get("hud") or {}).get("life") == "dead"), None)

        def state_at(j):                               # состояние сцены кадра лога j (+ поза для StreamerStage)
            sj = log.frames[j].get("stage")
            p = extra[j].get("pose") if j < len(extra) else None
            hj = log.frames[j].get("hud") or {}
            life = hj.get("life")
            if life and life != "alive" and isinstance(sj, dict):
                sj = dict(sj, life=life, life_t=float(hj.get("life_t") or 0.0))
                if k_dead is not None and life in ("dead", "respawn"):
                    sj["ball_q"] = (log.frames[k_dead].get("stage") or {}).get("qpos")
            return sj if p is None else dict(sj or {}, pose=p)

        def geom_at(j):                                # геометрия кадра j (для реакций: кадры их жизни)
            j = min(max(int(j), 0), log.n - 1)
            if j not in gcache:
                if len(gcache) > 4000:
                    gcache.clear()
                sj = state_at(j)
                if sj is not None:
                    stage.set_state(sj)
                gcache[j] = _stage_geometry(stage, SW, SH)
            return gcache[j]

        caps_at = lambda j: extra[j].get("captions") if 0 <= j < len(extra) else None
        done_q.put(("ready", wid))
        while True:
            i = task_q.get()
            if i is None:
                break
            slot = i % R
            k = k0 + i
            fr = log.frames[k]
            t0 = time.perf_counter()
            caps = extra[k].get("captions")
            cap_pos = place(caps, geom_at, caps_at, memo, fps=log.fps) if caps and place is not None else None
            st = state_at(k)
            if st is not None:
                stage.set_state(st)
            scr = yuv420_to_rgb(scr_ring[slot], scr_info) if scr_info else None
            set_screen(scr if scr is not None else black)
            scene = stage.render(SW, SH, **rkw)
            hud = dict(base_hud)
            hud.update(_stage_geometry(stage, SW, SH))
            hud.update(fr.get("hud") or {})
            hud.update(extra[k])
            hud["caption_pos"] = cap_pos
            hud["act_idx"], hud["act_val"] = log.act(k)
            if need_rgb:
                hud["screen_rgb"] = scr
            t1 = time.perf_counter()
            comp.compose(scene, hud, k / log.fps, out=out_ring[slot])
            done_q.put(("ok", i, wid, t1 - t0, time.perf_counter() - t1))
        try:
            stage.close()
        except Exception:
            pass
        del out_ring, scr_ring
        shm_out.close()
        shm_scr.close()
        done_q.put(("bye", wid))
    except Exception:
        done_q.put(("err", wid, traceback.format_exc()))


class _Writer(threading.Thread):
    """Пишет готовые кадры в ffmpeg строго по порядку и сразу отдаёт освободившийся слот кольца
    следующему кадру (i + R). Запись в pipe отпускает GIL -> идёт параллельно с приёмом сообщений."""

    def __init__(self, ff, out_ring, R, N, dispatch):
        super().__init__(daemon=True)
        self.ff, self.out_ring, self.R, self.N, self.dispatch = ff, out_ring, R, N, dispatch
        self.q = queue.Queue()
        self.err = None
        self.written = 0
        self.t_write = 0.0
        self.start()

    def run(self):
        try:
            while True:
                i = self.q.get()
                if i is None:
                    break
                tw = time.perf_counter()
                self.ff.stdin.write(self.out_ring[i % self.R].data)
                self.t_write += time.perf_counter() - tw
                self.written = i + 1
                if i + self.R < self.N:
                    self.dispatch(i + self.R)
        except Exception:
            self.err = traceback.format_exc()


# ====================================================================== главный процесс
def run_pass2(*args, encoder="auto", **kw):
    """Проход 2 (описание — у _run_pass2_once). Если h264_nvenc упал (видеокарта занята другим рендером или
    программой: «CreateBitstreamBuffer failed: out of memory»), один раз перерендериваем на процессоре (libx264)."""
    try:
        return _run_pass2_once(*args, encoder=encoder, **kw)
    except RuntimeError as e:
        msg = str(e)
        if encoder not in ("auto", "h264_nvenc") or not ("nvenc" in msg or "Broken pipe" in msg or "BrokenPipe" in msg):
            raise
        print("  видеокарта не дала кодировщик (занята/нет памяти) — кодирую на процессоре (libx264), "
              "это медленнее", flush=True)
        _ENC_OK["h264_nvenc"] = False
        return _run_pass2_once(*args, encoder="libx264", **kw)


def _run_pass2_once(state_log_path, out_mp4, source_audio_path, stage_factory, n_workers=None, width=1080, height=1920,
                    fps=30, source_video_path=None, frames=None, encoder="auto", crf=20, scene_h=None,
                    screen_max=960, ring=None, cv_threads=1, verbose=True, start=0, fmt=None):
    """Рендер лога в вертикальный mp4 с оригинальным звуком.

    state_log_path  — папка лога (StateLogWriter) или путь к её meta.json
    out_mp4         — итоговый файл (h264 yuv420p + AAC)
    source_audio_path — откуда звук (обычно исходный клип); None — без звука
    stage_factory   — вызываемое без аргументов, должно пиклиться (класс Stage или функция модуля);
                      объект: set_state(dict), set_screen(rgb) или set_screen_frame(rgb), render(w, h);
                      по желанию project()/_screen_corners() или fly_screen_xy()/screen_box() — чтобы
                      подписи-реакции вставали рядом с мухой и не закрывали экран
    source_video_path — откуда кадры для монитора (по умолчанию meta['source'] или source_audio_path)
    frames          — сколько кадров рендерить (None — до конца лога)
    start           — с какого кадра лога начать (превью окна); экран и звук сдвигаются так же
    fmt             — formats.fmt_cfg(...): формат 'base' | 'streamer', вариант комнаты, врезка «как видит
                      муха»; None — базовый без врезки (как раньше). Размер сцены — из модуля раскладки
    encoder         — 'auto' (h264_nvenc, если работает, иначе libx264) | 'h264_nvenc' | 'libx264'
    -> dict со статистикой (fps, время, кодек ...)."""
    sys.path.insert(0, str(HERE))
    import importlib
    import formats as FM
    fmt = dict(FM.fmt_cfg(), **(fmt or {}))
    cmod = FM.compose_module(fmt)
    C = importlib.import_module(cmod)
    scene_w = getattr(C, "SCENE_W", C.W)
    scene_h = C.SCENE_H if scene_h is None else scene_h
    assert (width, height, scene_h) == (C.W, C.H, C.SCENE_H), \
        f"раскладка {cmod} рассчитана на {C.W}x{C.H}, сцена {scene_w}x{C.SCENE_H}"
    T0 = time.perf_counter()
    log = StateLog(state_log_path)
    start = max(0, int(start or 0))
    N = log.n - start if frames is None else min(int(frames), log.n - start)
    if N <= 0:
        raise ValueError("пустой лог" if start == 0 else f"окно за концом лога: кадр {start} из {log.n}")
    fps = float(fps or log.fps)
    if abs(fps - log.fps) > 1e-6:
        raise ValueError(f"fps лога {log.fps} != {fps}")
    # 5: дальше упираемся в одну видеокарту (8 контекстов OpenGL медленнее 5 — см. замеры в отчёте)
    n_workers = int(n_workers or max(1, min(5, (os.cpu_count() or 4) // 2)))
    R = int(ring or max(6, 2 * n_workers + 2))
    src_video = source_video_path or log.meta.get("source") or source_audio_path
    if src_video and not Path(src_video).exists():      # относительный путь в meta — от корня проекта / лога
        for base in (ROOT, log.path, HERE):
            if (base / src_video).exists():
                src_video = base / src_video
                break
        else:
            raise FileNotFoundError(f"ролик для экрана не найден: {src_video}")
    t_src0 = float(log.meta.get("src_t0", 0.0))
    if encoder == "auto":
        encoder = "h264_nvenc" if encoder_available("h264_nvenc") else "libx264"
    # таймлайн интерфейса — один раз по логу с его начала (сглаживания и реакции как в полном ролике)
    C.configure(gauge_max=log.meta.get("gauge_max"))
    K = start + N
    huds = [fr.get("hud") or {} for fr in log.frames[:K]]
    stages = [fr.get("stage") or {} for fr in log.frames[:K]]
    extra = C.prepare_timeline(huds, fps, stages)
    if fmt["format"] == "streamer":                   # поза сидящей мухи (StreamerStage) — из нейронов и куклы
        for e, p in zip(extra, FM.pose_track(huds, stages, fps, log.meta.get("gauge_max"))):
            e["pose"] = p
    if fmt["flyeye"]:                                 # признаки зрения, которые получил мозг
        for e, v in zip(extra, FM.vis_rows(log.frames, src_video, K, t_src0, fps)):
            e["vis"] = v
    t_win = t_src0 + start / fps                      # с какой секунды ролика начинается окно
    dec = ScreenDecoder(src_video, screen_max, t_win) if src_video else None
    scr_shape = dec.shape if dec else (3, 2)
    shm_out = shared_memory.SharedMemory(create=True, size=R * height * width * 3)
    shm_scr = shared_memory.SharedMemory(create=True, size=R * int(np.prod(scr_shape)))
    out_ring = np.ndarray((R, height, width, 3), np.uint8, buffer=shm_out.buf)
    scr_ring = np.ndarray((R, *scr_shape), np.uint8, buffer=shm_scr.buf)
    ctx = mp.get_context("spawn")
    task_q, done_q = ctx.Queue(), ctx.Queue()
    cfg = dict(log=str(log.path), extra=extra, shm_out=shm_out.name, shm_scr=shm_scr.name, ring=R,
               width=width, height=height, scene_h=scene_h, scr_shape=scr_shape,
               scr_info=dec.info() if dec else None, stage_factory=stage_factory, cv_threads=cv_threads,
               start=start, fmt=fmt, compose_module=cmod, scene_w=scene_w)
    procs = [ctx.Process(target=_worker, args=(k, cfg, task_q, done_q), daemon=True) for k in range(n_workers)]
    for p in procs:
        p.start()
    ff = writer = None
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    err_path = out_mp4.with_suffix(".ffmpeg.log")
    stats = dict(frames=N, workers=n_workers, encoder=encoder, ring=R)
    if start or not FM.is_default(fmt):
        stats.update(start=start, fmt=FM.describe(fmt))
    try:
        # ждём готовности воркеров (Stage грузится несколько секунд)
        ready = 0
        while ready < n_workers:
            msg = _get(done_q, procs)
            if msg[0] == "ready":
                ready += 1
            elif msg[0] == "err":
                raise RuntimeError(f"воркер {msg[1]}:\n{msg[2]}")
        t_ready = time.perf_counter()
        stats["startup_s"] = round(t_ready - T0, 2)
        audio = source_audio_path if source_audio_path else None
        comment = None if FM.is_default(fmt) else "муха: " + FM.describe(fmt)
        ff = subprocess.Popen(_ffmpeg_cmd(out_mp4, width, height, fps, audio, t_win, N / fps, encoder, crf,
                                          comment=comment),
                              stdin=subprocess.PIPE, stderr=open(err_path, "wb"))
        pre = _Prefetch(dec, fps, N, R) if dec is not None else None
        t_dec = [0.0]

        def dispatch(j):
            if pre is not None:
                td = time.perf_counter()
                scr_ring[j % R] = pre.get()
                t_dec[0] += time.perf_counter() - td
            task_q.put(j)

        for j in range(min(R, N)):
            dispatch(j)
        writer = _Writer(ff, out_ring, R, N, dispatch)

        def check():
            if writer.err:
                raise RuntimeError("запись в ffmpeg: " + writer.err + "\n" +
                                   err_path.read_text(errors="replace")[-1500:])

        done = set()
        nxt = 0
        tr = tc = 0.0
        t_last = time.perf_counter()
        while nxt < N:
            msg = _get(done_q, procs, check=check)
            if msg[0] == "ok":
                done.add(msg[1])
                tr += msg[3]
                tc += msg[4]
            elif msg[0] == "err":
                raise RuntimeError(f"воркер {msg[1]}:\n{msg[2]}")
            while nxt in done:
                done.discard(nxt)
                writer.q.put(nxt)
                nxt += 1
            if verbose and (time.perf_counter() - t_last > 5 or nxt == N):
                el = time.perf_counter() - t_ready
                print(f"  [{writer.written}/{N}] {writer.written / max(el, 1e-6):.1f} кадр/с", flush=True)
                t_last = time.perf_counter()
        writer.q.put(None)
        writer.join()
        check()
        for _ in procs:
            task_q.put(None)
        ff.stdin.close()
        rc = ff.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg вернул {rc}: {err_path.read_text(errors='replace')[-2000:]}")
        el = time.perf_counter() - t_ready
        stats.update(render_s=round(el, 2), fps=round(N / el, 2), total_s=round(time.perf_counter() - T0, 2),
                     stage_ms=round(tr / N * 1e3, 2), compose_ms=round(tc / N * 1e3, 2),
                     pipe_write_ms=round(writer.t_write / N * 1e3, 2), screen_wait_ms=round(t_dec[0] / N * 1e3, 2),
                     out=str(out_mp4))
        for p in procs:
            p.join(timeout=10)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        if writer is not None and writer.is_alive():
            writer.q.put(None)
        if ff is not None and ff.poll() is None:
            try:
                ff.stdin.close()
            except Exception:
                pass
            ff.kill()
        if writer is not None:
            writer.join(timeout=5)
            writer.out_ring = None
        if dec is not None:
            dec.close()
        out_ring = scr_ring = None
        shm_out.close()
        shm_out.unlink()
        shm_scr.close()
        shm_scr.unlink()
    try:
        if err_path.exists() and err_path.stat().st_size == 0:
            err_path.unlink()
    except OSError:
        pass
    if verbose:
        print("  проход 2:", stats, flush=True)
    return stats


def _get(q, procs, timeout=180.0, check=None):
    """Сообщение от воркеров; если кто-то умер молча — ошибка, а не вечное ожидание."""
    t0 = time.time()
    while True:
        try:
            return q.get(timeout=1.0)
        except queue.Empty:
            if check is not None:
                check()
            dead = [k for k, p in enumerate(procs) if p.exitcode not in (None, 0)]
            if dead:
                raise RuntimeError(f"воркер(ы) {dead} упали (exitcode {[procs[k].exitcode for k in dead]})")
            if time.time() - t0 > timeout:
                raise TimeoutError("воркеры молчат > %.0f с" % timeout)


def probe(path):
    """Проверка результата: размер, fps, число кадров, звук (PyAV)."""
    import av
    c = av.open(str(path))
    v = c.streams.video[0]
    a = c.streams.audio[0] if c.streams.audio else None
    info = dict(width=v.codec_context.width, height=v.codec_context.height, fps=float(v.average_rate),
                frames=v.frames, codec=v.codec_context.name, pix_fmt=v.codec_context.pix_fmt,
                duration=float(c.duration / 1e6) if c.duration else None,
                audio=None if a is None else dict(codec=a.codec_context.name, rate=a.rate, channels=a.channels,
                                                  duration=float(a.duration * a.time_base) if a.duration else None))
    if a is not None:        # звук не тишина?
        peak = 0.0
        for k, fr in enumerate(c.decode(a)):
            peak = max(peak, float(np.abs(fr.to_ndarray()).max()))
            if k > 200:
                break
        info["audio"]["peak"] = round(peak, 4)
    c.close()
    return info


# ====================================================================== самопроверка
class MockStage:
    """Заглушка Stage для проверки конвейера: рисует cv2 «комнату», монитор с роликом
    (альбом/портрет по кадру), шар и муху, которая машет крыльями по состоянию."""

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.st = {}
        self.frame = None
        self._bg = None

    def set_state(self, s):
        self.st = s

    def set_screen_frame(self, rgb):
        self.frame = rgb

    def _layout(self, w, h):
        cam = int(self.st.get("cam", 0))
        portrait = self.frame is not None and self.frame.shape[0] > self.frame.shape[1]
        if portrait:
            sh = h * 0.56
            sw = sh * 9 / 16
        else:
            sw = w * 0.78
            sh = sw * 9 / 16
        cx = w * (0.5 + (0.08 if cam % 2 else -0.08))
        cy = max(h * 0.33, sh / 2 + 24)
        box = (cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2)
        fly = (w * (0.5 - (0.1 if cam % 2 else -0.1)), h * 0.72)
        return box, fly

    def screen_box(self, w, h):
        return self._layout(w, h)[0]

    def fly_box(self, w, h):
        fx, fy = self._layout(w, h)[1]
        return (fx - 90, fy - 50, fx + 110, fy + 40)

    def fly_screen_xy(self, w, h):
        return self._layout(w, h)[1]

    def render(self, w, h, bloom=0.0):
        cv2 = self.cv2
        if self._bg is None or self._bg.shape[:2] != (h, w):
            y = np.linspace(0, 1, h)[:, None, None]
            self._bg = (np.array([40, 22, 70]) * (1 - y) + np.array([8, 10, 22]) * y).astype(np.uint8) \
                .repeat(w, 1)
        img = self._bg.copy()
        (x0, y0, x1, y1), (fx, fy) = self._layout(w, h)
        t = float(self.st.get("t", 0.0))
        cv2.rectangle(img, (int(x0) - 14, int(y0) - 14), (int(x1) + 14, int(y1) + 14), (20, 20, 26), -1)
        if self.frame is not None:
            img[int(y0):int(y0) + int(y1 - y0), int(x0):int(x0) + int(x1 - x0)] = \
                cv2.resize(self.frame, (int(x1 - x0), int(y1 - y0)), interpolation=cv2.INTER_AREA)
            cv2.putText(img, "@xyz_web3", (int(x1) - 150, int(y1) - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (235, 235, 235), 2, cv2.LINE_AA)
        for k in range(12):                                   # RGB-лента
            col = tuple(int(v) for v in (np.array([np.sin(t + k), np.sin(t + k + 2), np.sin(t + k + 4)]) * 110 + 140))
            cv2.line(img, (k * w // 12, int(y1) + 60), ((k + 1) * w // 12, int(y1) + 60), col, 6, cv2.LINE_AA)
        cv2.circle(img, (int(fx), int(fy) + 330), 320, (215, 210, 220), -1, cv2.LINE_AA)       # шар
        flap = float(self.st.get("flying", 0)) * np.sin(t * 60)
        ang = int(25 + 40 * flap)
        cv2.ellipse(img, (int(fx), int(fy)), (60, 26), 0, 0, 360, (40, 30, 30), -1, cv2.LINE_AA)
        cv2.ellipse(img, (int(fx) - 10, int(fy) - 20), (70, 22), -ang, 0, 360, (200, 220, 255), 2, cv2.LINE_AA)
        cv2.ellipse(img, (int(fx) + 10, int(fy) - 20), (70, 22), ang, 0, 360, (200, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (int(fx) + 58, int(fy) - 8), 18, (150, 30, 30), -1, cv2.LINE_AA)
        if float(self.st.get("proboscis", 0)) > 0.3:
            cv2.line(img, (int(fx) + 70, int(fy)), (int(fx) + 95, int(fy) + 25), (60, 40, 40), 5, cv2.LINE_AA)
        return img

    def close(self):
        pass


def make_mock_log(path, secs=10.0, fps=30, title="VILLIAN, madk1d — ДИНАСТИЯ 2 ft. Андрей Гайдулян (клип 2026)",
                  source=DEFAULT_SRC, src_t0=30.0, real_stage=False, seed=0, portrait=False):
    """Фальшивый проход 1: правдоподобные шкалы, события, разреженная активность мозга.
    real_stage=True — состояния тела пишет настоящий stage.Stage (step_puppet + get_state)."""
    sys.path.insert(0, str(HERE))
    import compose as C
    rng = np.random.default_rng(seed)
    grp = C._brain_cache(1040, 380)["grp"]
    pools = [np.nonzero(grp == g)[0] for g in range(5)]
    n = int(secs * fps)
    stg = None
    if real_stage:
        from stage import Stage
        stg = Stage()
        presets = Stage.PRESET_ORDER
        if portrait and hasattr(stg, "set_orientation"):
            stg.set_orientation("portrait", t_anim=0)
    w = StateLogWriter(path, meta=dict(fps=fps, title=title, fly_no=7, source=str(source), src_t0=src_t0))
    for i in range(n):
        t = i / fps
        fly = 3.0 < t < 3.6
        groom = 6.0 < t < 7.5
        prob = t > 8.2
        hud = dict(fear_hz=100 * np.exp(-((t - 3.1) / 0.4) ** 2) + 6 + 3 * np.sin(t * 3),
                   interest=22 + 18 * np.sin(t * 0.9), hearing_hz=5 + 8 * max(0.0, np.sin(t * 2 * np.pi * 2)),
                   hunger=0.75 - 0.05 * max(0.0, t - 8.2), steer=float(np.sin(t * 0.7)),
                   gf_hz=140 if fly else 2, mn9_hz=13 if prob else 0.5, groom_hz=80 if groom else 3,
                   flying=fly, grooming=groom, proboscis=1.0 if prob else 0.0,
                   event={int(2.9 * fps): "резкая вспышка — угроза",
                          int(8.0 * fps): "бас-дроп! — капля сахара"}.get(i))
        # активность: зрение ~ интерес, органы чувств ~ слух, центр — фон, команды ~ страх
        hz = np.zeros(len(grp), np.float32)
        for g, k in ((0, int(60 + 8 * hud["interest"])), (1, int(10 * hud["hearing_hz"])), (2, 90),
                     (3, int(0.4 * hud["fear_hz"])), (4, 5 if prob else 1)):
            sel = rng.choice(pools[g], min(k, len(pools[g])), replace=False)
            hz[sel] = rng.uniform(20, 120, len(sel))
        if stg is not None:
            k = int(t // 3.3)
            stg.set_camera(presets[k % len(presets)], presets[(k + 1) % len(presets)],
                           blend=float(np.clip((t % 3.3 - 2.5) / 0.8, 0, 1)), shake=0.5)
            stg.step_puppet(1 / fps, drive_L=0.0 if fly or groom else 0.4 * (1 + 0.5 * hud["steer"]),
                            drive_R=0.0 if fly or groom else 0.4 * (1 - 0.5 * hud["steer"]),
                            head_turn=0.5 * hud["steer"], proboscis=1.0 if prob else 0.0,
                            wing_spread=1.0 if fly else 0.0, wing_flap=1.0 if fly else 0.0,
                            groom_t=(t - 6.0) if groom else None, drop=1.0 if 8.0 < t < 9.6 else 0.0)
            state = stg.get_state()
        else:
            state = dict(t=t, cam=int(t // 3.3), flying=fly, proboscis=float(prob))
        w.add(stage=state, hud=hud, hz=hz)
    w.close()
    if stg is not None:
        stg.close()
    return path


def _selftest(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=10)
    ap.add_argument("--workers", default="1,4,8")
    ap.add_argument("--stage", default="mock", choices=["mock", "real"])
    ap.add_argument("--encoder", default="auto")
    ap.add_argument("--out", default=str(SCRATCH / "mock_vertical.mp4"))
    ap.add_argument("--keep_log", action="store_true")
    ap.add_argument("--source", default=str(DEFAULT_SRC), help="ролик для экрана и звука")
    ap.add_argument("--src_t0", type=float, default=30.0)
    ap.add_argument("--portrait", action="store_true", help="монитор Stage в портрет (для вертикального ролика)")
    a = ap.parse_args(argv)
    SCRATCH.mkdir(exist_ok=True)
    logp = SCRATCH / f"mock_log_{a.stage}{'_portrait' if a.portrait else ''}"
    if not (a.keep_log and (logp / "meta.json").exists()):
        t0 = time.time()
        make_mock_log(logp, a.secs, real_stage=a.stage == "real", source=a.source, src_t0=a.src_t0,
                      portrait=a.portrait)
        print(f"мок-лог: {logp} ({time.time() - t0:.1f} с)")
    if a.stage == "real":
        from stage import Stage as factory
    else:
        factory = MockStage
    res = []
    for nw in [int(x) for x in a.workers.split(",")]:
        print(f"== воркеров: {nw}")
        st = run_pass2(logp, a.out, a.source, factory, n_workers=nw, encoder=a.encoder)
        info = probe(a.out)
        print("  результат:", info)
        res.append((nw, st, info))
    print("\nитог:")
    for nw, st, info in res:
        print(f"  {nw} воркеров: {st['fps']} кадр/с (запуск {st['startup_s']} с), stage {st['stage_ms']} мс, "
              f"compose {st['compose_ms']} мс, запись {st['pipe_write_ms']} мс, {st['encoder']}; "
              f"{info['width']}x{info['height']} {info['fps']} fps, {info['frames']} кадров, звук: "
              f"{info['audio']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest(sys.argv[2:])
    else:
        print(__doc__)
