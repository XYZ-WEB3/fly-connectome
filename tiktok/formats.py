"""Форматы ролика («Базовый» / «Стример») — общий код прохода 2 (см. FORMATS.md).

    import formats as F
    fmt = F.fmt_cfg("streamer", F.resolve_variant("random"), flyeye=True, watermark="@ник")
    F.compose_module(fmt)                      # 'compose' | 'compose_streamer'
    track = F.pose_track(huds, stages, fps)    # поза для StreamerStage по кадрам лога
    vis = F.vis_rows(log.frames, source, n)    # признаки зрения кадра (то, что получил мозг)
    with F.RenderLock("метка"):                # один рендер на машине + ждём свободной памяти
        ...

"""
import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "_cache"
LOCK_PATH = HERE / "_render.lock"
QPOS_MAP = CACHE / "stage_qpos_map.json"

FORMATS = ("base", "streamer")
FORMAT_RU = {"base": "Базовый", "streamer": "Стример"}
VARIANT_MAX = 999_999
VIS_KEYS = ("obj_L", "obj_R", "loom_L", "loom_R", "flow_x")      # признаки зрения -> LC10 / LPLC2+LC4 / поток

# что муха делает сейчас (подпись «сейчас: …»): ключ -> (текст, нейроны). Только по состояниям из нейронов.
DOING = {
    "takeoff": ("рвётся взлететь!", "гигантское волокно DNp01"),
    "startle": ("вздрогнула", "гигантское волокно DNp01"),
    "drink": ("пьёт сахар", "хоботок · мотонейрон MN9"),
    "proboscis": ("выпустила хоботок", "мотонейрон MN9"),
    "groom": ("чистит лапки", "DNg84 / DNg35"),
    "listen": ("слушает", "орган Джонстона"),
    "track": ("следит за объектом", "LC10 · руль DNa01/02"),
    "watch": ("смотрит ролик", "зрение: упрощённая сетчатка"),
    "seizure": ("приступ: мозг разогнался", "защита отключена"),          # подпись — с числом нейронов
    "dead": ("муха выбыла", "мозг всё ещё в приступе"),
    "respawn": ("новая муха", "мозг с нуля, защита на месте"),
}
STARTLE_SHOW = 0.6          # с: столько держим «вздрогнула» после события startle


# ---------------------------------------------------------------- конфиг формата
def resolve_variant(v):
    """'random' / '' / None -> случайный номер; число или строка-число -> int (0..VARIANT_MAX)."""
    if v is None or str(v).strip().lower() in ("", "random", "случайный"):
        return random.SystemRandom().randint(1, VARIANT_MAX)
    n = int(str(v).strip())
    if not 0 <= n <= VARIANT_MAX:
        raise ValueError(f"вариант должен быть 0..{VARIANT_MAX}: {v}")
    return n


def fmt_cfg(fmt="base", variant=None, flyeye=False, watermark=""):
    """Пиклящийся конфиг для воркеров рендера. variant — уже разрешённый int (или None у «Базового»)."""
    if fmt not in FORMATS:
        raise ValueError(f"формат {fmt!r}: нужен один из {FORMATS}")
    return dict(format=fmt, variant=None if variant is None else int(variant), flyeye=bool(flyeye),
                watermark=watermark or "")


def is_default(fmt):
    return not fmt or (fmt.get("format", "base") == "base" and not fmt.get("flyeye"))


def compose_module(fmt):
    return "compose_streamer" if (fmt or {}).get("format") == "streamer" else "compose"


def describe(fmt):
    """Строка для метаданных mp4 и лога."""
    fmt = fmt or {}
    s = f"формат={FORMAT_RU.get(fmt.get('format', 'base'), '?')}"
    if fmt.get("variant") is not None:
        s += f" вариант={fmt['variant']}"
    return s + f" как-видит-муха={'вкл' if fmt.get('flyeye') else 'выкл'}"


def name_tag(fmt):
    """Кусок имени файла: '' у базового, 'стример_v123' у стримера."""
    fmt = fmt or {}
    if fmt.get("format") != "streamer":
        return ""
    return "стример" + (f"_v{fmt['variant']}" if fmt.get("variant") is not None else "")


# ---------------------------------------------------------------- зрение для «как видит муха»
def vis_rows(frames, source=None, n=None, src_t0=0.0, fps=30):
    """Признаки зрения по кадрам лога (VIS_KEYS -> float) — то, что получил мозг в проходе 1.
    Сначала hud['vis'] (пишет новый проход 1), иначе кэш analyze (_cache/<имя ролика>.json — тот же
    файл читал проход 1). Нет данных -> None у кадра."""
    n = len(frames) if n is None else min(n, len(frames))
    out = [None] * n
    tl = None
    k0 = int(round(float(src_t0) * fps))
    for k in range(n):
        v = (frames[k].get("hud") or {}).get("vis")
        if v:
            out[k] = {kk: float(v.get(kk, 0.0)) for kk in VIS_KEYS}
            continue
        if tl is None and source:
            try:
                import analyze as A
                tl = A.load(source) or False
            except Exception:
                tl = False
        if tl:
            j = k + k0
            if j < len(tl["t"]):
                out[k] = {kk: float(tl[kk][j]) for kk in VIS_KEYS}
    return out


# ---------------------------------------------------------------- поза из лога (для StreamerStage)
def _stage_hash():
    return hashlib.md5((HERE / "stage.py").read_bytes()).hexdigest()[:12]


def base_qpos_map(nq=None):
    """Адреса qpos в модели stage.Stage (из неё писался лог): голова, крылья, ноги, хоботок.
    Кэш _cache/stage_qpos_map.json; нет/устарел — один раз строим Stage (без рендера, ~5 с)."""
    h = _stage_hash()
    if QPOS_MAP.exists():
        try:
            d = json.loads(QPOS_MAP.read_text(encoding="utf-8"))
            if d.get("stage_hash") == h and (nq is None or d["nq"] == nq):
                return d
        except Exception:
            pass
    import stage as SG
    st = SG.Stage()
    try:
        d = dict(stage_hash=h, nq=int(st.m.nq), head=int(st.head_yaw_adr), ball=int(st.ball_adr),
                 legs=[int(a) for a in st.leg_adr], leg_joints=list(st.fly.actuated_joints),
                 wings={s: [int(st.pose.wings[s][0][0]), int(st.pose.wings[s][1][0])] for s in "LR"},
                 rostrum=int(st.pose.ros[0]), haustellum=int(st.pose.hau[0]))
    finally:
        st.close()
    if nq is not None and d["nq"] != nq:
        raise ValueError(f"лог писался другой моделью Stage (nq {nq} != {d['nq']})")
    CACHE.mkdir(exist_ok=True)
    QPOS_MAP.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def pose_track(huds, stages, fps=30, gauge_max=None):
    """По логу прохода 1 (списки hud и stage.get_state() по кадрам) -> список поз по кадрам.
    Последовательно (время с начала чистки и т.п.), поэтому в главном процессе; воркер получает
    позу кадра в state['pose']. Всё — из нейронов (hud) и позы куклы прохода 1 (qpos). Ключи:
      i, t                     кадр лога и его время, с
      head                     поворот головы, рад (+ влево) — как в Stage (qpos joint_Head_roll)
      prob, drop, audio        хоботок 0..1 (MN9), капля сахара 0..1, бас ролика 0..1 (косметика комнаты)
      wing_yaw, wing_roll      (L, R) углы крыльев, рад (раскрытие/взмахи при startle/взлёте)
      legs                     float32[42] углы ног куклы (порядок leg_joints в base_qpos_map) или None
      flying, flight_age       полёт (GF > порога взлёта) и с его начала, с (None — не было)
      startle_age              с последнего вздрагивания, с (None — не было)
      grooming, groom_t        чистка и с её начала, с (None — не чистит)
      eating, eat_t            хоботок выдвинут (prob > 0.3) и с какого момента, с
      groom_end_age            с конца последней чистки, с (None — чистит сейчас или не было): плавный возврат лап
      listen, track            слух / интерес выше порога 0.3 с (как реакции в compose)
      doing, doing_age         ключ DOING — подпись «сейчас: …», и сколько секунд он уже держится
                               (кадры рисуются вразнобой: анимации смены — только по doing_age)"""
    gm = dict(hearing_hz=14.0, interest=40.0)
    gm.update({k: float(v) for k, v in (gauge_max or {}).items() if k in gm})
    dt = 1.0 / fps
    alpha = 1 - np.exp(-dt / 0.09)
    q0 = next((s.get("qpos") for s in stages if isinstance(s, dict) and s.get("qpos") is not None), None)
    qm = base_qpos_map(len(q0)) if q0 is not None else None
    g_hear = g_int = 0.0
    hold_hear = hold_int = 0.0
    groom_t0 = flight_t0 = eat_t0 = t_startle = groom_end = None
    was_flying = False
    doing_prev, doing_t0 = None, 0.0
    out = []
    for i, h in enumerate(huds):
        h = h or {}
        s = stages[i] if i < len(stages) and isinstance(stages[i], dict) else {}
        t = i * dt
        q = s.get("qpos")
        fx = s.get("fx")
        p = dict(i=i, t=t, head=0.0, prob=0.0, drop=0.0, audio=0.0, wing_yaw=(0.0, 0.0), wing_roll=(0.0, 0.0),
                 legs=None)
        if fx is not None and len(fx) >= 4:
            p["audio"], p["drop"], p["prob"] = float(fx[1]), float(fx[2]), float(fx[3])
        else:
            p["prob"] = float(h.get("proboscis", 0.0) or 0.0)
        if q is not None and qm is not None:
            p["head"] = float(q[qm["head"]])
            p["wing_yaw"] = (float(q[qm["wings"]["L"][0]]), float(q[qm["wings"]["R"][0]]))
            p["wing_roll"] = (float(q[qm["wings"]["L"][1]]), float(q[qm["wings"]["R"][1]]))
            p["legs"] = np.asarray(q, np.float32)[qm["legs"]]
        # события и состояния — только из hud (нейроны)
        flying = bool(h.get("flying", False))
        if h.get("takeoff") or (flying and not was_flying):
            flight_t0 = t
        was_flying = flying
        if h.get("startle"):
            t_startle = t
        grooming = bool(h.get("grooming", False))
        if grooming:
            groom_end = None
        elif groom_t0 is not None:
            groom_end = t                              # чистка только что кончилась
        groom_t0 = (groom_t0 if groom_t0 is not None else t) if grooming else None
        eating = p["prob"] > 0.3
        eat_t0 = (eat_t0 if eat_t0 is not None else t) if eating else None
        g_hear += alpha * (float(h.get("hearing_hz", 0.0) or 0.0) - g_hear)
        g_int += alpha * (float(h.get("interest", 0.0) or 0.0) - g_int)
        hold_hear = hold_hear + dt if g_hear / gm["hearing_hz"] > 0.7 else 0.0
        hold_int = hold_int + dt if g_int / gm["interest"] > 0.55 else 0.0
        p.update(flying=flying, flight_age=None if flight_t0 is None else t - flight_t0,
                 startle_age=None if t_startle is None else t - t_startle,
                 grooming=grooming, groom_t=None if groom_t0 is None else t - groom_t0,
                 groom_end_age=None if groom_end is None else t - groom_end,
                 eating=eating, eat_t=None if eat_t0 is None else t - eat_t0,
                 listen=hold_hear >= 0.3, track=hold_int >= 0.3)
        life = str(h.get("life") or "alive")                # приступ -> смерть -> новая муха (react.py)
        p["life"], p["life_t"] = life, float(h.get("life_t") or 0.0)
        if life in ("dead", "respawn"):
            p["drop"] = p["prob"] = 0.0
        if life != "alive":
            doing = life
        elif flying:
            doing = "takeoff"
        elif p["startle_age"] is not None and p["startle_age"] < STARTLE_SHOW:
            doing = "startle"
        elif eating:
            doing = "drink" if abs(p["drop"]) > 0.03 else "proboscis"
        elif grooming:
            doing = "groom"
        elif p["listen"]:
            doing = "listen"
        elif p["track"]:
            doing = "track"
        else:
            doing = "watch"
        if doing != doing_prev:
            doing_prev, doing_t0 = doing, t
        p["doing"], p["doing_age"] = doing, t - doing_t0
        out.append(p)
    return out


# ---------------------------------------------------------------- один рендер на машине
class RenderLock:
    """Файл-замок tiktok/_render.lock: одновременно рендерит один процесс (памяти мало).
    Замок создаётся атомарно (O_CREAT|O_EXCL), внутри — метка и pid; пока держим — раз в минуту
    обновляем время файла. Чужой замок старше stale_s или от умершего процесса — снимаем.
    После захвата ждём, пока свободной памяти станет больше min_free_gb."""

    def __init__(self, label, stale_s=20 * 60, min_free_gb=3.0, poll_s=10.0, log=print):
        self.label = re.sub(r"\s+", " ", str(label)).strip() or "рендер"
        self.stale_s, self.min_free_gb, self.poll_s, self.log = stale_s, min_free_gb, poll_s, log
        self._stop = threading.Event()
        self._hb = None
        self._mine = None

    def _stale(self):
        try:
            age = time.time() - os.path.getmtime(LOCK_PATH)
            text = LOCK_PATH.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        if age > self.stale_s:
            return True
        m = re.search(r"pid=(\d+)", text)
        if m:
            try:
                import psutil
                return not psutil.pid_exists(int(m.group(1)))
            except Exception:
                return False
        return False

    def __enter__(self):
        said = 0.0
        while True:
            try:
                fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._stale():
                    try:
                        os.remove(LOCK_PATH)
                    except OSError:
                        pass
                    continue
                if time.time() - said > 60:
                    try:
                        who = LOCK_PATH.read_text(encoding="utf-8", errors="replace").strip()
                    except OSError:
                        who = "?"
                    self.log(f"ЖДУ: идёт другой рендер ({who})", flush=True)
                    said = time.time()
                time.sleep(self.poll_s)
                continue
            self._mine = f"{self.label} pid={os.getpid()} t={int(time.time())}"
            os.write(fd, self._mine.encode("utf-8"))
            os.close(fd)
            break
        self._hb = threading.Thread(target=self._beat, daemon=True)
        self._hb.start()
        try:
            import psutil
            said = 0.0
            while psutil.virtual_memory().available / 1e9 < self.min_free_gb:
                if time.time() - said > 60:
                    self.log(f"ЖДУ: свободной памяти меньше {self.min_free_gb:.0f} ГБ", flush=True)
                    said = time.time()
                time.sleep(self.poll_s)
        except ImportError:
            pass
        return self

    def _beat(self):
        while not self._stop.wait(60.0):
            try:
                os.utime(LOCK_PATH)
            except OSError:
                pass

    def __exit__(self, *a):
        self._stop.set()
        try:
            if LOCK_PATH.read_text(encoding="utf-8", errors="replace") == self._mine:
                os.remove(LOCK_PATH)
        except OSError:
            pass
