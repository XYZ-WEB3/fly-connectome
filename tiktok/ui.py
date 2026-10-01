"""Окно утилиты «Муха смотрит ролики»: выбрать ролики, задать вотермарку, запустить рендер.

Запуск: двойной клик по «Муха-рендер.bat» в папке проекта
(или: .venv\\Scripts\\pythonw.exe tiktok\\ui.py).

Каждый ролик рендерится отдельным процессом tiktok/react.py (окно не зависает, «Стоп» убивает
всё дерево процессов). Пока ролик не готов, он пишется в «<имя>.part.mp4»: при стопе/ошибке
недописанный файл удаляется, а прошлый готовый ролик с тем же именем не портится. Номер мухи
тратится только на готовый полный ролик. Настройки (вотермарка, качество, процессы, формат ролика,
вариант комнаты, «как видит муха», замеры скорости для оценки времени) запоминаются в ui_settings.json.

Формат (tiktok/FORMATS.md): «Базовый» (как раньше) или «Стример» (кресло, наушники, комната —
своя на каждый номер варианта); врезка «как видит муха» — для обоих. Рендер окна идёт под общим
замком tiktok/_render.lock (--lock окно): пока рендерит кто-то другой, окно показывает «ждёт».
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:                                                          # PIL есть в .venv (им рисует stage.py)
    from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageTk
except Exception:                                             # без PIL просто не будет превью вотермарки
    Image = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:                                 # formats/style лежат рядом (запуск не из tiktok/)
    sys.path.insert(0, str(HERE))
import formats as FM                                          # noqa: E402  (формат, вариант, имя файла — как react.py)

PY = ROOT / ".venv" / "Scripts" / "python.exe"
REACT = HERE / "react.py"
IN_DIR = ROOT / "видео на реакцию"
OUT_DIR = ROOT / "готово"
SETTINGS = HERE / "ui_settings.json"
COUNTER = ROOT / "data" / "fly_counter.json"
LOG_CACHE = HERE / "_cache"
LOCK_LABEL = "окно"                                           # метка в tiktok/_render.lock (formats.RenderLock)
WM_FONT = "C:/Windows/Fonts/bahnschrift.ttf"                  # тот же шрифт, что у вотермарки в stage.py
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi")
NO_WINDOW = 0x08000000 if os.name == "nt" else 0          # CREATE_NO_WINDOW
SUSPENDED = 0x00000004                                    # CREATE_SUSPENDED: сначала в job, потом пуск
APP_TITLE = "Муха смотрит ролики — рендер"
# Приступ (защиту выключают, мозг разгоняется, муха выбывает, появляется новая) случается САМ — только по
# содержанию ролика: в самый грустный момент (слова), а без слов — на самом резком басе (react.py --seizure auto).
# Задать его в окне нельзя. Для исследователей: True — появятся выключатель «Приступ» и поле «на секундах».
SHOW_SEIZURE = False
# Сборка .exe: знак проекта на мониторе зашит и в окне не меняется (None — поле можно править).
WATERMARK_LOCK = None

DEFAULTS = dict(watermark="@xyz_web3", crf="24 — стандарт", workers=5, preview_secs=0, preview_t0=0,
                format="base", flyeye=False, variant="random", seizure=False)
QUALITY = ["20 — лучше, тяжелее", "24 — стандарт", "28 — легче файл"]
# оценка этапов: запуск+анализ (с), мозг (кадр/с), запуск рендера (с), рендер (кадр/с);
# после каждого готового ролика уточняется по замерам и хранится в настройках — отдельно по формату
TIMING = dict(boot1=20.0, r1=300.0, boot2=5.5, fps2=80.0)
TIMING_STREAMER = dict(boot1=20.0, r1=300.0, boot2=8.0, fps2=60.0)   # мозг тот же; рендер комнаты — свой
FORMAT_CARDS = (("base", "Базовый", "муха на шаре у монитора, как раньше"),
                ("streamer", "Стример", "кресло, наушники, комната стримера"))
ACTIVE = ("подготовка", "мозг", "рендер", "ждёт")             # статусы «ролик в работе»
EDITABLE = ("в очереди", "ошибка", "остановлено")             # эти ролики рендерятся по «Начать»
ERR_RE = re.compile(r"^\s*(Traceback \(most recent call last\)|[\w.]*(Error|Exception)(:|$))")
# комната варианта словами (поля style.RoomStyle) — для журнала
ROOM_RU = dict(chair={"racer": "гоночное кресло", "office": "офисное кресло", "egg": "кресло-яйцо",
                      "throne": "кресло-трон"},
               headphones={"over_ear": "большие наушники", "cat_ears": "наушники с ушками",
                           "studio": "студийные наушники", "headset_mic": "гарнитура с микрофоном"},
               mood={"night_neon": "ночной неон", "warm_cozy": "тёплый свет", "cool_studio": "студийный свет",
                     "sunset": "закатный свет"})

# цвета тёмной темы
BG, PANEL, FG, MUTED, ACCENT, OK, ERR = "#12131a", "#1b1d27", "#e8e8ef", "#8a8ca0", "#b46cff", "#6cffb0", "#ff6c7a"
PANEL2, LINE, SEL, HOVER = "#262a3a", "#30344a", "#3b2d57", "#323650"
CARD_ON, DIM, WAIT, RING = "#231d36", "#555868", "#ffcf6c", "#5b6080"   # выбранная карточка, выключенное, «ждёт», кружок


def load_settings():
    try:
        return {**DEFAULTS, **json.loads(SETTINGS.read_text(encoding="utf8"))}
    except Exception:
        return dict(DEFAULTS)


def save_settings(d):
    try:
        SETTINGS.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf8")
    except Exception:
        pass


def next_fly_no():
    try:
        return json.loads(COUNTER.read_text())["n"] + 1
    except Exception:
        return 1


def take_fly_no(n):
    # номер мухи тратится только на готовый полный ролик (счётчик назад не откатываем)
    try:
        COUNTER.parent.mkdir(parents=True, exist_ok=True)
        COUNTER.write_text(json.dumps({"n": max(n, next_fly_no() - 1)}))
    except Exception:
        pass


def rm_file(p, tries=20):
    # после taskkill файл может быть ещё занят умирающим ffmpeg — пробуем несколько раз
    for _ in range(tries):
        try:
            Path(p).unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            time.sleep(0.25)
    return False


def job_object():
    """Windows Job «убить всех при закрытии»: если окно упадёт или его снимут в Диспетчере задач,
    рендер (react.py, воркеры, ffmpeg) умрёт вместе с ним, а не останется грузить видеокарту."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes as W

        class Basic(ctypes.Structure):                        # JOBOBJECT_BASIC_LIMIT_INFORMATION
            _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64), ("LimitFlags", W.DWORD),
                        ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("e", W.DWORD), ("f", ctypes.c_size_t),
                        ("g", W.DWORD), ("h", W.DWORD)]

        class Ext(ctypes.Structure):                          # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
            _fields_ = [("Basic", Basic), ("Io", ctypes.c_uint64 * 6), ("m", ctypes.c_size_t * 4)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = W.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, W.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD]
        job = k32.CreateJobObjectW(None, None)
        info = Ext()
        info.Basic.LimitFlags = 0x2000                    # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if job and k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            return job
    except Exception:
        pass
    return None


def job_start(job, proc):
    """Процесс запущен приостановленным: кладём в job и отпускаем (всегда, даже если job не вышел)."""
    import ctypes
    from ctypes import wintypes as W
    h = W.HANDLE(int(proc._handle))
    try:
        k32 = ctypes.WinDLL("kernel32")
        k32.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
        k32.AssignProcessToJobObject(job, h)
    finally:
        nt = ctypes.WinDLL("ntdll")
        nt.NtResumeProcess.argtypes = [W.HANDLE]
        if nt.NtResumeProcess(h) != 0:
            raise OSError("не удалось запустить приостановленный процесс")


def open_path(p):
    try:
        os.startfile(str(p))                                  # Windows: открыть файл/папку
    except Exception as e:
        messagebox.showerror("Не открывается", str(e))


def show_in_folder(p):
    try:
        subprocess.Popen(f'explorer /select,"{Path(p)}"')    # проводник с выделенным файлом
    except Exception:
        open_path(Path(p).parent)


def clip_secs(path):
    """Длина ролика, с (None — не прочиталась)."""
    try:
        import av
        with av.open(str(path)) as c:
            if c.duration:
                return c.duration / 1e6
    except Exception:
        pass
    return None


def clip_frames(path, fps=30):
    """Сколько кадров выйдет из ролика — для оценки времени до первых замеров."""
    s = clip_secs(path)
    return int(s * fps) if s else 90 * fps


def norm_variant(v):
    """Вариант комнаты из настроек: 'random' или int 0..999999 (кривое -> 'random')."""
    try:
        n = int(str(v).strip())
    except (TypeError, ValueError):
        return "random"
    return n if 0 <= n <= FM.VARIANT_MAX else "random"


def plan_job(video, cfg, n, variant=None, title=""):
    """Команда react.py и имена файлов одного ролика: (cmd, final, part).
    cfg — настройки окна (App._settings_now); variant — уже разрешённый номер комнаты
    (formats.resolve_variant, только Стример), чтобы он попал в имя файла. Имя — как в react.py."""
    video = Path(video)
    prev = cfg["preview_secs"] > 0
    fmt = FM.fmt_cfg(cfg.get("format", "base"), variant if cfg.get("format") == "streamer" else None,
                     flyeye=cfg.get("flyeye", False))
    tag = FM.name_tag(fmt)                                    # "" у Базового — имя как раньше
    final = OUT_DIR / f"{'проба' if prev else 'муха'}_{n:03d}_{tag + '_' if tag else ''}{video.stem[:40].strip(" .")}.mp4"
    part = final.with_name(final.stem + ".part.mp4")          # пока не готово — только .part
    # «ключ=значение»: заголовок/вотермарка с «-» в начале не спутаются с ключом
    cmd = [str(PY), "-W", "ignore", str(REACT), str(video), "--watermark=" + (WATERMARK_LOCK or cfg["watermark"] or " "),
           "--crf", cfg["crf"].split()[0], "--workers", str(cfg["workers"]), "--fly_no", str(n),
           "--out", str(part), "--format", fmt["format"], "--flyeye", "on" if fmt["flyeye"] else "off",
           "--lock", LOCK_LABEL, "--seizure", (((cfg.get("seizure_t") or "auto") if cfg.get("seizure") else "off") if SHOW_SEIZURE else "auto")]
    if fmt["variant"] is not None:
        cmd += ["--variant", str(fmt["variant"])]
    if title:
        cmd.append("--title=" + title)
    if prev:
        cmd += ["--secs", str(cfg["preview_secs"])]
        if cfg.get("preview_t0", 0) > 0:
            cmd += ["--t0", str(cfg["preview_t0"])]
    return cmd, final, part


def room_text(n):
    """Какая комната у номера варианта (style.RoomStyle), словами; не вышло — ''."""
    try:
        import style
        s = style.RoomStyle(int(n))
        ch, hp, mood = s.chair.get("model"), s.headphones.get("model"), s.light.get("mood")
        parts = [ROOM_RU["chair"].get(ch, ch), ROOM_RU["headphones"].get(hp, hp), ROOM_RU["mood"].get(mood, mood)]
        return ", ".join(str(p) for p in parts if p)
    except Exception:
        return ""


def drop_own_lock(pid):
    """Рендер окна убит «Стопом» и замок не снял: снимаем его замок (метка окна + его pid) сразу,
    чтобы следующий ролик не ждал. Чужие замки не трогаем."""
    try:
        txt = FM.LOCK_PATH.read_text(encoding="utf-8", errors="replace")
        if txt.startswith(LOCK_LABEL + " ") and re.search(rf"\bpid={int(pid)}\b", txt):
            FM.LOCK_PATH.unlink()
    except (OSError, ValueError, TypeError):
        pass


def fmt_s(s):
    s = max(0, int(round(s)))
    return f"{s // 60} мин {s % 60:02d} с" if s >= 60 else f"{s} с"


def short(t, n=48):
    return t if len(t) <= n else t[:n - 1] + "…"


def default_title(path):
    return Path(path).stem.replace("_", " ")                 # как react.py без --title


# ---------------------------------------------------------------- превью вотермарки
_WM_FONTS = {}


def wm_font(size):
    f = _WM_FONTS.get(size)
    if f is None:
        try:
            f = ImageFont.truetype(WM_FONT, size)
            try:
                f.set_variation_by_name("Bold")
            except Exception:
                pass
        except Exception:
            f = ImageFont.load_default()
        _WM_FONTS[size] = f
    return f


def wm_preview_image(text, w, h):
    """Правый нижний угол экрана монитора с вотермаркой — как рисует stage.Stage._watermark
    (холст 960x540, Bahnschrift Bold 8% высоты, белый 65% + мягкая тень). Угол берём такой ширины,
    чтобы ник влез целиком. Возвращает (картинка w x h, шрифт пришлось уменьшить?)."""
    W, H = 960, 540
    text = (text or "").strip()
    size = max(14, int(0.08 * min(W, H)))
    f = wm_font(size)
    meas = ImageDraw.Draw(Image.new("L", (1, 1)))
    tl = meas.textlength(text, font=f) if text else 0.0
    room = W * 0.9 - int(size * 0.35) - 8
    small = bool(text) and tl > room
    if small:                                                 # длинный ник — мельче шрифт, как в stage.py
        size = max(8, int(size * room / tl))
        f = wm_font(size)
        tl = meas.textlength(text, font=f)
    tw, th, pad = int(tl) + 8, int(size * 1.35), int(size * 0.35)
    # под знаком — тёмный «кадр ролика»
    scr = ImageOps.colorize(Image.linear_gradient("L").resize((W, H)), "#34405e", "#141019")
    if text:
        a = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(a).text((4, 0), text, font=f, fill=255)
        sh = a.filter(ImageFilter.GaussianBlur(max(1.0, size * 0.06)))
        x0, y0 = max(0, W - tw - pad), max(0, H - th - pad // 2)
        roi = scr.crop((x0, y0, x0 + tw, y0 + th))
        roi = Image.composite(Image.new("RGB", roi.size, 0), roi, sh.point(lambda v: v * 45 // 100))
        roi = Image.composite(Image.new("RGB", roi.size, (255, 255, 255)), roi, a.point(lambda v: v * 65 // 100))
        scr.paste(roi, (x0, y0))
    bz = max(3, h // 14)                                      # рамка монитора справа и снизу
    vw, vh = w - bz, h - bz
    cw = int(min(W, max(W * 0.5, tw + pad + 90)))
    ch = min(H, int(cw * vh / vw))
    img = Image.new("RGB", (w, h), "#07080b")
    img.paste(scr.crop((W - cw, H - ch, W, H)).resize((vw, vh), Image.LANCZOS), (0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, w - 1, h - 1), outline="#2a2d3a")
    d.ellipse((w - bz - 14, h - bz // 2 - 2, w - bz - 10, h - bz // 2 + 2), fill="#3fd9e6")   # огонёк
    return img, small


def work_area(root):
    """Рабочая область экрана (без панели задач): x0, y0, x1, y1 в пикселях экрана."""
    try:
        import ctypes
        from ctypes import wintypes as W
        rc = W.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(rc), 0):   # SPI_GETWORKAREA
            return rc.left, rc.top, rc.right, rc.bottom
    except Exception:
        pass
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight() - 40


def dark_titlebar(root):
    # Windows 10/11: тёмный заголовок окна под тёмную тему (не вышло — не страшно)
    try:
        import ctypes
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        on = ctypes.c_int(1)
        for attr in (20, 19):                                 # DWMWA_USE_IMMERSIVE_DARK_MODE (новые/старые сборки)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(on), 4) == 0:
                break
        col = ctypes.c_int(int(BG[5:7] + BG[3:5] + BG[1:3], 16))   # COLORREF = 0x00BBGGRR
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(col), 4)   # цвет заголовка (Win11)
    except Exception:
        pass


class Job:
    def __init__(self, path):
        self.path = Path(path)
        self.title = default_title(self.path)
        self.status = "в очереди"
        self.out = None
        self.target = None                                    # имя файла, в который сейчас рендерим
        self.removed = False                                  # убран из списка — поток его пропустит
        self.variant = None                                   # номер комнаты последнего рендера (Стример)


class App:
    def __init__(self, root):
        self.root = root
        self.k = max(1.0, root.winfo_fpixels("1i") / 96.0)    # масштаб Windows (125% -> 1.25)
        self.cfg = load_settings()
        if self.cfg.get("crf") not in QUALITY:
            self.cfg["crf"] = DEFAULTS["crf"]
        if self.cfg.get("format") not in FM.FORMATS:
            self.cfg["format"] = DEFAULTS["format"]
        self.cfg["variant"] = norm_variant(self.cfg.get("variant"))
        t, ts = self.cfg.get("timing"), self.cfg.get("timing_streamer")
        base = {**TIMING, **(t if isinstance(t, dict) else {})}
        # замеры по формату; мозг у Стримера тот же — до первых его замеров берём мозг Базового
        strm = {**TIMING_STREAMER, "boot1": base["boot1"], "r1": base["r1"], **(ts if isinstance(ts, dict) else {})}
        self.timings = dict(base=base, streamer=strm)
        self.timing = base                                    # замеры формата текущего ролика
        self.jobs = []
        self.proc = None
        self.q = queue.Queue()
        self.running = False
        self.stop_flag = False
        self.lock = threading.Lock()                          # «Стоп» против запуска процесса
        self.cur = None                                       # ролик, чьи строки сейчас разбирает окно
        self.cur_run = None                                   # ролик, который сейчас в работе у потока
        self.job = job_object()
        self.worker = None
        self.last_out = None
        self.ph = {}                                          # замеры текущего ролика (прогресс, ETA)
        self.run_stats = dict(ok=0, err=0)
        self._t_tick = 0.0
        self._wm_after = None
        self._wm_img = None
        self._copy_after = None
        self._build()
        self._clean_parts()
        self._poll_id = self.root.after(100, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.report_callback_exception = self._tk_error

    def px(self, v):
        return int(round(v * self.k))

    # ------------------------------------------------------------ интерфейс
    def _style(self):
        r, px = self.root, self.px
        st = ttk.Style(r)
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=FG, fieldbackground=PANEL, font=("Segoe UI", 10),
                     bordercolor=LINE, lightcolor=PANEL, darkcolor=PANEL, troughcolor=BG,
                     selectbackground=SEL, selectforeground=FG, insertcolor=FG, focuscolor=ACCENT)
        st.configure("TFrame", background=BG)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Muted.TLabel", foreground=MUTED, font=("Segoe UI", 9))
        st.configure("Head.TLabel", font=("Segoe UI Semibold", 15))
        st.configure("Sec.TLabel", foreground=ACCENT, font=("Segoe UI Semibold", 11))
        st.configure("Stage.TLabel", font=("Segoe UI Semibold", 10))
        st.configure("Empty.TLabel", background=PANEL, foreground=MUTED)
        st.configure("TButton", background=PANEL2, foreground=FG, bordercolor=PANEL2, lightcolor=PANEL2,
                     darkcolor=PANEL2, padding=(px(12), px(6)))
        st.map("TButton", background=[("disabled", "#181a22"), ("pressed", SEL), ("active", HOVER)],
               foreground=[("disabled", "#555868")], bordercolor=[("focus", ACCENT)],
               lightcolor=[("pressed", SEL), ("active", HOVER)], darkcolor=[("pressed", SEL), ("active", HOVER)])
        st.configure("Go.TButton", background=ACCENT, foreground="#0b0b10", bordercolor=ACCENT, lightcolor=ACCENT,
                     darkcolor=ACCENT, font=("Segoe UI Semibold", 11), padding=(px(18), px(8)))
        st.map("Go.TButton", background=[("disabled", "#3a3050"), ("pressed", "#9d52ea"), ("active", "#c992ff")],
               foreground=[("disabled", "#7d7590")], bordercolor=[("disabled", "#3a3050"), ("focus", "#e6d2ff")],
               lightcolor=[("disabled", "#3a3050"), ("active", "#c992ff")],
               darkcolor=[("disabled", "#3a3050"), ("active", "#c992ff")])
        st.configure("Small.TButton", padding=(px(10), px(4)))
        st.configure("TEntry", fieldbackground=PANEL, foreground=FG, padding=px(5))
        st.map("TEntry", bordercolor=[("focus", ACCENT)], fieldbackground=[("disabled", BG)],
               foreground=[("disabled", MUTED)])
        st.configure("TCombobox", fieldbackground=PANEL, background=PANEL2, foreground=FG, arrowcolor=FG,
                     padding=px(4), arrowsize=px(12))
        st.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", FG)],
               selectbackground=[("readonly", PANEL)], selectforeground=[("readonly", FG)],
               background=[("active", HOVER), ("readonly", PANEL2)], bordercolor=[("focus", ACCENT)])
        st.configure("TSpinbox", fieldbackground=PANEL, background=PANEL2, foreground=FG, arrowcolor=FG,
                     padding=px(4), arrowsize=px(11))
        st.map("TSpinbox", background=[("active", HOVER)], bordercolor=[("focus", ACCENT)])
        st.configure("Vertical.TScrollbar", background=PANEL2, troughcolor=BG, bordercolor=BG, lightcolor=PANEL2,
                     darkcolor=PANEL2, arrowcolor=MUTED, gripcount=0, arrowsize=px(13))
        st.map("Vertical.TScrollbar", background=[("active", HOVER)], arrowcolor=[("active", FG)])
        st.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=PANEL, bordercolor=PANEL,
                     lightcolor=ACCENT, darkcolor=ACCENT, thickness=px(12))
        # переключатели: обычные (на фоне окна) и в карточках формата (фон карточки, выбранная — подсвечена)
        ind = dict(indicatorbackground=PANEL2, indicatorforeground=ACCENT, upperbordercolor=RING,
                   lowerbordercolor=RING, indicatormargin=(0, 0, px(6), 0), indicatorsize=px(12))
        ind_map = dict(indicatorbackground=[("disabled", BG), ("pressed", SEL), ("active", HOVER)],
                       upperbordercolor=[("disabled", PANEL2), ("selected", ACCENT)],
                       lowerbordercolor=[("disabled", PANEL2), ("selected", ACCENT)],
                       indicatorforeground=[("disabled", DIM)], foreground=[("disabled", DIM)])
        for w in ("TRadiobutton", "TCheckbutton"):
            st.configure(w, background=BG, foreground=FG, **ind)
            st.map(w, background=[("active", BG)], **ind_map)
            st.configure("Card." + w, background=PANEL, foreground=FG, font=("Segoe UI Semibold", 10), **ind)
            st.map("Card." + w, background=[("selected", CARD_ON), ("active", PANEL)], **ind_map)
        self._check_images(st)
        st.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG, rowheight=px(26),
                     bordercolor=LINE, lightcolor=PANEL, darkcolor=PANEL)
        st.map("Treeview", background=[("selected", SEL)], foreground=[("selected", FG)])
        st.configure("Treeview.Heading", background="#23263a", foreground=MUTED, bordercolor=LINE,
                     lightcolor="#23263a", darkcolor="#23263a", relief="flat", padding=(px(6), px(3)))
        st.map("Treeview.Heading", background=[("active", "#2a2d40")])
        # выпадающий список Combobox — обычный tk Listbox, красим через базу опций
        for k, v in (("background", PANEL), ("foreground", FG), ("selectBackground", SEL),
                     ("selectForeground", FG), ("font", "{Segoe UI} 10"), ("relief", "flat")):
            r.option_add(f"*TCombobox*Listbox.{k}", v)

    def _check_images(self, st):
        """Флажок карточки — галочка (у темы clam вместо неё крестик, похожий на «закрыть»). Без PIL — как в теме."""
        if Image is None:
            return
        try:
            n, s = self.px(13), 4                             # рисуем в 4 раза крупнее и уменьшаем — сглаживание

            def box(fill, edge, tick=None):
                im = Image.new("RGBA", (n * s, n * s), (0, 0, 0, 0))
                d = ImageDraw.Draw(im)
                d.rounded_rectangle((s, s, n * s - s - 1, n * s - s - 1), radius=3 * s, fill=fill, outline=edge,
                                    width=max(s, int(1.2 * s)))
                if tick:
                    d.line([(0.27 * n * s, 0.52 * n * s), (0.43 * n * s, 0.68 * n * s), (0.74 * n * s, 0.34 * n * s)],
                           fill=tick, width=int(1.9 * s), joint="curve")
                return ImageTk.PhotoImage(im.resize((n, n), Image.LANCZOS), master=self.root)

            self._imgs = dict(off=box(PANEL2, RING), on=box(ACCENT, ACCENT, "#0b0b10"),
                              off_dis=box(BG, PANEL2), on_dis=box(DIM, DIM, BG))
            i = self._imgs
            st.element_create("Fly.check", "image", i["off"], ("selected", "disabled", i["on_dis"]),
                              ("selected", i["on"]), ("disabled", i["off_dis"]), width=n + self.px(7),
                              sticky="w")                   # ширина с запасом = зазор до подписи
            st.layout("Card.TCheckbutton", [("Checkbutton.padding", {"sticky": "nswe", "children": [
                ("Fly.check", {"side": "left", "sticky": ""}),
                ("Checkbutton.focus", {"side": "left", "sticky": "w", "children": [
                    ("Checkbutton.label", {"sticky": "nswe"})]})]})])
        except Exception:
            pass                                              # не вышло — остаётся флажок темы

    def _build(self):
        r, px = self.root, self.px
        r.title(APP_TITLE)
        w, h = px(940), px(800)                               # окно по центру, с учётом масштаба Windows
        x0, y0, sw, sh = work_area(r)                         # без панели задач
        w, h = min(w, sw - x0 - 40), min(h, sh - y0 - px(40))  # px(40) — заголовок окна
        r.geometry(f"{w}x{h}+{x0 + max(0, (sw - x0 - w) // 2)}+{y0 + max(0, (sh - y0 - h - px(32)) // 2)}")
        r.minsize(min(px(900), w), min(px(740), h))
        r.configure(bg=BG)
        self._style()
        r.columnconfigure(0, weight=1)
        pad = px(16)

        top = ttk.Frame(r, padding=(pad, px(12), pad, px(2)))
        top.grid(row=0, sticky="ew")
        ttk.Label(top, text="Муха смотрит ролики", style="Head.TLabel").pack(side="left")
        self.lbl_fly = ttk.Label(top, text="", style="Muted.TLabel")
        self.lbl_fly.pack(side="right", anchor="s")

        # ---- очередь роликов
        ttk.Label(r, text="Ролики", style="Sec.TLabel", padding=(pad, px(6), 0, px(2))).grid(row=1, sticky="w")
        tf = ttk.Frame(r, padding=(pad, 0, pad, 0))
        tf.grid(row=2, sticky="nsew")
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(tf, columns=("title", "status", "file"), show="headings", height=5,
                                 selectmode="browse")
        self.tree.heading("title", text="Ролик / заголовок в видео", anchor="w")
        self.tree.heading("status", text="Статус")
        self.tree.heading("file", text="Файл результата", anchor="w")
        self.tree.column("title", width=px(330), minwidth=px(150), stretch=True)
        self.tree.column("status", width=px(120), minwidth=px(100), anchor="center", stretch=False)
        self.tree.column("file", width=px(270), minwidth=px(120), stretch=True)
        self.tree.tag_configure("ok", foreground=OK)
        self.tree.tag_configure("err", foreground=ERR)
        self.tree.tag_configure("run", foreground="#dcc2ff")
        self.tree.tag_configure("stop", foreground=MUTED)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", self._on_double)
        self.tree.bind("<Return>", self._on_double)
        self.tree.bind("<Delete>", lambda e: self._remove())
        self.tree.bind("<Button-3>", self._menu_popup)
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        self.lbl_empty = ttk.Label(tf, style="Empty.TLabel", justify="center",
                                   text="Список пуст.\nДобавь ролики кнопкой «Добавить ролики…»\n"
                                        "или возьми всё из папки «видео на реакцию».")
        self.menu = tk.Menu(r, tearoff=0, bg=PANEL2, fg=FG, activebackground=SEL, activeforeground=FG,
                            disabledforeground="#5b5e70", bd=0, relief="flat", font=("Segoe UI", 10))

        bar = ttk.Frame(r, padding=(pad, px(6), pad, 0))
        bar.grid(row=3, sticky="ew")
        ttk.Button(bar, text="Добавить ролики…", command=self._add_files).pack(side="left")
        ttk.Button(bar, text="Всё из «видео на реакцию»", command=self._add_folder).pack(side="left", padx=px(6))
        ttk.Button(bar, text="Убрать", command=self._remove).pack(side="left")
        ttk.Button(bar, text="Открыть папку «готово»", command=lambda: (OUT_DIR.mkdir(exist_ok=True), open_path(OUT_DIR))
                   ).pack(side="right")

        tl = ttk.Frame(r, padding=(pad, px(8), pad, 0))
        tl.grid(row=4, sticky="ew")
        ttk.Label(tl, text="Заголовок выбранного ролика:").pack(side="left")
        self.var_title = tk.StringVar()
        self.ent_title = ttk.Entry(tl, textvariable=self.var_title, state="disabled")
        self.ent_title.pack(side="left", fill="x", expand=True, padx=(px(8), 0))
        self.var_title.trace_add("write", self._on_title_edit)

        # ---- настройки: слева поля, справа превью вотермарки
        ttk.Label(r, text="Настройки", style="Sec.TLabel", padding=(pad, px(10), 0, px(2))).grid(row=5, sticky="w")
        sf = ttk.Frame(r, padding=(pad, 0, pad, 0))
        sf.grid(row=6, sticky="ew")
        sf.columnconfigure(0, weight=1)
        form = ttk.Frame(sf)
        form.grid(row=0, column=0, sticky="nw")
        only_digits = (r.register(lambda s: s.isdigit() and len(s) <= 3 or s == ""), "%P")

        def row(i, text):
            ttk.Label(form, text=text).grid(row=i, column=0, sticky="w", pady=px(3))
            cell = ttk.Frame(form)
            cell.grid(row=i, column=1, sticky="w", padx=(px(8), 0))
            return cell

        def hint(cell, text):
            ttk.Label(cell, text=text, style="Muted.TLabel").pack(side="left", padx=(px(8), 0))

        c = row(0, "Вотермарка на мониторе:")
        self.var_wm = tk.StringVar(value=WATERMARK_LOCK or self.cfg["watermark"])
        ttk.Entry(c, textvariable=self.var_wm, width=24, state="disabled" if WATERMARK_LOCK else "normal").pack(side="left")
        hint(c, "ник TikTok")
        c = row(1, "Качество файла:")
        self.var_q = tk.StringVar(value=self.cfg["crf"])
        ttk.Combobox(c, textvariable=self.var_q, values=QUALITY, state="readonly", width=22).pack(side="left")
        c = row(2, "Процессов рендера:")
        self.var_w = tk.IntVar(value=self._cfg_int("workers", 5))
        ttk.Spinbox(c, from_=1, to=8, textvariable=self.var_w, width=5, validate="key",
                    validatecommand=only_digits).pack(side="left")
        hint(c, "5 — оптимум для этой видеокарты")
        c = row(3, "Пробный прогон, секунд:")
        self.var_prev = tk.IntVar(value=self._cfg_int("preview_secs", 0))
        ttk.Spinbox(c, from_=0, to=600, increment=5, textvariable=self.var_prev, width=5, validate="key",
                    validatecommand=only_digits).pack(side="left")
        self.lbl_t0 = ttk.Label(c, text="с секунды")
        self.lbl_t0.pack(side="left", padx=(px(8), px(6)))
        self.var_t0 = tk.IntVar(value=self._cfg_int("preview_t0", 0))
        four_digits = (r.register(lambda s: s.isdigit() and len(s) <= 4 or s == ""), "%P")
        self.sp_t0 = ttk.Spinbox(c, from_=0, to=3600, increment=5, textvariable=self.var_t0, width=5,
                                 validate="key", validatecommand=four_digits)
        self.sp_t0.pack(side="left")
        hint(c, "0 — весь ролик; проба без номера мухи")
        self.var_prev.trace_add("write", lambda *_: self._sync_t0())

        pv = ttk.Frame(sf)
        pv.grid(row=0, column=1, sticky="ne", padx=(px(16), 0), pady=(px(3), 0))
        self.var_wm_note = tk.StringVar(value="")
        ttk.Label(pv, textvariable=self.var_wm_note, style="Muted.TLabel").pack(anchor="w")
        self.wm_view = tk.Label(pv, bd=0, bg=BG, highlightthickness=0)
        self.wm_view.pack(anchor="w", pady=(px(2), 0))
        if Image is None:
            pv.grid_remove()
        self.var_wm.trace_add("write", self._wm_changed)

        # ---- формат ролика: карточки «Базовый» / «Стример», врезка «как видит муха», вариант комнаты
        ttk.Label(r, text="Формат ролика", style="Sec.TLabel", padding=(pad, px(10), 0, px(4))).grid(row=7, sticky="w")
        ff = ttk.Frame(r, padding=(pad, 0, pad, 0))
        ff.grid(row=8, sticky="ew")
        self.var_fmt = tk.StringVar(value=self.cfg["format"])
        self.var_eye = tk.BooleanVar(value=bool(self.cfg.get("flyeye")))
        self.var_seiz = tk.BooleanVar(value=bool(self.cfg.get("seizure")))
        self.cards = {}
        cards = ttk.Frame(ff)
        cards.pack(fill="x")
        for i, (key, name, desc) in enumerate(FORMAT_CARDS):
            self.cards[key] = self._card(cards, name, desc, ttk.Radiobutton, variable=self.var_fmt, value=key)
            self.cards[key]["frame"].grid(row=0, column=i, sticky="nsew", padx=(0, px(8)))
        cards.columnconfigure(2, minsize=px(14))              # зазор: формат — выбор из двух, врезка — отдельно
        self.cards["flyeye"] = self._card(cards, "Как видит муха", "врезка: ролик глазами мухи",
                                          ttk.Checkbutton, variable=self.var_eye)
        self.cards["flyeye"]["frame"].grid(row=0, column=3, sticky="nsew", padx=(0, px(8)))
        self.var_seiz_t = tk.StringVar(value=str(self.cfg.get("seizure_t") or ""))
        if SHOW_SEIZURE:                                      # спрятано по умолчанию (см. SHOW_SEIZURE вверху)
            self.cards["seizure"] = self._card(cards, "Приступ", "в грустный момент: смерть и новая муха",
                                               ttk.Checkbutton, variable=self.var_seiz)
            self.cards["seizure"]["frame"].grid(row=0, column=4, sticky="nsew")
            sr = ttk.Frame(ff)
            sr.pack(fill="x", pady=(px(6), 0))
            ttk.Label(sr, text="Приступ на секундах:").pack(side="left")
            ttk.Entry(sr, textvariable=self.var_seiz_t, width=16).pack(side="left", padx=(px(8), px(8)))
            ttk.Label(sr, text="можно несколько: 30, 75. Пусто — сам: по грустным словам, а без слов — на самом резком басе",
                      foreground=MUTED).pack(side="left")
        for col in ((0, 1, 3, 4) if SHOW_SEIZURE else (0, 1, 3)):
            cards.columnconfigure(col, weight=1, uniform="card")

        vr = ttk.Frame(ff)
        vr.pack(fill="x", pady=(px(8), 0))
        self.lbl_var = ttk.Label(vr, text="Вариант комнаты:")
        self.lbl_var.pack(side="left")
        self.var_vmode = tk.StringVar(value="random" if self.cfg["variant"] == "random" else "fixed")
        self.var_vnum = tk.StringVar(value="" if self.cfg["variant"] == "random" else str(self.cfg["variant"]))
        self.rb_vrand = ttk.Radiobutton(vr, text="случайный", value="random", variable=self.var_vmode)
        self.rb_vrand.pack(side="left", padx=(px(10), px(12)))
        self.rb_vnum = ttk.Radiobutton(vr, text="номер", value="fixed", variable=self.var_vmode)
        self.rb_vnum.pack(side="left")
        six_digits = (r.register(lambda s: s.isdigit() and len(s) <= 6 or s == ""), "%P")
        self.ent_vnum = ttk.Entry(vr, textvariable=self.var_vnum, width=8, validate="key", validatecommand=six_digits)
        self.ent_vnum.pack(side="left", padx=(px(6), 0))
        self.var_vnote = tk.StringVar()
        ttk.Label(vr, textvariable=self.var_vnote, style="Muted.TLabel").pack(side="left", padx=(px(10), 0))
        self.btn_vrep = ttk.Button(vr, text="Повторить", style="Small.TButton", command=self._repeat_variant)
        self.btn_vrep.pack(side="right")
        self.btn_vcopy = ttk.Button(vr, text="Копировать", style="Small.TButton", command=self._copy_variant)
        self.btn_vcopy.pack(side="right", padx=(0, px(6)))
        self.var_vlast = tk.StringVar()
        self.lbl_vlast = ttk.Label(vr, textvariable=self.var_vlast, font=("Segoe UI Semibold", 10))
        self.lbl_vlast.pack(side="right", padx=(0, px(10)))
        ttk.Label(vr, text="последний:", style="Muted.TLabel").pack(side="right", padx=(0, px(4)))
        self.var_vnum.trace_add("write", self._vnum_typed)
        for v in (self.var_fmt, self.var_eye, self.var_seiz, self.var_vmode):
            v.trace_add("write", lambda *_: self._sync_format())

        # ---- запуск
        run = ttk.Frame(r, padding=(pad, px(12), pad, px(4)))
        run.grid(row=9, sticky="ew")
        self.btn_go = ttk.Button(run, text="Начать рендер", style="Go.TButton", command=self._start)
        self.btn_go.pack(side="left")
        self.btn_stop = ttk.Button(run, text="Стоп", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=px(8))
        self.btn_open = ttk.Button(run, text="Открыть готовое видео", command=self._open_last, state="disabled")
        self.btn_open.pack(side="right")

        pf = ttk.Frame(r, padding=(pad, px(2), pad, 0))
        pf.grid(row=10, sticky="ew")
        self.var_stage = tk.StringVar(value="Готово к работе")
        ttk.Label(pf, textvariable=self.var_stage, style="Stage.TLabel").pack(anchor="w")
        self.pb = ttk.Progressbar(pf, maximum=1000)
        self.pb.pack(fill="x", pady=(px(4), px(2)))
        er = ttk.Frame(pf)
        er.pack(fill="x")
        self.var_eta = tk.StringVar(value="")
        ttk.Label(er, textvariable=self.var_eta, style="Muted.TLabel").pack(side="left")
        self.var_pct = tk.StringVar(value="")
        ttk.Label(er, textvariable=self.var_pct, style="Muted.TLabel").pack(side="right")

        lf = ttk.Frame(r, padding=(pad, px(4), pad, px(14)))
        lf.grid(row=11, sticky="nsew")
        lf.columnconfigure(0, weight=1)
        lf.rowconfigure(0, weight=1)
        self.log = tk.Text(lf, height=5, bg="#0d0e13", fg="#b9bbcc", insertbackground=FG, relief="flat", bd=0,
                           highlightthickness=0, selectbackground=SEL, font=("Consolas", 9), wrap="word",
                           padx=px(6), pady=px(4), takefocus=0, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        lsb = ttk.Scrollbar(lf, orient="vertical", command=self.log.yview)
        lsb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=lsb.set)
        self.log.tag_configure("ok", foreground=OK)
        self.log.tag_configure("err", foreground=ERR)
        self.log.tag_configure("wait", foreground=WAIT)
        self.log.tag_configure("info", foreground="#dcc2ff")

        # при нехватке места ужимаются список и журнал, а не кнопки и настройки
        r.rowconfigure(2, weight=3, minsize=px(26) * 4)
        r.rowconfigure(11, weight=2, minsize=px(60))
        r.bind("<Control-o>", lambda e: self._add_files())
        self._refresh_fly_no()
        self._empty_hint()
        self._draw_wm()
        self._sync_t0()
        self._show_last_variant()
        self._sync_format()
        dark_titlebar(r)

    def _cfg_int(self, key, dflt):
        try:
            return int(self.cfg[key])
        except Exception:
            return dflt

    def _sync_t0(self):
        # «с секунды» — только для пробы (весь ролик всегда с начала)
        try:
            on = int(self.var_prev.get()) > 0
        except (tk.TclError, ValueError):
            on = False
        self.sp_t0.configure(state="normal" if on else "disabled")
        self.lbl_t0.configure(foreground=FG if on else DIM)

    # ------------------------------------------------------------ формат ролика
    def _card(self, parent, title, desc, kind, **opts):
        """Карточка выбора: переключатель + строка пояснения; клик по всей карточке = клик по переключателю."""
        px = self.px
        f = tk.Frame(parent, bg=PANEL, highlightthickness=max(1, px(1)), highlightbackground=LINE,
                     highlightcolor=LINE, cursor="hand2")
        b = kind(f, text=title, style=f"Card.T{kind.__name__}", **opts)   # Card.TRadiobutton / Card.TCheckbutton
        b.pack(anchor="w", padx=(px(10), px(10)), pady=(px(6), 0))
        lb = tk.Label(f, text=desc, bg=PANEL, fg=MUTED, font=("Segoe UI", 9), anchor="w", justify="left", cursor="hand2")
        lb.pack(anchor="w", fill="x", padx=(px(30), px(10)), pady=(0, px(7)))   # под текстом, не под кружком
        for w in (f, lb):
            w.bind("<Button-1>", lambda e: b.invoke())
        # узкое окно: пояснение переносится, а не обрезается
        f.bind("<Configure>", lambda e: lb.configure(wraplength=max(px(60), e.width - px(44))))
        return dict(frame=f, btn=b, lb=lb)

    def _sync_format(self):
        """Подсветка выбранных карточек; строка «Вариант комнаты» активна только у Стримера."""
        fmt = self.var_fmt.get()
        if fmt != getattr(self, "_fmt_drawn", fmt):           # у Стримера ник — в углу ролика, а не монитора
            self._wm_changed()
        self._fmt_drawn = fmt
        for key, c in self.cards.items():
            on = (bool(self.var_eye.get()) if key == "flyeye" else bool(self.var_seiz.get()) if key == "seizure"
                  else fmt == key)
            bg = CARD_ON if on else PANEL
            c["frame"].configure(bg=bg, highlightbackground=ACCENT if on else LINE, highlightcolor=ACCENT if on else LINE)
            c["lb"].configure(bg=bg, fg="#c9bfe0" if on else MUTED)
        strm = fmt == "streamer"
        st = "normal" if strm else "disabled"
        self.lbl_var.configure(foreground=FG if strm else DIM)
        self.rb_vrand.configure(state=st)
        self.rb_vnum.configure(state=st)
        self.ent_vnum.configure(state=st)
        has = self._last_variant() is not None
        self.lbl_vlast.configure(foreground=(ACCENT if has else FG) if strm else DIM)
        self.btn_vrep.configure(state="normal" if strm and has else "disabled")
        self.btn_vcopy.configure(state="normal" if has else "disabled")
        if not strm:
            self.var_vnote.set("только для «Стримера»")
        elif self.var_vmode.get() == "random":
            self.var_vnote.set("свой у каждого ролика")
        else:
            self.var_vnote.set("один на всю очередь")

    def _last_variant(self):
        v = norm_variant(self.cfg.get("variant_last"))
        return None if v == "random" else v

    def _show_last_variant(self):
        v = self._last_variant()
        self.var_vlast.set("—" if v is None else str(v))

    def _vnum_typed(self, *_):
        # вписал номер — значит, нужен этот номер
        if self.var_vnum.get().strip() and self.var_vmode.get() != "fixed":
            self.var_vmode.set("fixed")

    def _repeat_variant(self):
        v = self._last_variant()
        if v is not None:
            self.var_vnum.set(str(v))
            self.var_vmode.set("fixed")

    def _copy_variant(self):
        v = self._last_variant()
        if v is None:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(str(v))
        self.btn_vcopy.configure(text="Скопировано")
        if self._copy_after:
            self.root.after_cancel(self._copy_after)
        self._copy_after = self.root.after(1400, self._copy_back)

    def _copy_back(self):
        self._copy_after = None
        self.btn_vcopy.configure(text="Копировать")

    def _got_variant(self, j, n):
        # react.py напечатал «ВАРИАНТ: N» — это номер комнаты ролика
        if j is not None:
            j.variant = n
        self.cfg["variant_last"] = n
        self._show_last_variant()
        self._sync_format()
        room = room_text(n)
        self._log(f"ВАРИАНТ: {n}" + (f" — {room}" if room else ""), "info")

    # ------------------------------------------------------------ превью вотермарки
    def _wm_changed(self, *_):
        if self._wm_after:
            self.root.after_cancel(self._wm_after)
        self._wm_after = self.root.after(120, self._draw_wm)

    def _draw_wm(self):
        self._wm_after = None
        if Image is None:
            return
        text = self.var_wm.get()
        try:
            img, small = wm_preview_image(text, self.px(240), self.px(100))
            self._wm_img = ImageTk.PhotoImage(img, master=self.root)
        except Exception as e:
            self.var_wm_note.set(f"превью не рисуется: {e}")
            return
        self.wm_view.configure(image=self._wm_img)
        where = "ролика" if getattr(self, "var_fmt", None) and self.var_fmt.get() == "streamer" else "монитора"
        self.var_wm_note.set("Без вотермарки" if not text.strip() else
                             "Длинный ник: шрифт уменьшен, чтобы влез" if small else f"Так ник ляжет в угол {where}")

    # ------------------------------------------------------------ очередь
    def _refresh_fly_no(self):
        self.lbl_fly.configure(text=f"следующая муха: №{next_fly_no()}")

    def _empty_hint(self):
        if self.jobs:
            self.lbl_empty.place_forget()
        else:
            self.lbl_empty.place(relx=0.5, rely=0.58, anchor="center")

    def _clean_parts(self):
        # хвосты прошлого запуска (окно сняли силой): недописанный .part — не видео
        try:
            for p in list(OUT_DIR.glob("*.part.mp4")) + list(OUT_DIR.glob("*.part.ffmpeg.log")):
                rm_file(p, tries=1)                           # занятый чужим рендером не удалится — и не надо
        except Exception:
            pass

    def _shown_title(self, j):
        return j.title.strip() or default_title(j.path)

    def _row(self, j):
        iid = str(id(j))
        if not self.tree.exists(iid):
            return
        s = j.status
        tag = ("ok" if s == "готово" else "err" if s == "ошибка" else "stop" if s == "остановлено"
               else "run" if s.startswith(ACTIVE) else "")
        f = Path(j.out).name if j.out else (j.target or "") if s.startswith(ACTIVE) else ""
        self.tree.item(iid, values=(self._shown_title(j), s, f), tags=(tag,) if tag else ())

    def _add(self, paths):
        have = {j.path for j in self.jobs}
        added = 0
        for p in paths:
            p = Path(p)
            if p.suffix.lower() in VIDEO_EXT and p not in have:
                have.add(p)
                j = Job(p)
                self.jobs.append(j)
                self.tree.insert("", "end", iid=str(id(j)), values=("", "", ""))
                self._row(j)
                added += 1
        if paths and not added:
            self._log("Эти ролики уже в списке (или это не видео).")
        self._empty_hint()
        if self.jobs and not self.tree.selection():
            self.tree.selection_set(str(id(self.jobs[0])))

    def _add_files(self):
        ps = filedialog.askopenfilenames(title="Выбери ролики", initialdir=str(IN_DIR if IN_DIR.exists() else ROOT),
                                         filetypes=[("Видео", " ".join("*" + e for e in VIDEO_EXT)), ("Все файлы", "*.*")])
        self._add(ps)

    def _add_folder(self):
        if not IN_DIR.exists():
            IN_DIR.mkdir(parents=True)
        files = sorted(p for p in IN_DIR.iterdir() if p.suffix.lower() in VIDEO_EXT)
        if not files:
            messagebox.showinfo("Пусто", f"В папке «{IN_DIR.name}» нет роликов.")
        self._add(files)

    def _job_by_iid(self, iid):
        return next((j for j in self.jobs if str(id(j)) == iid), None)

    def _selected(self):
        sel = self.tree.selection()
        return self._job_by_iid(sel[0]) if sel else None

    def _busy(self, j):
        return j is self.cur_run or j.status.startswith(ACTIVE)

    def _editable(self, j):
        return j is not None and j.status in EDITABLE and j is not self.cur_run

    def _remove(self):
        kids = list(self.tree.get_children())
        for iid in self.tree.selection():
            j = self._job_by_iid(iid)
            with self.lock:                                   # не убрать ролик, который поток как раз запускает
                if j and self._busy(j):
                    continue
                if j:
                    j.removed = True
            i = kids.index(iid) if iid in kids else 0
            self.tree.delete(iid)
            if j:
                self.jobs.remove(j)
            rest = self.tree.get_children()
            if rest:                                          # выделить соседний — удобно убирать подряд
                self.tree.selection_set(rest[min(i, len(rest) - 1)])
        self._empty_hint()
        self._on_select()

    def _requeue(self, j):
        if not self.running and j and not self._busy(j):
            self._set_status(j, "в очереди")

    def _on_select(self, _=None):
        j = self._selected()
        self._editing = True
        self.var_title.set(j.title if j else "")
        self._editing = False
        self._sync_title_state()

    def _sync_title_state(self):
        self.ent_title.configure(state="normal" if self._editable(self._selected()) else "disabled")

    def _on_title_edit(self, *_):
        if getattr(self, "_editing", False):
            return
        j = self._selected()
        if self._editable(j):
            j.title = self.var_title.get()
            self._row(j)

    def _set_status(self, j, s):
        j.status = s
        self._row(j)
        if j is self._selected():
            self._sync_title_state()

    def _on_double(self, e=None):
        j = self._selected()
        if j and j.out and Path(j.out).exists():
            open_path(j.out)                                  # двойной клик — открыть готовое видео
        elif self._editable(j):
            self.ent_title.focus_set()
            self.ent_title.select_range(0, "end")

    def _menu_popup(self, e):
        iid = self.tree.identify_row(e.y)
        if not iid:
            return
        self.tree.selection_set(iid)
        self.tree.focus(iid)
        j = self._job_by_iid(iid)
        if j is None:
            return
        has = bool(j.out and Path(j.out).exists())
        on = lambda ok: "normal" if ok else "disabled"
        m = self.menu
        m.delete(0, "end")
        m.add_command(label="Открыть видео", command=lambda: open_path(j.out), state=on(has))
        m.add_command(label="Показать в папке", command=lambda: show_in_folder(j.out), state=on(has))
        m.add_command(label="Рендерить снова", command=lambda: self._requeue(j),
                      state=on(not self.running and j.status == "готово"))
        m.add_separator()
        m.add_command(label="Убрать из списка", command=self._remove, state=on(not self._busy(j)))
        try:
            m.tk_popup(e.x_root, e.y_root)
        finally:
            m.grab_release()

    def _open_last(self):
        if self.last_out and Path(self.last_out).exists():
            open_path(self.last_out)

    # ------------------------------------------------------------ запуск
    def _seiz_times(self):
        """«30, 75, 120» -> «30,75,120»; пусто/0/мусор -> "" (сам выберет)"""
        out = []
        for x in self.var_seiz_t.get().replace(";", ",").replace(" ", ",").split(","):
            try:
                v = float(x.replace("с", ""))
            except ValueError:
                continue
            if v > 0:
                out.append(f"{v:g}")
        return ",".join(out)

    @staticmethod
    def _int(var, dflt, lo, hi):
        # пустое/кривое число в Spinbox не должно ронять «Начать» и закрытие окна
        try:
            v = int(var.get()) or dflt
        except (tk.TclError, ValueError):
            v = dflt
        v = max(lo, min(hi, v))
        var.set(v)
        return v

    def _variant_setting(self):
        # «случайный» или вписанный номер (пустой номер = «случайный»)
        v = self.var_vnum.get().strip()
        return norm_variant(v) if self.var_vmode.get() == "fixed" and v else "random"

    def _settings_now(self):
        fmt = self.var_fmt.get()
        return dict(watermark=WATERMARK_LOCK or self.var_wm.get().strip(), crf=self.var_q.get(),
                    workers=self._int(self.var_w, 5, 1, 8), preview_secs=self._int(self.var_prev, 0, 0, 600),
                    preview_t0=self._int(self.var_t0, 0, 0, 3600),
                    format=fmt if fmt in FM.FORMATS else "base", flyeye=bool(self.var_eye.get()), seizure=bool(self.var_seiz.get()),
                    seizure_t=self._seiz_times(),
                    variant=self._variant_setting())

    def _save(self, cfg=None):
        self.cfg.update(cfg or self._settings_now())
        self.cfg["timing"] = self.timings["base"]
        self.cfg["timing_streamer"] = self.timings["streamer"]
        save_settings(self.cfg)

    def _start(self):
        if self.running:
            return
        todo = [j for j in self.jobs if j.status in EDITABLE]
        if not todo:
            messagebox.showinfo("Нечего рендерить", "Все ролики в списке уже готовы.\nПравый клик → «Рендерить снова»."
                                if self.jobs else "Добавь ролики в очередь.")
            return
        cfg = self._settings_now()
        if cfg["format"] == "streamer" and self.var_vmode.get() == "fixed" and cfg["variant"] == "random":
            messagebox.showinfo("Нет номера комнаты", f"Впиши номер варианта комнаты (0–{FM.VARIANT_MAX}) "
                                                      "или выбери «случайный».")
            self.ent_vnum.focus_set()
            return
        if not cfg["watermark"]:
            if not messagebox.askyesno("Без вотермарки?", "Поле вотермарки пустое. Рендерить без неё?"):
                return
        self._save(cfg)                                       # сохраняем только после подтверждения
        for j in [j for j in todo if not j.path.exists()]:
            self._set_status(j, "ошибка")
            self._log(f"Нет файла: {j.path}", "err")
            todo.remove(j)
        if not todo:
            return
        for j in todo:
            self._set_status(j, "в очереди")
        self.running, self.stop_flag = True, False
        self.run_stats = dict(ok=0, err=0)
        self.btn_go.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.worker = threading.Thread(target=self._run_queue, args=(todo, cfg), daemon=True)
        self.worker.start()

    def _run_queue(self, todo, cfg):
        for i, j in enumerate(todo):
            if self.stop_flag:
                break
            if j.removed:                                     # убрали из списка, пока ждал очереди
                continue
            prev = cfg["preview_secs"] > 0
            n = next_fly_no()
            jcfg = dict(cfg)
            dur = clip_secs(j.path)
            frames = int(dur * 30) if dur else 90 * 30
            k0, win = 0, None                                 # окно пробы в кадрах: с какого, сколько
            if prev:
                t0 = cfg.get("preview_t0", 0)
                if dur and t0 > max(0, int(dur) - 1):         # «с секунды» за концом ролика — берём конец
                    t0 = max(0, int(dur) - max(1, cfg["preview_secs"]))
                    self.q.put(("line", f"проба: ролик короче — начинаю с {t0} с", "wait"))
                jcfg["preview_t0"] = t0
                k0, win = t0 * 30, cfg["preview_secs"] * 30
                frames = min(frames, k0 + win)                # мозг считает с начала до конца окна
            # «случайный» разрешаем тут: номер нужен в имени файла; у каждого ролика очереди — свой
            variant = FM.resolve_variant(cfg["variant"]) if cfg["format"] == "streamer" else None
            title = j.title.replace("\r", " ").replace("\n", " ").strip()
            cmd, final, part = plan_job(j.path, jcfg, n, variant=variant, title=title)
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            err = p = None
            with self.lock:                                   # «Стоп» не проскочит между проверкой и запуском
                go = not (self.stop_flag or j.removed)
                if go:
                    self.cur_run = j
                    try:
                        p = subprocess.Popen(
                            cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            creationflags=NO_WINDOW | (SUSPENDED if self.job else 0), bufsize=1, text=True,
                            encoding="utf-8", errors="replace")
                        if self.job:
                            job_start(self.job, p)
                    except Exception as e:
                        if p is not None:                     # не оставлять висеть приостановленный процесс
                            try:
                                p.kill()
                            except Exception:
                                pass
                        p, err = None, e
                    self.proc = p
            if not go:
                if self.stop_flag:
                    break
                continue
            self.q.put(("job", j, i + 1, len(todo), final.name, frames, k0, win, cfg["format"], variant))
            if err is not None:
                self.cur_run = None
                self.q.put(("line", f"не удалось запустить: {err}", "err"))
                self.q.put(("done", j, 1, None))
                continue
            out = None
            for line in p.stdout:
                line = line.rstrip()
                if line.startswith("ИТОГ:"):
                    out = line.split(":", 1)[1].strip()
                self.q.put(("line", line, None))
            rc = p.wait()
            with self.lock:
                self.proc = None
            ok = rc == 0 and bool(out) and Path(out).exists()
            if ok:
                try:
                    os.replace(out, final)                    # готово: .part -> итоговое имя
                    out = str(final)
                except OSError as e:                          # итоговый файл открыт в плеере и т.п.
                    self.q.put(("line", f"не переименовать в {final.name}: {e}", "err"))
                if not prev:
                    take_fly_no(n)                            # номер мухи — только за готовый ролик
            else:
                drop_own_lock(p.pid)                          # убит «Стопом» — замок за собой не снял
                self._cleanup(j, part)
            self.cur_run = None
            self.q.put(("done", j, rc if ok else (rc or 1), out if ok else None))
        self.q.put(("all_done",))

    def _cleanup(self, j, part):
        # стоп/ошибка: недописанный mp4 без moov не открывается — удаляем (это только наш .part)
        rm_file(part)
        log = part.with_suffix(".ffmpeg.log")
        try:
            if self.stop_flag or log.stat().st_size == 0:
                rm_file(log)
        except OSError:
            pass
        if self.stop_flag:                                    # хвосты прохода 1 (перепишутся, но место занимают)
            for f in (LOG_CACHE / f"log_{j.path.stem[:40].strip(" .")}").glob("*.raw"):
                rm_file(f, tries=1)

    def _stop(self):
        if not self.running:
            return
        with self.lock:
            self.stop_flag = True
            p = self.proc
        if p and p.poll() is None:
            # убить всё дерево (react.py запускает процессы рендера)
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True,
                           creationflags=NO_WINDOW)
            if self.job:                                      # добить отставших (кто родился во время taskkill)
                try:
                    import ctypes
                    ctypes.windll.kernel32.TerminateJobObject(ctypes.c_void_p(self.job), 1)
                except Exception:
                    pass
        self.btn_stop.configure(state="disabled")
        self.var_stage.set("Останавливаю…")

    # ------------------------------------------------------------ события из потока
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                try:
                    self._handle(msg)
                except Exception as e:                        # одна ошибка не должна остановить опрос
                    self._log(f"ошибка окна: {e!r}", "err")
        except queue.Empty:
            pass
        if self.running and self.ph and time.time() - self._t_tick > 0.25:
            self._t_tick = time.time()
            try:
                self._progress()                              # тикают «прошло/осталось» и полоса рендера
            except Exception:
                pass
        self._poll_id = self.root.after(100, self._poll)

    def _stage(self, j, text):
        ph = self.ph
        pre = f"[{ph['i']}/{ph['total']}] " if ph.get("total", 1) > 1 else ""
        self.var_stage.set(f"{pre}{short(self._shown_title(j), 36)}: {text}")

    def _handle(self, msg):
        kind = msg[0]
        if kind == "job":
            j, i, total, name, frames, k0, win, fk, variant = msg[1:]
            self.cur = j
            j.target = name
            self.timing = self.timings.get(fk, self.timings["base"])   # оценка времени — по замерам этого формата
            n = max(1, frames)
            self.ph = dict(t0=time.time(), phase="boot", n=n, k0=k0, win=win, i=i, total=total,
                           n2=max(1, n - k0 if win is None else min(win, n - k0)))
            self.pb["value"] = 0
            self._set_status(j, "подготовка…")
            self._stage(j, "подготовка — анализ ролика, загрузка мозга мухи")
            self._log(f"\n=== {self._shown_title(j)} → {name} ===", "ok")
            self._progress()
        elif kind == "line":
            line, tag = msg[1], msg[2]
            ph = self.ph
            if line.startswith("ЖДУ"):                        # замок рендера: ждём, это не ошибка
                if ph and self.cur is not None:
                    ph["wait"] = "ждёт свободной памяти" if "памят" in line else "ждёт: идёт другой рендер"
                    self._progress()
                self._log(line, "wait")
                return
            if ph.get("wait") and line.strip():               # после ожидания пошёл вывод — работа началась
                ph.pop("wait")
                ph["t_run"] = time.time()
            m = re.match(r"ВАРИАНТ:\s*(\d+)", line)
            if m:
                self._got_variant(self.cur, int(m.group(1)))
                return
            if self._parse_progress(line):                    # служебные PROGRESS-строки в журнал не пишем
                return
            if tag is None and ERR_RE.match(line):
                tag = "err"
            self._log(line, tag)
        elif kind == "done":
            j, rc, out = msg[1], msg[2], msg[3]
            el = time.time() - self.ph.get("t0", time.time())
            if rc == 0 and out:
                j.out = out
                self.last_out = out
                self.btn_open.configure(state="normal")
                self._set_status(j, "готово")
                self.pb["value"] = 1000
                self.var_pct.set("100%")
                self._stage(j, f"готово за {fmt_s(el)}")
                self._log(f"Готово: {out}", "ok")
                self.run_stats["ok"] += 1
                self._learn()
            elif self.stop_flag:
                self._set_status(j, "остановлено")
                self.pb["value"] = 0
                self.var_pct.set("")
                self._stage(j, "остановлено")
                self._log("Остановлено.", "err")
            else:
                self._set_status(j, "ошибка")
                self._stage(j, f"ошибка (код {rc}) — подробности в журнале ниже")
                self._log(f"Ошибка (код {rc}) — подробности выше.", "err")
                self.run_stats["err"] += 1
            j.target = None
            self._row(j)
            self.ph = {}
            self.cur = None
            self.var_eta.set(f"файл: {Path(out).name}" if rc == 0 and out else "")
            self.root.title(APP_TITLE)
            self._refresh_fly_no()
        elif kind == "all_done":
            self.running = False
            self.ph = {}
            self.btn_go.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.root.title(APP_TITLE)
            self._sync_title_state()
            ok, bad = self.run_stats["ok"], self.run_stats["err"]
            if self.stop_flag:
                self.var_stage.set("Остановлено")
                self.var_eta.set("")
            elif ok + bad > 1 or bad:
                self.var_stage.set(f"Очередь готова: готово {ok}" + (f", с ошибкой {bad}" if bad else ""))
            elif not ok:
                self.var_stage.set("Очередь готова")
            # один готовый ролик — остаётся строка «готово за … → файл»

    # ------------------------------------------------------------ прогресс и оценка времени
    def _parse_progress(self, line):
        ph, j = self.ph, self.cur
        if not ph or j is None:
            return line.startswith("PROGRESS")
        now = time.time()
        m = re.match(r"PROGRESS 1 (\d+) (\d+)", line)
        if m:
            k, n = int(m.group(1)), max(1, int(m.group(2)))
            if ph["phase"] == "boot":                         # запуск считаем без ожидания замка
                ph.update(phase="p1", p1_t0=now, p1_k0=k, boot1=now - ph.get("t_run", ph["t0"]))
            k0, win = ph.get("k0", 0), ph.get("win")
            ph.update(n=n, k1=k, n2=max(1, n - k0 if win is None else min(win, n - k0)))
            if k >= n:
                ph["p1_t1"] = now
            self._progress()
            return True
        if line.startswith("PROGRESS 2"):
            ph.update(phase="p2", p2_t0=now)
            ph.setdefault("p1_t1", now)
            self._progress()
            return True
        if line.startswith("PROGRESS"):
            return True
        m = re.search(r"\[(\d+)/(\d+)\]\s+([\d.]+) кадр/с", line)
        if m and ph["phase"] == "p2":
            ph.update(k2=int(m.group(1)), N=max(1, int(m.group(2))), fps2=float(m.group(3)), t2=now)
            self._progress()
        m = re.search(r"'startup_s': ([\d.]+).*'fps': ([\d.]+)", line)
        if m:                                                 # итог прохода 2 — замеры на следующий раз
            ph.update(boot2_m=float(m.group(1)), fps2_m=float(m.group(2)))
        return False

    def _progress(self):
        """Полоса = доля оценённой работы; «осталось» — по этапам: реальная скорость текущего этапа
        + оценка следующих (по замерам прошлых роликов)."""
        ph, T, j = self.ph, self.timing, self.cur
        if not ph or j is None or self.stop_flag:
            return
        now = time.time()
        el = now - ph["t0"]
        n = ph["n"]
        d_b1, d_p1, d_b2, d_p2 = T["boot1"], n / T["r1"], T["boot2"], ph.get("n2", n) / T["fps2"]
        phase = ph["phase"]
        if ph.get("wait"):                                    # другой рендер / мало памяти: стоим, не считаем
            self._set_status(j, "ждёт…")
            self._stage(j, ph["wait"])
            self.var_eta.set(f"прошло {fmt_s(el)} · начнётся само, когда освободится")
            self.root.title(f"ждёт · {APP_TITLE}")
            return
        if phase == "boot":
            eb = now - ph.get("t_run", ph["t0"])              # с конца ожидания
            done = min(eb, 0.95 * d_b1)
            left = max(d_b1 - eb, 1.0) + d_p1 + d_b2 + d_p2
            self._set_status(j, "подготовка…")
            self._stage(j, "подготовка — анализ ролика, загрузка мозга мухи")
        elif phase == "p1":
            k, t = ph["k1"], now - ph["p1_t0"]
            dk = k - ph["p1_k0"]
            r = dk / t if dk >= 60 and t > 0.3 else T["r1"]
            done = d_b1 + d_p1 * k / n
            left = (n - k) / r + d_b2 + d_p2
            self._set_status(j, f"мозг {100 * k // n}%")
            self._stage(j, f"мозг мухи смотрит ролик ({k}/{n} кадров)")
        else:
            t = now - ph["p2_t0"]
            if "k2" in ph:                                    # есть замер: тянем полосу по реальной скорости
                N, fps = ph["N"], max(ph["fps2"], 0.5)
                k = min(N, ph["k2"] + fps * (now - ph["t2"]))
                done = d_b1 + d_p1 + d_b2 + d_p2 * k / N
                left = (N - k) / fps
                self._set_status(j, f"рендер {min(99, int(100 * k // N))}%")
                self._stage(j, f"рендер видео ({int(k)}/{N} кадров, {fps:.0f} кадр/с)")
            else:
                done = d_b1 + d_p1 + min(t, 0.97 * (d_b2 + d_p2))
                left = max(d_b2 + d_p2 - t, 1.0)
                self._set_status(j, "рендер…")
                self._stage(j, "рендер — запуск процессов" if t < d_b2 else "рендер видео")
        v = min(995.0, 1000.0 * done / (d_b1 + d_p1 + d_b2 + d_p2))
        v = max(float(self.pb["value"]), v)                  # полоса только вперёд
        self.pb["value"] = v
        self.var_pct.set(f"{v / 10:.0f}%")
        self.root.title(f"{v / 10:.0f}% · {APP_TITLE}")
        tail = "почти готово" if left < 2 else f"осталось ≈ {fmt_s(left)}"
        self.var_eta.set(f"прошло {fmt_s(el)} · {tail}")

    def _learn(self):
        # замеры готового ролика -> оценка времени для следующих (плавно, чтобы один выброс не сбил)
        ph, T = self.ph, self.timing

        def upd(key, v, lo, hi):
            if v and lo <= v <= hi:
                T[key] = round(0.3 * T[key] + 0.7 * v, 2)

        upd("boot1", ph.get("boot1"), 2, 900)
        n = ph.get("n", 0)
        if n >= 600 and "p1_t0" in ph and "p1_t1" in ph:      # короткие пробы скорость мозга не показывают
            upd("r1", (n - ph.get("p1_k0", 0)) / max(0.1, ph["p1_t1"] - ph["p1_t0"]), 5, 5000)
        upd("boot2", ph.get("boot2_m"), 0.5, 180)
        if ph.get("N", n) >= 600:
            upd("fps2", ph.get("fps2_m"), 1, 2000)
        self._save(dict(self.cfg))

    def _log(self, text, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag or ())
        if int(self.log.index("end-1c").split(".")[0]) > 3000:
            self.log.delete("1.0", "500.0")
        self.log.configure(state="disabled")
        self.log.see("end")

    def _tk_error(self, exc, val, tb):
        # под pythonw консоли нет — ошибки окна показываем в журнале
        self._log("ошибка окна:\n" + "".join(traceback.format_exception(exc, val, tb)).rstrip(), "err")

    def _on_close(self):
        if self.running and not messagebox.askyesno("Идёт рендер", "Остановить рендер и закрыть?"):
            return
        self._stop()
        if self.worker is not None:
            self.worker.join(timeout=6)                       # дать потоку убрать .part до выхода
        try:
            self._save()
        except Exception:
            pass                                              # настройки не важнее закрытия окна
        for aid in (self._poll_id, self._wm_after, self._copy_after):   # без «invalid command name» после закрытия
            try:
                if aid:
                    self.root.after_cancel(aid)
            except Exception:
                pass
        self.root.destroy()


def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)               # до создания окна, иначе Tk считает экран 96 dpi
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
