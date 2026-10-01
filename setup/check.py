"""Проверка, что всё установлено: python setup/check.py

Идёт по шагам и на каждом пишет «ок» или что сделать. Последний шаг — 1 секунда жизни мозга мухи.
"""
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
fails = []


def step(name, fn):
    print(f"• {name} … ", end="", flush=True)
    try:
        msg = fn()
        print("ок" + (f" ({msg})" if msg else ""))
    except Exception as e:
        print(f"НЕТ\n    {type(e).__name__}: {e}")
        fails.append(name)


def py():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"нужен Python 3.10+ (проверено на 3.12), у тебя {v.major}.{v.minor}")
    return f"{v.major}.{v.minor}.{v.micro}"


def packages():
    need = ["numpy", "pandas", "pyarrow", "scipy", "torch", "mujoco", "flygym", "dm_control", "cv2", "PIL", "av",
            "imageio", "imageio_ffmpeg", "matplotlib", "psutil", "tqdm"]
    miss = []
    for m in need:
        try:
            importlib.import_module(m)
        except Exception:
            miss.append(m)
    if miss:
        raise RuntimeError("не хватает пакетов: " + ", ".join(miss) + " — запусти setup\\install.bat")
    try:
        importlib.import_module("faster_whisper")
    except Exception:
        return "без faster-whisper: режим «по смыслу речи» не будет работать, «зрение и слух» — будет"
    return None


def gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("видеокарта NVIDIA с CUDA не найдена. Мозг на процессоре работает, но в сотни раз "
                           "медленнее (ролик в 1 минуту — часы). Проверь драйвер NVIDIA и что torch ставился с cu128")
    p = torch.cuda.get_device_properties(0)
    return f"{p.name}, {p.total_memory / 2**30:.0f} ГБ"


def data():
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "setup" / "get_data.py"), "--check", "--required"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("нет данных коннектома — запусти: python setup/get_data.py")


def ffmpeg():
    import imageio_ffmpeg
    return Path(imageio_ffmpeg.get_ffmpeg_exe()).name


def render():
    import mujoco
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
    r = mujoco.Renderer(m, 64, 64)
    r.update_scene(mujoco.MjData(m))
    img = r.render()
    r.close()
    return f"OpenGL, кадр {img.shape[1]}x{img.shape[0]}"


def brain():
    """Мухе показывают «объект слева» (детекторы LC10 левого глаза) — руль DNa должен ответить."""
    import brain as B
    t0 = time.time()
    ann = B.load_annotations()
    b = B.FlyBrain(params=dict(U=0.1, tau_d=300, a_inc=0.1, tau_a=3000), seed=0)
    t1 = time.time()

    def cells(types, side):
        m = ann.cell_type.isin(types) & (ann.side == side)
        return [b.idx[r] for r in ann.index[m] if r in b.idx]

    b.set_input(cells(["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"], "left"), 50.0)
    hz = b.run(1000)
    t2 = time.time()
    dl = hz[cells(["DNa01", "DNa02"], "left")].mean()
    dr = hz[cells(["DNa01", "DNa02"], "right")].mean()
    sp = lambda x: f"{int(x):,}".replace(",", " ")  # noqa: E731
    print(f"\n    нейронов {sp(len(b.ids))}; загрузка {t1 - t0:.0f} с; 1 с жизни мозга посчитана за {t2 - t1:.2f} с"
          f"\n    «объект слева» -> горело нейронов {sp((hz > 0).sum())}; руль DNa: слева {dl:.0f} Гц, справа {dr:.0f} Гц"
          f"\n    ", end="")
    if dl <= dr:
        raise RuntimeError("руль не повернул к объекту — что-то не так с данными или видеокартой")
    return None


if __name__ == "__main__":
    print("Проверка установки «муха с мозгом из коннектома»\n")
    step("Python", py)
    step("пакеты", packages)
    step("видеокарта", gpu)
    step("данные коннектома", data)
    step("ffmpeg (видео)", ffmpeg)
    step("рендер MuJoCo", render)
    if not fails:
        step("мозг мухи: 1 секунда", brain)
    print()
    if fails:
        print("Не готово: " + ", ".join(fails) + ". Подробности — в docs/install.md, раздел «Если что-то не так».")
        sys.exit(1)
    print("Всё готово. Дальше — README.md, раздел «Первый ролик».")
