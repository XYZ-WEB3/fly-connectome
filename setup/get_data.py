"""Скачать данные, которых нет в репозитории (они чужие — берём из первоисточников).

    python setup/get_data.py            # всё нужное (~135 МБ)
    python setup/get_data.py --check    # только проверить, что всё на месте

Версии закреплены: ровно те файлы, на которых получены все числа в research/ и learning/.
Каждый файл сверяется по sha256; битый или недокачанный файл скачивается заново.

Источники:
  * модель мозга Shiu et al. 2024 (MIT): github.com/philshiu/Drosophila_brain_model, коммит 91bdd1e
    - Connectivity_783.parquet — связи коннектома FlyWire v783 (кто на кого, сколько синапсов, знак)
    - Completeness_783.csv     — список нейронов (порядок нейронов в модели)
    - model.py, utils.py, example.ipynb, LICENSE, Readme.md — исходная модель на Brian2 (для сверки)
  * аннотации FlyWire (Schlegel et al. 2024 и др.): github.com/flyconnectome/flywire_annotations, коммит 17fc577
    - neuron_annotations.tsv — типы клеток, классы, сторона, медиатор, координаты
"""
import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIU = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/"
ANN = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/17fc57722002e1a7d38cdd0c89ac382bf92718da/"
       "supplemental_files/Supplemental_file1_neuron_annotations.tsv")

# (куда, откуда, sha256, нужен ли для работы)
FILES = [
    ("brain_model/Connectivity_783.parquet", SHIU + "Connectivity_783.parquet",
     "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347", True),
    ("brain_model/Completeness_783.csv", SHIU + "Completeness_783.csv",
     "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311", True),
    ("data/neuron_annotations.tsv", ANN,
     "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be", True),
    ("brain_model/model.py", SHIU + "model.py",
     "fc45837d7122c6ce2a7f3f2f23c515992e4b232aadb919efabb72337fac88e4e", False),
    ("brain_model/utils.py", SHIU + "utils.py",
     "f378cf0b464f2ac5e1daf3609ba25fd51c7e34f294846281707ffe006c3e9ef1", False),
    ("brain_model/example.ipynb", SHIU + "example.ipynb",
     "1737c3043af700504c4791c4bfc02d1856bbeb0d36c469832e4b9399dfddda3a", False),
    ("brain_model/LICENSE", SHIU + "LICENSE",
     "3621f6d6476189190e2960fa43f11b275ba4eca848fdac50a1ba2925de6223e8", False),
    ("brain_model/Readme.md", SHIU + "Readme.md",
     "0654ac53f9af85000b50f6a5d2297aade8602c507a44e811f8aff602629c2625", False),
]
# Completeness_783.csv после git на Windows может прийти с переводами строк CRLF — тот же файл
CRLF_OK = {"brain_model/Completeness_783.csv": "52b0ac6094cd32c546f8d4c341e094376f48f4e791f8db9b166de5dff8199ea4"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ok(rel, want):
    p = ROOT / rel
    if not p.exists():
        return False
    got = sha256(p)
    return got == want or got == CRLF_OK.get(rel)


def download(url, dst, rel, want, tries=8):
    """Качает в <имя>.part с докачкой после обрыва (HTTP Range). Файл встаёт на место,
    только когда он скачан целиком и контрольная сумма совпала."""
    tmp = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = None
    for attempt in range(1, tries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:                  # сервер не умеет докачку — с начала
                    have = 0
                n = int(r.headers.get("Content-Length") or 0)
                total = have + n if n else total
                with open(tmp, "ab" if have else "wb") as f:
                    done, t0 = have, time.time()
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            sp = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                            print(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} МБ  ({sp:.2f} МБ/с)   ",
                                  end="", flush=True)
            print()
        except Exception as e:                                # сеть мигнула — докачаем
            print(f"\n    связь прервалась ({e}), докачиваю: попытка {attempt} из {tries}")
            time.sleep(3)
            continue
        size = tmp.stat().st_size
        if total and size < total:                            # соединение закрылось раньше времени
            print(f"    скачано {size / 1e6:.1f} из {total / 1e6:.1f} МБ, докачиваю: попытка {attempt} из {tries}")
            time.sleep(3)
            continue
        got = sha256(tmp)
        if got == want or got == CRLF_OK.get(rel):
            tmp.replace(dst)
            return
        print("    файл скачался с ошибкой, качаю заново")
        tmp.unlink()
    raise SystemExit(f"\nНе удалось скачать {rel}. Проверь интернет и запусти ещё раз — скачанное не пропадёт.\n"
                     f"Если ошибка повторяется, файл в источнике мог измениться: {url}")


def main():
    ap = argparse.ArgumentParser(description="скачать данные коннектома и аннотаций")
    ap.add_argument("--check", action="store_true", help="только проверить")
    ap.add_argument("--required", action="store_true", help="только нужное для работы (без исходной модели Brian2)")
    a = ap.parse_args()
    missing = []
    for rel, url, want, need in FILES:
        if a.required and not need:
            continue
        if ok(rel, want):
            print(f"  есть     {rel}")
            continue
        if a.check:
            print(f"  НЕТ      {rel}")
            missing.append(rel)
            continue
        print(f"  качаю    {rel}")
        download(url, ROOT / rel, rel, want)
        print(f"  готово   {rel}")
    if missing:
        print("\nНе хватает файлов — запусти: python setup/get_data.py")
        sys.exit(1)
    print("\nДанные на месте.")


if __name__ == "__main__":
    main()
