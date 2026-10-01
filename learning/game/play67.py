"""Муха играет в 67COIN со своей памятью (после train67.py): партия на счёт -> журнал + счёт rules67.

Во время партии НЕТ награды и НЕТ обучения: играет то, что муха выучила (веса KC->MBON из memory/fly_<id>.npz).
Тап = родной руль DNa01+02 L−R (декодер game67.FlyPlayer, тот же, что при обучении, train67.DEC).
Счёт = rules67.score_log(log, seed) — официальный модуль (только чтение), сверка с живым счётом (live_match).

CLI (из корня проекта):
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -W ignore learning/game/play67.py --fly_id 7 --seconds 60
  --naive      — сыграть без памяти (для сравнения; та же партия по seed)
  --seed N     — seed партии (шум мозга + первая подсказка); по умолчанию следующий свободный для мухи
  --out PATH   — куда писать JSON (по умолчанию learning/game/logs/play_fly<id>[_naive]_seed<seed>.json)
JSON: log [[t_ms, 'L'|'R'], ...], seed, score (score_log), summary, файл памяти и его sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train67 as T  # noqa: E402
import game67 as G  # noqa: E402

LOGS = HERE / "logs"


def play_scored(rig, fly_id, seconds=60.0, seed=None, naive=False, mem_dir=T.MEM_DIR, record=False):
    """-> (json-словарь партии, ses, fly). Память грузится из mem_dir/fly_<id>.npz (если naive=False)."""
    fmem = Path(mem_dir) / f"fly_{fly_id}.npz"
    rig.pl.reset_memory()
    sha = None
    if not naive:
        if not fmem.exists():
            raise FileNotFoundError(f"нет памяти {fmem}: сначала train67.py --fly_id {fly_id} (или --naive)")
        rig.pl.load_memory(fmem)
        sha = hashlib.sha256(fmem.read_bytes()).hexdigest()
    if seed is None:
        seed = 30000 + 100 * int(fly_id)
        while (LOGS / f"play_fly{fly_id}{'_naive' if naive else ''}_seed{seed}.json").exists():
            seed += 1
    dur = float(seconds) * 1000.0
    ses, fly, sm = rig.play(seed, dur, mode=None, learn=False, record=record)
    sm.update(G.reference_scores(ses, dur, n_shuffle=100))
    mem = rig.memory()
    out = dict(kind="67coin_fly_game", fly_id=int(fly_id), naive=bool(naive), seed=int(seed), dur_ms=dur,
               up=ses.up, energy0=ses.energy0, rules_version=G.R67.VERSION,
               memory_file=None if naive else str(fmem), memory_sha256=sha,
               memory_MBON27=dict(L=mem["MBON27_L"], R=mem["MBON27_R"]),
               cue=rig.cue, decoder=rig.dec, learning_during_game=False, reward_during_game=False,
               created=time.strftime("%Y-%m-%d %H:%M:%S"),
               log=ses.log, score=ses.result, summary=sm, events=ses.events)
    return out, ses, fly


def main():
    ap = argparse.ArgumentParser(description="партия 67COIN обученной мухой (журнал + счёт)")
    ap.add_argument("--fly_id", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--naive", action="store_true")
    ap.add_argument("--mem_dir", default=str(T.MEM_DIR))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rig = T.Rig(verbose=False)
    out, ses, fly = play_scored(rig, a.fly_id, a.seconds, a.seed, a.naive, a.mem_dir)
    LOGS.mkdir(exist_ok=True)
    path = Path(a.out) if a.out else LOGS / f"play_fly{a.fly_id}{'_naive' if a.naive else ''}_seed{out['seed']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    sc, sm = out["score"], out["summary"]
    print(f"муха {a.fly_id} ({'без памяти' if a.naive else 'с памятью'}), seed {out['seed']}, {a.seconds:.0f} с")
    print(f"  счёт rules67: coins {sc['coins']} | тапов засчитано {sc['taps']} | сбросов {sc['resets']} | "
          f"best_combo {sc['best_combo']} | энергия в конце {sc['energy_end']} | HUD=rules {sc['live_match']}")
    print(f"  верно {sm['correct']:.3f} (L {sm['correct_cueL']:.3f}, R {sm['correct_cueR']:.3f}) | coins/min "
          f"{sm['coins_per_min']:.0f} | тап/мин {sm['taps_per_min']:.0f} | реакция L/R {sm['rt_med_L']:.0f}/{sm['rt_med_R']:.0f} мс")
    print(f"  журнал: {path}")


if __name__ == "__main__":
    main()
