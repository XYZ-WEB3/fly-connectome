"""Базовая линия 67COIN: НЕобученная муха играет родным рулём (без обучения, без награды).

Подсказка: LC10 со стороны, куда надо тапнуть (упрощённая сетчатка: «мелкий объект с этой стороны»).
Тап: сглаженный руль DNa01+02 L−R пересёк ±порог (триггер Шмитта, рефрактерность). Декодер не знает подсказку.

Этапы (запуск из корня проекта):
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -W ignore learning/game/baseline.py --tune   # подбор, seed 100-102, 30 с
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -W ignore learning/game/baseline.py --eval   # оценка, seed 0-4, 60 с
Подбор и оценка на РАЗНЫХ seed (подбор не подглядывает в оценку).
Логи партий: learning/game/logs/baseline_<условие>_seed<k>.json; сводки: learning/game/_scratch/*.csv|log.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import game67 as G  # noqa: E402
import brain as B  # noqa: E402
import senses as S  # noqa: E402

SCR = HERE / "_scratch"
LOGS = HERE / "logs"
TUNE_SEEDS = [100, 101, 102]
EVAL_SEEDS = [0, 1, 2, 3, 4]
DEFAULT = dict(hz=30.0, th=15.0, rearm=0.5, refr=60.0, bias=None)
WEAK = dict(hz=10.0, th=20.0, rearm=0.5, refr=60.0, bias=None)       # лучшее для 10 Гц на этапе 1 подбора


def log(msg, fh=None):
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n"); fh.flush()


class Rig:
    """Один мозг на весь прогон; seed меняется через G.reseed (побитно как новый мозг, c1_reseed_check.log)."""

    def __init__(self):
        t0 = time.time()
        self.ann = B.load_annotations()
        self.b = B.FlyBrain(params=G.PRM, seed=0)
        self.b.enable_graph()
        self.groups = S.build_groups(self.b, self.ann)
        self.pl = None
        print(f"мозг загружен за {time.time() - t0:.1f} с", flush=True)

    def dan_silenced(self, on):
        """on=True: подключить MBPlasticity с enabled=False — выход DAN заглушен, веса не меняются
        (так будет у обучаемой мухи; это честная база для сравнения с обучением)."""
        if on and self.pl is None:
            from plasticity import MBPlasticity
            self.pl = MBPlasticity(self.b)
            self.pl.enabled = False
        elif not on and self.pl is not None:
            self.pl.detach()
            self.pl = None

    def session(self, seed, cfg, dur_ms, record=False, mirror=False):
        G.reseed(self.b, seed)
        if self.pl is not None:
            self.pl.reset_traces()
        cue = G.LC10Cue(self.b, self.groups, hz=cfg["hz"], mirror=mirror)
        bias = cfg.get("bias")
        if isinstance(bias, list):
            bias = tuple(bias)
        fly = G.FlyPlayer(self.b, self.groups, cue, threshold=cfg["th"], refractory_ms=cfg["refr"],
                          rearm=cfg["rearm"], bias=bias, record=record,
                          plasticity=self.pl if self.pl is not None else None)
        ses = G.Game67Session(seed)
        fly.play(ses, dur_ms)
        ses.finish()
        sm = ses.summary(dur_ms)
        sm.update(G.reference_scores(ses, dur_ms, n_shuffle=100))
        sm.update(n_active_max=fly.n_active_max, n_active_mean=fly.n_active_sum / max(1, fly.n_steps),
                  step_ms=fly.step_ms)
        return ses, fly, sm


def cfg_name(c):
    b = c.get("bias")
    bs = "none" if b is None else (f"adapt{int(b[1])}" if isinstance(b, (tuple, list)) else f"c{b:g}")
    return f"hz{c['hz']:g}_th{c['th']:g}_re{c['rearm']:g}_rf{c['refr']:g}_{bs}"


# ---------------------------------------------------------------------------
def tune(rig, fh):
    dur = 30000.0
    rows = []

    def run_grid(grid, stage):
        for c in grid:
            t0 = time.time()
            sms = [rig.session(s, c, dur)[2] for s in TUNE_SEEDS]
            d = pd.DataFrame(sms)
            r = dict(stage=stage, name=cfg_name(c), **{k: v for k, v in c.items() if k != "bias"},
                     bias=str(c.get("bias")),
                     cpm=d.coins_per_min.mean(), cpm_sd=d.coins_per_min.std(), tpm=d.taps_per_min.mean(),
                     correct=d.correct.mean(), corrL=d.correct_cueL.mean(), corrR=d.correct_cueR.mean(),
                     resets=d.resets.mean(), pause=d.pause_resets.mean(), best=d.best_combo.mean(),
                     mean_mult=d.mean_mult.mean(), rtL=d.rt_med_L.mean(), rtR=d.rt_med_R.mean(),
                     live_ok=bool(d.live_match.all()), sec=time.time() - t0)
            rows.append(r)
            log(f"[{stage}] {r['name']:<40} coins/min {r['cpm']:7.0f} ±{r['cpm_sd']:5.0f} | тап/мин {r['tpm']:5.0f} | "
                f"верно {r['correct']:.3f} (L {r['corrL']:.3f} R {r['corrR']:.3f}) | сбросов {r['resets']:5.1f} "
                f"пауз {r['pause']:4.1f} | RT L/R {r['rtL']:.0f}/{r['rtR']:.0f} | ×ср {r['mean_mult']:.2f} "
                f"| {r['sec']:.1f} с", fh)

    # этап 1: сила подсказки x порог (остальное по умолчанию), с поправкой крена и без
    g1 = [dict(hz=hz, th=th, rearm=0.5, refr=60.0, bias=bias)
          for hz, th, bias in itertools.product([10.0, 20.0, 30.0, 50.0], [5.0, 10.0, 20.0, 30.0, 40.0],
                                                [None, ("adapt", 2000.0)])]
    run_grid(g1, "1")
    d1 = pd.DataFrame(rows)
    top = d1.sort_values("cpm", ascending=False).head(3)
    log("\nэтап 1, лучшие:\n" + top[["name", "cpm", "tpm", "correct", "resets"]].to_string(index=False), fh)
    # этап 2: вокруг двух лучших (hz, bias) — порог мельче, гистерезис, рефрактерность
    seen = set(d1.name)
    g2 = []
    for _, r in top.head(2).iterrows():
        bias = None if r["bias"] == "None" else ("adapt", 2000.0)
        for th, rearm, refr in itertools.product(sorted({max(3.0, r["th"] - 5), r["th"], r["th"] + 5}),
                                                 [0.25, 0.5, 0.8], [60.0, 100.0, 150.0]):
            c = dict(hz=r["hz"], th=th, rearm=rearm, refr=refr, bias=bias)
            if cfg_name(c) not in seen:
                seen.add(cfg_name(c)); g2.append(c)
    run_grid(g2, "2")
    d = pd.DataFrame(rows)
    d.to_csv(SCR / "tune.csv", index=False)
    best = d.sort_values("cpm", ascending=False).iloc[0]
    bias = None if best["bias"] == "None" else ("adapt", 2000.0)
    cfg = dict(hz=float(best["hz"]), th=float(best["th"]), rearm=float(best["rearm"]), refr=float(best["refr"]),
               bias=bias)
    with open(SCR / "best_cfg.json", "w", encoding="utf-8") as f:
        json.dump(dict(cfg=cfg, name=cfg_name(cfg), tune_seeds=TUNE_SEEDS, dur_ms=dur,
                       cpm=float(best["cpm"])), f, ensure_ascii=False, indent=1)
    log(f"\nЛУЧШЕЕ (по coins/min на seed {TUNE_SEEDS}): {cfg_name(cfg)} -> {best['cpm']:.0f} coins/min", fh)
    log("топ-10:\n" + d.sort_values("cpm", ascending=False).head(10)[
        ["stage", "name", "cpm", "cpm_sd", "tpm", "correct", "corrL", "corrR", "resets", "pause"]].to_string(index=False), fh)
    return cfg


# ---------------------------------------------------------------------------
def fmt_row(name, d):
    def ms(k, p=3):
        return f"{d[k].mean():.{p}f} ± {d[k].std():.{p}f}"
    return (f"{name:<22} тап/мин {ms('taps_per_min', 0)} | верно {ms('correct')} | L {ms('correct_cueL')} | "
            f"R {ms('correct_cueR')} | RT L {ms('rt_med_L', 0)} R {ms('rt_med_R', 0)} мс | сбросов {ms('resets', 1)} "
            f"| пауз {ms('pause_resets', 1)} | best_combo {ms('best_combo', 1)} | coins/min {ms('coins_per_min', 0)} "
            f"| идеал {ms('ideal_cpm', 0)} | случ. {ms('random_cpm', 0)} | энергия {ms('energy_used', 0)} "
            f"(конец {ms('energy_end', 0)}) | нет энергии {ms('no_energy', 0)} | активных/9 мс ср {ms('n_active_mean', 0)} "
            f"макс {d.n_active_max.max():.0f} | HUD=rules {d.live_match.all()}")


def evaluate(rig, cfg, fh, dur=60000.0):
    LOGS.mkdir(exist_ok=True)
    conds = [
        ("base", cfg, False, False, dur),              # главная база: мозг как есть
        ("base_dansil", cfg, True, False, dur),        # то же, выход DAN заглушен (как у обучаемой мухи)
        ("default", DEFAULT, False, False, dur),       # неподобранные параметры (для прозрачности)
        ("weak10", WEAK, False, False, dur),           # слабая подсказка LC10 10 Гц (видна левая асимметрия)
        ("nocue", dict(cfg, hz=0.0), False, False, dur),   # контроль: подсказки нет
        ("mirror", cfg, False, True, dur),             # контроль: объект на противоположной стороне
        ("long180", cfg, False, False, 180000.0),      # 3 мин: энергия и привыкание
    ]
    allrows = []
    for name, c, dansil, mirror, dms in conds:
        rig.dan_silenced(dansil)
        sms = []
        for s in EVAL_SEEDS:
            ses, fly, sm = rig.session(s, c, dms, record=(name in ("base", "weak10", "mirror")), mirror=mirror)
            sm.update(cond=name, cfg=cfg_name(c), mirror=mirror, dan_silenced=dansil)
            # ход партии по 10 с: темп и доля верных (привыкание)
            ev = ses.events
            bins = []
            for k in range(int(dms // 10000)):
                e = [x for x in ev if k * 10000 <= x["t"] < (k + 1) * 10000]
                bins.append(dict(t0=k * 10, taps=len(e), correct=float(np.mean([x["correct"] for x in e])) if e else None,
                                 coins=int(sum(x["gain"] for x in e))))
            sm["bins"] = bins
            sms.append(sm)
            out = dict(condition=name, cfg=c if not isinstance(c.get("bias"), tuple) else dict(c, bias=list(c["bias"])),
                       mirror=mirror, dan_silenced=dansil, dur_ms=dms, summary=sm, **ses.to_json())
            if fly.trace:
                out["trace_first10s"] = [list(x) for x in fly.trace if x[0] <= 10000]
            with open(LOGS / f"baseline_{name}_seed{s}.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
            log(f"  {name} seed {s}: тапов {sm['taps']}, верно {sm['correct']:.3f}, coins {sm['coins']}, "
                f"best_combo {sm['best_combo']}, сбросов {sm['resets']}, HUD=rules {sm['live_match']}", fh)
        d = pd.DataFrame(sms)
        log(fmt_row(name, d), fh)
        allrows += sms
    rig.dan_silenced(False)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "bins"} for r in allrows])
    df.to_csv(SCR / "baseline_eval.csv", index=False)
    with open(SCR / "baseline_bins.json", "w", encoding="utf-8") as f:
        json.dump([dict(cond=r["cond"], seed=r["seed"], bins=r["bins"]) for r in allrows], f, ensure_ascii=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    SCR.mkdir(exist_ok=True)
    rig = Rig()
    if a.tune:
        with open(SCR / "tune.log", "w", encoding="utf-8") as fh:
            tune(rig, fh)
    if a.eval:
        with open(SCR / "best_cfg.json", encoding="utf-8") as f:
            cfg = json.load(f)["cfg"]
        if isinstance(cfg.get("bias"), list):
            cfg["bias"] = tuple(cfg["bias"])
        with open(SCR / "baseline_eval.log", "w", encoding="utf-8") as fh:
            log(f"конфигурация (подобрана на seed {TUNE_SEEDS}): {cfg_name(cfg)}; оценка seed {EVAL_SEEDS}", fh)
            evaluate(rig, cfg, fh)


if __name__ == "__main__":
    main()
