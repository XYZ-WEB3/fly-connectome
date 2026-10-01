"""Задача A (игра 67COIN): может ли награда в грибовидном теле РОДНЫМ путём изменить руль к зрительной метке?

Родной путь (анатомия: _scratch/a1_route_anatomy.log):
  VPN стороны X -> KCg-d стороны X (92% синапсов VPN->KC ипсилатерально) -> MBON27 стороны X (100% его KC — своей
  стороны) -> DNa ПРОТИВОПОЛОЖНОЙ стороны (левый MBON27 -> руль вправо). То есть MB-путь метки толкает ОТ метки.
  Учитель отсека γ5d: MBON27_L — PAM01 слева 85%, MBON27_R — PAM01 справа 76%.
Прогноз: награда + метка слева -> депрессия KC_L->MBON27_L -> меньше толчка вправо -> руль сдвигается К метке.
Потолок (без обучения, веса KC->MBON27 x0.1; _scratch/a2_probe.log): метка справа (LC10 5 + VPN 250 Гц) -4.1 Гц,
метка слева -0.3..+1.1 Гц (MBON32_L (ГАМК) сам тормозит DNa_R и гасит вклад MBON27_L).

Протокол на «муху» (seed x обучаемая сторона):
  претест: метка L и R по REPS раз x 1 с (обучение выкл.) -> обучение N_TRAIN проб: метка 2 с, US 0.5-1.5 с
  (PAM 50 Гц, «как оптогенетика»), 0.5 с после -> посттест как претест.
  «К метке» T = руль x (+1 для метки слева, -1 справа), Гц. ΔT = после - до.
Запуск: .venv/Scripts/python.exe -W ignore learning/game/native_route.py [--analyze]  (~9 мин GPU)

ИЗМЕРЕНО (5 seed, метка LC10 5 + VPN 250 Гц; _scratch/a3_native_route.log, a3b_per_side.log, a4, a5):
  * метка СПРАВА: ΔT_R +1.90 ± 0.23 Гц (пул 5 условий, где PAM01 достаёт до γ5d справа, n = 25, p = 2e-8, >0 у 23/25);
    контроли (noUS, alt_noUS, PAM01 с другой стороны; n = 15) +0.10 ± 0.21. T_R 3.5 -> ~5.4 Гц. MBON27_R на метку 56 -> 1-12 Гц.
    PAM01 с двух сторон +2.17 ± 0.46 (p = 0.009, 5/5), только своей стороны +1.72 ± 0.22 (p = 0.001, 5/5),
    награда из 4 PAM +1.53 ± 0.58 (p = 0.06), PAM01 с чужой стороны -0.06 (память 0: учитель там 13-24% -> ниже порога 20 Гц).
  * метка СЛЕВА: эффекта нет (-0.14 ± 0.18, p = 0.44), хотя MBON27_L 56 -> 0 Гц. Причина (a5): MBON32_L (γ2, ГАМК,
    учит PPL103 — наказание, не награда) на ту же метку 34 Гц и тормозит DNa_R; с заглушённым MBON32 потолок слева
    +3.7 Гц, как справа. К тому же при LC10 5 Гц DNa_R на метку L уже ~1.5 Гц — ниже некуда.
  * метка слабее (VPN 150, MBON27 31 Гц): сдвига нет (+0.15 ± 0.28) — MBON27 ниже порога, при котором он двигает DNa.
  * чередование L/R как в игре: обе стороны учатся независимо (KC->MBON27 строго своей стороны). С «DA без KC»
    (lr_pot 0.5) US с двух сторон стирает память другой стороны (0.46 -> 0.15-0.19), US только своей стороны — нет (0.35-0.43).
  * потолок (a4, все KC->MBON27 x0.1): справа +1.3..+4.3 Гц, слева +0.3..+0.9 Гц; знак руля обучение не переворачивает
    (без LC10 муха отворачивает от метки справа: -11 -> -7 Гц).
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "_scratch" / "a3_native_route.pkl"

SEEDS = [0, 1, 2, 3, 4]
N_TRAIN = 12          # проб обучения на сторону
REPS = 12             # повторов теста на сторону
US_HZ = 50.0
KEYS = ["steer", "DNa_L", "DNa_R", "MBON27_L", "MBON27_R", "MBON32_L", "MBON32_R", "MBON26_L", "MBON26_R",
        "MBON07_L", "MBON07_R", "MBON01_L", "MBON01_R", "nKC_L", "nKC_R"]

# имя: (сила метки VPN, расписание, типы DAN для US, стороны US, lr_pot)
CONDS = {
    "noUS":           (250, "one", [], "", 0.0),
    "PAM01_bi":       (250, "one", ["PAM01"], "bi", 0.0),
    "PAM01_ipsi":     (250, "one", ["PAM01"], "ipsi", 0.0),
    "PAM01_contra":   (250, "one", ["PAM01"], "contra", 0.0),
    "REWARD4_bi":     (250, "one", ["PAM01", "PAM02", "PAM10", "PAM11"], "bi", 0.0),
    "noUS_v150":      (150, "one", [], "", 0.0),
    "PAM01_bi_v150":  (150, "one", ["PAM01"], "bi", 0.0),
    "alt_noUS":       (250, "alt", [], "", 0.0),
    "alt_PAM01_bi":   (250, "alt", ["PAM01"], "bi", 0.0),
    "alt_bi_pot":     (250, "alt", ["PAM01"], "bi", 0.5),
    "alt_ipsi_pot":   (250, "alt", ["PAM01"], "ipsi", 0.5),
}
LC10_HZ = 5.0


def us_sides(mode, side):
    other = "R" if side == "L" else "L"
    return {"bi": "LR", "ipsi": side, "contra": other, "": ""}[mode]


def test(f, vpn):
    """REPS раз метка L и R по 1 с; обучение выключено. -> {side: {key: массив}}."""
    f.pl.enabled = False
    out = {"L": {k: [] for k in KEYS}, "R": {k: [] for k in KEYS}}
    for _ in range(REPS):
        for s in "LR":
            f.fresh(); f.set_cue(s, lc10=LC10_HZ, vpn=vpn)
            r = f.rates(f.b.run(1000, tensor=True), 1000)
            for k in KEYS:
                out[s][k].append(r[k])
    f.set_cue(None)
    return {s: {k: np.asarray(v) for k, v in d.items()} for s, d in out.items()}


def train_trial(f, side, vpn, types, mode):
    f.fresh(); f.pl.enabled = True
    f.set_cue(side, lc10=LC10_HZ, vpn=vpn)
    f.pl.run(500)
    if types:
        f.set_us(types, us_sides(mode, side), US_HZ)
    f.pl.run(1000)
    f.us_off()
    f.pl.run(500)
    f.set_cue(None)
    f.pl.run(500)
    f.pl.enabled = False


def run_fly(f, cond, side):
    vpn, sched, types, mode, pot = CONDS[cond]
    f.pl.reset_memory()
    f.pl.p["lr_pot"] = pot
    pre = test(f, vpn)
    t0 = time.time()
    if sched == "one":
        for _ in range(N_TRAIN):
            train_trial(f, side, vpn, types, mode)
    else:                                   # чередование L, R, L, R ... как в игре
        for _ in range(N_TRAIN):
            for s in "LR":
                train_trial(f, s, vpn, types, mode)
    t_train = time.time() - t0
    post = test(f, vpn)
    rec = dict(cond=cond, side=side, pre=pre, post=post, mem27=f.mem27(),
               mem=f.pl.memory_strength()["by_mbon"], t_train=t_train)
    f.pl.p["lr_pot"] = 0.0
    f.pl.reset_memory()
    return rec


def collect():
    from fly_setup import Fly
    t0 = time.time()
    f = Fly(seed=SEEDS[0])
    print(f"мозг готов за {time.time() - t0:.1f} с", flush=True)
    recs = []
    for seed in SEEDS:
        f.reseed(seed)
        for cond, (vpn, sched, *_rest) in CONDS.items():
            for side in (("L", "R") if sched == "one" else ("LR",)):
                t1 = time.time()
                r = run_fly(f, cond, side)
                r["seed"] = seed
                recs.append(r)
                tr = side if side != "LR" else "L"
                dT = tow(r["post"], tr) - tow(r["pre"], tr)
                print(f"seed {seed} {cond:14s} {side:2s}: ΔT({tr}) {dT:+6.2f}  mem27 L {r['mem27']['L']:.2f} "
                      f"R {r['mem27']['R']:.2f}  ({time.time() - t1:.1f} с)", flush=True)
        with open(OUT, "wb") as fh:
            pickle.dump(recs, fh)
    print(f"всего {time.time() - t0:.0f} с", flush=True)
    return recs


def tow(d, s):
    """Руль к метке стороны s (Гц), среднее по повторам."""
    return float(np.mean(d[s]["steer"])) * (1 if s == "L" else -1)


def m(d, s, k):
    return float(np.mean(d[s][k]))


def analyze(recs):
    print("\n" + "=" * 110)
    print("ИТОГ. T = руль к метке (Гц); ΔT = после - до; среднее ± SE по мухам; p — парный t-тест ΔT против 0;")
    print("p(vs noUS) — Уэлч против noUS той же силы метки. «обуч.» = сторона, которую учили; «необуч.» = другая.")
    print("=" * 110)
    base = {}
    for vpn in (250, 150):
        nm = "noUS" if vpn == 250 else "noUS_v150"
        base[vpn] = np.array([tow(r["post"], r["side"]) - tow(r["pre"], r["side"]) for r in recs if r["cond"] == nm])
    rows = []
    for cond, (vpn, sched, types, mode, pot) in CONDS.items():
        rr = [r for r in recs if r["cond"] == cond]
        if not rr:
            continue
        if sched == "one":
            tr = [r["side"] for r in rr]
            un = ["R" if s == "L" else "L" for s in tr]
        else:
            tr, un = ["L"] * len(rr) + ["R"] * len(rr), [None] * (2 * len(rr))
            rr = rr + rr
        dT = np.array([tow(r["post"], s) - tow(r["pre"], s) for r, s in zip(rr, tr)])
        T0 = np.array([tow(r["pre"], s) for r, s in zip(rr, tr)])
        dU = np.array([tow(r["post"], s) - tow(r["pre"], s) for r, s in zip(rr, un) if s])
        d27 = np.array([m(r["post"], s, f"MBON27_{s}") - m(r["pre"], s, f"MBON27_{s}") for r, s in zip(rr, tr)])
        m27pre = np.array([m(r["pre"], s, f"MBON27_{s}") for r, s in zip(rr, tr)])
        mem = np.array([r["mem27"][s] for r, s in zip(rr, tr)])
        memU = np.array([r["mem27"][s] for r, s in zip(rr, un) if s])
        p = stats.ttest_1samp(dT, 0).pvalue if len(dT) > 1 else np.nan
        pb = stats.ttest_ind(dT, base[vpn], equal_var=False).pvalue if cond not in ("noUS", "noUS_v150") else np.nan
        bySide = {s: dT[np.array(tr) == s] for s in "LR"}
        rows.append((cond, len(dT), T0.mean(), dT.mean(), dT.std(ddof=1) / np.sqrt(len(dT)), p, pb, int((dT > 0).sum()),
                     bySide["L"].mean(), bySide["R"].mean(),
                     dU.mean() if len(dU) else np.nan, (dU.std(ddof=1) / np.sqrt(len(dU))) if len(dU) > 1 else np.nan,
                     m27pre.mean(), d27.mean(), mem.mean(), memU.mean() if len(memU) else np.nan))
    hdr = ("условие", "n", "T до", "ΔT", "SE", "p", "p(vs noUS)", "ΔT>0", "ΔT L", "ΔT R", "ΔT необуч.", "SE",
           "M27 до", "ΔM27", "mem27 обуч.", "mem27 необуч.")
    print(("{:15s}" + "{:>11s}" * (len(hdr) - 1)).format(*hdr))
    for r in rows:
        print(("{:15s}{:11d}" + "{:11.2f}" * 3 + "{:11.4f}" * 2 + "{:11d}" + "{:11.2f}" * 8).format(*r))

    # по seed (среднее по сторонам) — устойчивость
    print("\nΔT по seed (обучаемая сторона, среднее L и R):")
    for cond in CONDS:
        rr = [r for r in recs if r["cond"] == cond]
        line = []
        for sd in SEEDS:
            x = [r for r in rr if r["seed"] == sd]
            if CONDS[cond][1] == "one":
                v = [tow(r["post"], r["side"]) - tow(r["pre"], r["side"]) for r in x]
            else:
                v = [tow(r["post"], s) - tow(r["pre"], s) for r in x for s in "LR"]
            line.append(np.mean(v) if v else np.nan)
        print(f"  {cond:15s} " + " ".join(f"{v:+6.2f}" for v in line))

    # детали: DNa и MBON до/после для лучших условий
    print("\nДетали (метка обучаемой стороны), средние по мухам, Гц: до -> после")
    for cond in ("noUS", "PAM01_bi", "PAM01_ipsi", "REWARD4_bi", "alt_PAM01_bi"):
        rr = [r for r in recs if r["cond"] == cond]
        for s in "LR":
            x = [r for r in rr if r["side"] in (s, "LR")]
            if not x:
                continue
            parts = []
            for k in ("DNa_L", "DNa_R", "MBON27_L", "MBON27_R", "MBON32_L", "MBON32_R", "MBON07_L", "MBON07_R", "MBON01_L", "MBON01_R"):
                a = np.mean([m(r["pre"], s, k) for r in x]); b = np.mean([m(r["post"], s, k) for r in x])
                parts.append(f"{k} {a:.1f}->{b:.1f}")
            print(f"  {cond:13s} метка {s}: " + "; ".join(parts))
    return rows


def plot(recs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [c for c in CONDS if any(r["cond"] == c for r in recs)]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, c in enumerate(conds):
        rr = [r for r in recs if r["cond"] == c]
        if CONDS[c][1] == "one":
            dT = [tow(r["post"], r["side"]) - tow(r["pre"], r["side"]) for r in rr]
            sides = [r["side"] for r in rr]
        else:
            dT = [tow(r["post"], s) - tow(r["pre"], s) for r in rr for s in "LR"]
            sides = [s for r in rr for s in "LR"]
        x = i + np.random.default_rng(i).uniform(-0.15, 0.15, len(dT))
        ax.scatter(x, dT, s=14, c=["#3b78c2" if s == "L" else "#d0612b" for s in sides], alpha=0.8, zorder=3)
        ax.errorbar(i, np.mean(dT), yerr=np.std(dT, ddof=1) / np.sqrt(len(dT)), fmt="_", color="k", ms=18, capsize=4, zorder=4)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(conds)), conds, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("ΔT руль к метке, Гц (после - до)")
    ax.set_title("Награда в MB -> родной руль к метке (синие: метка слева, оранжевые: справа); 5 seed")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"график: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        with open(OUT, "rb") as fh:
            recs = pickle.load(fh)
    else:
        recs = collect()
    analyze(recs)
    plot(recs, HERE / "_scratch" / "a3_native_route.png")
