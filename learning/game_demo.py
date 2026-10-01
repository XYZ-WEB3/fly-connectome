"""Демо для игры «лево/право» (67COIN): муха выбирает сторону, за монетку получает сахар,
и по ходу игры её выбор меняется из-за памяти в грибовидном теле. Всё измерено, мосты подписаны.

Раунд:
  1) слева и справа по картинке: «монетка» и «пусто» (сторона монетки случайна);
     картинка = свой набор 5% KC по 20 Гц (МОДЕЛЬ: «картинка -> KC напрямую, как оптогенетика»;
     настоящее зрение VPN->KC даёт лишь 1.5-1.8% KC и MBON молчат — задача B);
  2) муха по очереди «смотрит» влево и вправо (по 1 с, МОДЕЛЬ: последовательный осмотр), считаем выход MB;
  3) выбор (МОДЕЛЬ-мост; в коннектоме модели выученное изменение MBON до руля не доходит):
       мост "V": P(лево) = σ(β·(V_L − V_R)), V — MBON с известной валентностью (Aso 2014b), β = 0.1/Гц;
       мост "R": P(лево) = σ(β·(R_R − R_L)), R — MBON отсеков награды (MBON01/06/07/27), β = 0.3/Гц
                 (меньше ответ «избегающих» MBON отсеков награды -> больше хочется);
  4) исход: 1.5 с картинки выбранной стороны (+0.5 с тишины); если это монетка — на 0.5–1.5 с сахар
     (20 LB3 по 200 Гц, настоящий вкусовой путь -> MN9 = хоботок) + PAM01/02/10/11 по 50 Гц
     (МОДЕЛЬ-мост «как оптогенетика»: сахар до PAM в модели не доходит, задача C);
     если «пусто» — та же картинка без награды. Обучение включено всё время исхода.
Реципрокно: у половины мух монетка = набор kcA, у другой — kcB. 5 seed.
Запуск: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -W ignore learning/game_demo.py [--rounds 30] [--seeds 0 1 2 3 4]
"""
import argparse
import io
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "tiktok"))
import brain as B  # noqa: E402
import conditioning as C  # noqa: E402
import plasticity as PL  # noqa: E402
from senses import SUGAR_IDS_630, MN9_ID  # noqa: E402

SCR = HERE / "_scratch"
BRIDGES = {"V": 0.1, "R": 0.3}
SUGAR_HZ = 200.0


def play(fly, coin, decoy, bridge, beta, rounds, rng, sugar, mn9):
    """Одна игра. -> список раундов (dict)."""
    pl, b = fly.pl, fly.b
    pl.reset_memory()
    ridx = torch.as_tensor([fly.vr.types.index(t) for t in fly.rew_mbon], device=b.dev)
    kc_coin, kc_decoy = fly.kcset[coin], fly.kcset[decoy]
    log = []
    for k in range(rounds):
        coin_left = bool(rng.random() < 0.5)
        items = {"L": coin if coin_left else decoy, "R": decoy if coin_left else coin}
        val = {}
        pl.enabled = False
        for side, it in items.items():                      # осмотр: по 1 с на сторону
            b.reset(); pl.reset_traces()
            fly.cues.set({it: 1.0})
            c = b.run(1000.0, tensor=True)
            r = fly.vr.rates_t(c, 1000.0)
            val[side] = float(fly.vr.sign @ r) if bridge == "V" else -float(r[ridx].sum())
        pL = 1.0 / (1.0 + np.exp(-beta * (val["L"] - val["R"])))
        left = bool(rng.random() < pL)
        chosen = items["L" if left else "R"]
        got = chosen == coin
        # исход: 1.5 с картинки выбранной стороны; монетка -> сахар + PAM 0.5–1.5 с
        pl.enabled = True
        b.reset(); pl.reset_traces()
        fly.cues.set({chosen: 1.0})
        mn = torch.zeros((), device=b.dev)
        state = None
        for i in range(int(round(2000.0 / C.BIN))):
            t = i * C.BIN
            st = (t < 1500.0, got and 500.0 <= t < 1500.0)          # (картинка, награда)
            if st != state:
                fly.cues.set({chosen: 1.0} if st[0] else {})
                b.rate_in[fly.us] = C.US_HZ if st[1] else 0.0
                b.rate_in[sugar] = SUGAR_HZ if st[1] else 0.0
                state = st
            c = b.run(C.BIN, tensor=True)
            pl.update(c, C.BIN)
            if st[1]:
                mn += c[mn9]
        mn = float(mn)
        pl.enabled = False
        pc = pL if coin_left else 1 - pL
        log.append(dict(round=k + 1, coin_left=coin_left, chose_left=left, got=got, p_coin=pc,
                        v_coin=val["L" if coin_left else "R"], v_decoy=val["R" if coin_left else "L"],
                        mn9_hz=mn / 1.0 if got else np.nan,
                        mem_coin=pl.memory_strength(kc=kc_coin, comps=fly.rew_comps)["total"],
                        mem_decoy=pl.memory_strength(kc=kc_decoy, comps=fly.rew_comps)["total"]))
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    pkl = SCR / "game_results.pkl"
    if not a.analyze:
        ann = B.load_annotations()
        m = PL.load_map()
        runs = []
        for seed in a.seeds:
            fly = C.Fly(seed, ann, m)
            fly.set_mode("kc")
            sugar = torch.as_tensor([fly.b.idx[r] for r in SUGAR_IDS_630 if r in fly.b.idx], device=fly.b.dev)
            mn9 = fly.b.idx[MN9_ID]
            for bridge, beta in BRIDGES.items():
                for coin, decoy in (("kcA", "kcB"), ("kcB", "kcA")):
                    t0 = time.time()
                    rng = np.random.default_rng(10 * seed + (coin == "kcB"))
                    log = play(fly, coin, decoy, bridge, beta, a.rounds, rng, sugar, mn9)
                    df = pd.DataFrame(log)
                    runs.append(dict(seed=seed, bridge=bridge, coin=coin, log=df))
                    q = len(df) // 3
                    print(f"  seed {seed} мост {bridge} монетка={coin}: P(монетка) раунды 1-{q}: {df.p_coin[:q].mean():.2f}, "
                          f"последние {q}: {df.p_coin[-q:].mean():.2f}; выбрал монетку {int(df.got.sum())}/{len(df)}; "
                          f"MN9 при сахаре {df.mn9_hz.mean():.0f} Гц; память монетки {df.mem_coin.iloc[-1]:.2f}, "
                          f"пустышки {df.mem_decoy.iloc[-1]:.2f}; {time.time() - t0:.1f} с", flush=True)
            del fly
            torch.cuda.empty_cache()
        with open(pkl, "wb") as f:
            pickle.dump(runs, f)
    with open(pkl, "rb") as f:
        runs = pickle.load(f)

    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for bridge, colr in (("V", "#1f77b4"), ("R", "#2ca02c")):
        rr = [r for r in runs if r["bridge"] == bridge]
        P = np.stack([r["log"].p_coin.values for r in rr])
        G = np.stack([r["log"].got.values.astype(float) for r in rr])
        Mc = np.stack([r["log"].mem_coin.values for r in rr]); Md = np.stack([r["log"].mem_decoy.values for r in rr])
        n, T = P.shape
        x = np.arange(1, T + 1)
        q = T // 3
        first, last = P[:, 0], P[:, -q:].mean(1)            # раунд 1 — до любой награды (наивно)
        t = stats.ttest_rel(last, first)
        print(f"мост {bridge} (β={BRIDGES[bridge]}), игр {n} ({len({r['seed'] for r in rr})} seed x 2): P(монетка) в раунде 1 (наивно) "
              f"{first.mean():.3f} ± {first.std(ddof=1) / np.sqrt(n):.3f} -> последние {q} раундов {last.mean():.3f} ± "
              f"{last.std(ddof=1) / np.sqrt(n):.3f} (парный t={t.statistic:+.2f}, p={t.pvalue:.2g}, рост у {int((last > first).sum())}/{n}); "
              f"доля выборов монетки {G[:, :q].mean():.2f} -> {G[:, -q:].mean():.2f}; память монетки {Mc[:, -1].mean():.3f}, "
              f"пустышки {Md[:, -1].mean():.3f}; MN9 при сахаре {np.nanmean([r['log'].mn9_hz.mean() for r in rr]):.1f} Гц")
        ax[0].errorbar(x, P.mean(0), yerr=P.std(0, ddof=1) / np.sqrt(n), color=colr, marker=".", capsize=2,
                       label=f"мост {bridge} (β={BRIDGES[bridge]}/Гц)")
        k = 5
        Gs = np.stack([np.convolve(g, np.ones(k) / k, mode="valid") for g in G])
        ax[1].plot(np.arange(k, T + 1), Gs.mean(0), color=colr, label=f"мост {bridge}")
        ax[2].plot(x, Mc.mean(0), color=colr, label=f"мост {bridge}: KC «монетки»")
        ax[2].plot(x, Md.mean(0), color=colr, ls="--", label=f"мост {bridge}: KC «пусто»")
    ax[0].axhline(0.5, color="gray", lw=0.5); ax[1].axhline(0.5, color="gray", lw=0.5)
    ax[0].set_ylim(0, 1); ax[1].set_ylim(0, 1)
    ax[0].set_title("P(выбор монетки) по модели, ±SEM", fontsize=10); ax[0].set_xlabel("раунд"); ax[0].legend()
    ax[1].set_title("фактические выборы монетки\n(скользящее среднее по 5 раундам)", fontsize=10); ax[1].set_xlabel("раунд")
    ax[2].set_title("сила памяти в отсеках награды\n(1 − Σw/Σw0, связи KC картинки)", fontsize=10)
    ax[2].set_xlabel("раунд"); ax[2].legend(fontsize=8)
    fig.suptitle("Игра лево/право: картинка -> 5% KC (как оптогенетика); монетка -> сахар (LB3->MN9) + PAM 50 Гц (мост); "
                 "5 seed x 2 реципрокно", fontsize=10, y=1.05)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    with open(SCR / "game_curve.png", "wb") as f:
        f.write(buf.getvalue())
    print("график:", SCR / "game_curve.png")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"всего {time.time() - t0:.0f} с")
