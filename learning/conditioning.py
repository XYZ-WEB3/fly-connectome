"""Классическое (павловское) обусловливание мухи в модели FlyWire (PRM): CS + «награда» -> пластичность KC->MBON.

БИОЛОГИЯ: схема опыта как у Tempel 1983 / Aso & Rubin 2016 — запах CS+ в паре с наградой, CS- без неё,
  реципрокно (у половины мух CS+ = A, у другой половины CS+ = B), потом тест выбора.
МОДЕЛЬ (наш выбор, подписывать):
  * US «награда» = прямое включение PAM с проверенной наградой (Aso & Rubin 2016: PAM01 γ5, PAM02 β'2a,
    PAM10 β1, PAM11 α1) по 50 Гц — «как оптогенетика». Настоящий сахар до PAM в модели НЕ доходит (задача C).
  * CS «запах» = ORN (обе антенны) + две правки усика (learning/cues.py: eLN заглушены, PN без утомления);
    CS «KC» = прямо 5% клеток Кеньона по 20 Гц (как оптогенетика KC).
  * выход всех DAN заглушен (быстрый дофамин модели — артефакт), их спайки учат через plasticity.MBPlasticity.
  * между пробами brain.reset() (как длинный перерыв; адаптация и утомление синапсов обнуляются, веса — нет).
  * тесты — с выключенным обучением (замер не пишет память).
  * «выбор» (а) мост: муха по очереди «нюхает» CS+ и CS- (по 1 с), выбирает сторону с большим V
    (ValenceReadout, валентность MBON по Aso 2014b); (б) родной путь без моста: CS+ в одну антенну, CS- в другую,
    руль = DNa01+DNa02 слева − справа.
Условия: paired (US 0.5–1.5 с внутри CS+ 0–2 с), unpaired (US 10–11 с, через 8 с после конца CS+), noUS.

Запуск:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -W ignore learning/conditioning.py [--seeds 0 1 2 3 4]
         [--trials 6] [--ntest 6] [--modes odor kc] [--analyze]   (--analyze — только графики/таблицы из pkl)
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
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import brain as B  # noqa: E402
import plasticity as PL  # noqa: E402
from cues import Cues, kc_sets  # noqa: E402

PRM = dict(U=0.1, tau_d=300, a_inc=0.1, tau_a=3000)
SCR = HERE / "_scratch"
PKL = SCR / "cond_results.pkl"
REWARD = ["PAM01", "PAM02", "PAM10", "PAM11"]      # награда проверена (Aso & Rubin 2016)
US_HZ = 50.0
BIN = 18.0                                          # мс: шаг обучения (10 блоков по 1.8 мс)
CS_MS, US_PAIRED, US_UNPAIRED = 2000.0, (500.0, 1500.0), (10000.0, 11000.0)
TAIL_PAIRED, TAIL_UNPAIRED = 3500.0, 11500.0
PRE_US_MS = 500.0                                   # окно кривой обучения: до начала US
TEST_MS = 1000.0
NATIVE_REPS = 4
CONDS = ["paired", "unpaired", "noUS"]


class Fly:
    """Один мозг (seed) + пластичность + CS. reset_memory() делает из него «новую муху»."""

    def __init__(self, seed, ann, m):
        t0 = time.time()
        self.b = b = B.FlyBrain(params=PRM, seed=seed)
        b.enable_graph()
        self.pl = PL.MBPlasticity(b, m)
        self.vr = PL.ValenceReadout(b, m)
        self.cues = Cues(b, ann)
        self._pn_all = self.cues.pn
        self.us = torch.as_tensor(sum((m["dan"][t]["idx_left"] + m["dan"][t]["idx_right"] for t in REWARD), []),
                                  device=b.dev)
        ty = lambda ts, s: [b.idx[r] for r in ann.index[ann.cell_type.isin(ts) & (ann.side == s)] if r in b.idx]
        self.dna = {s: torch.as_tensor(ty(["DNa01", "DNa02"], side), device=b.dev) for s, side in (("L", "left"), ("R", "right"))}
        kcL = sorted(i for v in m["kc"]["by_subtype"].values() for i in v["idx_left"])
        kcR = sorted(i for v in m["kc"]["by_subtype"].values() for i in v["idx_right"])
        self.kc_side = (np.asarray(kcL), np.asarray(kcR))
        sA, sB = kc_sets(self.pl.kc_idx_np, 2, 0.05, seed=1000 + seed)
        for k, ix in (("kcA", sA), ("kcB", sB)):
            self.cues.add_kc_cue(k, ix, 20.0)
            self.cues.add_kc_cue(k + "@L", np.intersect1d(ix, self.kc_side[0]), 20.0)
            self.cues.add_kc_cue(k + "@R", np.intersect1d(ix, self.kc_side[1]), 20.0)
        self.kcset = {"kcA": sA, "kcB": sB}
        # отсеки, которые учит наша «награда», и MBON, у которых >= 50% веса KC->MBON в этих отсеках
        self.rew_comps = self.pl.taught_by(REWARD)
        w0 = self.pl.w0.cpu().numpy()
        inrew = np.isin(self.pl.edge_comp, [self.pl.comp_names.index(c) for c in self.rew_comps])
        self.rew_mbon = [t for k, t in enumerate(self.pl.mbon_types)
                         if w0[(self.pl.edge_mbon == k) & inrew].sum() >= 0.5 * w0[self.pl.edge_mbon == k].sum()]
        print(f"  отсеки награды: {self.rew_comps}; их MBON: {self.rew_mbon}", flush=True)
        print(f"  мозг seed={seed} готов за {time.time() - t0:.1f} с", flush=True)

    def set_mode(self, mode):
        """odor: eLN заглушены + PN без утомления; kc: усик как в исходной модели (PN-трюк и глушение eLN сняты)."""
        c = self.cues
        if mode == "kc":
            c.pn = None
            self.b.silenced[c.eln] = False
        else:
            c.pn = self._pn_all
            self.b.silenced[c.eln] = True

    # ---------- пробы ----------
    @torch.no_grad()
    def test(self, cue, ms=TEST_MS):
        """Одна тестовая проба без обучения: -> (ставки MBON по типам, V, активные KC)."""
        b = self.b
        b.reset(); self.pl.reset_traces()
        self.cues.set(cue)
        c = b.run(ms, tensor=True)
        r = self.vr.rates_t(c, ms)
        v = float(self.vr.sign @ r)
        steer = float((c[self.dna["L"]].mean() - c[self.dna["R"]].mean()) * 1000.0 / ms)
        return r.cpu().numpy(), v, (c[self.pl.kc_idx] > 0).cpu().numpy(), steer, int((c > 0).sum())

    @torch.no_grad()
    def train_trial(self, cue, us=None, total=TAIL_PAIRED):
        """Проба обучения с включённой пластичностью. -> ставки MBON (типы) в окне до US (0–500 мс)."""
        b, pl = self.b, self.pl
        b.reset(); pl.reset_traces()
        pre = torch.zeros(b.N, device=b.dev)
        state = None
        for i in range(int(round(total / BIN))):
            t = i * BIN
            st = (t < CS_MS, us is not None and us[0] <= t < us[1])
            if st != state:
                self.cues.set(cue if st[0] else {})
                b.rate_in[self.us] = US_HZ if st[1] else 0.0
                state = st
            c = b.run(BIN, tensor=True)
            pl.update(c, BIN)
            if t + BIN <= PRE_US_MS:
                pre += c
        r = self.vr.rates_t(pre, BIN * int(PRE_US_MS // BIN))
        return r.cpu().numpy(), float(self.vr.sign @ r)


def run_fly(fly, mode, csp, csm, cond, n_trials, n_test):
    """Одна «муха»: претест -> обучение -> посттест. -> dict с сырыми данными."""
    pl = fly.pl
    pl.reset_memory()
    pl.enabled = False
    out = dict(mode=mode, csp=csp, csm=csm, cond=cond)
    t0 = time.time()

    def tests(tag):
        for k, name in (("p", csp), ("m", csm)):
            rr = [fly.test({name: 1.0}) for _ in range(n_test)]
            out[f"{tag}_{k}_rates"] = np.stack([x[0] for x in rr])
            out[f"{tag}_{k}_V"] = np.array([x[1] for x in rr])
            out[f"{tag}_{k}_kc"] = np.stack([x[2] for x in rr]).mean(0)
            out[f"{tag}_{k}_nact"] = np.array([x[4] for x in rr])
        # родной путь: CS+ в одну сторону, CS- в другую (без моста), руль L−R
        sl = [fly.test({csp + "@L": 1.0, csm + "@R": 1.0})[3] for _ in range(NATIVE_REPS)]
        sr = [fly.test({csm + "@L": 1.0, csp + "@R": 1.0})[3] for _ in range(NATIVE_REPS)]
        out[f"{tag}_native"] = (np.mean(sl) - np.mean(sr)) / 2       # > 0 = поворот к CS+
        out[f"{tag}_native_raw"] = (np.array(sl), np.array(sr))

    tests("pre")
    kc_p = out["pre_p_kc"] >= 0.5 if mode == "odor" else np.isin(pl.kc_idx_np, fly.kcset[csp])
    kc_m = out["pre_m_kc"] >= 0.5 if mode == "odor" else np.isin(pl.kc_idx_np, fly.kcset[csm])
    out["n_kc_p"], out["n_kc_m"] = int(kc_p.sum()), int(kc_m.sum())
    out["n_kc_overlap"] = int((kc_p & kc_m).sum())
    kcp_idx, kcm_idx = pl.kc_idx_np[kc_p], pl.kc_idx_np[kc_m]
    pl.enabled = True
    curve = []
    for k in range(n_trials):
        if cond == "paired":
            rp, vp = fly.train_trial({csp: 1.0}, US_PAIRED, TAIL_PAIRED)
        elif cond == "unpaired":
            rp, vp = fly.train_trial({csp: 1.0}, US_UNPAIRED, TAIL_UNPAIRED)
        else:
            rp, vp = fly.train_trial({csp: 1.0}, None, TAIL_PAIRED)
        rm, vm = fly.train_trial({csm: 1.0}, None, TAIL_PAIRED)
        curve.append(dict(trial=k + 1, Vp=vp, Vm=vm, rates_p=rp, rates_m=rm,
                          mem_p=pl.memory_strength(kc=kcp_idx)["total"], mem_m=pl.memory_strength(kc=kcm_idx)["total"],
                          mem_p_rew=pl.memory_strength(kc=kcp_idx, comps=fly.rew_comps)["total"],
                          mem_m_rew=pl.memory_strength(kc=kcm_idx, comps=fly.rew_comps)["total"],
                          mem_all=pl.memory_strength()["total"]))
    pl.enabled = False
    out["curve"] = curve
    out["mem_p"] = pl.memory_strength(kc=kcp_idx)
    out["mem_m"] = pl.memory_strength(kc=kcm_idx)
    out["mem_all"] = pl.memory_strength()
    out["mem_p_rew"] = pl.memory_strength(kc=kcp_idx, comps=fly.rew_comps)["total"]
    out["mem_m_rew"] = pl.memory_strength(kc=kcm_idx, comps=fly.rew_comps)["total"]
    out["rew_comps"], out["rew_mbon"] = fly.rew_comps, fly.rew_mbon
    dose = pl.da_dose.cpu().numpy()
    out["da_dose"] = {n: float(dose[2 * c] + dose[2 * c + 1]) for c, n in enumerate(pl.comp_names)}
    tests("post")
    out["sec"] = time.time() - t0
    return out


def collect(args):
    ann = B.load_annotations()
    m = PL.load_map()
    res = []
    for seed in args.seeds:
        fly = Fly(seed, ann, m)
        for mode in args.modes:
            fly.set_mode(mode)
            pair = ("odor_A", "odor_B") if mode == "odor" else ("kcA", "kcB")
            for cond in CONDS:
                for csp, csm in (pair, pair[::-1]):
                    r = run_fly(fly, mode, csp, csm, cond, args.trials, args.ntest)
                    r["seed"] = seed
                    res.append(r)
                    c = r["curve"]
                    print(f"  seed {seed} {mode:4s} {cond:8s} CS+={csp}: V(CS+) {r['pre_p_V'].mean():+.1f} -> "
                          f"{r['post_p_V'].mean():+.1f}, V(CS-) {r['pre_m_V'].mean():+.1f} -> {r['post_m_V'].mean():+.1f} Гц; "
                          f"память CS+ {c[-1]['mem_p']:.2f} / CS- {c[-1]['mem_m']:.2f}; руль к CS+ "
                          f"{r['pre_native']:+.2f} -> {r['post_native']:+.2f}; {r['sec']:.1f} с", flush=True)
        del fly
        torch.cuda.empty_cache()
    with open(PKL, "wb") as f:
        pickle.dump(dict(res=res, args=vars(args),
                         mbon_types=list(m["mbon"].keys())), f)
    return res


# ======================= анализ =======================
BETA = 0.1          # 1/Гц: «решительность» моста выбора, P(CS+) = σ(β·(V(CS+) − V(CS−))) — МОДЕЛЬ
BETAS = (0.05, 0.1, 0.2)


def save_fig(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    with open(SCR / name, "wb") as f:        # open() понимает кириллицу в пути
        f.write(buf.getvalue())


def p_choose(r, tag, beta=BETA):
    """Вероятность выбрать CS+: по всем парам проб (CS+ i, CS− j), σ(β·(V+ − V−))."""
    vp, vm = r[f"{tag}_p_V"][:, None], r[f"{tag}_m_V"][None, :]
    return float(np.mean(1.0 / (1.0 + np.exp(-beta * (vp - vm)))))


def recip_pi(rc, tag, beta=BETA):
    """Реципрокный PI по seed: 2·(ср. P(CS+) в группах A+ и B+) − 1 (врождённое предпочтение запаха сокращается)."""
    s = pd.Series([p_choose(r, tag, beta) for r in rc], index=[r["seed"] for r in rc]).groupby(level=0).mean()
    return 2 * s.values - 1


def sem(x):
    x = np.asarray(x, dtype=float)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def rew_resp(rates, types, rew):
    j = [types.index(t) for t in rew]
    return rates[..., j].sum(-1)


def analyze():
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(PKL, "rb") as f:
        D = pickle.load(f)
    res, types = D["res"], D["mbon_types"]
    m = PL.load_map()
    vr_ap = [t for t, v in m["mbon"].items() if v["valence"] == "approach" and v["valence_status"] in ("tested", "group")]
    vr_av = [t for t, v in m["mbon"].items() if v["valence"] == "avoid" and v["valence_status"] in ("tested", "group")]
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    modes = sorted({r["mode"] for r in res}, key=["odor", "kc"].index)
    seeds = sorted({r["seed"] for r in res})
    rew = res[0]["rew_mbon"]
    print(f"\n==== АНАЛИЗ: seeds {seeds}, режимы {modes}, запусков «мух» {len(res)} ====")
    print(f"V = Σ притяжение {vr_ap} − Σ избегание {vr_av} (Гц, ср. по нейронам типа)")
    print(f"отсеки награды {res[0]['rew_comps']}; MBON этих отсеков (R = сумма их частот): {rew}")
    print(f"мост выбора: P(CS+) = σ(β·ΔV), β = {BETA}/Гц (МОДЕЛЬ); PI = 2·P − 1, реципрокно по seed")

    summary = []
    for mode in modes:
        print(f"\n######## CS = {mode} ########")
        rr = [r for r in res if r["mode"] == mode]
        print(f"KC в коде (активен в >= 50% проб претеста): CS+ {np.mean([r['n_kc_p'] for r in rr]):.0f}, "
              f"CS- {np.mean([r['n_kc_m'] for r in rr]):.0f}, общих {np.mean([r['n_kc_overlap'] for r in rr]):.1f}")
        for csp in sorted({r["csp"] for r in rr}):
            rg = [r for r in rr if r["csp"] == csp]
            print(f"  врождённо (претест, все условия): V({csp}) {np.mean([r['pre_p_V'].mean() for r in rg]):+.1f} Гц, "
                  f"R({csp}) {np.mean([rew_resp(r['pre_p_rates'], types, rew).mean() for r in rg]):.1f} Гц")
        for cond in CONDS:
            rc = [r for r in rr if r["cond"] == cond]
            n = len(rc)
            dVp = np.array([r["post_p_V"].mean() - r["pre_p_V"].mean() for r in rc])
            dVm = np.array([r["post_m_V"].mean() - r["pre_m_V"].mean() for r in rc])
            dV = dVp - dVm
            Rp0 = np.array([rew_resp(r["pre_p_rates"], types, rew).mean() for r in rc])
            Rp1 = np.array([rew_resp(r["post_p_rates"], types, rew).mean() for r in rc])
            Rm0 = np.array([rew_resp(r["pre_m_rates"], types, rew).mean() for r in rc])
            Rm1 = np.array([rew_resp(r["post_m_rates"], types, rew).mean() for r in rc])
            dR = (Rp1 - Rp0) - (Rm1 - Rm0)
            pi_pre, pi_post = recip_pi(rc, "pre"), recip_pi(rc, "post")
            nat = np.array([r["post_native"] - r["pre_native"] for r in rc])
            mp = np.array([r["mem_p_rew"] for r in rc]); mm = np.array([r["mem_m_rew"] for r in rc])
            t_dV = stats.ttest_1samp(dV, 0.0); t_dR = stats.ttest_1samp(dR, 0.0)
            t_pi = stats.ttest_rel(pi_post, pi_pre) if len(pi_pre) > 1 else None
            t_nat = stats.ttest_1samp(nat, 0.0)
            print(f"\n--- {cond}: {n} запусков = {len(seeds)} seed x 2 реципрокные группы ---")
            print(f"  V: ΔV(CS+) {dVp.mean():+.2f} ± {sem(dVp):.2f}, ΔV(CS-) {dVm.mean():+.2f} ± {sem(dVm):.2f}; "
                  f"разность {dV.mean():+.2f} ± {sem(dV):.2f} Гц (SEM), d={dV.mean() / dV.std(ddof=1):+.2f}, "
                  f"t={t_dV.statistic:+.2f}, p={t_dV.pvalue:.2g}, > 0 у {int((dV > 0).sum())}/{n}")
            print(f"  R (MBON отсеков награды): CS+ {Rp0.mean():.1f} -> {Rp1.mean():.1f} Гц, CS- {Rm0.mean():.1f} -> {Rm1.mean():.1f}; "
                  f"ΔR(CS+) − ΔR(CS-) {dR.mean():+.2f} ± {sem(dR):.2f}, d={dR.mean() / dR.std(ddof=1):+.2f}, "
                  f"p={t_dR.pvalue:.2g}, < 0 у {int((dR < 0).sum())}/{n}")
            print(f"  выбор PI (β={BETA}): до {pi_pre.mean():+.3f} ± {sem(pi_pre):.3f}, после {pi_post.mean():+.3f} ± {sem(pi_post):.3f}"
                  + (f"; парный t={t_pi.statistic:+.2f}, p={t_pi.pvalue:.2g}, рост у {int((pi_post > pi_pre).sum())}/{len(pi_pre)} seed"
                     if t_pi is not None else ""))
            print("  P(выбор CS+) по группам (без реципрокного усреднения): " + "; ".join(
                f"CS+={g}: {np.mean([p_choose(r, 'pre') for r in rc if r['csp'] == g]):.2f} -> "
                f"{np.mean([p_choose(r, 'post') for r in rc if r['csp'] == g]):.2f}" for g in sorted({r['csp'] for r in rc})))
            print("  чувствительность к β: " + "; ".join(
                f"β={bb}: PI {recip_pi(rc, 'pre', bb).mean():+.3f} -> {recip_pi(rc, 'post', bb).mean():+.3f}" for bb in BETAS))
            print(f"  родной руль к CS+ без моста (после − до): {nat.mean():+.2f} ± {sem(nat):.2f} Гц, t={t_nat.statistic:+.2f}, "
                  f"p={t_nat.pvalue:.2g}  (сам руль до обучения: {np.mean([r['pre_native'] for r in rc]):+.2f})")
            print(f"  сила памяти в отсеках награды: связи KC(CS+) {mp.mean():.3f} ± {mp.std(ddof=1):.3f}, "
                  f"KC(CS-) {mm.mean():.3f} ± {mm.std(ddof=1):.3f}; по всем обучаемым связям мозга "
                  f"{np.mean([r['mem_all']['total'] for r in rc]):.4f}")
            na = {f"{t}_{k}": np.mean([r[f"{t}_{k}_nact"].mean() for r in rc]) for t in ("pre", "post") for k in "pm"}
            nx = max(max(r[f"{t}_{k}_nact"].max() for t in ("pre", "post") for k in "pm") for r in rc)
            print(f"  устойчивость: активных нейронов за 1 с теста CS+ {na['pre_p']:.0f} -> {na['post_p']:.0f}, "
                  f"CS- {na['pre_m']:.0f} -> {na['post_m']:.0f} (макс. {nx} из 138639)")
            dose = pd.DataFrame([r["da_dose"] for r in rc]).mean()
            print("  доза дофамина за обучение (∫d̂ dt, с) > 0.01:",
                  ", ".join(f"{k} {v:.2f}" for k, v in dose.sort_values(ascending=False).items() if v > 0.01) or "нет")
            bym = pd.DataFrame([r["mem_p"]["by_mbon"] for r in rc]).mean()
            print("  память KC(CS+)->MBON по типам > 0.01:",
                  ", ".join(f"{k} {v:.2f}" for k, v in bym.sort_values(ascending=False).items() if v > 0.01) or "нет")
            bym = pd.DataFrame([r["mem_m"]["by_mbon"] for r in rc]).mean()
            print("  память KC(CS-)->MBON по типам > 0.01:",
                  ", ".join(f"{k} {v:.2f}" for k, v in bym.sort_values(ascending=False).items() if v > 0.01) or "нет")
            summary.append(dict(mode=mode, cond=cond, n=n, dV=dV.mean(), dV_sem=sem(dV), dV_d=dV.mean() / dV.std(ddof=1),
                                dV_p=t_dV.pvalue, dR=dR.mean(), dR_sem=sem(dR), dR_p=t_dR.pvalue,
                                PI_pre=pi_pre.mean(), PI_post=pi_post.mean(), PI_pre_sem=sem(pi_pre), PI_post_sem=sem(pi_post),
                                PI_p=t_pi.pvalue if t_pi is not None else np.nan,
                                native=nat.mean(), native_sem=sem(nat), native_p=t_nat.pvalue,
                                mem_p_rew=mp.mean(), mem_m_rew=mm.mean()))
            # по типам MBON (L+R) — отдельно по реципрокным группам (коды A и B задевают разные MBON)
            for csp in sorted({r["csp"] for r in rc}):
                rg = [r for r in rc if r["csp"] == csp]
                pre_p = np.stack([r["pre_p_rates"].mean(0) for r in rg]); post_p = np.stack([r["post_p_rates"].mean(0) for r in rg])
                pre_m = np.stack([r["pre_m_rates"].mean(0) for r in rg]); post_m = np.stack([r["post_m_rates"].mean(0) for r in rg])
                dd = (post_p - pre_p) - (post_m - pre_m)
                rows = []
                for j, t in enumerate(types):
                    if max(pre_p[:, j].mean(), pre_m[:, j].mean()) < 1.0:
                        continue
                    sd = dd[:, j].std(ddof=1)
                    pct = lambda a, b: 100 * (b.mean() / a.mean() - 1) if a.mean() >= 1.0 else np.nan
                    rows.append(dict(MBON=t, валентн=m["mbon"][t]["valence"], CSp_до=pre_p[:, j].mean(), CSp_после=post_p[:, j].mean(),
                                     CSp_pct=pct(pre_p[:, j], post_p[:, j]), CSm_до=pre_m[:, j].mean(), CSm_после=post_m[:, j].mean(),
                                     CSm_pct=pct(pre_m[:, j], post_m[:, j]), dd=dd[:, j].mean(),
                                     d=dd[:, j].mean() / sd if sd > 0 else np.nan,
                                     p=stats.ttest_1samp(dd[:, j], 0.0).pvalue if sd > 0 else np.nan,
                                     mem=np.mean([r["mem_p"]["by_mbon"].get(t, 0.0) for r in rg])))
                print(f"  по типам MBON, CS+={csp}, CS-={rg[0]['csm']} (n={len(rg)} seed; Гц; pct — % изменения; "
                      f"dd = ΔCS+ − ΔCS-; d = dd/SD; mem — память связей KC(CS+)->MBON):")
                print("    " + pd.DataFrame(rows).round(2).to_string(index=False).replace("\n", "\n    "))
    S = pd.DataFrame(summary)
    S.round(4).to_csv(SCR / "cond_summary.csv", index=False, encoding="utf-8-sig")

    # ---------- графики ----------
    col = {"p": "#1f77b4", "m": "#d62728"}
    for mode in modes:
        rr = [r for r in res if r["mode"] == mode]
        fig, ax = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey="row")
        for j, cond in enumerate(CONDS):
            rc = [r for r in rr if r["cond"] == cond]
            nt = len(rc[0]["curve"])
            x = np.arange(1, nt + 1)
            for k, lab in (("p", "CS+"), ("m", "CS−")):
                V = np.array([[c[f"V{k}"] for c in r["curve"]] for r in rc])
                V = V - V[:, :1]
                ax[0, j].errorbar(x, V.mean(0), yerr=V.std(0, ddof=1) / np.sqrt(len(V)), color=col[k], marker="o", capsize=3, label=lab)
                R = np.array([[rew_resp(c[f"rates_{k}"], types, rew) for c in r["curve"]] for r in rc])
                ok = R[:, 0] >= 1.0
                Rn = 100 * R[ok] / R[ok, :1]
                if ok.sum() > 1:
                    ax[1, j].errorbar(x, Rn.mean(0), yerr=Rn.std(0, ddof=1) / np.sqrt(ok.sum()), color=col[k], marker="o",
                                      capsize=3, label=f"{lab} (мух с R ≥ 1 Гц: {int(ok.sum())})")
                M = np.array([[c[f"mem_{k}_rew"] for c in r["curve"]] for r in rc])
                ax[2, j].errorbar(x, M.mean(0), yerr=M.std(0, ddof=1) / np.sqrt(len(M)), color=col[k], marker="o", capsize=3,
                                  label=f"связи KC({lab})→MBON")
            ax[0, j].set_title(f"{cond}  (n={len(rc)})")
            for i in range(3):
                ax[i, j].axhline(0 if i != 1 else 100, color="gray", lw=0.5)
            ax[1, j].legend(fontsize=8)
            ax[2, j].set_xlabel("проба обучения")
            ax[2, j].set_ylim(-0.02, 0.9)
        ax[0, 0].set_ylabel("V(проба k) − V(проба 1), Гц\n(окно 0–0.5 с, до US этой пробы)")
        ax[1, 0].set_ylabel("R: MBON отсеков награды,\n% от пробы 1 (окно до US)")
        ax[2, 0].set_ylabel("сила памяти 1 − Σw/Σw0\n(отсеки награды, после пробы k)")
        ax[0, 0].legend(); ax[2, 0].legend(fontsize=8)
        fig.suptitle(f"Обучение: CS = {'запах (ORN; 2 правки усика)' if mode == 'odor' else '5% KC по 20 Гц (как оптогенетика)'}; "
                     f"US = PAM01+02+10+11 по 50 Гц (как оптогенетика); {len(seeds)} seed x 2 реципрокные группы; ±SEM")
        save_fig(fig, f"cond_curves_{mode}.png"); plt.close(fig)

        rc = [r for r in rr if r["cond"] == "paired"]
        groups = sorted({r["csp"] for r in rc})
        fig, ax = plt.subplots(len(groups), 1, figsize=(15, 4.2 * len(groups)))
        ax = np.atleast_1d(ax)
        for g, csp in enumerate(groups):
            rg = [r for r in rc if r["csp"] == csp]
            pre_p = np.stack([r["pre_p_rates"].mean(0) for r in rg]); post_p = np.stack([r["post_p_rates"].mean(0) for r in rg])
            pre_m = np.stack([r["pre_m_rates"].mean(0) for r in rg]); post_m = np.stack([r["post_m_rates"].mean(0) for r in rg])
            keep = [j for j in range(len(types)) if max(pre_p[:, j].mean(), pre_m[:, j].mean()) >= 1.0]
            xx = np.arange(len(keep))
            for off, a0, a1, lab, cc in ((-0.2, pre_p, post_p, "CS+", col["p"]), (0.2, pre_m, post_m, "CS−", col["m"])):
                d = (a1 - a0)[:, keep]
                ax[g].bar(xx + off, d.mean(0), 0.38, yerr=d.std(0, ddof=1) / np.sqrt(len(rg)), color=cc, capsize=2,
                          label=f"{lab}: после − до")
            ax[g].set_xticks(xx)
            ax[g].set_xticklabels([f"{types[j]}{' *' if types[j] in rew else ''}\n{m['mbon'][types[j]]['valence']}" for j in keep], fontsize=7)
            ax[g].axhline(0, color="gray", lw=0.5)
            ax[g].set_ylabel("Δ частоты на CS, Гц")
            ax[g].set_title(f"paired, CS+ = {csp}, CS− = {rg[0]['csm']}: n={len(rg)} seed; типы с ответом ≥ 1 Гц; * = отсек награды")
            ax[g].legend()
        save_fig(fig, f"cond_mbon_{mode}.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    w = 0.18
    for i, mode in enumerate(modes):
        s = S[S["mode"] == mode].set_index("cond").loc[CONDS]
        for k, tag in enumerate(("pre", "post")):
            xx = np.arange(len(CONDS)) + (2 * i + k - 1.5) * w
            ax[0].bar(xx, s[f"PI_{tag}"], w, yerr=s[f"PI_{tag}_sem"], capsize=3,
                      color=("#bbbbbb" if tag == "pre" else ("#1f77b4" if mode == "odor" else "#2ca02c")),
                      hatch="" if mode == "odor" else "//", edgecolor="k", lw=0.3,
                      label=f"{mode}: {'до' if tag == 'pre' else 'после'} обучения")
        ax[1].bar(np.arange(len(CONDS)) + (i - 0.5) * 0.35, s["native"], 0.35, yerr=s["native_sem"], capsize=3,
                  hatch="" if mode == "odor" else "//", edgecolor="k", lw=0.3, label=mode)
    for a in ax:
        a.set_xticks(np.arange(len(CONDS))); a.set_xticklabels(CONDS); a.axhline(0, color="gray", lw=0.5)
    ax[0].set_ylabel("PI = 2·P(выбор CS+) − 1")
    ax[0].set_title(f"Выбор через мост: P = σ(β·ΔV), β={BETA}/Гц;\n"
                    "реципрокный PI (до обучения ≈ 0 по построению); ±SEM по seed", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].set_ylabel("Δ руля к CS+ (после − до), Гц")
    ax[1].set_title("Родной путь без моста: CS+ с одной стороны, CS− с другой;\n"
                    "руль = DNa01+02 L − R, после − до; ±SEM", fontsize=10)
    ax[1].legend()
    save_fig(fig, "cond_choice.png"); plt.close(fig)
    print("\nграфики: " + ", ".join(str(SCR / f"cond_{x}.png") for x in
                                    [f"curves_{mm}" for mm in modes] + [f"mbon_{mm}" for mm in modes] + ["choice"]))
    return S


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--ntest", type=int, default=6)
    ap.add_argument("--modes", nargs="+", default=["odor", "kc"])
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if not a.analyze:
        print(f"правило: {PL.DEFAULTS}\nUS: {REWARD} {US_HZ} Гц; CS {CS_MS} мс; paired US {US_PAIRED}, unpaired US {US_UNPAIRED}",
              flush=True)
        collect(a)
    analyze()
    print(f"всего {time.time() - t0:.0f} с")
