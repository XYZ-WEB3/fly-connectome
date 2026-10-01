"""Обучение мухи играть в 67COIN РОДНЫМ путём (задача D): сначала учим, потом играет (play67.py).

Схема (всё родное, кроме подписанного):
  метка на стороне X (cues67.CueSet): LC10 X (объект -> руль К метке) + VPN чашечки X -> KCg-d X -> MBON27 X
  -> DNa ПРОТИВОПОЛОЖНОЙ стороны (толкает ОТ метки). Тап = родной руль DNa01+02 L−R пересёк порог (game67.FlyPlayer).
  Верный засчитанный тап -> награда: PAM01 (учитель γ5/γ5d/γ4γ5) обеих сторон 50 Гц на REWARD['ms'] мс —
  «как оптогенетика» (в модели сахар до PAM не доходит). MBPlasticity: KC(метка, след 1.5 с) + дофамин -> ослабление
  KC->MBON (γ5d: MBON27). Меньше толчка «от метки» -> руль сильнее к метке. Мостов «память -> поведение» НЕТ:
  декодер читает только спайки 4 нейронов DNa и не знает ни подсказку, ни память.
  Наш выбор (модель): отсек calyx заморожен (метка VPN 150 Гц сама гонит PPL2 выше порога; cues67.freeze_compartments).

CLI (из корня проекта):
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -W ignore learning/game/train67.py --fly_id 7 --sessions 10 --seconds 60
  -> learning/game/memory/fly_7.npz (память KC->MBON) + fly_7.json (история сессий); повторный запуск продолжает.
  --mode operant|yoked|noUS|sham (по умолчанию operant; остальные — контроли, см. TRAINING.md).
  --pretrain 12 — классическое предобучение (метка + PAM01 без тапов) перед первой сессией.

ИЗМЕРЕНО (5 мух, 10 сессий x 60 с; пробы 2 x 30 с без награды и обучения; TRAINING.md, _scratch/d2-d4):
  * проба: coins/min 498 ± 93 -> 1214 ± 206 (p = 0.001, 5/5); верно 0.898 -> 0.981; справа 0.863 -> 0.964;
    сбросов за 30 с 13.6 -> 2.5; MBON27 на подсказку 13-23 -> 0 Гц; плато после 4-5 сессий.
  * контроли: noUS и sham — память 0, проба побитно = наивной; yoked = operant (разница кривых p >= 0.59):
    MB учит «метка -> награда» (классика), награда за ОШИБКИ тоже улучшает игру (0.898 -> 0.927, p = 0.025).
  * носитель — только KC->MBON27: reset -> наивная побитно, load -> выученная побитно; память вне MBON27 — 0 эффекта.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LEARN = HERE.parent
ROOT = LEARN.parent
for _p in (str(ROOT), str(LEARN), str(HERE), str(ROOT / "tiktok")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import game67 as G  # noqa: E402
import brain as B  # noqa: E402
import senses as S  # noqa: E402
from plasticity import MBPlasticity  # noqa: E402
from cues67 import CueSet, freeze_compartments  # noqa: E402

MEM_DIR = HERE / "memory"
PRM = G.PRM

# --- метка: LC10 strength x 10 Гц + VPN чашечки 150 Гц (задача B, GAME_STRENGTH 0.4: наивно ошибается) ---
CUE = dict(strength=0.4, lc10_hz=10.0, cal_hz=150.0)
# --- декодер: подобран на НАИВНЫХ мухах seed 100-102 до обучения (_scratch/d1_pilot.log), дальше не меняется ---
DEC = dict(threshold=8.0, rearm=0.5, refractory_ms=100.0, tau_ms=60.0)
# --- награда после верного засчитанного тапа: PAM01 обеих сторон (γ5d учит только он), «как оптогенетика» ---
REWARD = dict(types=["PAM01"], hz=50.0, ms=100.0)
# --- правило: MBPlasticity (порог DA 20 Гц, след 1.5 с), но lr_dep 0.02/с вместо 0.2: в игре ~250 наград в минуту,
#     при 0.2 память насыщается за первую минуту (_scratch/d1_pilot.log, d1b_speed.log); 0.02 растягивает кривую
#     на несколько сессий. Итоговый уровень тот же (MBON27 на метку замолкает; веса при этом ещё выше пола w_min,
#     ~0.16·w0 после 10 сессий — ревизия E), меняется только скорость — модельный выбор ---
PLAST = dict(lr_dep=0.02)
FREEZE = ["calyx"]
PL_BIN_MS = 18.0          # правило KC->MBON: бин 18 мс (2 шага декодера)

# сиды: у каждой мухи свои; пробы (без награды и обучения) — одни и те же для мухи во всех сессиях и условиях,
# поэтому разница проб = только разница весов
def train_seed(fly, k):
    return 10000 + 100 * int(fly) + int(k)


def probe_seeds(fly, n=2):
    return [20000 + 100 * int(fly) + j for j in range(n)]


class CueFn:
    """Адаптер CueSet -> cue_fn(side) для FlyPlayer."""

    def __init__(self, cs, strength):
        self.cs, self.strength = cs, float(strength)

    def __call__(self, side):
        if side is None:
            self.cs.off()
        else:
            self.cs.apply(side, self.strength)


class Player(G.FlyPlayer):
    """FlyPlayer + режимы награды:
       'operant' — награда после ВЕРНОГО засчитанного тапа (как в game67);
       'yoked'   — та же награда в ЗАДАННЫЕ моменты (reward_times, мс партии) — не зависит от того, что делает муха;
       'wrong'   — стимуляция после НЕВЕРНОГО засчитанного тапа (контроль «награда за ошибки» / наказание за ошибки);
       None      — без награды.
    Записывает моменты наград (self.reward_times) для «сцепленного» контроля."""

    def __init__(self, *a, reward_mode=None, reward_times=None, extra=None, **kw):
        # extra: {имя: индексы нейронов} — средняя частота по стороне подсказки (MBON27 и т.п.), только чтение
        self.extra_names = list(extra or {})
        super().__init__(*a, **kw)
        assert reward_mode in (None, "operant", "yoked", "wrong")
        self.reward_mode = reward_mode if self.reward is not None else None
        self.sched = sorted(float(x) for x in (reward_times or []))
        dev = self.b.dev
        ex = [torch.as_tensor(np.asarray(extra[k]), dtype=torch.long, device=dev) for k in self.extra_names]
        self.ex_n = np.array([len(x) for x in ex], dtype=np.float64)
        self.ex_idx = torch.cat(ex) if ex else torch.zeros(0, dtype=torch.long, device=dev)
        self.ex_seg = np.repeat(np.arange(len(ex)), [len(x) for x in ex])

    def reset_state(self):
        super().reset_state()
        self.reward_times = []
        self._yi = 0
        self._acc, self._acc_ms = None, 0.0
        k = len(getattr(self, "extra_names", []))
        self.by_cue = {s: dict(ms=0.0, L=0.0, R=0.0, ex=np.zeros(k)) for s in "LR"}   # спайки по стороне подсказки

    def _brain_step(self, ms):
        # как G.FlyPlayer._brain_step, но правило KC->MBON считается бинами по PL_BIN_MS (как MBPlasticity.run):
        # вдвое меньше вызовов update на общем GPU; спайки не теряются (копятся)
        c = self.b.run(ms, tensor=True)
        if self.pl is not None:
            if self._acc is None:
                self._acc, self._acc_ms = c.clone(), ms
            else:
                self._acc += c
                self._acc_ms += ms
            if self._acc_ms >= PL_BIN_MS - 1e-6:
                self.pl.update(self._acc, self._acc_ms)
                self._acc, self._acc_ms = None, 0.0
        v = torch.cat([c[self.dna], (c > 0).sum().view(1).to(c.dtype), c[self.ex_idx]]).cpu().numpy()
        nd = self.nL + self.nR
        L = float(v[:self.nL].sum()) / self.nL * (1000.0 / ms)
        Rr = float(v[self.nL:nd].sum()) / self.nR * (1000.0 / ms)
        na = int(v[nd])
        cs = self._cue_set
        if cs in ("L", "R"):
            bc = self.by_cue[cs]
            bc["ms"] += ms
            bc["L"] += float(v[:self.nL].sum())
            bc["R"] += float(v[self.nL:nd].sum())
            if len(self.ex_seg):
                bc["ex"] += np.bincount(self.ex_seg, weights=v[nd + 1:], minlength=len(self.ex_n))
        self.n_active_max = max(self.n_active_max, na)
        self.n_active_sum += na
        self.n_steps += 1
        return L, Rr

    def step(self, ses):
        # тот же шаг, что G.FlyPlayer.step, но награда по режиму
        if ses.cue != self._cue_set:
            self.cue_fn(ses.cue)
            self._cue_set = ses.cue
        L, Rr = self._brain_step(self.step_ms)
        ses.advance(self.step_ms)
        t = ses.t
        a = min(1.0, self.step_ms / self.tau_ms)
        self.f += a * ((L - Rr) - self.f)
        y = self.f - self._bias_now()
        tap = None
        if not self.armed and abs(y) < self.rearm * self.th:
            self.armed = True
        if self.armed and t - self.last_tap >= self.refr:
            if y > self.th:
                tap = "L"
            elif y < -self.th:
                tap = "R"
        ev = None
        ms = float(self.reward.get("ms", 300.0)) if self.reward is not None else 0.0
        if tap is not None:
            self.armed = False
            self.last_tap = t
            ev = ses.tap(tap)
            if ev["status"] == "ok" and ((self.reward_mode == "operant" and ev["correct"]) or
                                         (self.reward_mode == "wrong" and not ev["correct"])):
                self.reward_until = t + ms
                self.n_reward += 1
                self.reward_times.append(t)
        if self.reward_mode == "yoked":
            while self._yi < len(self.sched) and self.sched[self._yi] <= t + 1e-6:
                self.reward_until = max(self.reward_until, self.sched[self._yi] + ms)
                self.n_reward += 1
                self.reward_times.append(self.sched[self._yi])
                self._yi += 1
        if self.reward_mode is not None:
            on = t < self.reward_until
            if on:
                self.reward_ms_total += self.step_ms
            self._set_reward(on)
        if self.record:
            self.trace.append((t, L, Rr, self.f, y, ses.cue, tap or ""))
        return ev


class Rig:
    """Один мозг на всё; муха = память (веса KC->MBON) + сиды шума. Смена seed через G.reseed (побитно как новый мозг)."""

    def __init__(self, cue=None, dec=None, reward=None, plast=None, freeze=FREEZE, verbose=True):
        t0 = time.time()
        self.cue = dict(CUE, **(cue or {}))
        self.dec = dict(DEC, **(dec or {}))
        self.rew = dict(REWARD, **(reward or {}))
        self.ann = B.load_annotations()
        self.b = B.FlyBrain(params=PRM, seed=0)
        self.b.enable_graph()
        self.groups = S.build_groups(self.b, self.ann)
        self.pl = MBPlasticity(self.b, **dict(PLAST, **(plast or {})))      # выход DAN заглушен (артефакт модели)
        self.frozen = freeze_compartments(self.pl, list(freeze or []))
        self.pl.enabled = False
        self.cs = CueSet(self.b, self.ann, lc10_hz=self.cue["lc10_hz"], cal_hz=self.cue["cal_hz"])
        self.cue_fn = CueFn(self.cs, self.cue["strength"])
        self.rew_idx = self.pl.dan_indices(self.rew["types"])
        # связи для чтения памяти
        m = json.load(open(LEARN / "mb_map.json", encoding="utf-8"))
        self.mbon_side = {}
        for t in ("MBON27", "MBON01", "MBON21", "MBON29", "MBON24"):
            for s, key in (("L", "idx_left"), ("R", "idx_right")):
                self.mbon_side[f"{t}_{s}"] = np.isin(self.pl.post, m["mbon"][t][key])
        self.kcgd = {s: np.asarray(m["kc"]["by_subtype"]["g-d"][key])
                     for s, key in (("L", "idx_left"), ("R", "idx_right"))}
        self.extra = {}
        for t in ("MBON27", "MBON32", "MBON01"):
            for s, key in (("L", "idx_left"), ("R", "idx_right")):
                self.extra[f"{t}_{s}"] = m["mbon"][t][key]
        for s in "LR":
            self.extra[f"KCgd_{s}"] = self.kcgd[s]
        if verbose:
            print(f"мозг + пластичность готовы за {time.time() - t0:.1f} с; заморожены {self.frozen}; "
                  f"награда {self.rew['types']} ({len(self.rew_idx)} нейронов)", flush=True)

    # ---------- одна партия ----------
    def play(self, seed, dur_ms, mode=None, learn=False, reward_times=None, record=False, first_cue=None,
             reward_idx=None, up=None, energy0=None, on_step=None):
        """mode: None | 'operant' | 'yoked' | 'wrong'; learn: включить правило KC->MBON; reward_idx — другие DAN
        вместо PAM01 (например, PPL1 для наказания). -> (ses, player, summary).
        В summary ещё: tow_L/tow_R — родной руль К подсказке (DNa, Гц, без сглаживания) по стороне подсказки;
        <MBON>_<сторона нейрона>@<сторона подсказки> — средние частоты (только чтение)."""
        G.reseed(self.b, seed)
        self.pl.reset_traces()
        self.pl.enabled = bool(learn)
        idx = self.rew_idx if reward_idx is None else reward_idx
        reward = dict(idx=idx, hz=self.rew["hz"], ms=self.rew["ms"]) if mode else None
        fly = Player(self.b, self.groups, self.cue_fn, threshold=self.dec["threshold"],
                     refractory_ms=self.dec["refractory_ms"], rearm=self.dec["rearm"], tau_ms=self.dec["tau_ms"],
                     reward=reward, reward_mode=mode, reward_times=reward_times,
                     plasticity=self.pl if learn else None, record=record, extra=self.extra)
        ses = G.Game67Session(seed, up=up, energy0=energy0, first_cue=first_cue)   # up/energy0 — улучшения и энергия на старте (необязательно)
        fly.play(ses, dur_ms, on_step=on_step)       # on_step — для повтора/живого окна (только чтение)
        self.pl.enabled = False
        ses.finish()
        sm = ses.summary(dur_ms)
        sm.update(n_reward=fly.n_reward, reward_s=fly.reward_ms_total / 1000.0, n_active_max=fly.n_active_max,
                  n_active_mean=fly.n_active_sum / max(1, fly.n_steps))
        for s in "LR":
            bc = fly.by_cue[s]
            sec = bc["ms"] / 1000.0
            if sec <= 0:
                continue
            dl, dr = bc["L"] / fly.nL / sec, bc["R"] / fly.nR / sec
            sm[f"dnaL@{s}"], sm[f"dnaR@{s}"] = dl, dr
            sm[f"tow_{s}"] = (dl - dr) if s == "L" else (dr - dl)
            for j, k in enumerate(fly.extra_names):
                sm[f"{k}@{s}"] = float(bc["ex"][j] / fly.ex_n[j] / sec)
        return ses, fly, sm

    # ---------- память ----------
    def memory(self):
        """Сила памяти (1 − Σw/Σw0) по MBON-сторонам + у кода метки (KCg-d своей стороны) в отсеках PAM01."""
        w = self.pl.w.cpu().numpy().astype(np.float64)
        w0 = self.pl.w0.cpu().numpy().astype(np.float64)
        out = {}
        for k, m in self.mbon_side.items():
            out[k] = float(1 - w[m].sum() / w0[m].sum())
        for s in "LR":
            m = np.isin(self.pl.pre, self.kcgd[s]) & self.pl.plastic_np
            out[f"KCgd_{s}_all"] = float(1 - w[m].sum() / w0[m].sum())
        ch = np.abs(w - w0) > 1e-9 * w0
        out["n_changed"] = int(ch.sum())
        out["total"] = float(1 - w[self.pl.plastic_np].sum() / w0[self.pl.plastic_np].sum())
        by = self.pl.memory_strength()["by_comp"]
        out["by_comp"] = {c: v for c, v in by.items() if v > 1e-4}
        return out


def classical(rig, seed, n_pairs=12, cue_ms=2000.0, us=(500.0, 1500.0), gap_ms=500.0, learn=True):
    """Классическое предобучение (необязательный этап): метка L, R, L, R ... по cue_ms, награда PAM01 внутри метки
    (us, мс от начала метки), тапов нет (руль не читается). -> секунд награды."""
    G.reseed(rig.b, seed)
    rig.pl.reset_traces()
    rig.pl.enabled = bool(learn)
    idx, hz = rig.rew_idx, float(rig.rew["hz"])
    for _ in range(int(n_pairs)):
        for side in "LR":
            rig.cue_fn(side)
            rig.pl.run(us[0])
            rig.b.rate_in[idx] = hz
            rig.pl.run(us[1] - us[0])
            rig.b.rate_in[idx] = 0.0
            rig.pl.run(cue_ms - us[1])
            rig.cue_fn(None)
            rig.pl.run(gap_ms)
    rig.pl.enabled = False
    return 2 * int(n_pairs) * (us[1] - us[0]) / 1000.0


def probe(rig, fly, dur_ms=30000.0, n=2):
    """Проба: без награды и без обучения, фиксированные сиды мухи -> список summary."""
    return [rig.play(s, dur_ms, mode=None, learn=False)[2] for s in probe_seeds(fly, n)]


# ---------------------------------------------------------------------------
# CLI: обучить муху (продолжает с сохранённой памяти)
# ---------------------------------------------------------------------------
def _short(sm):
    return (f"тапов {sm['taps']:4d} | верно {sm.get('correct', float('nan')):.3f} "
            f"(L {sm.get('correct_cueL', float('nan')):.2f} R {sm.get('correct_cueR', float('nan')):.2f}) | "
            f"сбросов {sm['resets']:3d} пауз {sm['pause_resets']:3d} | best_combo {sm['best_combo']} | "
            f"coins/min {sm['coins_per_min']:6.0f}")


def main():
    ap = argparse.ArgumentParser(description="обучение мухи 67COIN (родной путь MB)")
    ap.add_argument("--fly_id", type=int, required=True)
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--mode", default="operant", choices=["operant", "yoked", "noUS", "sham"],
                    help="operant: награда за верный тап + обучение; yoked: награда в моменты из --yoke_from, "
                         "обучение; noUS: без награды, обучение; sham: награда за верный тап, обучение ВЫКЛ")
    ap.add_argument("--yoke_from", type=int, default=None, help="fly_id мухи-ведущей (для --mode yoked)")
    ap.add_argument("--probe", action="store_true", help="после каждой сессии проба 2 x 30 с без награды/обучения")
    ap.add_argument("--pretrain", type=int, default=0,
                    help="новой мухе: классическое предобучение, N пар метка+PAM01 на сторону (0 = нет)")
    ap.add_argument("--mem_dir", default=str(MEM_DIR))
    a = ap.parse_args()

    mem_dir = Path(a.mem_dir)
    mem_dir.mkdir(parents=True, exist_ok=True)
    fmem, fjs = mem_dir / f"fly_{a.fly_id}.npz", mem_dir / f"fly_{a.fly_id}.json"
    rig = Rig()
    hist = dict(fly_id=a.fly_id, cue=rig.cue, dec=rig.dec, reward=rig.rew, frozen=rig.frozen, sessions=[])
    if fjs.exists():
        with open(fjs, encoding="utf-8") as f:
            hist = json.load(f)
    if fmem.exists():
        n = rig.pl.load_memory(fmem)
        print(f"память загружена: {fmem.name} ({n} связей), сессий уже было {len(hist['sessions'])}")
    else:
        print("новая муха: память пустая")
    yoke = None
    if a.mode == "yoked":
        assert a.yoke_from is not None, "--mode yoked требует --yoke_from"
        with open(mem_dir / f"fly_{a.yoke_from}.json", encoding="utf-8") as f:
            yoke = json.load(f)["sessions"]
    k0 = len(hist["sessions"])
    dur = a.seconds * 1000.0
    if a.pretrain and k0 == 0 and not hist.get("pretrain"):
        us_s = classical(rig, train_seed(a.fly_id, 99), n_pairs=a.pretrain)
        rig.pl.save_memory(fmem)
        mem = rig.memory()
        hist["pretrain"] = dict(n_pairs=a.pretrain, us_s=us_s, memory=mem)
        print(f"классическое предобучение: {a.pretrain} пар на сторону, награда {us_s:.0f} с; память MBON27 "
              f"L {mem['MBON27_L']:.3f} R {mem['MBON27_R']:.3f}")
    for k in range(k0, k0 + a.sessions):
        seed = train_seed(a.fly_id, k)
        t0 = time.time()
        if a.mode == "operant":
            ses, fly, sm = rig.play(seed, dur, mode="operant", learn=True)
        elif a.mode == "yoked":
            ses, fly, sm = rig.play(seed, dur, mode="yoked", learn=True,
                                    reward_times=yoke[k]["reward_times"] if k < len(yoke) else [])
        elif a.mode == "noUS":
            ses, fly, sm = rig.play(seed, dur, mode=None, learn=True)
        else:
            ses, fly, sm = rig.play(seed, dur, mode="operant", learn=False)
        rig.pl.save_memory(fmem)
        mem = rig.memory()
        rec = dict(k=k, seed=seed, mode=a.mode, dur_s=a.seconds, summary=sm, memory=mem,
                   reward_times=[float(x) for x in fly.reward_times], log=ses.log)
        if a.probe:
            rec["probe"] = probe(rig, a.fly_id)
        hist["sessions"].append(rec)
        with open(fjs, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False)
        print(f"сессия {k:3d} [{a.mode}] {_short(sm)} | наград {sm['n_reward']:4d} | память MBON27 "
              f"L {mem['MBON27_L']:.3f} R {mem['MBON27_R']:.3f} | {time.time() - t0:.1f} с", flush=True)
        if a.probe:
            pr = rec["probe"]
            print(f"   проба: верно {np.mean([p['correct'] for p in pr]):.3f} "
                  f"(R {np.nanmean([p['correct_cueR'] for p in pr]):.3f}) coins/min "
                  f"{np.mean([p['coins_per_min'] for p in pr]):.0f}", flush=True)
    print(f"память сохранена: {fmem}")


if __name__ == "__main__":
    main()
