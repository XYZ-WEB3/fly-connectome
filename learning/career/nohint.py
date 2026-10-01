"""67COIN БЕЗ ПОДСКАЗКИ (задача A «карьеры»): муха видит только то, что видит человек на экране игры.

Что на экране (наш выбор «упрощённой сетчатки», как cues67.CueSet — те же настоящие нейроны):
  * обе кнопки видны всегда (фон, необязательно): слабый предмет С ДВУХ сторон — LC10 bg_lc10 Гц + VPN чашечки
    bg_cal Гц на обе стороны;
  * нажатая кнопка вспыхивает на flash_ms мс: на стороне ТОЛЬКО ЧТО сделанного тапа LC10 fl_lc10 Гц + VPN чашечки
    fl_cal Гц (вместо фона этой стороны).
  Вспышку зажигает ТАП МУХИ (выход декодера), а НЕ сторона, которую ждёт игра (ses.cue здесь не читается нигде,
  кроме статистики). Подсказки «жми сюда» нет.
Тап = родной руль DNa01+02 L−R (декодер train67.DEC, как у официальной мухи; ничего не подбирали под эту задачу).
Счёт = rules67.score_log (только чтение). Память — только веса KC->MBON (MBPlasticity, выход DAN заглушен).

Режимы дофамина (всё — «как оптогенетика», стимулы подписаны):
  None      — без дофамина;
  'shaped'  — после засчитанного тапа PAM01 (награда) 50 Гц на rew_ms_per_coin × (монет за тап) мс
              (монет за тап = множитель серии); после сброса серии (та же рука) — PPL1 (PPL101-103, наказание)
              50 Гц на pun_ms мс;
  'yoked'   — та же программа PAM/PPL1 (моменты и длительности) из журнала другой партии: НЕ зависит от тапов мухи;
  'sched'   — то же, что yoked (для своих расписаний).

Пример:
    import nohint as N
    rig = N.make_rig()
    ses, fly, sm = N.play(rig, seed=1, dur_ms=30000, fb=N.FB_VPN5)
    print(sm['alt'], sm['taps_per_min'], sm['coins_per_min'], sm['random_cpm'])
CLI (из корня проекта):
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -W ignore learning/career/nohint.py --screen vpn5 --seconds 30 --seed 1 [--official] [--out x.json]

ИЗМЕРЕНО. Этап 1 — игра без подсказки, без дофамина и обучения: 5 мух (сиды 80000 + 100·f), партии 30 с; «случайные» =
те же моменты тапов со случайными сторонами (rules67), «идеал» = те же моменты с идеальным чередованием
(_scratch/n0_pilot.py, n1_scan.log, n1_analyze.log, n3_flashresp.log).
  * Кнопок не видно (только вспышка после тапа) — 0 тапов: первого тапа нет, начать нечем.
  * Только кнопки (LC10 с двух сторон, без вспышки): тапы идут от шума, есть врождённый левый крен. Фон 2/3/4/5 Гц:
    тап/мин 84/170/225/270, чередование 0.22/0.29/0.44/0.49, доля L 0.87/0.82/0.67/0.64.
  * Вспышка = только VPN чашечки (KC-код кнопки; LC10 не растёт) повышает чередование при любом фоне: +0.11..+0.37,
    p ≤ 0.007, 5/5. Лучше всего фон 5 + вспышка 300 мс (FB_VPN5):
      - чередование 0.662 ± 0.045 (к 0.5: p = 0.001), 298 тап/мин, 50 сбросов за 30 с;
      - coins/min 323 против 303 у случайных сторон (+20, p = 0.008, 5/5), средний множитель 1.08, best_combo 2.2;
      - идеальное чередование в те же моменты дало бы 1837. Для сравнения, с подсказкой: наивная 498, обученная 1214.
  * Вспышка с LC10 (предмет целиком: + LC10 2 Гц, с VPN или без) ПРИТЯГИВАЕТ к только что нажатой кнопке.
      - Чередование 0.11–0.42, то есть ниже 0.5 (фон 5, LC10+VPN 300 мс: 0.323, p = 9e-5); монет чуть меньше случайных.
      - «Кнопка = предмет старой подсказки» (LC10 4 Гц + VPN 150, фон 2): чередование 0.18–0.30, 95–149 тап/мин.
  * Руль на одиночную вспышку (вне игры, 5 сидов × 10 вспышек на сторону):
      - вспышка VPN толкает ОТ кнопки на +4.9..+6.2 Гц через 100–300 мс (p = 0.003); почти всё даёт вспышка справа
        (+10..+12 Гц), слева ≈ 0;
      - нокаут MBON27 уменьшает толчок до +1.4..+2.2, нокаут KCg-d — до +1.8..+3.3 (остаток идёт не через MB);
      - вспышка с LC10 тянет К кнопке: −9..−13 Гц.
  * Официальная память fly67_v1 = нокаут MBON27 ПОБИТНО (140 партий из 140): MBON27 на KC-код кнопки молчит.
    Поэтому без подсказки она МЕШАЕТ: на FB_VPN5 чередование 0.662 → 0.590 (p = 0.11, 0/5↑), coins/min 323 → 285
    (p = 0.036, 0/5↑) — уже не выше случайных (285 против 280, p = 0.11). Чередование ниже, чем у наивной, на 15 из 16
    экранов с KC-кодом во вспышке. Если KC-кода во вспышке нет (только LC10 или вспышки нет) — побитно как наивная.
  Итог этапа 1: гипотеза верна наполовину. Врождённый толчок MBON27 «от предмета» даёт чередование выше случайного,
  но только если вспышка возбуждает KC и не возбуждает LC10 — это наш выбор «сетчатки», не факт о мухе. Выигрыш мал:
  серии чередования в среднем 3 тапа, множитель почти не растёт.

Этап 2 — «лепка» без подсказки (_scratch/n2_shape.py, n2_*_analyze.log). 5 мух, 8 сессий × 60 с, обучение с нуля.
После каждой сессии — пробы 2 × 30 с без дофамина и без обучения, у каждой мухи на одних и тех же сидах.
Режим shaped: PAM01 на 60 мс × множитель после каждого засчитанного тапа, PPL1 на 300 мс за каждый сброс.
  * Экран FB_VPN5 (все 5 условий). За сессию ~295 наград (20 с PAM) и ~100 наказаний (26 с PPL1).
      - shaped: чередование на пробах 0.687 → 0.671 после 8 сессий; в среднем по сессиям −0.025 (p = 0.035, 0/5↑).
        Лучше не стало. coins/min 331 → 314 (p = 0.13).
      - yoked (та же программа PAM/PPL1 по времени, свой шум) = shaped: разница кривых −0.003 (p = 0.36),
        память та же (MBON27 L/R 0.27/0.21). Если шум общий, yoked = shaped побитно (муха 0, 8 сессий): дофамин
        внутри сессии на поведение не влияет.
      - только награда — хуже: 0.687 → 0.610 (p = 0.004, 5/5↓), сбросов на +10.6 за 30 с больше (p = 0.003).
        PAM01 на фоне KC-кода вспышки ослабляет KC→MBON27: ответ MBON27 на свою кнопку падает с 17–22 до 3–5 Гц,
        и врождённый толчок «от кнопки» пропадает. Это то же, что официальная память на этапе 1.
      - только наказание: 0.687 → 0.699 (p = 0.5). MBON32_L на кнопку L падает с 6.9 до 0.3 Гц (γ2),
        а чередование не меняется.
      - noUS: изменённых связей 0, пробы побитно равны наивной.
  * Экран FB_LV5 (shaped, yoked, noUS, только наказание): вспышка притягивает к нажатой кнопке, чередование 0.31.
      - Ни одно условие его не меняет: shaped −0.003 (p = 0.9), yoked +0.013 (p = 0.39), только наказание +0.003 (p = 0.87).
      - Множитель всё время 1.00; coins/min 229 → 217–222 (везде p ≥ 0.19).
  Итог этапа 2: обучение павловское — «экран = дофамин», а не «мой тап → награда».
    - Кнопки вспыхивают каждые ~200 мс, а след KC длится 1.5 с, поэтому дофамин всегда совпадает с кодами ОБЕИХ кнопок.
    - Правило KC→MBON не видит порядок тапов («L, потом R»), а в нём и есть правило 67.
    - Поэтому награда за чередование в этой модели научить чередованию не может. Для этого нужен механизм, которого
      в модели нет: оперантное «self-learning» вне MB или код последовательности.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LEARN = HERE.parent
GAME = LEARN / "game"
ROOT = LEARN.parent
for _p in (str(ROOT), str(LEARN), str(GAME), str(ROOT / "tiktok")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import game67 as G  # noqa: E402
import train67 as T  # noqa: E402

OFFICIAL = GAME / "official" / "fly67_v1.npz"
OFFICIAL_JSON = GAME / "official" / "fly67_v1.json"

# экран: «нажатая кнопка» = тот же предмет, что старая подсказка (strength 0.4: LC10 4 Гц + VPN чашечки 150 Гц),
# вспышка 200 мс; обе кнопки видны всегда тусклыми (LC10 2 Гц, VPN 0) — наш выбор, варианты — в scan
FB_MAIN = dict(bg_lc10=2.0, bg_cal=0.0, fl_lc10=4.0, fl_cal=150.0, flash_ms=200.0)
FB_NOBG = dict(bg_lc10=0.0, bg_cal=0.0, fl_lc10=4.0, fl_cal=150.0, flash_ms=200.0)     # без фона: 0 тапов
# лучший экран сетки (_scratch/n1_scan.log): кнопки LC10 5 Гц с двух сторон, вспышка = только VPN чашечки 150 Гц 300 мс
FB_VPN5 = dict(bg_lc10=5.0, bg_cal=0.0, fl_lc10=5.0, fl_cal=150.0, flash_ms=300.0)
# тот же фон, вспышка = полный предмет (+ LC10 2 Гц + VPN 150 Гц): вспышка ПРИТЯГИВАЕТ
FB_LV5 = dict(bg_lc10=5.0, bg_cal=0.0, fl_lc10=7.0, fl_cal=150.0, flash_ms=300.0)
SCREENS = dict(main=FB_MAIN, vpn5=FB_VPN5, lv5=FB_LV5, nobg=FB_NOBG)
# дофамин при «лепке»: PAM01 (учитель γ5d/γ5/γ4γ5), PPL1 (учителя γ1/γ2/α'1)
SHAPE = dict(rew_types=["PAM01"], rew_hz=50.0, rew_ms_per_coin=60.0,
             pun_types=["PPL101", "PPL102", "PPL103"], pun_hz=50.0, pun_ms=300.0)


def make_rig(verbose=False):
    """Тот же стенд, что train67.Rig (мозг PRM, MBPlasticity lr_dep 0.02, DAN заглушены, calyx заморожен)."""
    return T.Rig(verbose=verbose)


def load_official(rig):
    """Память официальной мухи (sha256 сверяется с fly67_v1.json). -> sha256."""
    sha = hashlib.sha256(OFFICIAL.read_bytes()).hexdigest()
    ref = json.load(open(OFFICIAL_JSON, encoding="utf-8"))["memory_sha256"]
    assert sha == ref, "память официальной мухи не совпадает с fly67_v1.json"
    rig.pl.reset_memory()
    rig.pl.load_memory(OFFICIAL)
    return sha


# ---------------------------------------------------------------------------
# Экран игры -> входы мозга
# ---------------------------------------------------------------------------
class Screen:
    """Обратная связь игры: кнопки (фон) + вспышка нажатой. Знает только тапы мухи и время."""

    def __init__(self, cs, bg_lc10=0.0, bg_cal=0.0, fl_lc10=4.0, fl_cal=150.0, flash_ms=200.0):
        self.b = cs.b
        self.g = {k: cs.groups[k] for k in ("LC10_L", "LC10_R", "CAL_L", "CAL_R")}
        self.bg = (float(bg_lc10), float(bg_cal))
        self.fl = (float(fl_lc10), float(fl_cal))
        self.flash_ms = float(flash_ms)
        self.reset()

    def reset(self):
        self.lit, self.until, self._state = None, -1e9, "unset"

    def on_tap(self, side, t):
        self.lit, self.until = side, float(t) + self.flash_ms

    def _put(self, lit):
        r = self.b.rate_in
        for s in "LR":
            lc, cal = self.fl if s == lit else self.bg
            r[self.g["LC10_" + s]] = lc
            r[self.g["CAL_" + s]] = cal

    def tick(self, t):
        """Экран на шаг, начинающийся в t: вспышка горит, пока t < until."""
        if self.lit is not None and t >= self.until:
            self.lit = None
        if self.lit != self._state:
            self._put(self.lit)
            self._state = self.lit
        return self.lit

    def __call__(self, side):
        # для FlyPlayer.warmup: до партии на экране только кнопки (фон)
        self.lit = None
        self._put(None)
        self._state = None

    def off(self):
        r = self.b.rate_in
        for v in self.g.values():
            r[v] = 0.0
        self._state = "unset"


# ---------------------------------------------------------------------------
# Муха-игрок без подсказки
# ---------------------------------------------------------------------------
class NoHintPlayer(T.Player):
    """Как train67.Player, но вход — только экран (Screen), и дофамин по режиму 'shaped' / 'yoked'.
    Декодер тот же (читает только 4 DNa). ses.cue не используется для входа."""

    def __init__(self, brain, groups, screen, dopa=None, dopa_mode=None, schedule=None, **kw):
        self.screen = screen
        self.dopa = dopa or {}
        assert dopa_mode in (None, "shaped", "yoked", "sched")
        self.dopa_mode = dopa_mode
        self.sched_in = sorted((float(t), k, float(ms)) for t, k, ms in (schedule or []))
        super().__init__(brain, groups, screen, reward=None, reward_mode=None, **kw)

    def reset_state(self):
        super().reset_state()
        self.until = {"R": -1e9, "P": -1e9}         # R = награда PAM, P = наказание PPL1
        self.on = {"R": False, "P": False}
        self.sched_out = []                          # (t, 'R'|'P', ms) — для yoked-контроля
        self._si = 0
        self.n_dopa = {"R": 0, "P": 0}
        self.dopa_ms = {"R": 0.0, "P": 0.0}
        self.flash_steer = []                        # (t_tap, side) — для усреднения руля после вспышки

    def _dopa_set(self, k, on):
        if on == self.on[k]:
            return
        idx = self.dopa["rew_idx"] if k == "R" else self.dopa["pun_idx"]
        hz = self.dopa["rew_hz"] if k == "R" else self.dopa["pun_hz"]
        self.b.rate_in[idx] = float(hz) if on else 0.0
        self.on[k] = on

    def _give(self, k, t, ms):
        self.until[k] = max(self.until[k], t + ms)
        self.n_dopa[k] += 1
        self.sched_out.append((float(t), k, float(ms)))

    def step(self, ses):
        lit = self.screen.tick(ses.t)
        self._cue_set = lit                          # только для чтения MBON по горящей кнопке
        L, Rr = self._brain_step(self.step_ms)
        ses.advance(self.step_ms)
        t = ses.t
        a = min(1.0, self.step_ms / self.tau_ms)
        self.f += a * ((L - Rr) - self.f)
        y = self.f
        tap = None
        if not self.armed and abs(y) < self.rearm * self.th:
            self.armed = True
        if self.armed and t - self.last_tap >= self.refr:
            if y > self.th:
                tap = "L"
            elif y < -self.th:
                tap = "R"
        ev = None
        if tap is not None:
            self.armed = False
            self.last_tap = t
            r0 = ses.live.res["resets"]
            ev = ses.tap(tap)
            ev["reset"] = ses.live.res["resets"] > r0
            self.screen.on_tap(tap, t)               # нажатая кнопка вспыхивает (любой тап мухи)
            if self.dopa_mode == "shaped" and ev["status"] == "ok":
                self._give("R", t, self.dopa["rew_ms_per_coin"] * ev["gain"])
                if ev["reset"]:
                    self._give("P", t, self.dopa["pun_ms"])
        if self.dopa_mode in ("yoked", "sched"):
            while self._si < len(self.sched_in) and self.sched_in[self._si][0] <= t + 1e-6:
                t0, k, ms = self.sched_in[self._si]
                self._give(k, t0, ms)
                self._si += 1
        if self.dopa_mode is not None:
            for k in "RP":
                on = t < self.until[k]
                if on:
                    self.dopa_ms[k] += self.step_ms
                self._dopa_set(k, on)
        if self.record:
            self.trace.append((t, L, Rr, self.f, y, lit or "", tap or ""))
        return ev

    def play(self, ses, dur_ms, on_step=None, warmup=True):
        self.reset_state()
        self.screen.reset()
        if warmup and self.warmup_ms > 0:
            self.warmup()                            # до партии на экране только кнопки
        self.n_active_max, self.n_active_sum, self.n_steps = 0, 0, 0
        n = int(math.floor(dur_ms / self.step_ms + 1e-9))
        for _ in range(n):
            ev = self.step(ses)
            if on_step is not None:
                on_step(self, ses, ev)
        for k in "RP":
            if self.dopa:
                self._dopa_set(k, False)
        self.screen.off()
        return ses


def dopa_cfg(rig, shape=None):
    s = dict(SHAPE, **(shape or {}))
    return dict(s, rew_idx=rig.pl.dan_indices(s["rew_types"]), pun_idx=rig.pl.dan_indices(s["pun_types"]))


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------
def alt_stats(log):
    """Чередование по журналу: доля пар соседних тапов на РАЗНЫХ сторонах; P(след. другая | этот L/R)."""
    s = [x[1] for x in log]
    if len(s) < 2:
        return dict(alt=float("nan"), alt_afterL=float("nan"), alt_afterR=float("nan"), n_pairs=0)
    a = np.array([s[i] != s[i - 1] for i in range(1, len(s))])
    prev = np.array(s[:-1])
    out = dict(alt=float(a.mean()), n_pairs=int(len(a)))
    for k in "LR":
        m = prev == k
        out[f"alt_after{k}"] = float(a[m].mean()) if m.any() else float("nan")
    # длины серий чередования (между сбросами «та же рука»)
    runs, cur = [], 1
    for x in a:
        if x:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    out["run_mean"] = float(np.mean(runs))
    out["run_max"] = int(max(runs))
    return out


def play(rig, seed, dur_ms, fb=None, dopa_mode=None, shape=None, schedule=None, learn=False, record=False,
         up=None, energy0=None):
    """Одна партия без подсказки. -> (ses, fly, summary).
    summary: taps_per_min, alt (доля чередований), resets, coins_per_min (rules67), random_cpm (те же моменты тапов,
    стороны случайно), ideal_cpm (те же моменты, идеальное чередование), frac_L_taps, n/сек дофамина,
    MBON27_L@L и т.п. — средние частоты, пока горит кнопка L/R (только чтение)."""
    G.reseed(rig.b, seed)
    rig.pl.reset_traces()
    rig.pl.enabled = bool(learn)
    scr = Screen(rig.cs, **dict(FB_MAIN, **(fb or {})))
    dopa = dopa_cfg(rig, shape) if dopa_mode else None
    fly = NoHintPlayer(rig.b, rig.groups, scr, dopa=dopa, dopa_mode=dopa_mode, schedule=schedule,
                       threshold=rig.dec["threshold"], refractory_ms=rig.dec["refractory_ms"],
                       rearm=rig.dec["rearm"], tau_ms=rig.dec["tau_ms"],
                       plasticity=rig.pl if learn else None, record=record, extra=rig.extra)
    ses = G.Game67Session(seed, up=up, energy0=energy0)
    fly.play(ses, dur_ms)
    rig.pl.enabled = False
    ses.finish()
    sm = ses.summary(dur_ms)
    sm.update(alt_stats(ses.log))
    sm.update(G.reference_scores(ses, dur_ms, n_shuffle=100))
    sm.update(n_rew=fly.n_dopa["R"], rew_s=fly.dopa_ms["R"] / 1000.0, n_pun=fly.n_dopa["P"],
              pun_s=fly.dopa_ms["P"] / 1000.0, n_active_max=fly.n_active_max)
    for s in "LR":
        bc = fly.by_cue[s]
        sec = bc["ms"] / 1000.0
        sm[f"lit_s_{s}"] = sec
        if sec <= 0:
            continue
        dl, dr = bc["L"] / fly.nL / sec, bc["R"] / fly.nR / sec
        sm[f"steer@{s}"] = dl - dr                   # руль (Гц, > 0 = влево), пока горит кнопка s
        for j, k in enumerate(fly.extra_names):
            sm[f"{k}@{s}"] = float(bc["ex"][j] / fly.ex_n[j] / sec)
    return ses, fly, sm


# ---------------------------------------------------------------------------
# Ответ руля на одиночную вспышку (вне игры): куда толкает вспыхнувшая кнопка
# ---------------------------------------------------------------------------
@torch.no_grad()
def flash_response(rig, seed, fb=None, n=6, pre_ms=504.0, post_ms=792.0, gap_ms=1008.0):
    """Кнопки фоном, вспышка слева / справа по очереди (n раз каждая), руль DNa01+02 L−R (Гц, без сглаживания)
    по шагам 9 мс от начала вспышки. Тапов нет, обучения нет.
    -> (руль: side -> массив [n, шагов], MBON27: side -> массив [n, 2] — Гц MBON27_L, MBON27_R за время вспышки)."""
    G.reseed(rig.b, seed)
    rig.pl.reset_traces()
    rig.pl.enabled = False
    scr = Screen(rig.cs, **dict(FB_MAIN, **(fb or {})))
    dna, nL, nR = G.steer_index(rig.b, rig.groups)
    m27 = {s: torch.as_tensor(rig.extra[f"MBON27_{s}"], device=rig.b.dev) for s in "LR"}
    step = G.STEP_MS
    scr(None)
    for _ in range(int(round(pre_ms / step))):
        rig.b.run(step, tensor=True)
    nst = int(round(post_ms / step))
    out = {s: [] for s in "LR"}
    mb = {s: [] for s in "LR"}
    t = 0.0
    for _ in range(n):
        for s in "LR":
            scr.on_tap(s, t)
            tr, m = [], np.zeros(2)
            for i in range(nst):
                scr.tick(t)
                c = rig.b.run(step, tensor=True)
                v = c[dna].cpu().numpy()
                tr.append((v[:nL].sum() / nL - v[nL:].sum() / nR) * 1000.0 / step)
                if t < scr.until:
                    m += np.array([float(c[m27["L"]].sum()), float(c[m27["R"]].sum())])
                t += step
            out[s].append(tr)
            mb[s].append(m / np.array([len(rig.extra["MBON27_L"]), len(rig.extra["MBON27_R"])]) / (scr.flash_ms / 1000.0))
            scr.tick(t)
            for _ in range(int(round(gap_ms / step))):
                rig.b.run(step, tensor=True)
                t += step
    scr.off()
    return {s: np.array(out[s]) for s in "LR"}, {s: np.array(mb[s]) for s in "LR"}


# ---------------------------------------------------------------------------
# CLI: партия без подсказки (без дофамина и обучения) -> счёт rules67 (+ JSON журнала)
# ---------------------------------------------------------------------------
def main():
    import argparse
    import time
    ap = argparse.ArgumentParser(description="67COIN без подсказки: только обратная связь экрана")
    ap.add_argument("--screen", default="vpn5", choices=list(SCREENS))
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--official", action="store_true", help="память official/fly67_v1.npz (иначе наивная)")
    ap.add_argument("--out", default=None, help="куда записать JSON партии")
    a = ap.parse_args()
    rig = make_rig()
    sha = load_official(rig) if a.official else None
    dur = a.seconds * 1000.0
    ses, fly, sm = play(rig, a.seed, dur, fb=SCREENS[a.screen])
    sc = ses.result
    print(f"экран {a.screen} {SCREENS[a.screen]} | {'official' if a.official else 'наивная'} | seed {a.seed}, {a.seconds:.0f} с")
    print(f"  тап/мин {sm['taps_per_min']:.0f} | чередование {sm['alt']:.3f} (после L {sm['alt_afterL']:.2f}, "
          f"после R {sm['alt_afterR']:.2f}) | сбросов {sc.get('resets', 0)} | coins {sc.get('coins', 0)} "
          f"({sm['coins_per_min']:.0f}/мин; случайные стороны {sm['random_cpm']:.0f}, идеал {sm['ideal_cpm']:.0f}) | "
          f"best_combo {sc.get('best_combo', 1)} | HUD=rules {sc.get('live_match')}")
    if a.out:
        out = dict(kind="67coin_fly_nohint", hint=False, screen=SCREENS[a.screen], seed=a.seed, dur_ms=dur,
                   memory_sha256=sha, learning_during_game=False, dopamine_during_game=False,
                   created=time.strftime("%Y-%m-%d %H:%M:%S"), log=ses.log, score=sc, summary=sm)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print(f"  журнал -> {a.out}")


if __name__ == "__main__":
    main()
