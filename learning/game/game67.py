"""67COIN для мухи: симулятор партии + муха-игрок на РОДНОМ руле (DNa01+DNa02, лево − право).

Что здесь есть:
  * Live67 — пошаговое зеркало официального счёта rules67.score_log (серия, множитель, энергия, криты) для HUD.
    В конце партии сверяется с score_log поле в поле (Game67Session.finish()).
  * Game67Session — партия: время (мс мозга = реальное время мухи), подсказка (сторона, куда надо тапнуть
    следующей: противоположная последнему ЗАСЧИТАННОМУ тапу; первая — случайная), журнал [[t_ms, 'L'|'R'], ...].
  * LC10Cue — подсказка = «мелкий объект» с этой стороны: вход в LC10 этой стороны (упрощённая сетчатка, senses.py).
  * FlyPlayer — каждые step_ms (по умолчанию 9 мс = 5 блоков по 1.8 мс): поставить подсказку, прогнать мозг,
    прочитать DNa L−R, сгладить (tau 60 мс, как senses.Filt), триггер Шмитта: тап 'L' при y > +th, 'R' при y < −th,
    снова «взведён», когда |y| < rearm·th; рефрактерность >= 60 мс. Декодер видит ТОЛЬКО нейроны (про подсказку и
    про сторону своего прошлого тапа он не знает — триггер симметричен).
    Необязательно: reward (включить дофаминовые PAM на R мс после верного засчитанного тапа — «как оптогенетика»)
    и plasticity (MBPlasticity.update на каждом шаге).

Пример:
    import brain as B, senses as S, game67 as G
    ann = B.load_annotations(); b = B.FlyBrain(params=G.PRM, seed=0); b.enable_graph()
    groups = S.build_groups(b, ann)
    ses = G.Game67Session(seed=0)
    fly = G.FlyPlayer(b, groups, G.LC10Cue(b, groups, hz=50), threshold=15, refractory_ms=60, rearm=0.8)
    fly.play(ses, 60_000)
    res = ses.finish()              # rules67.score_log + сверка с живым счётом
    print(ses.summary(60_000))

Правила счёта 67COIN — rules67.py рядом с этим файлом.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for _p in (str(ROOT), str(HERE.parent), str(ROOT / "tiktok")):
    if _p not in sys.path:
        sys.path.insert(0, _p)



def _import_rules():
    """Правила счёта 67COIN (rules67.py рядом с этим файлом)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import rules67  # noqa: E402
    return rules67


R67 = _import_rules()

PRM = dict(U=0.1, tau_d=300, a_inc=0.1, tau_a=3000)     # противоэпилептика — всегда
BLOCK_MS = 1.8
STEP_MS = 9.0                                           # 5 блоков графа; тап-метки целые мс
OTHER = {"L": "R", "R": "L"}
# подпись для зрителей (формулировка автора): что изменено в правилах и что муха выучила
RULES_NOTE = ("Правила изменены под муху: связь «тап = еда» и «комбо = больше еды» она понять не смогла, "
              "но научилась — если цифра светится, на неё надо нажать.")   # ≤160 символов: заметка повтора


# ---------------------------------------------------------------------------
# Живой счёт = зеркало rules67.score_log (та же целая арифметика)
# ---------------------------------------------------------------------------
class Live67:
    """Счёт по одному тапу. Логика строка в строку как в rules67.score_log (версия R67.VERSION)."""

    def __init__(self, seed: int, up: dict | None = None, energy0: int | None = None):
        up = up or {}
        self.up, self.seed, self.energy0 = up, int(seed), energy0
        self.cost, self.cap, self.reg = R67.tap_value(up), R67.energy_max(up), R67.regen(up)
        self.crit_pm = 20 * int(up.get("crit") or 0)
        self.e = (self.cap if energy0 is None else max(0, min(self.cap, int(energy0)))) * 1000
        self.s = (int(seed) & 0xFFFFFFFF) or 1
        self.streak, self.last_side, self.last_t, self.last_seen = 0, None, None, None
        self.res = {"coins": 0, "taps": 0, "paid": 0, "fast": 0, "no_energy": 0, "resets": 0,
                    "crits": 0, "best_combo": 1}
        self.max_streak = 0
        self.pause_resets = 0          # серия сброшена паузой > 1200 мс (в rules67 не считается отдельно)

    @property
    def mult(self) -> int:
        return min(R67.COMBO_MAX, 1 + self.streak // R67.COMBO_STEP)

    def tap(self, t: int, side: str) -> dict:
        """-> dict(status 'ok'|'fast'|'no_energy', mult, streak, gain, crit)."""
        if self.last_seen is not None:
            self.e = min(self.cap * 1000, self.e + self.reg * (t - self.last_seen))
        self.last_seen = t
        if self.last_t is not None and t - self.last_t < R67.MIN_GAP_MS:
            self.res["fast"] += 1
            return dict(status="fast", mult=self.mult, streak=self.streak, gain=0, crit=False)
        if self.e < self.cost * 1000:
            self.res["no_energy"] += 1
            return dict(status="no_energy", mult=self.mult, streak=self.streak, gain=0, crit=False)
        self.e -= self.cost * 1000
        if self.last_t is not None and t - self.last_t > R67.COMBO_GAP_MS:
            if self.streak > 0:
                self.pause_resets += 1
            self.streak = 0
        if self.last_side is not None and side != self.last_side:
            self.streak += 1
        elif side == self.last_side:
            self.streak = 0
            self.res["resets"] += 1
        self.last_side, self.last_t = side, t
        m = self.mult
        self.s = R67.xs32(self.s)
        crit = self.crit_pm > 0 and self.s % 1000 < self.crit_pm
        gain = self.cost * m * ((6 if side == "L" else 7) if crit else 1)
        self.res["coins"] += gain
        self.res["taps"] += 1
        self.res["paid"] += 1
        self.res["crits"] += 1 if crit else 0
        self.res["best_combo"] = max(self.res["best_combo"], m)
        self.max_streak = max(self.max_streak, self.streak)
        return dict(status="ok", mult=m, streak=self.streak, gain=gain, crit=crit)

    def energy_at(self, t: float) -> float:
        """Энергия для HUD в момент t (с накоплением с прошлого тапа)."""
        e = self.e if self.last_seen is None else min(self.cap * 1000, self.e + self.reg * (t - self.last_seen))
        return e / 1000.0

    def streak_at(self, t: float) -> int:
        """Серия для HUD: после паузы > 1200 мс она уже фактически сгорела (rules67 обнулит на следующем тапе)."""
        if self.last_t is not None and t - self.last_t > R67.COMBO_GAP_MS:
            return 0
        return self.streak


# ---------------------------------------------------------------------------
# Партия
# ---------------------------------------------------------------------------
class Game67Session:
    """Партия 67COIN для мухи. Время — мс мозга (муха живёт в реальном темпе, мозг считает быстрее)."""

    def __init__(self, seed: int, up: dict | None = None, energy0: int | None = None, first_cue: str | None = None):
        self.seed, self.up, self.energy0 = int(seed), up, energy0
        self.live = Live67(seed, up, energy0)
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 67])
        self.cue = first_cue or ("L" if rng.random() < 0.5 else "R")
        self.cue_t0 = 0.0             # когда подсказка встала на эту сторону
        self.t = 0.0
        self.log: list[list] = []     # [[t_ms, 'L'|'R'], ...] -> rules67.score_log
        self.events: list[dict] = []  # подробности по тапам (для статистики)
        self.cue_changes = [(0.0, self.cue)]
        self.result = None

    def advance(self, dt_ms: float):
        self.t += dt_ms

    def tap(self, side: str) -> dict:
        assert side in ("L", "R")
        t = int(round(self.t))
        if self.log and t < self.log[-1][0]:
            t = self.log[-1][0]
        prev = self.events[-1]["t"] if self.events else None
        since = t - max(self.cue_t0, prev if prev is not None else -1e9)
        r = self.live.tap(t, side)
        ev = dict(t=t, side=side, cue=self.cue, correct=(side == self.cue), rt=float(since),
                  cue_age=float(t - self.cue_t0), **r)
        self.log.append([t, side])
        self.events.append(ev)
        if r["status"] == "ok":
            new = OTHER[side]                       # подсказка: противоположная последнему засчитанному тапу
            if new != self.cue:
                self.cue, self.cue_t0 = new, float(t)
                self.cue_changes.append((float(t), new))
        return ev

    def hud(self) -> dict:
        lv = self.live
        return dict(t=self.t, cue=self.cue, coins=lv.res["coins"], streak=lv.streak_at(self.t),
                    mult=min(R67.COMBO_MAX, 1 + lv.streak_at(self.t) // R67.COMBO_STEP),
                    energy=lv.energy_at(self.t), taps=len(self.log))

    def finish(self) -> dict:
        """Официальный счёт rules67.score_log + сверка с живым счётом (должны совпасть все поля)."""
        if not self.log:
            self.result = {"ok": False, "why": "empty", "live_match": True, "coins": 0, "taps": 0, "resets": 0,
                           "best_combo": 1, "paid": 0, "no_energy": 0, "fast": 0, "crits": 0,
                           "energy_end": self.live.e // 1000}
            return self.result
        res = R67.score_log(self.log, self.seed, self.up, self.energy0)
        lv = self.live
        mism = {k: (res.get(k), lv.res[k]) for k in lv.res if res.get(k) != lv.res[k]}
        if res.get("energy_end") != lv.e // 1000:
            mism["energy_end"] = (res.get("energy_end"), lv.e // 1000)
        res["live_match"] = not mism
        res["live_mismatch"] = mism
        self.result = res
        return res

    # --- статистика партии ---
    def summary(self, dur_ms: float) -> dict:
        res = self.result or self.finish()
        ev = self.events
        mins = dur_ms / 60000.0
        out = dict(seed=self.seed, dur_s=dur_ms / 1000.0, taps=len(ev), taps_per_min=len(ev) / mins,
                   counted=int(res.get("taps", 0)), coins=int(res.get("coins", 0)),
                   coins_per_min=res.get("coins", 0) / mins, resets=int(res.get("resets", 0)),
                   pause_resets=self.live.pause_resets, best_combo=int(res.get("best_combo", 1)),
                   max_streak=self.live.max_streak, no_energy=int(res.get("no_energy", 0)),
                   fast=int(res.get("fast", 0)), energy_end=int(res.get("energy_end", 0)),
                   live_match=bool(res.get("live_match", False)))
        e0 = self.live.cap if self.energy0 is None else self.energy0
        out["energy_used"] = int(res.get("paid", 0)) * self.live.cost
        out["energy_start"] = int(e0)
        if ev:
            corr = np.array([e["correct"] for e in ev])
            cue = np.array([e["cue"] for e in ev])
            rt = np.array([e["rt"] for e in ev])
            side = np.array([e["side"] for e in ev])
            out["correct"] = float(corr.mean())
            out["frac_L_taps"] = float((side == "L").mean())
            for s in "LR":
                m = cue == s
                out[f"n_cue{s}"] = int(m.sum())
                out[f"correct_cue{s}"] = float(corr[m].mean()) if m.any() else float("nan")
                mc = m & corr
                out[f"rt_med_{s}"] = float(np.median(rt[mc])) if mc.any() else float("nan")
                out[f"rt_mean_{s}"] = float(np.mean(rt[mc])) if mc.any() else float("nan")
            mults = [e["mult"] for e in ev if e["status"] == "ok"]
            out["mean_mult"] = float(np.mean(mults)) if mults else 0.0
            out["frac_mult7"] = float(np.mean(np.array(mults) == R67.COMBO_MAX)) if mults else 0.0
        else:
            out.update(correct=float("nan"), frac_L_taps=float("nan"), mean_mult=0.0, frac_mult7=0.0,
                       n_cueL=0, n_cueR=0, correct_cueL=float("nan"), correct_cueR=float("nan"),
                       rt_med_L=float("nan"), rt_med_R=float("nan"), rt_mean_L=float("nan"),
                       rt_mean_R=float("nan"))
        # время, которое подсказка провисела на каждой стороне
        ch = self.cue_changes + [(float(dur_ms), None)]
        for s in "LR":
            out[f"cue_time_{s}"] = float(sum(b[0] - a[0] for a, b in zip(ch[:-1], ch[1:]) if a[1] == s)) / 1000.0
        return out

    def to_json(self) -> dict:
        return dict(seed=self.seed, up=self.up, energy0=self.energy0, log=self.log, events=self.events,
                    cue_changes=self.cue_changes, result=self.result)


# ---------------------------------------------------------------------------
# Подсказка (вход в мозг)
# ---------------------------------------------------------------------------
class LC10Cue:
    """Подсказка = мелкий движущийся объект на стороне подсказки -> LC10 этой стороны, hz Гц (Пуассон).
    mirror=True — контроль: объект на ПРОТИВОПОЛОЖНОЙ стороне. hz=0 — контроль «подсказки не видно»."""

    def __init__(self, brain, groups, hz=50.0, mirror=False):
        self.b, self.hz, self.mirror = brain, float(hz), bool(mirror)
        dev = brain.dev
        self.idx = {s: torch.as_tensor(groups["obj_" + s], dtype=torch.long, device=dev) for s in "LR"}
        self.name = f"LC10 {hz:g} Гц" + (" (зеркало)" if mirror else "")

    def __call__(self, side):
        if side is not None and self.mirror:
            side = OTHER[side]
        for s in "LR":
            self.b.rate_in[self.idx[s]] = self.hz if s == side else 0.0


# ---------------------------------------------------------------------------
# Муха-игрок
# ---------------------------------------------------------------------------
def steer_index(brain, groups):
    """Индексы DNa01+DNa02 слева и справа (руль senses.READOUTS)."""
    L, R = list(groups["steer_L"]), list(groups["steer_R"])
    return torch.as_tensor(L + R, dtype=torch.long, device=brain.dev), len(L), len(R)


class FlyPlayer:
    """Муха играет своим рулём. Декодер читает только нейроны DNa (без знания подсказки).

    bias: None — без поправки; число — вычесть постоянный сдвиг (Гц), измеренный заранее по нейронам;
          ('adapt', tau_ms) — вычитать медленное среднее самого сигнала (фильтр верхних частот).
    reward: None или dict(idx=тензор нейронов, hz=50.0, ms=500.0) — включается после ВЕРНОГО засчитанного тапа.
    По умолчанию — подобранная база (BASELINE.md): порог 15 Гц, перевзвод |y| < 0.8·порог, рефрактерность 60 мс.
    Узкий перевзвод (0.5) опасен: сглаженный руль прыгает до 26-35 Гц за шаг и проскакивает полосу -> декодер
    «залипает», пока муха держит поворот (_scratch/c7_diag.log).
    plasticity: None или MBPlasticity — update(counts, step_ms) на каждом шаге.
    """

    def __init__(self, brain, groups, cue_fn, threshold=15.0, refractory_ms=60.0, rearm=0.8,
                 step_ms=STEP_MS, tau_ms=60.0, bias=None, reward=None, plasticity=None, record=False,
                 warmup_ms=500.0):
        assert refractory_ms >= 60.0, "рефрактерность >= 60 мс (правила: тапы ближе 50 мс не считаются)"
        self.b, self.groups, self.cue_fn = brain, groups, cue_fn
        self.th, self.refr, self.rearm = float(threshold), float(refractory_ms), float(rearm)
        self.step_ms, self.tau_ms = float(step_ms), float(tau_ms)
        self.bias, self.reward, self.pl = bias, reward, plasticity
        self.record, self.warmup_ms = record, float(warmup_ms)
        self.dna, self.nL, self.nR = steer_index(brain, groups)
        self.k = 1000.0 / self.step_ms
        self.reset_state()

    def reset_state(self):
        self.f = 0.0                    # сглаженный руль L−R, Гц
        self.m = 0.0                    # медленное среднее (для bias='adapt')
        self.armed = True
        self.last_tap = -1e9
        self.reward_until = -1e9
        self.reward_on = False
        self._cue_set = "unset"
        self.trace = []                 # (t, L, R, f, y, cue, tap)
        self.n_reward = 0
        self.reward_ms_total = 0.0
        self.n_active_max, self.n_active_sum, self.n_steps = 0, 0, 0

    def _bias_now(self):
        if self.bias is None:
            return 0.0
        if isinstance(self.bias, (int, float)):
            return float(self.bias)
        return self.m

    def _set_reward(self, on):
        if self.reward is None or on == self.reward_on:
            return
        self.b.rate_in[self.reward["idx"]] = float(self.reward.get("hz", 50.0)) if on else 0.0
        self.reward_on = on

    def _brain_step(self, ms):
        c = self.b.run(ms, tensor=True)
        if self.pl is not None:
            self.pl.update(c, ms)
        # DNa + число активных нейронов за шаг (контроль «эпилепсии») — одной пересылкой
        v = torch.cat([c[self.dna], (c > 0).sum().view(1).to(c.dtype)]).cpu().numpy()
        L = float(v[:self.nL].sum()) / self.nL * (1000.0 / ms)
        Rr = float(v[self.nL:self.nL + self.nR].sum()) / self.nR * (1000.0 / ms)
        na = int(v[-1])
        self.n_active_max = max(self.n_active_max, na)
        self.n_active_sum += na
        self.n_steps += 1
        return L, Rr

    def warmup(self):
        """Разогрев без подсказки (не входит во время партии): мозг из reset() выходит из переходного режима."""
        self.cue_fn(None)
        self._cue_set = None
        n = int(round(self.warmup_ms / self.step_ms))
        for _ in range(n):
            L, Rr = self._brain_step(self.step_ms)
            a = min(1.0, self.step_ms / self.tau_ms)
            self.f += a * ((L - Rr) - self.f)

    def step(self, ses: Game67Session):
        if ses.cue != self._cue_set:
            self.cue_fn(ses.cue)
            self._cue_set = ses.cue
        L, Rr = self._brain_step(self.step_ms)
        ses.advance(self.step_ms)
        t = ses.t
        a = min(1.0, self.step_ms / self.tau_ms)
        self.f += a * ((L - Rr) - self.f)
        if isinstance(self.bias, tuple):
            self.m += min(1.0, self.step_ms / float(self.bias[1])) * (self.f - self.m)
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
        if tap is not None:
            self.armed = False
            self.last_tap = t
            ev = ses.tap(tap)
            if self.reward is not None and ev["status"] == "ok" and ev["correct"]:
                self.reward_until = t + float(self.reward.get("ms", 500.0))
                self.n_reward += 1
        if self.reward is not None:
            on = t < self.reward_until
            if on:
                self.reward_ms_total += self.step_ms
            self._set_reward(on)
        if self.record:
            self.trace.append((t, L, Rr, self.f, y, ses.cue, tap or ""))
        return ev

    def play(self, ses: Game67Session, dur_ms: float, on_step=None, warmup=True):
        """Сыграть партию dur_ms мс. on_step(player, session, ev) — для HUD/кадров."""
        self.reset_state()
        if warmup and self.warmup_ms > 0:
            self.warmup()
        self.n_active_max, self.n_active_sum, self.n_steps = 0, 0, 0     # разогрев не считаем
        n = int(math.floor(dur_ms / self.step_ms + 1e-9))
        for _ in range(n):
            ev = self.step(ses)
            if on_step is not None:
                on_step(self, ses, ev)
        self._set_reward(False)
        self.cue_fn(None)
        return ses


# ---------------------------------------------------------------------------
# Помощники
# ---------------------------------------------------------------------------
def reseed(brain, seed):
    """Тот же мозг как новый FlyBrain(seed=seed): ГСЧ (Philox: зерно + номер шага) с нуля, состояние reset(),
    граф перезаписан (зерно запечено в аргументы ядра). Веса не трогает. Проверено против нового мозга
    (_scratch/c1_reseed_check.py)."""
    brain.seed = int(seed)
    brain._args_key = None
    if getattr(brain, "_tdev", None) is not None:
        brain._tdev.zero_()
    brain.reset()
    if brain.graph is not None:
        brain.enable_graph()
    return brain


def reference_scores(ses: Game67Session, dur_ms: float, n_shuffle=200, rng=None) -> dict:
    """Контекст для coins/min при ТЕХ ЖЕ моментах тапов:
    ideal — всегда тап в сторону подсказки (сторона = противоположная последнему ЗАСЧИТАННОМУ тапу; незасчитанный
            из-за энергии тап подсказку не меняет) — идеальный игрок с тем же темпом;
    random — стороны случайны 50/50 (среднее по n_shuffle)."""
    if not ses.log:
        return dict(ideal_cpm=0.0, random_cpm=0.0)
    rng = rng or np.random.default_rng(ses.seed + 999)
    ts = [t for t, _ in ses.log]
    lv, ideal, last = Live67(ses.seed, ses.up, ses.energy0), [], None
    for t in ts:
        side = "L" if last is None else OTHER[last]
        if lv.tap(t, side)["status"] == "ok":
            last = side
        ideal.append([t, side])
    ri = R67.score_log(ideal, ses.seed, ses.up, ses.energy0)["coins"]
    rs = []
    for _ in range(n_shuffle):
        sd = rng.integers(0, 2, len(ts))
        rs.append(R67.score_log([[t, "LR"[k]] for t, k in zip(ts, sd)], ses.seed, ses.up, ses.energy0)["coins"])
    mins = dur_ms / 60000.0
    return dict(ideal_cpm=ri / mins, random_cpm=float(np.mean(rs)) / mins)
