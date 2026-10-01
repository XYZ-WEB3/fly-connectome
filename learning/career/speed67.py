"""Скорость мухи в 67COIN: что ограничивает темп тапов — наш декодер или нейроны (задача C, часть 1).

Декодер (game67.FlyPlayer / train67.DEC): руль DNa01+02 L−R по шагам 9 мс, сглаживание tau, порог ±th,
перевзвод |y| < rearm·th, рефрактерность refr. Здесь его параметры можно менять (в т.ч. refr < 60 мс —
game67 это запрещает assert-ом; ниже 50 мс не пускаем: правило rules67 MIN_GAP_MS).

Что пишем в каждой партии (сырые спайки DNa по шагам 9 мс, record=True плеера):
  * игра: тапы, верно, сбросы, coins/min (rules67.score_log, только чтение);
  * нейроны: латентность первого спайка DNa стороны подсказки после смены подсказки (родная реакция, без декодера);
  * задержка декодера = тап − первый спайк DNa той же стороны в том же эпизоде подсказки;
  * двойной счёт: два тапа подряд в одну сторону из ОДНОЙ пачки спайков DNa этой стороны
    (между ними нет паузы >= GAP_MS в спайках этой стороны и нет спайков другой стороны);
  * слишком ранние тапы: rt < RT_MIN_MS после смены подсказки (быстрее, чем нейроны вообще успевают ответить).
Контроли: cue='mirror' (метка с другой стороны: верно должно упасть к ~0), cue='none' (метки нет: тапов ~0).

Всё только читает нейроны DNa; память (веса KC->MBON) грузится из файла и во время партии не меняется.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEARN = HERE.parent
GAME = LEARN / "game"
ROOT = LEARN.parent
for _p in (str(ROOT), str(LEARN), str(GAME), str(ROOT / "tiktok")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import train67 as T  # noqa: E402
import game67 as G  # noqa: E402

OFFICIAL_MEM = GAME / "official" / "fly67_v1.npz"
GAP_MS = 36.0          # пауза в спайках стороны, после которой пачка считается новой (4 шага по 9 мс)
RT_MIN_MS = 30.0       # тап раньше этого после смены подсказки — не ответ на неё (см. латентность ниже)


class FastPlayer(T.Player):
    """train67.Player, но рефрактерность можно 50-60 мс (game67 требует >= 60)."""

    def __init__(self, *a, refractory_ms=100.0, **kw):
        super().__init__(*a, refractory_ms=max(60.0, float(refractory_ms)), **kw)
        assert refractory_ms >= G.R67.MIN_GAP_MS, "рефрактерность ниже MIN_GAP rules67 — такие тапы не засчитают"
        self.refr = float(refractory_ms)


class CueMode:
    """Обёртка метки: 'normal' | 'mirror' (метка с другой стороны) | 'none' (метки нет)."""

    def __init__(self, cue_fn, mode="normal"):
        assert mode in ("normal", "mirror", "none")
        self.f, self.mode = cue_fn, mode

    def __call__(self, side):
        if self.mode == "none" or side is None:
            self.f(None)
        else:
            self.f(G.OTHER[side] if self.mode == "mirror" else side)


def play_dec(rig, dec, seed, dur_ms=30000.0, cue="normal", up=None, energy0=None):
    """Партия без награды и без обучения с декодером dec (dict threshold, rearm, refractory_ms, tau_ms).
    Память — та, что сейчас в rig.pl. -> (ses, fly, sm)."""
    G.reseed(rig.b, seed)
    rig.pl.reset_traces()
    rig.pl.enabled = False
    fly = FastPlayer(rig.b, rig.groups, CueMode(rig.cue_fn, cue), threshold=dec["threshold"],
                     refractory_ms=dec["refractory_ms"], rearm=dec["rearm"], tau_ms=dec["tau_ms"],
                     reward=None, reward_mode=None, plasticity=None, record=True, extra=rig.extra)
    ses = G.Game67Session(seed, up=up, energy0=energy0)
    fly.play(ses, dur_ms)
    ses.finish()
    sm = ses.summary(dur_ms)
    sm.update(n_active_max=fly.n_active_max)
    sm.update(analyze(ses, fly))
    return ses, fly, sm


def raw_arrays(fly):
    """Сырые спайки DNa по шагам: t (мс, конец шага), sL, sR (число спайков стороны за шаг), tap ('', 'L', 'R')."""
    tr = fly.trace
    t = np.array([r[0] for r in tr])
    k = fly.step_ms / 1000.0
    sL = np.rint(np.array([r[1] for r in tr]) * fly.nL * k).astype(int)
    sR = np.rint(np.array([r[2] for r in tr]) * fly.nR * k).astype(int)
    tap = np.array([r[6] for r in tr])
    return t, sL, sR, tap


def analyze(ses, fly, gap_ms=GAP_MS, rt_min=RT_MIN_MS):
    """Честность декодера и нейронная латентность по сырым спайкам DNa одной партии."""
    t, sL, sR, tap = raw_arrays(fly)
    sp = {"L": sL, "R": sR}
    out = {}
    # --- латентность первого спайка DNa стороны подсказки (родная реакция на смену подсказки) ---
    lat = {"L": [], "R": []}
    lag = {"L": [], "R": []}                 # тап − первый спайк той же стороны в эпизоде (задержка декодера)
    ch = ses.cue_changes[1:]                 # первая подсказка — от разогрева, её не считаем
    ends = [c[0] for c in ch[1:]] + [float(t[-1]) if len(t) else 0.0]
    ev_t = np.array([e["t"] for e in ses.events]) if ses.events else np.zeros(0)
    ev_side = [e["side"] for e in ses.events]
    for (t0, side), t1 in zip(ch, ends):
        m = (t > t0) & (t <= t1 + 1e-6)
        idx = np.nonzero(m & (sp[side] > 0))[0]
        if len(idx) == 0:
            continue
        ts = float(t[idx[0]])
        lat[side].append(ts - t0)
        j = np.nonzero((ev_t > t0) & (ev_t <= t1 + 1e-6))[0]
        if len(j) and ev_side[j[0]] == side:
            lag[side].append(float(ev_t[j[0]]) - ts)
    for s in "LR":
        out[f"lat1_med_{s}"] = float(np.median(lat[s])) if lat[s] else float("nan")
        out[f"lag_med_{s}"] = float(np.median(lag[s])) if lag[s] else float("nan")
        out[f"lag_mean_{s}"] = float(np.mean(lag[s])) if lag[s] else float("nan")
    # --- двойной счёт: два тапа подряд в одну сторону из одной пачки спайков этой стороны ---
    ti = np.nonzero(tap != "")[0]
    dup, same = 0, 0

    def trig(i, s):
        # спайк, от которого сработал тап: последний шаг <= i со спайком этой стороны
        j = np.nonzero(sp[s][:i + 1] > 0)[0]
        return int(j[-1]) if len(j) else -1

    for a, b in zip(ti[:-1], ti[1:]):
        s = tap[a]
        if tap[b] != s:
            continue
        same += 1
        ta, tb = trig(a, s), trig(b, s)
        if ta < 0:
            continue
        if tb == ta:                                  # один и тот же спайк дал два тапа
            dup += 1
            continue
        o = sp[G.OTHER[s]][ta + 1:tb + 1]
        own = np.nonzero(sp[s][ta:tb + 1] > 0)[0]    # шаги со спайками своей стороны между «спусковыми» спайками
        gaps = np.diff(own) * fly.step_ms
        if o.sum() == 0 and gaps.max() < gap_ms:
            dup += 1
    out["same_side_pairs"] = same
    out["dup_taps"] = dup
    # каждый тап — от свежего спайка DNa своей стороны? (спайк в шаге тапа или шагом раньше; перевес своей стороны
    # за последние 3 шага = 27 мс)
    fresh = net = 0
    for i in ti:
        s = tap[i]
        lo = max(0, i - 1)
        fresh += int(sp[s][lo:i + 1].sum() > 0)
        lo3 = max(0, i - 2)
        net += int(sp[s][lo3:i + 1].sum() > sp[G.OTHER[s]][lo3:i + 1].sum())
    out["tap_fresh_spike"] = fresh / max(1, len(ti))
    out["tap_net_own"] = net / max(1, len(ti))
    # --- тапы раньше, чем нейроны успевают ответить на смену подсказки ---
    rts = np.array([e["rt"] for e in ses.events]) if ses.events else np.zeros(0)
    out["early_taps"] = int((rts < rt_min).sum())
    iti = np.diff(ev_t) if len(ev_t) > 1 else np.zeros(0)
    out["iti_min"] = float(iti.min()) if len(iti) else float("nan")
    out["iti_lt100"] = int((iti < 100).sum())
    # спайки DNa: сколько «пачек» (по GAP_MS) на сторону — верхняя граница честного числа тапов
    nb = 0
    for s in "LR":
        on = np.nonzero(sp[s] > 0)[0]
        if len(on):
            nb += 1 + int((np.diff(on) * fly.step_ms >= gap_ms).sum())
    out["dna_bursts"] = nb
    out["dna_spikes_L"], out["dna_spikes_R"] = int(sL.sum()), int(sR.sum())
    return out
