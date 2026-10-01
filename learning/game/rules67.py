"""Правила счёта игры 67COIN (тапалка «6 или 7»): партия пересчитывается по журналу тапов.

Журнал: [[t_ms, "L"|"R"], ...] — t_ms от начала партии, не убывает. L — шестёрка, R — семёрка.
  * пауза больше COMBO_GAP_MS — серия с нуля;
  * другая рука — серия +1, та же рука дважды — серия с нуля («жест сломан»);
  * множитель = min(7, 1 + серия // 6);
  * не хватает энергии — тап не засчитан и серию не трогает;
  * энергия копится REGEN в секунду, потолок — ENERGY_BASE (+ улучшение «Бак»);
  * крит — бросок из зерна партии (xorshift32): ×6 на левой руке, ×7 на правой.
Зависимостей нет, только целые числа — один и тот же журнал всегда даёт один и тот же счёт.
"""
from __future__ import annotations

VERSION = 1

COMBO_STEP, COMBO_MAX, COMBO_GAP_MS = 6, 7, 1200
ENERGY_BASE, ENERGY_TANK, REGEN_BASE = 670, 250, 3
MIN_GAP_MS = 50              # два тапа ближе 50 мс не считаем
MAX_EVENTS = 50000           # один журнал — не больше
MAX_MS = 2 * 3600 * 1000     # партия — не дольше двух часов


def xs32(s: int) -> int:
    """xorshift32."""
    s ^= (s << 13) & 0xFFFFFFFF
    s ^= s >> 17
    s ^= (s << 5) & 0xFFFFFFFF
    return s & 0xFFFFFFFF


def tap_value(up: dict) -> int:
    return 1 + int(up.get("multitap") or 0)


def energy_max(up: dict) -> int:
    return ENERGY_BASE + ENERGY_TANK * int(up.get("tank") or 0)


def regen(up: dict) -> int:
    return REGEN_BASE + int(up.get("regen") or 0)


def check(log) -> str | None:
    """Формат журнала. None — годится, иначе код отказа."""
    if not isinstance(log, list) or not log:
        return "empty"
    if len(log) > MAX_EVENTS:
        return "too_long"
    last = 0
    for ev in log:
        if not (isinstance(ev, (list, tuple)) and len(ev) == 2):
            return "bad_event"
        t, side = ev
        if isinstance(t, bool) or not isinstance(t, int) or side not in ("L", "R"):
            return "bad_event"
        if t < last:
            return "time_back"
        last = t
    if last > MAX_MS:
        return "too_long"
    return None


def score_log(log, seed: int, up: dict | None = None, energy0: int | None = None) -> dict:
    """Пересчитать партию. Возвращает итог; ok=False и why — если журнал негоден."""
    up = up or {}
    bad = check(log)
    if bad:
        return {"ok": False, "why": bad, "version": VERSION}
    cost, cap, reg = tap_value(up), energy_max(up), regen(up)
    crit_pm = 20 * int(up.get("crit") or 0)          # крит: 2% за уровень, в промилле
    e = (cap if energy0 is None else max(0, min(cap, int(energy0)))) * 1000   # энергия в тысячных
    s = (int(seed) & 0xFFFFFFFF) or 1
    streak, last_side, last_t, last_seen = 0, None, None, None
    res = {"ok": True, "version": VERSION, "coins": 0, "taps": 0, "paid": 0, "fast": 0,
           "no_energy": 0, "resets": 0, "crits": 0, "best_combo": 1}
    for t, side in log:
        if last_seen is not None:
            e = min(cap * 1000, e + reg * (t - last_seen))
        last_seen = t
        if last_t is not None and t - last_t < MIN_GAP_MS:
            res["fast"] += 1
            continue
        if e < cost * 1000:
            res["no_energy"] += 1
            continue
        e -= cost * 1000
        if last_t is not None and t - last_t > COMBO_GAP_MS:
            streak = 0
        if last_side is not None and side != last_side:
            streak += 1
        elif side == last_side:
            streak = 0
            res["resets"] += 1
        last_side, last_t = side, t
        m = min(COMBO_MAX, 1 + streak // COMBO_STEP)
        s = xs32(s)
        crit = crit_pm > 0 and s % 1000 < crit_pm
        gain = cost * m * ((6 if side == "L" else 7) if crit else 1)
        res["coins"] += gain
        res["taps"] += 1
        res["paid"] += 1
        res["crits"] += 1 if crit else 0
        if m > res["best_combo"]:
            res["best_combo"] = m
    res["energy_end"] = e // 1000
    res["duration_ms"] = log[-1][0] - log[0][0]
    return res
