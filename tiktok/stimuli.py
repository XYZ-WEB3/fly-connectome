"""«Движок интереса», часть 2: когда давать мухе настоящие сенсорные стимулы.

    import analyze as A, stimuli as S
    tl = A.analyze(path)
    plan = S.plan_stimuli(tl, seed=0)          # [{t, type, dur, strength, reason, caption, event_t, event_kind}]
    arr = S.plan_arrays(plan, tl["meta"]["n_frames_out"])   # покадрово: arr["sugar"], arr["puff"] (0..1)
    tick = S.ticker(tl, plan)                  # бегущая строка: [{t, dur, text, kind}]

Стимулы (время выбираем мы — «экспериментатор», а реакция — из мозга):
  'sugar' — капля сахара из пипетки у хоботка -> 21 сахарный GRN -> MN9 (хоботок). ~2.5 с.
            Награда за самые сильные моменты (дропы, пики). Не чаще раза в 8-10 с.
            t — начало вкусового входа; пипетку можно показывать заранее (~0.6 с до t).
  'puff'  — обдув головы (щетинки BM_Ant, BM_Fr, BM_FrOr) -> DNg84/DNg35 (чистка). ~0.6 с.
            После мощного баса/дропа или вспышки. Не чаще раза в 12 с.
strength 0..1 — множитель частоты входа (например сахар: 200 Гц * strength, обдув: 150 Гц * strength).

Непрерывные входы НЕ планируются — они идут из шкалы каждый кадр (см. CONTINUOUS):
  obj_L/obj_R -> LC10 (слежение за мелким объектом), loom_L/loom_R -> LPLC2 + LC4 (угроза, взлёт),
  hear -> JO-A/JO-B (слух: орган Джонстона).
"""
import numpy as np

SUGAR_DUR = 2.5
BITTER_DUR = 1.5
PUFF_DUR = 0.6
PUFF_DELAY = 0.15          # обдув — чуть ПОСЛЕ удара/вспышки
CONTINUOUS = {
    "obj_L": "LC10 левые (мелкий объект слева)",
    "obj_R": "LC10 правые (мелкий объект справа)",
    "loom_L": "LPLC2 + LC4 левые (надвигание слева)",
    "loom_R": "LPLC2 + LC4 правые (надвигание справа)",
    "hear": "JO-A / JO-B (слух, 100-500 Гц)",
}


def _ev_word(e):
    """Короткое «за что» для подписи."""
    return {"drop": "за бас-дроп" if e["strength"] >= 0.55 else "за подъём музыки",
            "peak": "за " + e["caption"],
            "flash": "после вспышки",
            "cut": "после смены кадра",
            "zoom": "после наезда",
            "beat": "в такт музыке"}.get(e["kind"], e["caption"])


def _candidates(tl, rng):
    """(оценка, t, тип, событие) — какие события годятся для какого стимула."""
    bass = np.asarray(tl["bass"])
    loud = np.asarray(tl["loud"])
    out = []
    for e in tl["events"]:
        k, s, i = e["kind"], e["strength"], e["frame"]
        b = float(bass[min(i, len(bass) - 1)])
        if k == "drop":
            out += [(1.2 + s, e["t"], "sugar", e), (0.6 + s * b, e["t"] + PUFF_DELAY, "puff", e)]
        elif k == "peak":
            out += [(0.35 + s, e["t"], "sugar", e), (0.1 + 0.9 * s * b, e["t"] + PUFF_DELAY, "puff", e)]
        elif k == "flash":
            out += [(0.55 + s, e["t"] + PUFF_DELAY, "puff", e), (0.1 + 0.5 * s, e["t"], "sugar", e)]
        elif k == "cut":
            out += [(0.15 + 0.3 * s * float(loud[min(i, len(loud) - 1)]), e["t"] + PUFF_DELAY, "puff", e)]
    return [(sc + rng.uniform(0, 0.02), t, ty, e) for sc, t, ty, e in out]


def _clean(w):
    """мат в подписи — первая буква и звёздочки"""
    import re
    return w[0] + "*" * (len(w) - 1) if re.match(r"^(бля|сук|пизд|ху|еба|ёба|пидор|муда|fuck|shit)", w, re.I) else w


def plan_stimuli(timeline, seed=0, sugar_gap=(8.0, 10.0), puff_gap=12.0, min_sep=3.5, hook=(1.0, 4.5),
                 max_gap=13.0, min_score=0.95, meaning=None, meaning_gap=4.0):
    """План стимулов по шкале analyze(). Возвращает список по времени."""
    tl = timeline
    rng = np.random.default_rng(seed)
    dur_total = float(tl["meta"]["duration"])
    fps = float(tl["meta"].get("fps_out", 30))
    loom = np.maximum(np.asarray(tl["loom_L"]), np.asarray(tl["loom_R"]))
    s_gap = float(rng.uniform(*sugar_gap))
    gaps = {"sugar": s_gap, "puff": float(puff_gap)}
    durs = {"sugar": SUGAR_DUR, "puff": PUFF_DUR, "bitter": BITTER_DUR}
    gaps["bitter"] = meaning_gap
    chosen = []

    def ok(t, ty):
        d = durs[ty]
        if t < 0.3 or t + d > dur_total - 0.2:
            return False
        for c in chosen:
            if abs(t - c["t"]) < min_sep:
                return False
            if c["type"] == ty and abs(t - c["t"]) < gaps[ty]:
                return False
            a0, a1 = sorted([(t, t + d), (c["t"], c["t"] + c["dur"])])      # не накладываем стимулы
            if a1[0] < a0[1] + 0.8:
                return False
        if ty == "sugar":        # в момент сильного надвигания муха занята побегом — сахар не даём
            i0, i1 = int((t - 0.2) * fps), int((t + 1.0) * fps)
            if loom[max(0, i0):max(1, i1)].max(initial=0) > 0.6:
                return False
        return True

    def _ok_meaning(t):                  # горькое: не внахлёст с другими каплями, не чаще meaning_gap
        if t < 0.3 or t + BITTER_DUR > dur_total - 0.2:
            return False
        return all(abs(t - c["t"]) >= (meaning_gap if c["type"] == "bitter" else c["dur"] + 0.5) for c in chosen)

    def add(t, ty, e, reason_extra=""):
        s = e["strength"] if e else 0.6
        strength = 0.7 + 0.3 * s if ty == "sugar" else 0.6 + 0.4 * s
        what = "капля сахара" if ty == "sugar" else "обдув головы"
        why = _ev_word(e) if e else "для затравки"
        chosen.append(dict(t=round(float(t), 3), type=ty, dur=durs[ty], strength=round(float(strength), 3),
                           reason=(f"{why} на {e['t']:.1f} с" if e else why) + reason_extra,
                           caption=f"{what} {why}", event_t=e["t"] if e else None,
                           event_kind=e["kind"] if e else None))

    # 0) смысл речи (meaning.py): «хорошо» -> сахар, «плохо» -> горькое. Первыми — у них приоритет
    for m in (meaning or []):
        ty = "sugar" if m["kind"] == "pos" else "bitter"
        t = float(m["t"]) + 0.3
        if ok(t, ty) if ty == "sugar" else _ok_meaning(t):
            what = "капля сахара" if ty == "sugar" else "горькая капля"
            chosen.append(dict(t=round(t, 3), type=ty, dur=durs[ty], strength=1.0,
                               reason=f"по смыслу: «{_clean(m['word'])}» на {m['t']:.1f} с",
                               caption=f"{what}: «{_clean(m['word'])}»", event_t=m["t"], event_kind="speech"))
    cands = sorted(_candidates(tl, rng), key=lambda c: -c[0])
    # 1) крючок: первая реакция в первые секунды. Сначала пробуем сахар на любом событии окна,
    #    потом — на доле такта, потом обдув, и только потом «просто так» на 2.5 с
    hk = sorted([c for c in cands if hook[0] <= c[1] <= hook[1]], key=lambda c: -c[0])
    beats = [dict(t=round(i / fps, 3), frame=int(i), kind="beat", strength=0.5, caption="в такт")
             for i in np.nonzero(np.asarray(tl["beat"]) > 0)[0] if hook[0] <= i / fps <= hook[1]]
    tries = [(t, "sugar", e) for sc, t, ty, e in hk if ty == "sugar"] + [(b["t"], "sugar", b) for b in beats] +             [(t, "puff", e) for sc, t, ty, e in hk if ty == "puff"] + [(2.5, "sugar", None)]
    for t, ty, e in tries:
        if ok(t, ty):
            add(t, ty, e, " (крючок)" if e else "")
            break
    # 2) жадно: сильнейшие события
    for sc, t, ty, e in cands:
        if sc >= min_score and ok(t, ty):
            add(t, ty, e)
    # 3) дыры длиннее max_gap — лучшее, что есть внутри (порог ниже)
    for _ in range(50):
        ts = [0.0] + sorted(c["t"] for c in chosen) + [dur_total]
        filled = False
        for a, b in zip(ts, ts[1:]):
            if b - a <= max_gap:
                continue
            for sc, t, ty, e in cands:
                if a < t < b and ok(t, ty):
                    add(t, ty, e, " (пауза)")
                    filled = True
                    break
            if filled:
                break
        if not filled:
            break
    chosen.sort(key=lambda c: c["t"])
    return chosen


def plan_arrays(plan, n_frames, fps_out=30):
    """Покадровые входы стимулов: {'sugar': float32[n], 'puff': float32[n]} (значение = strength внутри окна)."""
    out = {k: np.zeros(n_frames, np.float32) for k in ("sugar", "puff", "bitter")}
    for p in plan:
        i0 = int(round(p["t"] * fps_out))
        i1 = int(round((p["t"] + p["dur"]) * fps_out))
        out[p["type"]][max(0, i0):max(0, min(n_frames, i1))] = p["strength"]
    return out


def ticker(timeline, plan, min_gap=2.0, dur=1.6):
    """Бегущая строка событий без спама: стимулы, дропы, наезды, вспышки, пики, изредка — склейки."""
    items = [(3.0, p["t"], p["caption"], p["type"]) for p in plan]
    last_cut = -1e9
    for e in timeline["events"]:
        k, s = e["kind"], e["strength"]
        if k == "drop":
            items.append((2.5, e["t"], e["caption"], k))
        elif k == "zoom":
            items.append((2.0 + s, e["t"], e["caption"], k))
        elif k == "flash" and s >= 0.5:
            items.append((1.5 + s, e["t"], e["caption"], k))
        elif k == "peak" and s >= 0.6:
            items.append((1.0 + s, e["t"], e["caption"], k))
        elif k == "cut" and s >= 0.8 and e["t"] - last_cut >= 5.0:
            items.append((0.5, e["t"], e["caption"], k))
            last_cut = e["t"]
    items.sort(key=lambda x: -x[0])
    keep = []
    for pr, t, text, k in items:
        if all(abs(t - q[1]) >= min_gap for q in keep):
            keep.append((pr, t, text, k))
    keep.sort(key=lambda x: x[1])
    return [dict(t=round(t, 3), dur=dur, text=text, kind=k) for pr, t, text, k in keep]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze as A
    tl = A.analyze(sys.argv[1])
    plan = plan_stimuli(tl, seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    print(A.summary(tl, plan))
    print("бегущая строка:")
    for it in ticker(tl, plan):
        print(f"  {it['t']:6.2f}  {it['text']}")
