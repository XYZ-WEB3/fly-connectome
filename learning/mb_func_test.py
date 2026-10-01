"""Проверка карты MB на живой модели (GPU, короткие прогоны).

1) Позиции рёбер KC->MBON из mb_kc_mbon_edges.npz совпадают с массивами быстрого ядра
   (b._post / b._wv / b._cptr) и с find_edges() по живым массивам.
2) Функционально: включаем MBON одного типа и одной стороны (Пуассон, как оптогенетика),
   смотрим руль DNa01/02 L/R, ходьбу DNp09, гигантское волокно DNp01, MN9 — в 4 контекстах:
   none (только MBON), steer (LC10 с двух сторон), sugar (сахарные GRN), loom (LPLC2+LC4 с двух сторон).
   Тормозной MBON виден только там, где цель уже активна, поэтому нужны контексты.

    .venv\\Scripts\\python.exe learning\\mb_func_test.py [--calib]
    -> learning/mb_func.json (потом ещё раз mb_map.py -> MB_MAP.md)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import brain as B  # noqa: E402
import mb_edges as E  # noqa: E402

PRM = dict(U=0.1, tau_d=300, a_inc=0.1, tau_a=3000)      # противоэпилептика, как везде
MB = json.loads((HERE / "mb_map.json").read_text(encoding="utf-8"))
MBON_HZ = 80.0             # частота Пуассона на каждую клетку MBON (≈ её частота спайков: толчок 68.75 мВ)
WARM, MEAS = 50.0, 350.0   # мс: разгон контекста, окно измерения
N_BASE, N_TRIAL = 20, 6
CTX = {                    # контекст -> [(группа, Гц)]
    "none": [],
    "steer": [("lc10_L", 15.0), ("lc10_R", 15.0)],
    "sugar": [("sugar", 300.0)],
    "loom": [("loom_L", 8.0), ("loom_R", 8.0)],
}
READ = ["DNa_L", "DNa_R", "DNp09_L", "DNp09_R", "GF_L", "GF_R", "MN9"]
LC10 = ["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"]


def groups(b, ann):
    ct = ann.cell_type.astype(str)

    def sel(types, side=None):
        a = ann[ct.isin(types)]
        if side:
            a = a[a.side == side]
        return [b.idx[r] for r in a.index if r in b.idx]
    g = dict(lc10_L=sel(LC10, "left"), lc10_R=sel(LC10, "right"),
             loom_L=sel(["LPLC2", "LC4"], "left"), loom_R=sel(["LPLC2", "LC4"], "right"),
             sugar=[b.idx[r] for r in __import__("mb_map").SUGAR_IDS_630 if r in b.idx],
             DNa_L=sel(["DNa01", "DNa02"], "left"), DNa_R=sel(["DNa01", "DNa02"], "right"),
             DNp09_L=sel(["DNp09"], "left"), DNp09_R=sel(["DNp09"], "right"),
             GF_L=sel(["DNp01"], "left"), GF_R=sel(["DNp01"], "right"),
             MN9=[b.idx[__import__("mb_map").MN9_ID]])
    return {k: torch.tensor(v, device=b.dev) for k, v in g.items()}


def edge_check(b):
    z = np.load(HERE / "mb_kc_mbon_edges.npz")
    pos = torch.as_tensor(z["csc_pos"], device=b.dev)
    post_ok = (b._post[pos].cpu().numpy() == z["post_idx"])
    wv = b._wv[pos].cpu().numpy()
    w_ok = np.isclose(wv, z["weight"], rtol=1e-6, atol=0)
    cp = b._cptr.cpu().numpy().astype(np.int64)
    seg = (cp[z["pre_idx"]] <= z["csc_pos"]) & (z["csc_pos"] < cp[z["pre_idx"] + 1])
    p2, ok2 = E.find_edges(b, z["pre_idx"], z["post_idx"])
    same = bool(ok2.all()) and bool((p2.cpu().numpy() == z["csc_pos"]).all())
    r = dict(n_checked=int(len(pos)), post_match=int(post_ok.sum()), w_match=int(w_ok.sum()),
             max_abs_dw=float(np.abs(wv - z["weight"]).max()), in_segment=int(seg.sum()),
             find_edges_match=same, nnz_total=int(b._post.numel()))
    print("проверка рёбер:", r)
    return r


@torch.no_grad()
def inplace_check(b, n_rep=4):
    """Правка b._wv НА МЕСТЕ действует и при воспроизведении CUDA Graph (граф не пересобираем):
    5% KC (случайные, seed 0) 20 Гц -> частота всех MBON; потом веса KC->MBON x0, потом обратно."""
    ed = E.kc_mbon_edges(b)
    pos = ed["pos"]
    kc = np.array([i for v in MB["kc"]["by_subtype"].values() for i in v["idx_left"] + v["idx_right"]])
    sel = torch.tensor(np.random.default_rng(0).choice(kc, size=len(kc) // 20, replace=False), device=b.dev)
    mb = torch.tensor([i for v in MB["mbon"].values() for i in v["idx_left"] + v["idx_right"]], device=b.dev)
    w0 = b._wv[pos].clone()
    assert b._graph_ok()

    def rate():
        out = []
        for _ in range(n_rep):
            b.reset()
            b.set_input(sel, 20.0)
            b.run(50, tensor=True)
            c = b.run(350, tensor=True)
            out.append(float(c[mb].mean()) * 1000 / 350)
        return float(np.mean(out)), float(np.std(out))
    r = dict(n_kc_driven=int(len(sel)), kc_hz=20.0, graph_used=bool(b._graph_ok()))
    r["mbon_hz_orig"] = rate()
    b._wv[pos] = 0.0
    r["mbon_hz_zeroed"] = rate()
    b._wv[pos] = w0 * 2.0
    r["mbon_hz_x2"] = rate()
    b._wv[pos] = w0
    r["mbon_hz_restored"] = rate()
    r["restored_exact"] = bool(torch.equal(b._wv[pos], w0))
    print("правка весов на месте:", r)
    return r


def trial(b, g, ctx, act=None):
    b.reset()
    for k, hz in CTX[ctx]:
        b.set_input(g[k], hz)
    if act is not None:
        b.set_input(act, MBON_HZ)
    b.run(WARM, tensor=True)
    hz = b.run(MEAS, tensor=True) * (1000.0 / MEAS)
    out = {k: float(hz[g[k]].mean()) for k in READ}
    out["n_active"] = int((hz > 0).sum())
    if act is not None:
        out["mbon_hz"] = float(hz[act].mean())
    return out


def stats(rows):
    keys = [k for k in rows[0] if k not in ("n_active", "mbon_hz")]
    m = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    s = {k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0 for k in keys}
    st = [r["DNa_L"] - r["DNa_R"] for r in rows]
    return m, s, float(np.mean(st)), float(np.std(st, ddof=1)) if len(rows) > 1 else 0.0, \
        float(np.mean([r["n_active"] for r in rows])), float(max(r["n_active"] for r in rows))


def main():
    t0 = time.time()
    ann = B.load_annotations()
    b = B.FlyBrain(params=PRM, seed=11)
    b.enable_graph()
    print(f"мозг готов за {time.time() - t0:.1f} с")
    res = dict(desc="", edge_check=edge_check(b), readouts=READ)
    g = groups(b, ann)

    if "--inplace" in sys.argv:                 # только проверка правки весов, дописать в mb_func.json
        old = json.loads((HERE / "mb_func.json").read_text(encoding="utf-8"))
        old["edge_check"] = res["edge_check"]
        old["inplace_check"] = inplace_check(b)
        (HERE / "mb_func.json").write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    res["inplace_check"] = inplace_check(b)

    if "--calib" in sys.argv:
        for ctx in CTX:
            rows = [trial(b, g, ctx) for _ in range(4)]
            m, s, st, sst, na, mx = stats(rows)
            print(ctx, {k: f"{m[k]:.1f}±{s[k]:.1f}" for k in READ}, "руль", f"{st:.1f}±{sst:.1f}", "активных", na)
        return

    # фон
    base = {}
    for ctx in CTX:
        rows = [trial(b, g, ctx) for _ in range(N_BASE)]
        m, s, st, sst, na, mx = stats(rows)
        base[ctx] = dict(mean=m, sd=s, steer_mean=st, steer_sd=sst, n_active=na, n_active_max=mx, n=N_BASE)
        print("фон", ctx, {k: f"{m[k]:.1f}±{s[k]:.1f}" for k in READ}, f"руль {st:.1f}±{sst:.1f}", f"актив {na:.0f}")
    res["baseline"] = base

    # MBON: тип x сторона
    allres, hits = {}, []
    for t, v in MB["mbon"].items():
        for sd, key in (("L", "idx_left"), ("R", "idx_right")):
            if not v[key]:
                continue
            act = torch.tensor(v[key], device=b.dev)
            gname = f"{t}|{sd}"
            allres[gname] = {}
            for ctx in CTX:
                rows = [trial(b, g, ctx, act) for _ in range(N_TRIAL)]
                m, s, st, sst, na, mx = stats(rows)
                bb = base[ctx]
                d = {k: m[k] - bb["mean"][k] for k in READ}
                # значимо: |Δ| >= 3 Гц и > 3 стандартных ошибок (SD фона и SD условия)
                se = {k: np.sqrt(bb["sd"][k] ** 2 / bb["n"] + s[k] ** 2 / N_TRIAL) for k in READ}
                sig = {k: bool(abs(d[k]) >= 3.0 and abs(d[k]) > 3 * se[k]) for k in READ}
                dst = st - bb["steer_mean"]
                se_st = np.sqrt(bb["steer_sd"] ** 2 / bb["n"] + sst ** 2 / N_TRIAL)
                sig_st = bool(abs(dst) >= 3.0 and abs(dst) > 3 * se_st)
                mh = float(np.mean([r["mbon_hz"] for r in rows]))
                row = dict(group=gname, ctx=ctx, delta=d, sig=sig, d_steer=dst, sig_steer=sig_st, mbon_hz=mh,
                           n_active=na, n_active_max=mx, mean=m)
                allres[gname][ctx] = row
                if any(sig.values()) or sig_st:
                    hits.append(row)
        print(f"{t} готово ({time.time() - t0:.0f} с)")
    res["all"] = allres
    res["hits"] = hits
    # руль: тип, у которого левая и правая активация сдвигают руль в разные стороны
    lat = {}
    for t in MB["mbon"]:
        for ctx in ("none", "steer"):
            L, R = allres.get(f"{t}|L", {}).get(ctx), allres.get(f"{t}|R", {}).get(ctx)
            if L and R:
                lat.setdefault(t, {})[ctx] = dict(d_steer_L_act=L["d_steer"], d_steer_R_act=R["d_steer"],
                                                  sig_L=L["sig_steer"], sig_R=R["sig_steer"])
    res["steer_by_side"] = lat
    na_max = max(r["n_active_max"] for gg in allres.values() for r in gg.values())
    res["n_active_max_any"] = na_max
    res["desc"] = (
        f"FlyBrain(params={PRM}, seed=11), быстрое ядро + CUDA Graph. Каждый прогон: reset(), контекст + MBON "
        f"{MBON_HZ:.0f} Гц Пуассон на каждую клетку типа (одна сторона), {WARM:.0f} мс разгон, {MEAS:.0f} мс окно "
        f"измерения. Фон — {N_BASE} прогонов на контекст, условие — {N_TRIAL}. Контексты: steer = LC10 обе стороны 15 Гц, "
        "sugar = 20 сахарных GRN 300 Гц, loom = LPLC2+LC4 обе стороны 8 Гц. Значимо (жирным): |Δ| >= 3 Гц и > 3 SE. "
        f"DNa_L/R — среднее DNa01+DNa02 стороны; руль = DNa_L - DNa_R. Максимум активных нейронов за окно во всех "
        f"прогонах: {na_max:.0f} из {b.N} (лавины нет, если это не десятки тысяч). Время: {time.time() - t0:.0f} с.")
    (HERE / "mb_func.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("готово", time.time() - t0, "с; значимых строк", len(hits))


if __name__ == "__main__":
    main()
