"""Бенчмарк и сверка путей расчёта мозга (brain.FlyBrain).

Режимы:
  eager       — старый путь, step() по 0.1 мс (torch-операции + cuSPARSE mv)
  block       — старый путь, run(): одно SpMM на блок 18 шагов, 18x ~25 мелких ядер
  graph       — старый путь, run() после enable_graph(): блок записан в CUDA Graph
  fused       — новый путь, run() без графа: 2 своих ядра на запуск (до 18 шагов)
  fused+graph — новый путь, run() после enable_graph() (так работают show.py / prototype.py)

Что печатаем:
  1) побитная сверка: старый _dyn против нового ядра на одинаковых случайных числах;
  2) для каждого режима: стенное время на 1 с мозга, сумма спайков, число активных
     нейронов, корреляция частот по нейронам с eager, MN9 / гигантское волокно / поворот;
  3) шум: корреляция fused(seed A) против fused(seed B) — ниже неё различия режимов
     статистически не отличимы.

Запуск:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -W ignore tiktok/bench_brain.py [сек_мозга]
"""
import ctypes
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import brain as B  # noqa: E402

PRM = dict(U=0.1, tau_d=300, a_inc=0.1, tau_a=3000)
T_MS = float(sys.argv[1]) * 1000 if len(sys.argv) > 1 else 1000.0
MN9_ID = 720575940660219265


def groups(b, ann):
    ty = lambda ts, s=None: [b.idx[r] for r in ann[ann.cell_type.isin(ts) & ((ann.side == s) if s else True)].index
                             if r in b.idx]
    nb = json.load(open(ROOT / "brain_model" / "example.ipynb", encoding="utf8"))
    sugar = [b.idx[int(x)] for x in re.findall(r"7205\d+", "".join(nb["cells"][3]["source"])) if int(x) in b.idx]
    return dict(sugar=sugar, loom=ty(["LPLC2", "LC4"], "left"),
                lc10=ty(["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"], "left"),
                gf=ty(["DNp01"]), dnaL=ty(["DNa01", "DNa02"], "left"), dnaR=ty(["DNa01", "DNa02"], "right"),
                mn9=[b.idx[MN9_ID]])


def protocol(b, G, mode, T):
    """Смесь входов (сахар + LC10 слева + луминг слева) на T мс, как в шоу."""
    b.reset()
    b.set_input(G["sugar"], 150); b.set_input(G["lc10"], 50); b.set_input(G["loom"], 40)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if mode in ("eager", "fused-step"):
        c = torch.zeros(b.N, device=b.dev)
        for _ in range(int(round(T / b.dt))):
            c += b.step()
    elif mode == "frames":
        # как prototype.py: кадры по 33.3 мс (не кратно 1.8 мс)
        c = torch.zeros(b.N, device=b.dev)
        for _ in range(int(round(T / (1000 / 30)))):
            c += b.run(1000 / 30, tensor=True)
    else:
        c = b.run(T, tensor=True)
    torch.cuda.synchronize()
    return c.cpu().numpy().astype(np.float64), time.perf_counter() - t0


def setmode(b, fast, graph):
    b.fast = fast
    b.graph = None
    b.reset()
    if graph:
        b.enable_graph()


def corr(x, y):
    m = (x > 0) | (y > 0)
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def exact_check(b, T=3000):
    """Старый _dyn (torch) против нового ядра на одних и тех же случайных числах."""
    N, D = b.N, b.delay_steps
    gen = torch.Generator(device="cuda").manual_seed(11)
    Us = torch.rand((T, N), generator=gen, device="cuda")
    G = b._groups
    def inputs():
        b.set_input(G["sugar"], 150); b.set_input(G["lc10"], 50); b.set_input(G["loom"], 40)
    b.fast = False; b.graph = None; b.reset(); inputs()
    old = torch.zeros((T, N), dtype=torch.bool, device="cuda")
    for s in range(T):
        old[s] = b._dyn(torch.mv(b.W, b.buf[b.ptr]), rnd=Us[s])
    st_old = [z.clone() for z in (b.v, b.g, b.a, b.x, b.rfc)]
    b.fast = True; b.reset(); inputs()
    new = torch.zeros((T, N), device="cuda")
    for s in range(T):                       # по шагу, чтобы получить растр
        c = torch.zeros(N, device="cuda")
        b._fast_launch(1, c, rnd=Us[s:s + 1])
        new[s] = c
    st_new = [z.clone() for z in (b.v, b.g, b.a, b.x, b.rfc)]
    b.reset(); inputs()
    tot = torch.zeros(N, device="cuda"); s = 0
    for m in [18, 7, 18, 18, 11, 1, 18] * 10000:  # куски разной длины, с любой позиции кольца
        m = min(m, T - s)
        if m <= 0:
            break
        b._fast_launch(m, tot, rnd=Us[s:s + m].contiguous()); s += m
    torch.cuda.synchronize()
    mis = int((old.float() != new).sum())
    dv = float((st_old[0] - st_new[0]).abs().max())      # суммы синаптического тока в другом порядке
    dg = float((st_old[1] - st_new[1]).abs().max())      # -> расхождение в последних битах float32
    return dict(steps=T, spikes_old=int(old.sum()), spikes_new=int(new.sum()), raster_mismatch=mis,
                max_dv_mV=dv, max_dg=dg, chunked_total=int(tot.sum()),
                chunked_per_neuron_absdiff=int((old.float().sum(0) - tot).abs().sum()))


def main():
    t0 = time.time()
    ann = B.load_annotations()
    b = B.FlyBrain(params=PRM, seed=0)
    print(f"мозг загружен за {time.time() - t0:.1f} с; N={b.N}, синаптических пар {b.W.values().numel()}, fast={b.fast}", flush=True)
    G = b._groups = groups(b, ann)
    print(f"GPU: {torch.cuda.get_device_name(0)}; память занята torch: {torch.cuda.memory_allocated() / 2**20:.0f} МБ", flush=True)

    print("\n== 1) побитная сверка (одинаковые случайные числа) ==", flush=True)
    ex = exact_check(b)
    print(ex, flush=True)

    print(f"\n== 2) режимы: {T_MS / 1000:.2f} с мозга, вход: сахар 150 Гц + LC10 слева 50 Гц + LPLC2/LC4 слева 40 Гц ==", flush=True)
    modes = [("eager", False, False), ("block", False, False), ("graph", False, True),
             ("fused-step", True, False), ("fused", True, False), ("fused+graph", True, True),
             ("frames", True, True)]
    res = {}
    for name, fast, graph in modes:
        setmode(b, fast, graph)
        protocol(b, G, name, min(T_MS, 36.0))            # прогрев
        T = T_MS if name not in ("eager", "fused-step") else min(T_MS, 500.0)  # пошаговые медленнее: 0.5 с
        c, w = protocol(b, G, name, T)
        res[name] = (c * (1000.0 / T), w / (T / 1000.0), T)
    ref = res["eager"][0]
    print(f"{'режим':12s} {'с/1с мозга':>10s} {'x реалтайм':>10s} {'спайков/с':>10s} {'активных':>8s} "
          f"{'r(eager)':>8s} {'MN9 Гц':>7s} {'GF Гц':>6s} {'DNa L-R':>7s}")
    for name, (hz, wall, T) in res.items():
        r, _ = corr(hz, ref)
        print(f"{name:12s} {wall:10.3f} {1 / wall:10.1f} {hz.sum():10.0f} {int((hz > 0).sum()):8d} {r:8.3f} "
              f"{hz[G['mn9']].mean():7.1f} {hz[G['gf']].mean():6.1f} {hz[G['dnaL']].mean() - hz[G['dnaR']].mean():7.1f}",
              flush=True)

    print("\n== 3) шум: одинаковый режим, разные случайные числа ==", flush=True)
    setmode(b, True, True)
    a, _ = protocol(b, G, "fused+graph", T_MS)
    c, _ = protocol(b, G, "fused+graph", T_MS)             # ГСЧ идёт дальше -> другая реализация
    e2 = res["eager"]
    print(f"fused vs fused (другие случайные числа, {T_MS / 1000:.1f} с): r = {corr(a, c)[0]:.3f}")
    print(f"block vs graph (старый путь):                      r = {corr(res['block'][0], res['graph'][0])[0]:.3f}")
    print(f"fused+graph vs block:                              r = {corr(res['fused+graph'][0], res['block'][0])[0]:.3f}")
    print(f"(eager шёл {e2[2] / 1000:.1f} с мозга -> его r ниже из-за более короткого окна)")

    print("\n== 4) долгий прогон быстрого пути (10 с мозга, граф) ==", flush=True)
    setmode(b, True, True)
    b.reset(); b.set_input(G["sugar"], 150); b.set_input(G["lc10"], 50); b.set_input(G["loom"], 40)
    torch.cuda.synchronize(); t = time.perf_counter()
    c = b.run(10000, tensor=True)
    torch.cuda.synchronize(); w = time.perf_counter() - t
    print(f"10 с мозга за {w:.2f} с -> {w / 10:.3f} с на 1 с мозга (x{10 / w:.0f} реалтайма); спайков {int(c.sum())}")

    print("\n== 5) тяжёлый режим: исходная модель без противоэпилептики + запах (судорога) ==", flush=True)
    orn = [b.idx[r] for r in ann[ann.cell_type.isin(["ORN_DM1", "ORN_DM4", "ORN_VA2"]) & (ann.side == "left")].index
           if r in b.idx]
    b.P = dict(B.P)                                        # U=0, a_inc=0 (веса те же: w_syn не менялся)
    b._set_consts()
    out = {}
    for name, fast in [("graph", False), ("fused+graph", True)]:
        setmode(b, fast, True)
        b.set_input(orn, 30); b.run(500)                   # запах 0.5 с -> самоподдерживающаяся активность
        b.set_input(orn, 0)
        torch.cuda.synchronize(); t = time.perf_counter()
        c = b.run(1000, tensor=True)
        torch.cuda.synchronize(); w = time.perf_counter() - t
        c = c.cpu().numpy()
        out[name] = c
        print(f"{name:12s}: 1 с мозга после запаха за {w:.3f} с; спайков {c.sum():.0f}, активных {(c > 0).sum()}", flush=True)
    print(f"r(graph, fused+graph) = {corr(out['graph'], out['fused+graph'])[0]:.3f}")
    b.P = dict(B.P, **PRM); b._set_consts()


if __name__ == "__main__":
    with torch.no_grad():
        main()
