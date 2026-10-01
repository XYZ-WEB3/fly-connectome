"""Какие зрительные проекционные нейроны (из глаза в центральный мозг)
ведут к нейронам поворота DNa01/DNa02 и ходьбы — с учётом стороны.
Батч: много опытов параллельно на GPU."""
import numpy as np
import pandas as pd
import torch

import brain as B

ann = B.load_annotations()
b = B.FlyBrain()
P, dev, N = B.P, b.dev, b.N

READ = ["DNa01", "DNa02", "DNp09", "DNb05", "DNa03", "DNa06", "DNb06", "DNg13", "oDN1"]
read = {f"{t}_{s[0]}": [b.idx[r] for r in ann[(ann.cell_type == t) & (ann.side == s)].index if r in b.idx]
        for t in READ for s in ["left", "right"]}
read = {k: v for k, v in read.items() if v}
print("читаем:", list(read))

vpn = ann[ann.super_class == "visual_projection"]
types = [t for t, g in vpn.groupby("cell_type") if {"left", "right"} <= set(g.side)]
print(len(types), "типов VPN")


def run_batch(stims, ms=300, rate=80.0):
    """stims: список массивов индексов нейронов; каждый — отдельный опыт."""
    Bn = len(stims)
    v = torch.full((Bn, N), P["v_0"], device=dev); g = torch.zeros_like(v)
    rfc = torch.zeros((Bn, N), dtype=torch.int32, device=dev)
    buf = torch.zeros((b.delay_steps, N, Bn), device=dev); ptr = 0
    rate_in = torch.zeros((Bn, N), device=dev)
    for i, s in enumerate(stims):
        rate_in[i, s] = rate
    driven = rate_in > 0
    counts = torch.zeros((Bn, N), device=dev)
    decay = np.exp(-b.dt / P["tau"])
    for _ in range(int(ms / b.dt)):
        arr = buf[ptr]
        if arr.any():
            g += torch.sparse.mm(b.W, arr).T
        v += ((torch.rand((Bn, N), device=dev) < rate_in * b.dt * 1e-3) & driven) * (P["w_syn"] * P["f_poi"])
        act = rfc <= 0
        v = torch.where(act, v + (P["v_0"] - v + g) / P["t_mbr"] * b.dt, v)
        g = torch.where(act, g * decay, g)
        rfc -= 1
        spk = v > P["v_th"]
        v = torch.where(spk, torch.full_like(v, P["v_rst"]), v)
        g = torch.where(spk, torch.zeros_like(g), g)
        rfc = torch.where(spk & ~driven, torch.full_like(rfc, b.rfc_steps), rfc)
        buf[ptr] = spk.float().T
        ptr = (ptr + 1) % b.delay_steps
        counts += spk
    return (counts / (ms / 1000)).cpu().numpy()


rows = []
BATCH = 16
jobs = [(t, s) for t in types for s in ["left", "right"]]
for k in range(0, len(jobs), BATCH):
    chunk = jobs[k:k + BATCH]
    stims = [[b.idx[r] for r in vpn[(vpn.cell_type == t) & (vpn.side == s)].index if r in b.idx] for t, s in chunk]
    rates = run_batch(stims)
    for (t, s), r in zip(chunk, rates):
        row = dict(type=t, side=s, n_active=int((r > 0).sum()))
        row.update({kk: float(r[vv].mean()) for kk, vv in read.items()})
        rows.append(row)
    print(f"{k + len(chunk)}/{len(jobs)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results_vpn_scan.csv", index=False)
L = df[df.side == "left"].set_index("type"); R = df[df.side == "right"].set_index("type")
cols = [c for c in read]
asym = pd.DataFrame({"a02": (L["DNa02_l"] - L["DNa02_r"]) - (R["DNa02_l"] - R["DNa02_r"]) if "DNa02_l" in L else 0,
                     "a01": (L["DNa01_l"] - L["DNa01_r"]) - (R["DNa01_l"] - R["DNa01_r"]) if "DNa01_l" in L else 0,
                     "nL": L.n_active, "nR": R.n_active})
asym["score"] = asym.a02.abs() + asym.a01.abs()
print(asym.sort_values("score", ascending=False).head(30).round(1).to_string())
