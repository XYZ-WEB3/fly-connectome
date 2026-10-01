"""Раунд 4: утомление синапсов + медленная адаптация.
Хотим: MN9 > 25 Гц, взлёт > 50, поворот > 30, и после запаха — тишина (эхо ≈ 0, DNg35 молчит)."""
import json
import re

import numpy as np

import brain as B

ann = B.load_annotations()
nb = json.load(open("brain_model/example.ipynb", encoding="utf8"))
SUGAR = [int(x) for x in re.findall(r"7205\d+", "".join(nb["cells"][3]["source"]))]
MN9 = 720575940660219265

configs = {
    "U.1 + a.2/1000мс": dict(U=0.1, tau_d=300, a_inc=0.2, tau_a=1000),
    "U.1 + a.4/1000мс": dict(U=0.1, tau_d=300, a_inc=0.4, tau_a=1000),
    "U.1 + a.1/3000мс": dict(U=0.1, tau_d=300, a_inc=0.1, tau_a=3000),
    "U.1 + a.5/200мс": dict(U=0.1, tau_d=300, a_inc=0.5, tau_a=200),
}
for name, prm in configs.items():
    b = B.FlyBrain(params=prm)
    ty = lambda ts, s=None: [b.idx[r] for r in ann[ann.cell_type.isin(ts) & ((ann.side == s) if s else True)].index if r in b.idx]
    sugar = [b.idx[r] for r in SUGAR if r in b.idx]
    groom = ty(["DNg84", "DNg35"])
    r = {}
    b.reset(); b.set_input(sugar, 150); c = b.run(1500) / 1.5
    r["MN9"] = c[b.idx[MN9]]
    b.reset(); b.set_input(ty(["LPLC2", "LC4"], "left"), 60); c = b.run(300) / 0.3
    r["взлёт"] = c[ty(["DNp01"])].mean()
    b.reset(); b.set_input(ty(["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"], "left"), 50); c = b.run(1000)
    r["поворот"] = c[ty(["DNa01", "DNa02"], "left")].mean() - c[ty(["DNa01", "DNa02"], "right")].mean()
    b.reset(); b.set_input(ty(["ORN_DM1", "ORN_DM4", "ORN_VA2"], "left"), 30); b.run(1000)
    b.set_input(ty(["ORN_DM1", "ORN_DM4", "ORN_VA2"], "left"), 0); b.run(1500)
    c = b.run(500)
    r["эхо после запаха"] = int((c > 0).sum())
    r["чистка в покое, Гц"] = c[groom].mean() * 2
    print(name, {k: round(float(v), 1) for k, v in r.items()}, flush=True)
