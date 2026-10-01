"""Подбор «противоэпилептики»: адаптация нейронов (a_inc) + утомление синапсов (U).
Хотим: поведение сохранилось, а приступов нет."""
import itertools
import json
import re
import sys

import numpy as np
import pandas as pd

import brain as B

ann = B.load_annotations()
nb = json.load(open("brain_model/example.ipynb", encoding="utf8"))
SUGAR = [int(x) for x in re.findall(r"7205\d+", "".join(nb["cells"][3]["source"]))]
MN9 = 720575940660219265

# (inh_scale, a_inc, U, tau_d)
configs = [(1, 0, 0.1, 300), (1, 0, 0.2, 300), (1, 0, 0.3, 300), (1, 0, 0.2, 100), (1, 1, 0.2, 300), (1, 0, 0.3, 600)]
if len(sys.argv) > 1:
    configs = [tuple(map(float, c.split(","))) for c in sys.argv[1:]]
b = B.FlyBrain()
ty = lambda ts, s=None: [b.idx[r] for r in ann[ann.cell_type.isin(ts) & ((ann.side == s) if s else True)].index if r in b.idx]
sugar = [b.idx[r] for r in SUGAR if r in b.idx]
mn9 = [b.idx[MN9]]
gf = ty(["DNp01"])
dna_L, dna_R = ty(["DNa01", "DNa02"], "left"), ty(["DNa01", "DNa02"], "right")
lc10_L = ty(["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"], "left")
lc10_R = ty(["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"], "right")
loom_L = ty(["LPLC2", "LC4"], "left")
orn = ty(["ORN_DM1", "ORN_DM4", "ORN_VA2"], "left")


def trial(inputs, ms_on=500, ms_off=300):
    b.reset()
    for idx, r in inputs:
        b.set_input(idx, r)
    on = b.run(ms_on) * 1000 / ms_on
    for idx, r in inputs:
        b.set_input(idx, 0)
    off = b.run(ms_off)
    return on, int((on > 0).sum()), int((off[ms_off // 2:] if False else off > 0).sum())


rows = []
for inh, a_inc, U, tau_d in configs:
    b = B.FlyBrain(params=dict(inh_scale=inh, a_inc=a_inc, U=U, tau_d=tau_d))
    ty = lambda ts, s=None: [b.idx[r] for r in ann[ann.cell_type.isin(ts) & ((ann.side == s) if s else True)].index if r in b.idx]
    r = dict(inh=inh, a_inc=a_inc, U=U, tau_d=tau_d)
    on, n, aft = trial([(sugar, 150)]);                 r.update(sugar_MN9=on[mn9].mean(), sugar_act=n, sugar_after=aft)
    on, n, aft = trial([(loom_L, 60)], 300);            r.update(loom_GF=on[gf].mean(), loom_act=n)
    on, n, aft = trial([(lc10_L, 50)]);                 r.update(lc10_turn=on[dna_L].mean() - on[dna_R].mean(), lc10_act=n)
    on, n, aft = trial([(orn, 30)]);                    r.update(odor_act=n, odor_after=aft)
    on, n, aft = trial([(sugar, 200), (lc10_L, 60), (lc10_R, 60)], 1500)
    r.update(combo_act=n, combo_after=aft)
    rows.append(r)
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

pd.DataFrame(rows).to_csv("results_epilepsy.csv", index=False)
