"""Долгая проверка: возвращается ли мозг в покой после сильных стимулов.
Циклы: запах 1 с → пауза 1.5 с; сахар+еда перед глазами 1.5 с → пауза 1.5 с.
Меряем, сколько нейронов ещё горит в последние 0.5 с каждой паузы."""
import json
import re
import sys

import brain as B

ann = B.load_annotations()
nb = json.load(open("brain_model/example.ipynb", encoding="utf8"))
SUGAR = [int(x) for x in re.findall(r"7205\d+", "".join(nb["cells"][3]["source"]))]

configs = {"оригинал": {}, "U0.1_td300": dict(U=0.1, tau_d=300), "U0.2_td100": dict(U=0.2, tau_d=100)}
for name, prm in configs.items():
    b = B.FlyBrain(params=prm)
    ty = lambda ts, s=None: [b.idx[r] for r in ann[ann.cell_type.isin(ts) & ((ann.side == s) if s else True)].index if r in b.idx]
    sugar = [b.idx[r] for r in SUGAR if r in b.idx]
    orn = ty(["ORN_DM1", "ORN_DM4", "ORN_VA2"], "left")
    lc10 = ty(["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"])
    b.reset()
    rest = []
    for cyc in range(3):
        b.set_input(orn, 30); b.run(1000); b.set_input(orn, 0)
        b.run(1000); rest.append(int((b.run(500) > 0).sum()))
        b.set_input(sugar, 200); b.set_input(lc10, 60); b.run(1500)
        b.set_input(sugar, 0); b.set_input(lc10, 0)
        b.run(1000); rest.append(int((b.run(500) > 0).sum()))
    print(f"{name:12s} нейронов в покое после каждого стимула: {rest}", flush=True)
