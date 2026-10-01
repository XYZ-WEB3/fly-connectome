"""Перебор сенсорных входов: какой вход мозг обрабатывает без «лавины»
и при этом по-разному отвечает на стимул слева / справа на нисходящих нейронах
поворота (DNa01, DNa02) и ходьбы (DNp09, DNb05 и др.)."""
import numpy as np
import pandas as pd

import brain as B

ann = B.load_annotations()
b = B.FlyBrain()
dn = ann[ann.super_class == "descending"]
dn_idx = np.array([b.idx[r] for r in dn.index if r in b.idx])
dn_root = b.ids[dn_idx]
dn_name = [f"{ann.loc[r, 'cell_type']}_{str(ann.loc[r, 'side'])[0]}" for r in dn_root]

sens = ann[(ann.super_class.isin(["sensory", "sensory_ascending"])) & (ann.cell_class != "visual")]
types = [t for t, g in sens.groupby("cell_type") if {"left", "right"} <= set(g.side) and len(g) >= 2]
print(len(types), "типов сенсорных нейронов с левой и правой стороной")

rows = []
for t in types:
    res = {}
    for s in ["left", "right"]:
        idx = [b.idx[r] for r in sens[(sens.cell_type == t) & (sens.side == s)].index if r in b.idx]
        b.reset(); b.set_input(idx, 100)
        c = b.run(300) / 0.3
        b.set_input(idx, 0)
        c2 = b.run(100)
        res[s] = (c, int((c > 0).sum()), int((c2 > 0).sum()))
    L, R = res["left"][0][dn_idx], res["right"][0][dn_idx]
    row = dict(type=t, cls=sens[sens.cell_type == t].cell_class.iloc[0],
               activeL=res["left"][1], activeR=res["right"][1],
               afterL=res["left"][2], afterR=res["right"][2],
               dn_asym=float(np.abs(L - R).sum()), dn_total=float((L + R).sum()))
    for k in np.argsort(-np.abs(L - R))[:3]:
        row.setdefault("top", []).append(f"{dn_name[k]} {L[k]:.0f}/{R[k]:.0f}")
    row["top"] = "; ".join(row.get("top", []))
    rows.append(row)
    print(row, flush=True)

df = pd.DataFrame(rows)
df.to_csv("results_sense_scan.csv", index=False)
ok = df[(df.afterL < 50) & (df.afterR < 50)].sort_values("dn_asym", ascending=False)
print("\n=== Без лавины, по асимметрии DN ===")
print(ok.head(25).to_string())
