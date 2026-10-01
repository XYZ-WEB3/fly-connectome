"""Опыт 1: только мозг, без тела.

A) Проверка модели: «кормим» сахарные вкусовые нейроны -> должен
   заработать MN9 (мотонейрон хоботка), как в статье Shiu et al.
B) Даём запах на левую / правую антенну и смотрим, какие нисходящие
   нейроны (DN, «провода» от мозга к телу) реагируют и есть ли асимметрия.
"""
import time

import numpy as np
import pandas as pd

from brain import FlyBrain, load_annotations

T = 1000  # мс на один прогон
MN9 = 720575940660219265

brain = FlyBrain()
ann = load_annotations()
print(f"Мозг загружен: {brain.N} нейронов, {brain.n_syn:,} синапсов, устройство {brain.dev}")

def ids_to_idx(root_ids):
    return [brain.idx[r] for r in root_ids if r in brain.idx]

def group(cell_type=None, side=None, sub_class=None):
    a = ann
    if cell_type is not None:
        a = a[a.cell_type == cell_type]
    if sub_class is not None:
        a = a[a.cell_sub_class == sub_class]
    if side is not None:
        a = a[a.side == side]
    return ids_to_idx(a.index)

def trial(inputs, rate=150.0):
    brain.reset()
    for idx in inputs:
        brain.set_input(idx, rate)
    t0 = time.time()
    counts = brain.run(T)
    return counts * (1000.0 / T), time.time() - t0   # Гц

# ---------- A: сахар -> хоботок ----------
sugar_R = group(sub_class="sugar/water", side="right")
rates, wall = trial([sugar_R])
print(f"\n[A] Сахар справа ({len(sugar_R)} нейронов). 1 с мозга посчитана за {wall:.1f} с")
print(f"    MN9 (хоботок): {rates[brain.idx[MN9]]:.0f} Гц;  активных нейронов: {(rates > 0).sum()}")
rates0, _ = trial([])
print(f"    Без стимула:   MN9 {rates0[brain.idx[MN9]]:.0f} Гц;  активных нейронов: {(rates0 > 0).sum()}")

# ---------- B: запах слева vs справа ----------
dn_mask = ann.super_class == "descending"
dn_ids = ids_to_idx(ann[dn_mask].index)
dn_root = brain.ids[dn_ids]

res = {}
for glom in ["DM1", "DM4", "VA2", "DM2"]:          # «вкусные» запахи (уксус, фрукты)
    for side in ["left", "right"]:
        idx = group(cell_type=f"ORN_{glom}", side=side)
        r, _ = trial([idx])
        res[(glom, side)] = r[dn_ids]

df = pd.DataFrame({f"{g}_{s}": v for (g, s), v in res.items()}, index=dn_root)
df["type"] = ann.loc[df.index, "cell_type"].values
df["side"] = ann.loc[df.index, "side"].values
act = df[df.drop(columns=["type", "side"]).max(axis=1) > 0]
print(f"\n[B] Нисходящие нейроны, реагирующие на запах: {len(act)} из {len(dn_ids)}")
print(act.sort_values(act.columns[0], ascending=False).head(40).round(0).to_string())
df.to_csv("results_exp1_dn.csv")
