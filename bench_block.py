"""Пошаговый vs блочный vs CUDA Graph: те же ли спайки и насколько быстро."""
import time
import numpy as np
import torch
import brain as B

PRM = dict(U=0.1, tau_d=300, a_inc=0.1, tau_a=3000)
ann = B.load_annotations()
res = {}
for mode in ["step", "block", "graph"]:
    b = B.FlyBrain(params=PRM, seed=1)
    if mode == "graph":
        b.enable_graph()
    lc = [b.idx[r] for r in ann[ann.cell_type.isin(["LC10a", "LPLC2"])].index if r in b.idx]
    b.set_input(lc, 60)
    torch.cuda.synchronize(); t = time.time()
    if mode == "step":
        c = torch.zeros(b.N, device=b.dev)
        for _ in range(3600):
            c += b.step()
    else:
        c = b.run(360, tensor=True)
    torch.cuda.synchronize(); w = time.time() - t
    res[mode] = c.cpu().numpy()
    print(f"{mode:5s}: 1 с мозга = {w / 0.36:5.2f} с реального времени;  спайков {int(c.sum())}")
for m in ["block", "graph"]:
    d = np.abs(res["step"] - res[m])
    print(f"{m} vs step: расхождение {int(d.sum())} спайков")
