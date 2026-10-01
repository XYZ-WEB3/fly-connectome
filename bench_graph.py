"""Бенчмарк: обычный шаг мозга vs CUDA Graph (N шагов одним запуском)."""
import time

import numpy as np
import torch

import brain as B

PRM = dict(U=0.1, tau_d=300, a_inc=0.1, tau_a=3000)
b = B.FlyBrain(params=PRM)
b.set_input(list(range(200)), 50)

# --- обычный режим ---
for _ in range(100):
    b.step()
torch.cuda.synchronize(); t = time.time()
for _ in range(1500):
    b.step()
torch.cuda.synchronize()
eager = (time.time() - t) / 1500 * 1000
print(f"обычный шаг: {eager:.3f} мс  ->  1 с мозга = {eager * 10:.1f} с")

# --- CUDA Graph: захватываем K шагов ---
K = 150
cnt = torch.zeros(b.N, device=b.dev)


def block():
    cnt.zero_()
    for _ in range(K):
        cnt.add_(b.step().float())


try:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            block()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    g.register_generator_state(b.gen) if hasattr(g, "register_generator_state") else None
    with torch.cuda.graph(g):
        block()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    per = (time.time() - t) / (10 * K) * 1000
    print(f"CUDA Graph:  {per:.3f} мс/шаг ->  1 с мозга = {per * 10:.1f} с  (ускорение x{eager / per:.1f})")
except Exception as e:
    print("CUDA Graph не взлетел:", type(e).__name__, str(e)[:300])
