"""Найти синапсы KC->MBON (и любые другие) в быстром ядре brain.py — для модуля пластичности.

    import brain as B, mb_edges as E
    b = B.FlyBrain(params=PRM)
    ed = E.kc_mbon_edges(b)            # позиции всех 62261 связей KC->MBON в b._wv + метаданные
    b._wv[ed["pos"]] *= 0.5            # ослабить (на месте! граф CUDA пересобирать не надо)

Как хранит brain.py (_fast_init): исходящие синапсы нейрона j лежат в b._post / b._wv на позициях
b._cptr[j] .. b._cptr[j+1]-1, внутри отрезка отсортированы по постсинаптическому индексу.
b._wv[e] = знак x число синапсов x w_syn (0.275 мВ). Менять только на месте: адреса b._wv зашиты
в аргументы ядра и в CUDA Graph.
"""
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NPZ = HERE / "mb_kc_mbon_edges.npz"


@torch.no_grad()
def find_edges(b, pre_idx, post_idx):
    """Позиции рёбер pre -> post в b._post/b._wv по ЖИВЫМ массивам ядра (годится и при drop_kc_kc=True).
    -> (pos: LongTensor на устройстве мозга, ok: BoolTensor — ребро существует)."""
    n = b.N
    cptr = b._cptr.long()
    pre_of_e = torch.repeat_interleave(torch.arange(n, device=b.dev), cptr[1:] - cptr[:-1])
    key = pre_of_e * n + b._post.long()                  # строго растёт (сортировка по (pre, post))
    q = torch.as_tensor(np.asarray(pre_idx), device=b.dev).long() * n + \
        torch.as_tensor(np.asarray(post_idx), device=b.dev).long()
    pos = torch.searchsorted(key, q).clamp_(max=key.numel() - 1)
    ok = key[pos] == q
    del key, pre_of_e
    return pos, ok


def kc_mbon_edges(b):
    """Все связи KC->MBON из mb_kc_mbon_edges.npz с позициями, найденными заново в этом мозге."""
    z = np.load(NPZ)
    pos, ok = find_edges(b, z["pre_idx"], z["post_idx"])
    assert bool(ok.all()), f"не найдено {int((~ok).sum())} рёбер KC->MBON"
    return dict(pos=pos, pre=z["pre_idx"], post=z["post_idx"], n_syn=z["n_syn"], weight0=z["weight"],
                comp_id=z["comp_id"], comp_names=[str(s) for s in z["comp_names"]],
                mbon_type_id=z["mbon_type_id"], mbon_types=[str(s) for s in z["mbon_types"]],
                kc_subtype=z["kc_subtype"])
