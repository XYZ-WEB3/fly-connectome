"""Общая обвязка для игры 67COIN: мозг + пластичность MB + группы нейронов по сторонам.

    from fly_setup import Fly
    f = Fly(seed=0)                        # FlyBrain(PRM) + CUDA Graph + MBPlasticity (выход DAN заглушен)
    f.set_cue("L", lc10=5, vpn=150)        # зрительная метка слева: LC10 (объект) + VPN->KC (зрение в MB)
    f.set_us(["PAM01"], "LR", 50)          # награда = включение PAM «как оптогенетика»
    r = f.rates(f.b.run(1000, tensor=True), 1000)   # dict: DNa_L, DNa_R, steer, MBON27_L, ...

Что биология и что наш выбор:
  * LC10 (детекторы мелких объектов) одной стороны -> руль DNa ипсилатерально (родной путь, senses.py).
  * VPN->KC: зрительные проекционные нейроны с >= 20 синапсами на KCg-d/KCab-p (как cues.py, visual=True);
    92% их синапсов на KC — своей стороны (game/_scratch/a1_route_anatomy.log). Частоты — наш выбор.
  * руль = средняя DNa01+DNa02 слева минус справа (Гц), > 0 = влево.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
LEARN = HERE.parent
ROOT = LEARN.parent
for p in (ROOT, LEARN, ROOT / "tiktok"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import brain as B  # noqa: E402
from plasticity import MBPlasticity  # noqa: E402

PRM = dict(U=0.1, tau_d=300, a_inc=0.1, tau_a=3000)   # противоэпилептика — всегда
REWARD = ["PAM01", "PAM02", "PAM10", "PAM11"]
LC10 = ["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"]          # как senses.py obj_L/obj_R
MBON_READ = ["MBON27", "MBON26", "MBON31", "MBON32", "MBON35", "MBON12", "MBON21", "MBON01", "MBON07", "MBON09"]
VPN_CACHE = HERE / "_scratch" / "vpn_kc_idx.json"


def _vpn_kc(ann, idx):
    """VPN с >= 20 синапсами на KCg-d / KCab-p, по сторонам (кэш в _scratch)."""
    if VPN_CACHE.exists():
        with open(VPN_CACHE, encoding="utf-8") as f:
            return json.load(f)
    vp = ann[ann.super_class == "visual_projection"]
    viskc = set(ann.index[ann.cell_type.fillna("").isin(["KCg-d", "KCab-p"])])
    con = pd.read_parquet(B.PATH_CON, columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"],
                          filters=[("Presynaptic_ID", "in", [int(x) for x in vp.index])])
    cc = con[con.Postsynaptic_ID.isin(viskc)]
    syn = cc.groupby("Presynaptic_ID").Connectivity.sum()
    v = vp.loc[syn.index[syn >= 20]]
    out = {s: sorted(int(idx[r]) for r in v.index[v.side == side] if r in idx)
           for s, side in (("L", "left"), ("R", "right"))}
    VPN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(VPN_CACHE, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


class Fly:
    """Одна «муха»: мозг + пластичность + индексы групп. Все группы — по сторонам _L/_R."""

    def __init__(self, seed=0, graph=True, **pl_params):
        self.b = b = B.FlyBrain(params=PRM, seed=seed)
        if graph:
            b.enable_graph()
        self.pl = MBPlasticity(b, **pl_params)
        m = json.load(open(LEARN / "mb_map.json", encoding="utf-8"))
        ann = B.load_annotations()
        ct = ann.cell_type.fillna("").astype(str)
        g = {}

        def sel(mask, side):
            a = ann[mask & (ann.side == side)]
            return [b.idx[r] for r in a.index if r in b.idx]

        for s, side in (("L", "left"), ("R", "right")):
            g[f"LC10_{s}"] = sel(ct.isin(LC10), side)
            g[f"DNa_{s}"] = sel(ct.isin(["DNa01", "DNa02"]), side)
            for t in MBON_READ:
                g[f"{t}_{s}"] = m["mbon"][t]["idx_left" if s == "L" else "idx_right"]
            for t in ("PAM01", "PAM02", "PAM10", "PAM11", "PAM05", "PAM08"):
                g[f"{t}_{s}"] = m["dan"][t]["idx_left" if s == "L" else "idx_right"]
            key = "idx_left" if s == "L" else "idx_right"
            g[f"KC_{s}"] = sorted(i for v in m["kc"]["by_subtype"].values() for i in v[key])
            g[f"KCgd_{s}"] = m["kc"]["by_subtype"]["g-d"][key]
        vk = _vpn_kc(ann, b.idx)
        g["VPN_L"], g["VPN_R"] = vk["L"], vk["R"]
        self.names = list(g)
        self.g = {k: torch.as_tensor(np.asarray(v, dtype=np.int64), device=b.dev) for k, v in g.items()}
        # матрица средних по группам: одна операция на GPU, одна пересылка на CPU
        rows, cols, vals = [], [], []
        for j, k in enumerate(self.names):
            ix = g[k]
            rows += [j] * len(ix); cols += list(ix); vals += [1.0 / len(ix)] * len(ix)
        self.G = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(vals, dtype=torch.float32),
                                         (len(self.names), b.N)).to_sparse_csr().to(b.dev)
        # KC -> MBON27 рёбра (для «потолка» и чтения памяти)
        mb27 = set(m["mbon"]["MBON27"]["idx_left"] + m["mbon"]["MBON27"]["idx_right"])
        self.e27 = torch.as_tensor(np.flatnonzero(np.isin(self.pl.post, list(mb27))), device=b.dev)
        self.cue_on = None

    # ---------- входы ----------
    @torch.no_grad()
    def set_cue(self, side, lc10=0.0, vpn=0.0):
        """Зрительная метка на стороне side ('L'/'R'/None): LC10 и VPN->KC этой стороны."""
        r = self.b.rate_in
        for s in "LR":
            r[self.g[f"LC10_{s}"]] = 0.0
            r[self.g[f"VPN_{s}"]] = 0.0
        if side:
            r[self.g[f"LC10_{side}"]] = float(lc10)
            r[self.g[f"VPN_{side}"]] = float(vpn)
        self.cue_on = side

    @torch.no_grad()
    def set_us(self, types, sides="LR", hz=50.0):
        """US: DAN типов types на сторонах sides ('L', 'R', 'LR') -> hz (0 = выкл)."""
        for t in types:
            for s in sides:
                self.b.rate_in[self.g[f"{t}_{s}"]] = float(hz)

    @torch.no_grad()
    def us_off(self):
        for t in ("PAM01", "PAM02", "PAM10", "PAM11", "PAM05", "PAM08"):
            for s in "LR":
                self.b.rate_in[self.g[f"{t}_{s}"]] = 0.0

    def fresh(self):
        """Новая проба: сбросить состояние мозга и следы (память-веса остаются)."""
        self.b.reset(); self.pl.reset_traces(); self.cue_on = None

    def reseed(self, seed):
        """Тот же мозг как новый FlyBrain(seed): ГСЧ Philox с нуля, граф перезаписан; память сброшена.
        (Как game67.reseed; проверено там же против нового мозга: _scratch/c1_reseed_check.log.)"""
        b = self.b
        b.seed = int(seed); b._args_key = None
        if getattr(b, "_tdev", None) is not None:
            b._tdev.zero_()
        b.reset()
        if b.graph is not None:
            b.enable_graph()
        self.pl.reset_memory()
        self.cue_on = None

    # ---------- выходы ----------
    @torch.no_grad()
    def rates(self, counts, ms):
        """Средние частоты групп (Гц) + steer = DNa_L - DNa_R + число активных KC по сторонам."""
        v = (torch.mv(self.G, counts) * (1000.0 / ms)).cpu().numpy()
        out = dict(zip(self.names, v.tolist()))
        out["steer"] = out["DNa_L"] - out["DNa_R"]
        kc_act = torch.stack([(counts[self.g["KC_L"]] > 0).sum(), (counts[self.g["KC_R"]] > 0).sum()]).cpu().numpy()
        out["nKC_L"], out["nKC_R"] = int(kc_act[0]), int(kc_act[1])
        return out

    # ---------- память MBON27 ----------
    def mem27(self):
        """Сила памяти KC->MBON27 по сторонам MBON (1 - Σw/Σw0)."""
        w = self.pl.w[self.e27].cpu().numpy(); w0 = self.pl.w0[self.e27].cpu().numpy()
        post = self.pl.post[self.e27.cpu().numpy()]
        out = {}
        for s in "LR":
            mk = np.isin(post, self.g[f"MBON27_{s}"].cpu().numpy())
            out[s] = float(1 - w[mk].sum() / w0[mk].sum())
        return out
