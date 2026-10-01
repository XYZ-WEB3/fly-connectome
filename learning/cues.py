"""Условные стимулы (CS) для обучения в грибовидном теле. Основа — CueBank из задачи B
(learning/_scratch/cs_api.py, числа — cs_final.log); здесь добавлены запахи с одной стороны и CS прямо на KC.

Что биология и что наш выбор:
  * запах = пуассоновский вход в ORN своих гломерулов (обе антенны; «@L»/«@R» — только одна антенна).
    odor_A: DM1, DM4, VA2, DP1l, DP1m, DM2 («уксусоподобный» набор по литературе), 20 Гц;
    odor_B: DC1, VL2a, VA6, DL1, VA1d, DC3, VM6m, VM4, VC1, VA4 — просто другой набор, 40 Гц.
  * ВЫБОР МОДЕЛИ 1 (odor_fix=True): 134 возбуждающих локальных нейрона усика (eLN) заглушены. Без этого
    любой запах за 20-40 мс раскачивает весь усик -> ~50% KC и одинаковые коды (Жаккар 0.93-0.95).
    Обоснование (не доказательство): в мухе eLN возбуждают PN в основном щелевыми контактами, их в модели нет.
  * ВЫБОР МОДЕЛИ 2 (pn_nodep=True): синапсы PN без утомления — трюк: вход 1e-9 Гц делает PN «ведомыми»
    (побочно: у PN нет рефрактерности и изредка ложный спайк, ~0.4 спайка/с на все 685 PN).
    Без этого KC отвечают только первые ~100 мс запаха.
    С обеими правками (cs_final.log): A ~483 KC (9.3%), B ~436 (8.4%), Жаккар A-B 0.071, эхо 0-1.
  * CS прямо на KC (add_kc_cue) — как оптогенетика клеток Кеньона; 5% KC x 20 Гц: другие KC не
    подтягиваются, эхо 0 (задача C, c6).
После brain.reset() вызвать set(...) снова (reset обнуляет rate_in, а с ним и фон PN).
"""
import numpy as np
import pandas as pd
import torch

ODOR_GLOMS = {
    "odor_A": ["DM1", "DM4", "VA2", "DP1l", "DP1m", "DM2"],
    "odor_B": ["DC1", "VL2a", "VA6", "DL1", "VA1d", "DC3", "VM6m", "VM4", "VC1", "VA4"],
}
RATE = {"odor_A": 20.0, "odor_B": 40.0, "cue_L": 80.0, "cue_R": 80.0}
ELN_NT = ["acetylcholine", "serotonin", "dopamine", "octopamine"]


def _idx(brain, ids):
    return torch.tensor([brain.idx[r] for r in ids if r in brain.idx], dtype=torch.long, device=brain.dev)


class Cues:
    """Банк CS. set({имя: сила 0..1}) — включить (остальные CS в 0), другие входы мозга не трогает."""

    def __init__(self, brain, ann, odor_fix=True, pn_nodep=True, visual=False):
        self.b = brain
        ct = ann.cell_type.fillna("").astype(str)
        self.g, self.rate = {}, dict(RATE)
        for k, gl in ODOR_GLOMS.items():
            m = ct.isin(["ORN_" + x for x in gl])
            self.g[k] = _idx(brain, ann.index[m])
            for s, side in (("L", "left"), ("R", "right")):
                self.g[f"{k}@{s}"] = _idx(brain, ann.index[m & (ann.side == side)])
                self.rate[f"{k}@{s}"] = self.rate[k]
        if visual:            # зрительные метки (VPN -> KCg-d/KCab-p), см. cs_api.py; парк читается долго
            from brain import PATH_CON
            con = pd.read_parquet(PATH_CON, columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
            vp = ann[ann.super_class == "visual_projection"]
            viskc = set(ann.index[ct.isin(["KCg-d", "KCab-p"])])
            cc = con[con.Presynaptic_ID.isin(set(vp.index)) & con.Postsynaptic_ID.isin(viskc)]
            syn = cc.groupby("Presynaptic_ID").Connectivity.sum()
            vpn = vp.loc[syn.index[syn >= 20]]
            self.g["cue_L"] = _idx(brain, vpn.index[vpn.side == "left"])
            self.g["cue_R"] = _idx(brain, vpn.index[vpn.side == "right"])
        self.eln = None
        if odor_fix:
            ln = ann[ann.cell_class == "ALLN"]
            self.eln = _idx(brain, ln.index[ln.top_nt.isin(ELN_NT)])
            brain.silenced[self.eln] = True
        self.pn = _idx(brain, ann.index[ann.cell_class == "ALPN"]) if pn_nodep else None

    def add_kc_cue(self, name, kc_idx, hz=20.0):
        """CS прямо на набор KC (как оптогенетика KC)."""
        self.g[name] = torch.as_tensor(np.asarray(kc_idx), dtype=torch.long, device=self.b.dev)
        self.rate[name] = float(hz)

    def set(self, active):
        r = self.b.rate_in
        for idx in self.g.values():
            r[idx] = 0.0
        for k, s in active.items():
            idx = self.g[k]
            r[idx] = torch.clamp(r[idx], min=float(self.rate[k] * np.clip(s, 0, 1)))
        if self.pn is not None:
            r[self.pn] = torch.clamp(r[self.pn], min=1e-9)


def kc_sets(kc_idx, n_sets=2, frac=0.05, seed=0):
    """n_sets непересекающихся случайных наборов по frac всех KC (обе стороны)."""
    kc = np.asarray(kc_idx)
    perm = np.random.default_rng(seed).permutation(len(kc))
    n = int(frac * len(kc))
    return [np.sort(kc[perm[k * n:(k + 1) * n]]) for k in range(n_sets)]
