"""Обучение «как у собаки Павлова» в грибовидном теле (MB): пластичность синапсов KC -> MBON.

Меняет НАСТОЯЩИЕ веса быстрого ядра brain.py (b._wv, на месте — CUDA Graph пересобирать не надо).
Позиции 62261 связей KC->MBON и их отсеки — learning/mb_edges.py + mb_kc_mbon_edges.npz (задача A).

БИОЛОГИЯ (литература, что мы берём):
  * Hige et al. 2015: совпадение активности KC и дофамина (DAN) в отсеке -> долговременная ДЕПРЕССИЯ
    синапсов KC->MBON именно этого отсека. Отсек = DAN «пишет» и MBON «читает» в одном месте
    (Aso 2014a; Li 2020; в нашем коннектоме проверено синапсами DAN->MBON, MB_MAP.md).
  * Aso & Rubin 2016: DAN, включённый во время запаха, пишет память; повторное включение того же DAN
    БЕЗ запаха через несколько минут почти стирает её (их рис. 3D; по визуализации MBON-γ4 у Cohn 2015 —
    ответ MBON восстанавливается). Отсюда необязательный член «DA без KC -> вес назад к исходному».
  * Награда (PAM) ослабляет KC -> «избегающие» MBON (глутаматные, Aso 2014b), наказание (PPL1) —
    KC -> «притягивающие». Знак памяти задаёт отсек, а не само правило.
МОДЕЛЬ (наш выбор, в живой мухе так не доказано):
  * след KC:  e_k(t) = сумма спайков KC k с забыванием exp(-dt/tau_e), tau_e = 1.5 с;
              доля «готовности» ê = min(e_k / e_sat, 1), e_sat = 5 спайков.
  * дофамин отсека c на стороне s: d = взвешенная средняя частота DAN-учителей отсека (low-pass tau_da),
    вес DAN-нейрона = его синапсы на MBON этого отсека этой стороны (коннектом);
    учителя = DAN-типы отсека из mb_map.json; d̂ = max(d - da_thr, 0) / da_ref (порог: запах/CS сам
    чуть-чуть зажигает DAN — без порога было бы ложное «самообучение»).
  * Δw = -lr_dep · ê · d̂ · (w - w_min·w0) · dt         (депрессия; «мягкая граница» -> кривая насыщается)
         + lr_pot · d̂ · (1 - ê) · (w0 - w) · dt          (необязательно: DA без KC возвращает вес, «стирание»)
         + (w0 - w) · dt / tau_rec                      (необязательно: медленное забывание)
    w ∈ [w_min · w0, w0]. Связи KC->MBON без учителя (строки «вне(...)» карты, 2187 из 62261) не учатся.
  * Дофамин в модели Shiu — обычный быстрый ВОЗБУЖДАЮЩИЙ синапс (артефакт: в мухе — GPCR). Поэтому по
    умолчанию выход всех 331 DAN глушится (brain.silenced): спайки DAN считаются и идут в правило, но
    напрямую MBON/KC не раскачивают (проверено в _scratch/c8: PAM11 50 Гц -> MBON07 0 Гц).
ИЗМЕРЕНО (learning/conditioning.py, _scratch/conditioning.log, cond_analysis.log, d2_rule_checks.log; PRM, 5 seed):
  * запах A/B + PAM01/02/10/11 50 Гц, 6 пар: ответ MBON отсеков награды на CS+ 23.9 -> 6.1 Гц, на CS- 24.0 -> 22.8;
    MBON06 на A+ 24.8 -> 9.4 Гц (-62%), MBON07 17.2 -> 2.7 (-84%), MBON01 на B+ 5.7 -> 0 (-100%);
    unpaired (US через 8 с) — в 10 раз слабее: к началу US ê у KC(CS+) ещё 0.7% (запах) – 2.2% (5% KC),
    а не exp(-8/1.5) = 0.5% (ê насыщен: e >> e_sat); при разрыве 20 с ê = 0 и память 0 (ревизия E, e3_controls.log);
    noUS — 0.
  * побочное (без US): запах A сам зажигает PPL106 (α3) выше порога на пиках -> MBON14 на A -9..-21% во всех
    условиях, в noUS -14..-18% (неассоциативно, как «привыкание»; валентность MBON14 не проверена, в V не входит).
  * save/load побитно; перенос памяти в мозг с другим seed работает; lr_pot=0.5: 6 US без запаха стирают
    память 0.054 -> 0.005; tau_rec=60 с за 60 с -> x0.368 (как по формуле); update() 0.18 мс на бин;
    10 с мозга бинами по 18 мс: 0.12 с без обучения, 0.27 с с обучением (RTX 4070 Laptop).
Пример:
    b = B.FlyBrain(params=PRM, seed=0); b.enable_graph()
    pl = MBPlasticity(b)                       # веса пока исходные, DAN заглушены
    counts = pl.run(1000)                      # 1 с мозга бинами по 18 мс + обучение
    # или в своём цикле: c = b.run(ms, tensor=True); pl.update(c, ms)
    pl.memory_strength(kc=cs_plus_kc)          # насколько «запомнен» этот набор KC
    pl.save_memory("fly_memory.npz"); pl.load_memory("fly_memory.npz"); pl.reset_memory()
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import brain as B  # noqa: E402
import mb_edges as E  # noqa: E402

MAP_JSON = HERE / "mb_map.json"
VERSION = 1

DEFAULTS = dict(
    tau_e=1500.0,     # мс: след KC (окно совпадения KC и DA) — модельный выбор в диапазоне 1-2 с
    e_sat=5.0,        # спайков: KC с >= 5 недавними спайками «полностью готов»
    tau_da=100.0,     # мс: сглаживание частоты DAN
    da_thr=20.0,      # Гц: ниже — не учит (CS сам зажигает DAN: в среднем до ~17 Гц; d0_explore.log)
    da_ref=30.0,      # Гц над порогом -> d̂ = 1
    lr_dep=0.2,       # 1/с: скорость депрессии (доля остатка w - w_min·w0 за 1 с при ê = d̂ = 1)
    w_min=0.1,        # нижняя граница веса, доля от w0
    lr_pot=0.0,       # 1/с: «DA без KC -> назад к w0» (0 = выкл)
    tau_rec=None,     # мс: медленное забывание к w0 (None = выкл; память держится, пока её не сбросят)
)


def load_map(mb_map=None):
    if mb_map is None:
        mb_map = MAP_JSON
    if isinstance(mb_map, (str, Path)):
        with open(mb_map, encoding="utf-8") as f:
            return json.load(f)
    return mb_map


def _dan_to_mbon_syn(dan_idx, mbon_idx):
    """Синапсы DAN -> MBON по нейронам из коннектома: DataFrame (pre, post, n)."""
    cols = ["Presynaptic_Index", "Postsynaptic_Index", "Connectivity"]
    try:
        con = pd.read_parquet(B.PATH_CON, columns=cols,
                              filters=[("Presynaptic_Index", "in", [int(x) for x in dan_idx])])
    except Exception:                                   # старый pyarrow без filters
        con = pd.read_parquet(B.PATH_CON, columns=cols)
        con = con[con.Presynaptic_Index.isin(set(dan_idx))]
    con = con[con.Postsynaptic_Index.isin(set(mbon_idx))]
    return con.rename(columns={"Presynaptic_Index": "pre", "Postsynaptic_Index": "post", "Connectivity": "n"})


class MBPlasticity:
    """Трёхфакторное правило на синапсах KC->MBON по отсекам. Все массивы — на устройстве мозга."""

    @torch.no_grad()
    def __init__(self, brain, mb_map=None, silence_dan=True, **params):
        assert brain.fast, "пластичность работает с быстрым ядром (b._wv); fast=False считает через b.W"
        unknown = set(params) - set(DEFAULTS)
        assert not unknown, f"неизвестные параметры: {unknown}"
        self.b = b = brain
        self.p = dict(DEFAULTS, **params)
        m = load_map(mb_map)
        dev = b.dev

        # ---- связи KC->MBON: позиции в b._wv, исходные веса, отсек ----
        ed = E.kc_mbon_edges(b)
        self.pos = ed["pos"]
        self.w0 = b._wv[self.pos].clone()
        assert bool((self.w0 > 0).all()), "веса KC->MBON должны быть > 0 (KC возбуждающие)"
        self.pre = np.asarray(ed["pre"], dtype=np.int64)
        self.post = np.asarray(ed["post"], dtype=np.int64)
        self.comp_names = ed["comp_names"]
        self.mbon_types = ed["mbon_types"]
        self.edge_comp = np.asarray(ed["comp_id"])
        self.edge_mbon = np.asarray(ed["mbon_type_id"])
        self.kc_idx_np = np.unique(self.pre)
        self.kc_idx = torch.as_tensor(self.kc_idx_np, device=dev)
        self.pre_k = torch.as_tensor(np.searchsorted(self.kc_idx_np, self.pre), device=dev)

        # сторона MBON (по mb_map): 0 = left, 1 = right
        side_of = {}
        for t, v in m["mbon"].items():
            for i in v.get("idx_left", []):
                side_of[i] = 0
            for i in v.get("idx_right", []):
                side_of[i] = 1
        post_side = np.array([side_of[i] for i in self.post], dtype=np.int64)

        # ---- DAN (все 331: PAM, PPL1, PPL2) ----
        self.dan_types = list(m["dan"].keys())
        dan_idx, dan_type = [], []
        for t, v in m["dan"].items():
            for i in v["idx_left"] + v["idx_right"]:
                dan_idx.append(i); dan_type.append(t)
        self.dan_idx_np = np.asarray(dan_idx, dtype=np.int64)
        self.dan_type = np.asarray(dan_type)
        self.dan_idx = torch.as_tensor(self.dan_idx_np, device=dev)

        # ---- учителя: столбец = (отсек, сторона); вес DAN-нейрона = его синапсы на MBON отсека этой стороны ----
        comps = m["compartments"]
        mbon_idx_of = {t: v["idx_left"] + v["idx_right"] for t, v in m["mbon"].items()}
        all_mbon = sorted({i for v in mbon_idx_of.values() for i in v})
        syn = _dan_to_mbon_syn(self.dan_idx_np, all_mbon)
        dpos = {int(i): k for k, i in enumerate(self.dan_idx_np)}
        nC = len(self.comp_names)
        T = np.zeros((2 * nC + 1, len(self.dan_idx_np)), dtype=np.float32)   # последний столбец = «нет учителя»
        self.teachers = {}
        for c, name in enumerate(self.comp_names):
            info = comps.get(name)
            if not info or not info.get("dans"):
                continue
            tset = set(info["dans"])
            mb = {i for t in info["mbons"] for i in mbon_idx_of.get(t, [])}
            s = syn[syn.post.isin(mb)]
            s = s[np.array([self.dan_type[dpos[int(p)]] in tset for p in s.pre], dtype=bool)]   # пустой s тоже ок
            for (pre_i, post_i), n in s.groupby(["pre", "post"]).n.sum().items():
                T[2 * c + side_of[int(post_i)], dpos[int(pre_i)]] += n
            self.teachers[name] = sorted(tset)
        tot = T.sum(1, keepdims=True)
        self.teach_syn = tot[:, 0].copy()
        T = np.divide(T, tot, out=np.zeros_like(T), where=tot > 0)
        self.T = torch.as_tensor(T, device=dev)
        col = 2 * self.edge_comp + post_side
        col[self.teach_syn[col] == 0] = 2 * nC               # без учителя -> нулевой столбец
        self.col_np = col
        self.col = torch.as_tensor(col, device=dev)
        self.plastic_np = col != 2 * nC
        self.plastic = torch.as_tensor(self.plastic_np, device=dev)
        self.wmin_abs = self.w0 * float(self.p["w_min"])

        # ---- состояние ----
        self.w = self.w0.clone()
        self.e = torch.zeros(len(self.kc_idx_np), device=dev)       # след KC (спайки)
        self.da = torch.zeros(len(self.dan_idx_np), device=dev)     # сглаженная частота DAN (Гц)
        self.dhat = torch.zeros(2 * nC + 1, device=dev)             # d̂ по столбцам
        self.da_dose = torch.zeros(2 * nC + 1, device=dev)          # интеграл d̂ (с) — для отчёта
        self.enabled = True
        self.t_ms = 0.0
        self._silenced_by_me = None
        if silence_dan:
            self.set_dan_output(False)

    # ---------------- управление ----------------
    @torch.no_grad()
    def set_dan_output(self, on):
        """False: выход DAN заглушен (артефакт быстрого дофамина убран), спайки DAN всё равно считаются."""
        s = self.b.silenced
        if not on:
            if self._silenced_by_me is None:
                self._silenced_by_me = ~s[self.dan_idx]          # что глушим мы, а не кто-то раньше
            s[self.dan_idx] = True
        elif self._silenced_by_me is not None:
            s[self.dan_idx[self._silenced_by_me]] = False
            self._silenced_by_me = None

    @torch.no_grad()
    def reset_traces(self):
        """Обнулить следы (после brain.reset()). Память (веса) не трогает."""
        self.e.zero_(); self.da.zero_(); self.dhat.zero_()

    @torch.no_grad()
    def reset_memory(self):
        """Забыть всё: веса KC->MBON = исходные."""
        self.w.copy_(self.w0)
        self.b._wv[self.pos] = self.w
        self.da_dose.zero_()
        self.reset_traces()

    @torch.no_grad()
    def detach(self):
        """Вернуть мозг как был: исходные веса, выход DAN включён. Обучение выключается
        (иначе следующий update() снова записал бы выученные веса в b._wv)."""
        self.enabled = False
        self.b._wv[self.pos] = self.w0
        self.set_dan_output(True)

    # ---------------- правило ----------------
    @torch.no_grad()
    def update(self, counts, dt_ms):
        """counts — спайки всех нейронов за последние dt_ms (b.run(dt_ms, tensor=True)). Без синхронизации GPU."""
        p = self.p
        dt = dt_ms / 1000.0
        self.e.mul_(float(np.exp(-dt_ms / p["tau_e"]))).add_(counts[self.kc_idx])
        a = float(np.exp(-dt_ms / p["tau_da"]))
        self.da.mul_(a).add_(counts[self.dan_idx], alpha=(1.0 - a) * 1000.0 / dt_ms)
        d = torch.mv(self.T, self.da)
        self.dhat = ((d - p["da_thr"]).clamp_(min=0.0) / p["da_ref"])
        self.dhat[-1] = 0.0
        self.t_ms += dt_ms
        if not self.enabled:
            return
        self.da_dose.add_(self.dhat, alpha=dt)
        eh = (self.e[self.pre_k] / p["e_sat"]).clamp_(max=1.0)
        de = self.dhat[self.col]
        w = self.w
        w.addcmul_(eh * de, w - self.wmin_abs, value=-p["lr_dep"] * dt)
        if p["lr_pot"]:
            w.add_(de * (1.0 - eh) * (self.w0 - w).clamp(min=0.0), alpha=p["lr_pot"] * dt)
        if p["tau_rec"]:
            w.add_(self.w0 - w, alpha=dt_ms / p["tau_rec"])
        torch.maximum(w, self.wmin_abs, out=w)
        torch.minimum(w, self.w0, out=w)
        self.b._wv[self.pos] = w                   # на месте: адрес b._wv зашит в ядро и CUDA Graph

    @torch.no_grad()
    def run(self, ms, bin_ms=18.0, on_bin=None):
        """Прогнать мозг ms мс бинами (кратно 1.8 мс) с обучением. on_bin(t_ms) — поставить входы
        в начале бина. -> суммарные спайки (тензор N)."""
        b = self.b
        nb = int(round(ms / bin_ms))
        tot = torch.zeros(b.N, device=b.dev)
        for i in range(nb):
            if on_bin is not None:
                on_bin(i * bin_ms)
            c = b.run(bin_ms, tensor=True)
            self.update(c, bin_ms)
            tot += c
        return tot

    # ---------------- чтение памяти ----------------
    @torch.no_grad()
    def ratio(self):
        """w / w0 по всем 62261 связям (numpy)."""
        return (self.w / self.w0).cpu().numpy()

    def memory_strength(self, kc=None, comps=None):
        """Сила памяти = 1 - Σw/Σw0 по обучаемым связям (0 = как новорождённая, 1 - w_min = предел).
        kc — индексы KC (например, набор CS+): только их связи; comps — только эти отсеки.
        -> dict total, by_comp, by_mbon."""
        w = self.w.cpu().numpy().astype(np.float64)
        w0 = self.w0.cpu().numpy().astype(np.float64)
        m = self.plastic_np.copy()
        if kc is not None:
            m &= np.isin(self.pre, np.asarray(kc))
        if comps is not None:
            m &= np.isin(self.edge_comp, [self.comp_names.index(c) for c in comps])
        out = dict(total=float(1 - w[m].sum() / w0[m].sum()) if m.any() else 0.0, by_comp={}, by_mbon={})
        for c, name in enumerate(self.comp_names):
            mm = m & (self.edge_comp == c)
            if mm.any():
                out["by_comp"][name] = float(1 - w[mm].sum() / w0[mm].sum())
        for k, name in enumerate(self.mbon_types):
            mm = m & (self.edge_mbon == k)
            if mm.any():
                out["by_mbon"][name] = float(1 - w[mm].sum() / w0[mm].sum())
        return out

    def dopamine(self):
        """Текущий d (Гц, до порога) по отсекам: {отсек: (слева, справа)}."""
        d = torch.mv(self.T, self.da).cpu().numpy()
        return {n: (float(d[2 * c]), float(d[2 * c + 1])) for c, n in enumerate(self.comp_names)
                if self.teach_syn[2 * c] + self.teach_syn[2 * c + 1] > 0}

    # ---------------- сохранение (память мухи между сессиями) ----------------
    def save_memory(self, path):
        """Сохранить выученные веса как w/w0 по связям (pre_root, post_root) — переносимо между мозгами."""
        ids = self.b.ids
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:                # open() понимает кириллицу в пути
            np.savez_compressed(f, pre_root=ids[self.pre].astype(np.int64), post_root=ids[self.post].astype(np.int64),
                                ratio=self.ratio().astype(np.float32),
                                meta=np.array(json.dumps(dict(version=VERSION, saved=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                              params=self.p, t_ms=self.t_ms), ensure_ascii=False)))
        return path

    @torch.no_grad()
    def load_memory(self, path):
        """Загрузить память: связи сопоставляются по root_id; чего нет — остаётся исходным. -> число связей."""
        with open(path, "rb") as f:
            z = np.load(f)
            pre_r, post_r, r = z["pre_root"], z["post_root"], z["ratio"]
        idx = self.b.idx
        ok = np.array([(a in idx) and (c in idx) for a, c in zip(pre_r, post_r)])
        n = self.b.N
        key_saved = np.array([idx[a] * n + idx[c] for a, c in zip(pre_r[ok], post_r[ok])], dtype=np.int64)
        key_mine = self.pre * n + self.post
        order = np.argsort(key_mine)
        j = np.searchsorted(key_mine[order], key_saved).clip(max=len(order) - 1)
        hit = key_mine[order][j] == key_saved
        rr = np.ones(len(key_mine), dtype=np.float32)
        rr[order[j[hit]]] = r[ok][hit]
        rr = np.clip(rr, self.p["w_min"], 1.0)
        self.w.copy_(self.w0 * torch.as_tensor(rr, device=self.w0.device))
        self.b._wv[self.pos] = self.w
        self.reset_traces()
        return int(hit.sum())

    def dan_indices(self, dan_types):
        """Индексы нейронов этих DAN-типов (тензор) — например, для US: b.rate_in[pl.dan_indices(REWARD)] = 50."""
        return self.dan_idx[torch.as_tensor(np.isin(self.dan_type, list(dan_types)), device=self.dan_idx.device)]

    def taught_by(self, dan_types, min_frac=0.5):
        """Отсеки, где эти DAN дают >= min_frac синапсов учителей (в среднем по сторонам) -> список имён."""
        sel = np.isin(self.dan_type, list(dan_types))
        T = self.T.cpu().numpy()
        out = []
        for c, name in enumerate(self.comp_names):
            fr = [T[2 * c + s, sel].sum() for s in (0, 1) if self.teach_syn[2 * c + s] > 0]
            if fr and np.mean(fr) >= min_frac:
                out.append(name)
        return out

    # ---------------- справка ----------------
    def describe(self):
        """Кто кого учит: отсек -> учителя (синапсы DAN на MBON отсека слева/справа), число обучаемых связей."""
        rows = []
        for c, name in enumerate(self.comp_names):
            k = self.edge_comp == c
            rows.append(dict(comp=name, teachers=",".join(self.teachers.get(name, [])) or "—",
                             teach_syn_L=int(self.teach_syn[2 * c]), teach_syn_R=int(self.teach_syn[2 * c + 1]),
                             edges=int(k.sum()), plastic=int((k & self.plastic_np).sum())))
        return pd.DataFrame(rows)


class ValenceReadout:
    """«Нравится / не нравится» по выходу MB (МОДЕЛЬ-мост; в коннектоме модели выученное изменение MBON
    до руля почти не доходит — задача C). V = Σ ср. частота MBON «притяжения» − Σ «избегания», Гц
    (по типам, L+R вместе, каждый тип с весом 1). Валентность — Aso et al. 2014b (оптогенетика MBON в арене);
    берём только статусы tested/group без «?»: избегание MBON01, 03, 04, 05, 06; притяжение MBON09, 11, 12, 15, 16, 18."""

    def __init__(self, brain, mb_map=None, statuses=("tested", "group")):
        m = load_map(mb_map)
        self.b = brain
        self.approach = [t for t, v in m["mbon"].items() if v["valence"] == "approach" and v["valence_status"] in statuses]
        self.avoid = [t for t, v in m["mbon"].items() if v["valence"] == "avoid" and v["valence_status"] in statuses]
        self.types = list(m["mbon"].keys())
        rows, cols, vals = [], [], []
        for j, t in enumerate(self.types):
            ix = m["mbon"][t]["idx_left"] + m["mbon"][t]["idx_right"]
            rows += [j] * len(ix); cols += ix; vals += [1.0 / len(ix)] * len(ix)
        self.G = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(vals, dtype=torch.float32),
                                         (len(self.types), brain.N)).to_sparse_csr().to(brain.dev)
        sign = np.zeros(len(self.types), dtype=np.float32)
        for t in self.approach:
            sign[self.types.index(t)] = 1.0
        for t in self.avoid:
            sign[self.types.index(t)] = -1.0
        self.sign = torch.as_tensor(sign, device=brain.dev)

    @torch.no_grad()
    def rates_t(self, counts, ms):
        """Средняя частота по типам MBON (тензор, Гц)."""
        return torch.mv(self.G, counts) * (1000.0 / ms)

    def rates(self, counts, ms):
        return dict(zip(self.types, self.rates_t(counts, ms).cpu().numpy().tolist()))

    @torch.no_grad()
    def value(self, counts, ms):
        """V (Гц): > 0 — выход MB «за», < 0 — «против»."""
        return float(self.sign @ self.rates_t(counts, ms))
