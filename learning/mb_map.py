"""Карта грибовидного тела (mushroom body, MB) в коннектоме FlyWire v783 — основа модуля обучения.

Только чтение, CPU. Мозг не создаём (brain.py импортируем ради путей и load_annotations).

    .venv\\Scripts\\python.exe learning\\mb_map.py
    -> learning/mb_map.json            всё по типам: KC, MBON, DAN, компартменты, пути к DN
    -> learning/mb_kc_mbon_edges.npz   каждая связь KC->MBON: индексы, синапсы, вес, позиция в быстром ядре
    -> learning/MB_MAP.md              читаемая сводка (подхватывает learning/mb_func.json из mb_func_test.py)

Что здесь БИОЛОГИЯ (литература), а что ИЗМЕРЕНО на коннектоме:
  * компартменты MBON/DAN — из статей (Aso 2014a Table 1; Li 2020 для MBON23-35 и PAM15),
    но каждый проверен связями: какие DAN шлют синапсы на дендриты MBON (DAN->MBON) и
    какие подтипы KC дают вход (гамма / альфа-бета / альфа'-бета' доля);
  * валентность MBON — Aso 2014b (активация CsChrimson), DAN — Aso & Rubin 2016 (пары запах+DAN);
    где типа нет в проверенном тексте статьи — так и помечено (nt_rule / unknown);
  * все числа синапсов, путей, хопов — посчитаны здесь по Connectivity_783.parquet.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import brain as B  # noqa: E402  (только пути/аннотации)

OUT_JSON = HERE / "mb_map.json"
OUT_NPZ = HERE / "mb_kc_mbon_edges.npz"
OUT_MD = HERE / "MB_MAP.md"
FUNC_JSON = HERE / "mb_func.json"

MN9_ID = 720575940660219265          # как в tiktok/senses.py (Shiu et al. 2024)
SUGAR_IDS_630 = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
                 720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
                 720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
                 720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
                 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
                 720575940611875570]

# ============================ ЛИТЕРАТУРА ============================
SRC = {
    "Aso2014a": "Aso et al. 2014a eLife 3:e04577, Table 1 (The neuronal architecture of the mushroom body...)",
    "Aso2014b": "Aso et al. 2014b eLife 3:e04580 (MBONs encode valence...), CsChrimson-активация, 4-квадрантная арена",
    "AsoRubin2016": "Aso & Rubin 2016 eLife 5:e16135 (DANs write and update memories...), пары запах + CsChrimson в DAN",
    "Li2020": "Li et al. 2020 eLife 9:e62576 (hemibrain: connectome of the adult MB)",
    "Hige2015": "Hige et al. 2015 Neuron 88:985 (депрессия KC->MBON при совпадении KC + DAN в компартменте)",
}

# MBON: тип FlyWire -> (компартмент по литературе, источник). γ4>γ1γ2 = дендриты в γ4, аксон в γ1γ2.
MBON_LIT = {
    "MBON01": ("γ5β'2a", "Aso2014a"), "MBON02": ("β2β'2a", "Aso2014a"), "MBON03": ("β'2mp", "Aso2014a"),
    "MBON04": ("β'2mp_bilateral", "Aso2014a"), "MBON05": ("γ4>γ1γ2", "Aso2014a"), "MBON06": ("β1>α", "Aso2014a"),
    "MBON07": ("α1", "Aso2014a"), "MBON09": ("γ3β'1", "Aso2014a"), "MBON10": ("β'1", "Aso2014a"),
    "MBON11": ("γ1pedc>α/β", "Aso2014a"), "MBON12": ("γ2α'1", "Aso2014a"), "MBON13": ("α'2", "Aso2014a"),
    "MBON14": ("α3", "Aso2014a"), "MBON15": ("α'1", "Aso2014a"), "MBON15-like": ("α'1", "Li2020"),
    "MBON16": ("α'3ap", "Aso2014a"), "MBON17": ("α'3m", "Aso2014a"), "MBON17-like": ("α'3m", "Li2020"),
    "MBON18": ("α2sc", "Aso2014a"), "MBON19": ("α2p3p", "Aso2014a"), "MBON20": ("γ1γ2", "Aso2014a"),
    "MBON21": ("γ4γ5", "Aso2014a"), "MBON22": ("calyx", "Aso2014a"),
    "MBON23": ("α2sp", "hemibrain instance name (в тексте Li2020 не найдено — проверено связями)"),
    "MBON24": ("β2γ5", "Li2020"), "MBON25,MBON34": ("γ1γ2 (MBON25) / γ2 (MBON34)", "Li2020"),
    "MBON26": ("β'2d", "Li2020"), "MBON27": ("γ5d", "Li2020"), "MBON28": ("α'3", "Li2020"),
    "MBON29": ("γ4γ5", "Li2020"), "MBON30": ("γ1γ2γ3", "Li2020"), "MBON31": ("α'1a", "Li2020"),
    "MBON32": ("γ2", "Li2020"), "MBON33": ("γ2γ3", "Li2020"), "MBON35": ("γ2", "Li2020"),
}
TYPICAL = {"MBON01", "MBON02", "MBON03", "MBON04", "MBON05", "MBON06", "MBON07", "MBON09", "MBON11", "MBON12",
           "MBON13", "MBON14", "MBON15", "MBON15-like", "MBON16", "MBON17", "MBON17-like", "MBON18", "MBON19",
           "MBON21", "MBON22", "MBON23"}          # Li 2020: остальные — «атипичные» (дендриты и вне долей MB)

# Куда отнести вход от KC каждой доли (g = γ, ab = α/β, apbp = α'/β'). Чего нет в словаре — «вне компартмента».
LOBE_SPLIT = {
    "MBON01": {"g": "γ5", "apbp": "β'2a"}, "MBON02": {"ab": "β2", "apbp": "β'2a"},
    "MBON03": {"apbp": "β'2mp"}, "MBON04": {"apbp": "β'2mp"}, "MBON05": {"g": "γ4"},
    "MBON06": {"ab": "β1"}, "MBON07": {"ab": "α1"}, "MBON09": {"g": "γ3", "apbp": "β'1"},
    "MBON10": {"apbp": "β'1"}, "MBON11": {"g": "γ1pedc", "ab": "γ1pedc"}, "MBON12": {"g": "γ2", "apbp": "α'1"},
    "MBON13": {"apbp": "α'2"}, "MBON14": {"ab": "α3"}, "MBON15": {"apbp": "α'1"}, "MBON15-like": {"apbp": "α'1"},
    "MBON16": {"apbp": "α'3ap"}, "MBON17": {"apbp": "α'3m"}, "MBON17-like": {"apbp": "α'3m"},
    "MBON18": {"ab": "α2sc"}, "MBON19": {"ab": "α2p3p"}, "MBON20": {"g": "γ1γ2"}, "MBON21": {"g": "γ4γ5"},
    "MBON22": {"g": "calyx", "ab": "calyx", "apbp": "calyx"}, "MBON23": {"ab": "α2sp"},
    "MBON24": {"ab": "β2", "g": "γ5"}, "MBON25,MBON34": {"g": "γ1γ2"}, "MBON26": {"apbp": "β'2d"},
    "MBON27": {"g": "γ5d"}, "MBON28": {"apbp": "α'3"}, "MBON29": {"g": "γ4γ5"}, "MBON30": {"g": "γ1γ2γ3"},
    "MBON31": {"apbp": "α'1a"}, "MBON32": {"g": "γ2"}, "MBON33": {"g": "γ2γ3"}, "MBON35": {"g": "γ2"},
}

# Валентность MBON. status: tested = линия с одним типом значима; group = только в составе группы;
# nt_rule = НЕ проверено, догадка по правилу Aso 2014b («все отталкивающие — глутамат, все
# притягивающие — ГАМК или ACh»); unknown = данных нет.
MBON_VAL = {
    "MBON01": ("avoid", "group", "MB011B (MBON01+03+04) -> избегание; подмножества MB210B, MB002B не значимы"),
    "MBON03": ("avoid", "group", "MB011B (MBON01+03+04) -> избегание"),
    "MBON04": ("avoid", "group", "MB011B (MBON01+03+04) -> избегание"),
    "MBON05": ("avoid", "group", "MB434B (MBON05+06) -> избегание"),
    "MBON06": ("avoid", "group", "MB434B (MBON05+06) -> избегание"),
    "MBON02": ("avoid?", "nt_rule", "глутамат; отдельный результат MB399B в тексте не найден"),
    "MBON07": ("avoid?", "nt_rule", "глутамат; MB310C нужен для аппетитной памяти, валентность активации в тексте нет"),
    "MBON09": ("approach", "group", "MB083C (MBON08+09), ГАМК -> притяжение; в скрининге Fig.2C не значим к GAL4/+"),
    "MBON10": ("approach?", "nt_rule", "ГАМК по Aso 2014a (FlyWire: ГАМК/глутамат); не проверено"),
    "MBON11": ("approach", "tested", "MB112C (только MBON11) -> притяжение; MB112C+MB077B — самое сильное"),
    "MBON12": ("approach", "tested", "MB077B (только MBON12) -> притяжение"),
    "MBON13": ("approach?", "nt_rule", "ACh; нужен для 2-ч аппетитной памяти (MB018B), активация в тексте нет"),
    "MBON14": ("approach?", "nt_rule", "ACh; нужен для 2-ч аппетитной памяти (MB082C/MB093C), активация в тексте нет"),
    "MBON15": ("approach", "group", "MB052B (V2-кластер: MBON18,16,15,(19),(17)) -> притяжение"),
    "MBON15-like": ("approach?", "group", "как MBON15 (Li 2020 объединяет с MBON15)"),
    "MBON16": ("approach", "group", "MB052B (V2-кластер) -> притяжение"),
    "MBON17": ("approach?", "group", "в MB052B слабо (в скобках) -> не уверенно"),
    "MBON17-like": ("approach?", "group", "как MBON17"),
    "MBON18": ("approach", "group", "MB052B (V2-кластер) -> притяжение"),
    "MBON19": ("approach?", "group", "в MB052B слабо (в скобках) -> не уверенно"),
}
for _t in MBON_LIT:
    MBON_VAL.setdefault(_t, ("unknown", "unknown", "в Aso 2014b не тестировался (атипичный/новый тип)"))

# DAN: тип -> (компартмент, кластер, валентность, status, доказательство)
DAN_LIT = {
    "PAM01": ("γ5", "PAM", "reward", "tested", "MB315C (PAM-γ5) -> аппетитная память; с MB109B сильнее (AsoRubin2016)"),
    "PAM02": ("β'2a", "PAM", "reward", "tested", "MB109B (PAM-β'2a) -> аппетитная (AsoRubin2016)"),
    "PAM03": ("β2β'2a", "PAM", "reward?", "presumed", "кластер PAM; отдельно не проверено"),
    "PAM04": ("β2", "PAM", "reward?", "presumed", "кластер PAM; отдельно не проверено"),
    "PAM05": ("β'2p", "PAM", "reward?", "presumed", "кластер PAM; отдельно не проверено"),
    "PAM06": ("β'2m", "PAM", "reward?", "presumed", "кластер PAM; отдельно не проверено"),
    "PAM07": ("γ4<γ1γ2", "PAM", "?", "uncertain", "MB312C (PAM-γ4 + PAM-γ4<γ1γ2): слабая, но значимая память; знак в тексте не указан"),
    "PAM08": ("γ4", "PAM", "?", "uncertain", "как PAM07 (MB312C)"),
    "PAM09": ("β1ped", "PAM", "reward?", "presumed", "MB213B (аппетитная) включает PAM-β1 и ещё тип(ы); состав в тексте не проверен"),
    "PAM10": ("β1", "PAM", "reward", "tested", "MB063B (PAM-β1) слабо значим; MB213B сильно (AsoRubin2016)"),
    "PAM11": ("α1", "PAM", "reward", "tested", "MB043C (PAM-α1) -> аппетитная, долговременная (AsoRubin2016)"),
    "PAM12": ("γ3", "PAM", "?", "uncertain", "Li2020: два подтипа PAM12, вероятно, противоположной валентности"),
    "PAM13": ("β'1ap", "PAM", "?", "uncertain", "не проверено"),
    "PAM14": ("β'1m", "PAM", "?", "uncertain", "не проверено"),
    "PAM15": ("γ5β'2a", "PAM", "reward?", "presumed", "новый тип hemibrain (Li2020); не проверено"),
    "PPL101": ("γ1pedc", "PPL1", "punish", "tested", "MB320C (PPL1-γ1pedc) -> сильная аверсивная память (AsoRubin2016)"),
    "PPL102": ("γ1", "PPL1", "?", "uncertain", "не проверено"),
    "PPL103": ("γ2α'1", "PPL1", "punish", "tested", "MB296B (только PPL1-γ2α'1) слабо; MB099C (+PPL105) сильно (AsoRubin2016)"),
    "PPL104": ("α'3", "PPL1", "punish?", "group", "только в комбинациях MB060B/MB065B"),
    "PPL105": ("α'2α2", "PPL1", "punish", "group", "MB099C (PPL103+PPL105) -> аверсивная (AsoRubin2016)"),
    "PPL106": ("α3", "PPL1", "punish", "tested", "MB630B (PPL1-α3) -> аверсивная, долговременная (AsoRubin2016)"),
    "PPL107": ("— (нет выхода на KC в v783)", "PPL1", "?", "not_MB", "в v783 на KC 3 синапса -> не учитель долей MB"),
    "PPL108": ("— (нет выхода на KC в v783)", "PPL1", "?", "not_MB", "в v783 на KC 1 синапс -> не учитель долей MB"),
    "PPL201": ("calyx/LH (PPL2ab)", "PPL2", "?", "uncertain", "чашечка + латеральный рог"),
    "PPL202": ("calyx/LH (PPL2ab)", "PPL2", "?", "uncertain", "чашечка + латеральный рог"),
    "PPL203": ("calyx/LH (PPL2ab)", "PPL2", "?", "uncertain", ""),
    "PPL204": ("calyx/LH (PPL2ab)", "PPL2", "?", "uncertain", ""),
}

# Метка компартмента -> множество из 15 канонических (γ1-5, α1-3, β1-2, α'1-3, β'1-2) + pedc + calyx
_C = {
    "γ1pedc": "γ1 pedc", "γ1pedc>α/β": "γ1 pedc", "γ1": "γ1", "γ2": "γ2", "γ3": "γ3", "γ4": "γ4", "γ5": "γ5",
    "γ5d": "γ5", "γ1γ2": "γ1 γ2", "γ4γ5": "γ4 γ5", "γ1γ2γ3": "γ1 γ2 γ3", "γ2γ3": "γ2 γ3", "γ4>γ1γ2": "γ4",
    "γ4<γ1γ2": "γ4", "γ3β'1": "γ3 β'1", "γ5β'2a": "γ5 β'2", "γ2α'1": "γ2 α'1", "β2γ5": "β2 γ5",
    "α1": "α1", "α2sc": "α2", "α2sp": "α2", "α2p3p": "α2 α3", "α3": "α3", "β1": "β1", "β1>α": "β1",
    "β1ped": "β1 pedc", "β2": "β2", "β2β'2a": "β2 β'2", "β'1": "β'1", "β'1ap": "β'1", "β'1m": "β'1",
    "β'2a": "β'2", "β'2m": "β'2", "β'2p": "β'2", "β'2mp": "β'2", "β'2mp_bilateral": "β'2", "β'2d": "β'2",
    "α'1": "α'1", "α'1a": "α'1", "α'2": "α'2", "α'2α2": "α'2 α2", "α'3": "α'3", "α'3ap": "α'3", "α'3m": "α'3",
    "calyx": "calyx", "calyx/LH (PPL2ab)": "calyx", "γ1γ2 (MBON25) / γ2 (MBON34)": "γ1 γ2",
}
CANON_ORDER = ["γ1", "γ2", "γ3", "γ4", "γ5", "pedc", "β1", "β2", "α1", "α2", "α3",
               "β'1", "β'2", "α'1", "α'2", "α'3", "calyx"]


def canon(label):
    return set(_C.get(label, "").split())


def kc_sub(h):
    """hemibrain_type KC -> короткий подтип и доля (g / ab / apbp)."""
    h = str(h)
    if h.startswith("KCg-"):
        s = h[2:].split("-")[0] + "-" + h.split("-")[1][:1]          # g-m, g-d, g-s
        return s, "g"
    if h.startswith("KCab"):
        return "ab-" + h.split("-")[1], "ab"                          # ab-c, ab-s, ab-m, ab-p
    if h.startswith("KCa'b'"):
        return "a'b'-" + h.split("-")[1], "apbp"                      # a'b'-ap1, a'b'-ap2, a'b'-m
    return "?", "?"


LOBE_RU = {"g": "γ", "ab": "α/β", "apbp": "α'/β'"}


# ============================ ДАННЫЕ ============================
def load():
    t0 = time.time()
    ann = B.load_annotations()
    comp = pd.read_csv(B.PATH_COMP, index_col=0)
    con = pd.read_parquet(B.PATH_CON)
    ids = comp.index.values
    idx = pd.Series(np.arange(len(ids)), index=ids)          # root_id -> индекс мозга (как FlyBrain.idx)
    a = ann.reindex(ids)                                     # аннотации в порядке индексов мозга
    print(f"загрузка {time.time() - t0:.1f} с: {len(ids)} нейронов, {len(con)} связей")
    return a, con, idx


def lab_of(a):
    """Подпись нейрона для путей: cell_type, иначе hemibrain_type, иначе класс."""
    ct = a.cell_type.astype("string")
    lab = ct.fillna(a.hemibrain_type.astype("string")).fillna(a.cell_class.astype("string"))
    return lab.fillna("?").astype(str).values


def side_of(a):
    return a.side.fillna("?").astype(str).values


def bfs(G, sources, maxh=8):
    """Многоисточниковый BFS по CSR (строка = откуда, столбцы = куда). -> хопы (99 = недостижимо)."""
    n = G.shape[0]
    d = np.full(n, 99, np.int16)
    fr = np.unique(np.asarray(sources, dtype=np.int64))
    d[fr] = 0
    for h in range(1, maxh + 1):
        if len(fr) == 0:
            break
        nb = np.unique(G[fr].indices)
        nb = nb[d[nb] == 99]
        d[nb] = h
        fr = nb
    return d


# ============================ АНАЛИЗ ============================
def main():
    a, con, idx = load()
    N = len(a)
    ct = a.cell_type.fillna("").astype(str).values
    cls = a.cell_class.fillna("").astype(str).values
    side = side_of(a)
    lab = lab_of(a)
    pre = con.Presynaptic_Index.values.astype(np.int64)
    post = con.Postsynaptic_Index.values.astype(np.int64)
    nsyn = con.Connectivity.values.astype(np.int64)
    sgn = con.Excitatory.values.astype(np.int64)
    assert (a.index.values[pre[:1000]] == con.Presynaptic_ID.values[:1000]).all()

    # ---------- KC ----------
    is_kc = cls == "Kenyon_Cell"
    kc_idx = np.where(is_kc)[0]
    sub = np.array(["" for _ in range(N)], dtype=object)
    lobe = np.array(["" for _ in range(N)], dtype=object)
    for i in kc_idx:
        sub[i], lobe[i] = kc_sub(a.hemibrain_type.iat[i])
    kc = dict(total=int(is_kc.sum()), by_cell_type={}, by_subtype={},
              top_nt_flywire=a.top_nt[is_kc].value_counts().to_dict(),
              note=("cell_type FlyWire: KCg-m, KCg-d, KCg-s1..3, KCab, KCab-p, KCapbp-ap1/ap2/m, KCa'b'-ap1; "
                    "подтипы по hemibrain_type (KCab-c/-s/-m и т.д.). top_nt FlyWire у KC почти всегда "
                    "'dopamine' — это ошибка предсказателя; биологически KC холинергические. В модели все "
                    "выходные синапсы KC помечены +1 (возбуждающие) — проверено ниже."))
    for t in sorted(set(ct[is_kc])):
        m = is_kc & (ct == t)
        kc["by_cell_type"][t] = dict(n=int(m.sum()), n_left=int((m & (side == "left")).sum()),
                                     n_right=int((m & (side == "right")).sum()))
    for s in sorted(set(sub[kc_idx])):
        m = is_kc & (sub == s)
        kc["by_subtype"][s] = dict(lobe=LOBE_RU[lobe[np.where(m)[0][0]]], n=int(m.sum()),
                                   idx_left=np.where(m & (side == "left"))[0].tolist(),
                                   idx_right=np.where(m & (side == "right"))[0].tolist())
    kc_out = is_kc[pre]
    kc["out_rows"] = int(kc_out.sum())
    kc["out_rows_excitatory_in_model"] = int((kc_out & (sgn > 0)).sum())
    kc["kc_to_kc_rows"] = int((kc_out & is_kc[post]).sum())
    kc["kc_to_kc_synapses"] = int(nsyn[kc_out & is_kc[post]].sum())
    print(f"KC: {kc['total']}  выходных связей {kc['out_rows']} (возб. {kc['out_rows_excitatory_in_model']})")

    # ---------- MBON / DAN ----------
    is_mbon = cls == "MBON"
    mbon_types = sorted(set(ct[is_mbon]), key=lambda s: (s[:6], s))
    assert set(mbon_types) == set(MBON_LIT), set(mbon_types) ^ set(MBON_LIT)
    is_dan = pd.Series(ct).str.match(r"^(PAM|PPL)\d").values
    dan_types = sorted(set(ct[is_dan]))
    assert set(dan_types) == set(DAN_LIT), set(dan_types) ^ set(DAN_LIT)
    tot_in = np.bincount(post, weights=nsyn, minlength=N)
    tot_out = np.bincount(pre, weights=nsyn, minlength=N)

    # таблицы по типам
    df = pd.DataFrame(dict(pre=pre, post=post, n=nsyn, s=sgn))
    df["pre_t"] = ct[pre]
    df["post_t"] = ct[post]
    m_kc_mbon = kc_out & is_mbon[post]
    kcm = df[m_kc_mbon].copy()
    kcm["sub"] = sub[kcm.pre.values]
    kcm["lobe"] = lobe[kcm.pre.values]
    dan_mbon = df[is_dan[pre] & is_mbon[post]]
    mbon_dan = df[is_mbon[pre] & is_dan[post]]
    dan_kc = df[is_dan[pre] & is_kc[post]].copy()
    dan_kc["lobe"] = lobe[dan_kc.post.values]
    kc_dan = df[kc_out & is_dan[post]]
    mbon_mbon = df[is_mbon[pre] & is_mbon[post]]

    mbon = {}
    for t in mbon_types:
        m = is_mbon & (ct == t)
        ii = np.where(m)[0]
        lit, src = MBON_LIT[t]
        val, vst, vev = MBON_VAL[t]
        k = kcm[kcm.post_t == t]
        by_sub = k.groupby("sub").n.sum().to_dict()
        by_lobe = k.groupby("lobe").n.sum().to_dict()
        kc_syn = int(k.n.sum())
        in_syn = int(tot_in[ii].sum())
        d = dan_mbon[dan_mbon.post_t == t].groupby("pre_t").n.sum().sort_values(ascending=False)
        cm = canon(lit.split(" ")[0]) if t != "MBON25,MBON34" else canon(lit)
        match = [x for x in d.index if canon(DAN_LIT[x][0]) & cm]
        dm = int(d[match].sum()) if len(d) else 0
        out_rows = df[m[pre]]
        exc_frac = float(out_rows.n[out_rows.s > 0].sum() / max(1, out_rows.n.sum()))
        split = LOBE_SPLIT[t]
        kc_by_label = defaultdict(lambda: [0, 0])
        for (lb, g) in k.groupby("lobe"):
            key = split.get(lb, f"вне({LOBE_RU[lb]})")
            kc_by_label[key][0] += len(g)
            kc_by_label[key][1] += int(g.n.sum())
        mbon[t] = dict(
            n=len(ii), idx_left=[int(i) for i in ii if side[i] == "left"],
            idx_right=[int(i) for i in ii if side[i] == "right"],
            root_ids=[int(r) for r in a.index.values[ii]],
            compartment=lit, compartment_source=src, canonical=sorted(cm, key=CANON_ORDER.index),
            typical=t in TYPICAL,
            nt_flywire=",".join(sorted(set(a.top_nt.iloc[ii].dropna()))),
            out_excitatory_frac_in_model=round(exc_frac, 3),
            valence=val, valence_status=vst, valence_evidence=vev,
            kc_input_synapses=kc_syn, kc_input_rows=int(len(k)), all_input_synapses=in_syn,
            kc_input_frac=round(kc_syn / max(1, in_syn), 3),
            kc_input_by_subtype={s: int(v) for s, v in sorted(by_sub.items())},
            kc_input_by_lobe={LOBE_RU[s]: int(v) for s, v in by_lobe.items()},
            kc_input_by_compartment={kk: dict(rows=v[0], synapses=v[1]) for kk, v in kc_by_label.items()},
            dan_input_top={x: int(v) for x, v in d.head(6).items()},
            dan_input_total=int(d.sum()) if len(d) else 0,
            dan_input_same_compartment_frac=round(dm / max(1, int(d.sum()) if len(d) else 1), 3),
            dan_input_same_compartment_types=match,
            to_dan_top={x: int(v) for x, v in mbon_dan[mbon_dan.pre_t == t].groupby("post_t").n.sum()
                        .sort_values(ascending=False).head(5).items()},
            to_mbon_top={x: int(v) for x, v in mbon_mbon[mbon_mbon.pre_t == t].groupby("post_t").n.sum()
                         .sort_values(ascending=False).head(5).items()},
        )

    dan = {}
    for t in dan_types:
        m = is_dan & (ct == t)
        ii = np.where(m)[0]
        comp_, clus, val, vst, vev = DAN_LIT[t]
        dk = dan_kc[dan_kc.pre_t == t]
        tm = dan_mbon[dan_mbon.pre_t == t].groupby("post_t").n.sum().sort_values(ascending=False)
        dan[t] = dict(
            n=len(ii), idx_left=[int(i) for i in ii if side[i] == "left"],
            idx_right=[int(i) for i in ii if side[i] == "right"],
            root_ids=[int(r) for r in a.index.values[ii]],
            cluster=clus, compartment=comp_, canonical=sorted(canon(comp_), key=CANON_ORDER.index),
            valence=val, valence_status=vst, valence_evidence=vev,
            kc_output_synapses=int(dk.n.sum()),
            kc_output_by_lobe={LOBE_RU[s]: int(v) for s, v in dk.groupby("lobe").n.sum().items()},
            kc_input_synapses=int(kc_dan[kc_dan.post_t == t].n.sum()),
            mbon_output_synapses=int(tm.sum()),
            mbon_targets_top={x: int(v) for x, v in tm.head(5).items()},
            model_sign_of_outputs=("+1 (возбуждающие)" if (df[m[pre]].s > 0).mean() > 0.95 else
                                   f"возб. {100 * (df[m[pre]].s > 0).mean():.0f}% строк"),
        )

    other = {}
    for t in ["APL", "DPM"]:
        ii = np.where(ct == t)[0]
        other[t] = dict(n=len(ii), idx=[int(i) for i in ii], nt_flywire=",".join(sorted(set(a.top_nt.iloc[ii].dropna()))),
                        kc_input_synapses=int(nsyn[kc_out & np.isin(post, ii)].sum()),
                        kc_output_synapses=int(nsyn[np.isin(pre, ii) & is_kc[post]].sum()))
    other["APL"]["note"] = "ГАМК; обратное торможение KC (разреженный код). Держать как есть — это «противоэпилептика» самого MB."
    other["DPM"]["note"] = "серотонин/amnesiac; в модели знак синапсов по предсказанию NT."

    # ---------- компартменты ----------
    comps = defaultdict(lambda: dict(mbons=[], dans=[], kc_mbon_rows=0, kc_mbon_synapses=0,
                                     kc_mbon_rows_by_mbon={}, canonical=[]))
    for t in mbon_types:
        for lb, v in mbon[t]["kc_input_by_compartment"].items():
            c = comps[lb]
            c["mbons"].append(t)
            c["kc_mbon_rows"] += v["rows"]
            c["kc_mbon_synapses"] += v["synapses"]
            c["kc_mbon_rows_by_mbon"][t] = v["rows"]
            c["canonical"] = sorted(canon(lb), key=CANON_ORDER.index) if not lb.startswith("вне") else []
    dm_tab = dan_mbon.groupby(["pre_t", "post_t"]).n.sum()
    for lb, c in comps.items():
        cs = set(c["canonical"])
        c["dans"] = [d for d in dan_types if cs and (set(dan[d]["canonical"]) & cs)]
        c["dan_valence"] = {d: dan[d]["valence"] for d in c["dans"]}
        c["mbon_valence"] = {m_: mbon[m_]["valence"] for m_ in c["mbons"]}
        # проверка связями: синапсы DAN -> MBON этого компартмента (DAN и дендрит MBON рядом)
        c["dan_to_these_mbons_syn"] = {d: int(sum(dm_tab.get((d, m_), 0) for m_ in c["mbons"])) for d in c["dans"]}
    comps = dict(sorted(comps.items(), key=lambda kv: (kv[0].startswith("вне"),
                                                        min([CANON_ORDER.index(x) for x in kv[1]["canonical"]] or [99]), kv[0])))
    canon_tab = {}
    for cn in CANON_ORDER:
        canon_tab[cn] = dict(
            mbons=[t for t in mbon_types if cn in mbon[t]["canonical"]],
            dans=[d for d in dan_types if cn in dan[d]["canonical"]],
            kc_mbon_rows_exclusive=sum(c["kc_mbon_rows"] for c in comps.values() if c["canonical"] == [cn]),
            kc_mbon_rows_shared=sum(c["kc_mbon_rows"] for c in comps.values() if cn in c["canonical"] and len(c["canonical"]) > 1),
        )

    # ---------- рёбра KC->MBON и их место в быстром ядре ----------
    # brain.py _fast_init: Wc = coo([post, pre]).coalesce(); order = argsort(pre*N + post)
    # -> ребро (pre, post) лежит на позиции searchsorted(sorted(pre*N+post), pre*N+post) в b._post/b._wv,
    #    внутри отрезка b._cptr[pre] .. b._cptr[pre+1]. Дубликатов пар в parquet нет (проверено ниже).
    key_all = pre * N + post
    key_sorted = np.sort(key_all)
    n_dup = int(len(key_sorted) - np.count_nonzero(np.diff(key_sorted)) - 1)
    ep = np.where(m_kc_mbon)[0]
    e_pre, e_post, e_n = pre[ep], post[ep], nsyn[ep]
    e_pos = np.searchsorted(key_sorted, key_all[ep])
    assert (key_sorted[e_pos] == key_all[ep]).all()
    labels = sorted(comps.keys())
    lab_id = {lb: i for i, lb in enumerate(labels)}
    e_lab = np.array([lab_id[LOBE_SPLIT[ct[po]].get(lobe[pr], f"вне({LOBE_RU[lobe[pr]]})")]
                      for pr, po in zip(e_pre, e_post)], dtype=np.int16)
    mt_id = {t: i for i, t in enumerate(mbon_types)}
    e_mt = np.array([mt_id[ct[po]] for po in e_post], dtype=np.int16)
    w_syn = B.P["w_syn"]
    np.savez_compressed(
        OUT_NPZ, pre_idx=e_pre.astype(np.int32), post_idx=e_post.astype(np.int32), n_syn=e_n.astype(np.int32),
        weight=(e_n * w_syn).astype(np.float32), csc_pos=e_pos.astype(np.int64), comp_id=e_lab,
        comp_names=np.array(labels), mbon_type_id=e_mt, mbon_types=np.array(mbon_types),
        kc_subtype=np.array([sub[i] for i in e_pre]))
    edges = dict(
        rows=int(len(ep)), synapses=int(e_n.sum()), weight_sum_mV=float((e_n * w_syn).sum()),
        mean_syn_per_row=round(float(e_n.mean()), 2), median_syn_per_row=float(np.median(e_n)),
        max_syn_per_row=int(e_n.max()), rows_with_1_syn=int((e_n == 1).sum()),
        kc_with_any_mbon_output=int(len(np.unique(e_pre))), duplicate_pairs_in_parquet=n_dup,
        npz=OUT_NPZ.name,
        fast_path=dict(
            arrays="b._cptr (int32, N+1), b._post (int32, nnz), b._wv (float32, nnz) на GPU; сортировка по (pre, post)",
            weight="b._wv[e] = Excitatory x Connectivity x w_syn (0.275 мВ за синапс; x inh_scale если < 0). KC->MBON: все > 0",
            find=("исходящие синапсы нейрона j: e in [b._cptr[j], b._cptr[j+1]); b._post[e] = постсинаптический индекс. "
                  "Позиции в npz (csc_pos) верны для FlyBrain(drop_kc_kc=False); иначе learning/mb_edges.py: find_edges(b, pre, post) ищет по живым массивам ядра"),
            modify=("менять только НА МЕСТЕ: b._wv[pos] = новое (или b._wv.index_copy_/mul_). Не присваивать новый тензор: "
                    "адрес зашит в аргументы ядра и в CUDA Graph. Граф пересобирать не надо — ядро читает WV при каждом запуске"),
            spike_value=("при U>0 вклад спайка = WV[e] * x_pre (ресурс синапса пресинаптического нейрона, 0..1); "
                         "пластичность умножает статический вес, депрессия — поверх"),
            slow_path="старый путь (fast=False) считает через b.W (CSR [post, pre]) — там другие позиции; для обучения используем быстрый путь",
        ),
    )
    print(f"KC->MBON: {edges['rows']} строк, {edges['synapses']} синапсов; дубликатов пар {n_dup}")

    # ---------- пути от MBON к моторным нейронам ----------
    frac = nsyn / np.maximum(tot_in[post], 1)
    AT = sp.csr_matrix((frac, (pre, post)), shape=(N, N))              # AT[pre, post] = доля входа post от pre
    down = downstream(a, pre, post, nsyn, sgn, tot_in, ct, side, lab, is_mbon, mbon_types, N, AT)

    # ---------- сахар -> DAN (почему сахар не доходит до PAM) ----------
    Gout = sp.csr_matrix((np.ones(len(pre), np.int8), (pre, post)), shape=(N, N))
    strong = nsyn >= 5
    Gout5 = sp.csr_matrix((np.ones(strong.sum(), np.int8), (pre[strong], post[strong])), shape=(N, N))
    sug = [int(idx[r]) for r in SUGAR_IDS_630 if r in idx.index]
    d1, d5 = bfs(Gout, sug), bfs(Gout5, sug)
    sugar = dict(n_sugar_grn=len(sug), hops_to_dan_any_syn={}, hops_to_dan_ge5syn={}, path_strength_1to3={},
                 note="сила = сумма по путям длины 1-3 произведений долей входа (как для MBON->DN), от всех сахарных GRN")

    def strength_from(src, tgt):
        e = np.zeros(N)
        e[tgt] = 1.0
        r1 = AT @ e; r2 = AT @ r1; r3 = AT @ r2
        return float(r1[src].sum()), float(r2[src].sum()), float(r3[src].sum())
    for t in dan_types:
        ii = np.where(is_dan & (ct == t))[0]
        sugar["hops_to_dan_any_syn"][t] = int(d1[ii].min())
        sugar["hops_to_dan_ge5syn"][t] = int(d5[ii].min())
        if t.startswith("PAM"):
            h = strength_from(sug, ii)
            sugar["path_strength_1to3"][t] = dict(hop1=h[0], hop2=h[1], hop3=h[2], per_cell=sum(h) / len(ii))
    mn9 = [int(idx[MN9_ID])]
    h = strength_from(sug, mn9)
    sugar["path_strength_1to3"]["MN9"] = dict(hop1=h[0], hop2=h[1], hop3=h[2], per_cell=sum(h))
    sugar["hops_to_mn9_any"] = int(d1[mn9[0]])
    sugar["hops_to_mn9_ge5"] = int(d5[mn9[0]])

    res = dict(
        meta=dict(
            made=time.strftime("%Y-%m-%d %H:%M"), connectome="FlyWire v783 (Shiu et al. 2024 Connectivity_783.parquet)",
            n_neurons=N, n_connections=int(len(pre)), sources=SRC,
            index_note="индексы = позиция в Completeness_783.csv = FlyBrain.idx[root_id]",
            learning_rule_biology=("Hige 2015: совпадение активности KC и DAN в компартменте -> долговременная "
                                   "депрессия синапсов KC->MBON этого компартмента. Награда (PAM) давит KC-> «избегающие» "
                                   "MBON горизонтальной доли -> запах становится привлекательным; наказание (PPL1) давит "
                                   "KC-> «притягивающие» MBON (γ1pedc, γ2α'1, α2, α3...). Aso&Rubin 2016: знак зависит от "
                                   "времени — DAN в пределах ~30 с после начала запаха пишет память, DAN за 20-60 с ДО "
                                   "запаха — обратный знак."),
            model_caveats=[
                "В модели Shiu DAN->KC и DAN->MBON — обычные быстрые синапсы со знаком +1 (дофамин считан возбуждающим). "
                "В живой мухе дофамин — модулятор (учитель), а не быстрый возбудитель. Спайки DAN в модели напрямую "
                "возбуждают KC/MBON — это артефакт, его надо учитывать (или глушить при обучении).",
                "Глутаматные MBON (01-07 и др.) в модели тормозные (Excitatory=-1). В живой мухе глутамат может быть и "
                "возбуждающим (NMDA) — Li 2020 отмечает оба типа рецепторов у мишеней.",
                "Валентность MBON (approach/avoid) — поведенческий результат активации целого нейрона в арене, а не "
                "свойство отдельного синапса; в модели нет арены — есть только DN-выходы.",
            ],
        ),
        kc=kc, mbon=mbon, dan=dan, other_mb_neurons=other, compartments=comps, canonical_compartments=canon_tab,
        kc_mbon_edges=edges,
        connection_totals=dict(
            kc_to_mbon=dict(rows=int(m_kc_mbon.sum()), synapses=int(nsyn[m_kc_mbon].sum())),
            dan_to_kc=dict(rows=int(len(dan_kc)), synapses=int(dan_kc.n.sum())),
            kc_to_dan=dict(rows=int(len(kc_dan)), synapses=int(kc_dan.n.sum())),
            dan_to_mbon=dict(rows=int(len(dan_mbon)), synapses=int(dan_mbon.n.sum())),
            mbon_to_dan=dict(rows=int(len(mbon_dan)), synapses=int(mbon_dan.n.sum())),
            mbon_to_mbon=dict(rows=int(len(mbon_mbon)), synapses=int(mbon_mbon.n.sum())),
        ),
        downstream=down, sugar_to_dan=sugar,
    )
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("записал", OUT_JSON, OUT_NPZ)
    write_md(res)


# ============================ ПУТИ MBON -> DN ============================
TARGETS = {
    "steer_L": (["DNa01", "DNa02"], "left"), "steer_R": (["DNa01", "DNa02"], "right"),
    "walk_L": (["DNp09"], "left"), "walk_R": (["DNp09"], "right"),
    "gf_L": (["DNp01"], "left"), "gf_R": (["DNp01"], "right"),
    "mn9": ([MN9_ID], None),
}


def downstream(a, pre, post, nsyn, sgn, tot_in, ct, side, lab, is_mbon, mbon_types, N, AT):
    """Сила путей = произведение долей входа (синапсы от pre / все входные синапсы post) вдоль пути,
    сумма по всем путям длины k (k = 1, 2, 3). Знаковая версия: x Excitatory на каждом ребре —
    линейное приближение (порог нейронов не учитывает)."""
    t0 = time.time()
    frac = nsyn / np.maximum(tot_in[post], 1)
    ST = sp.csr_matrix((frac * sgn, (pre, post)), shape=(N, N))        # ST[pre, post]
    Acsc = AT.T.tocsc()                                                 # A[post, pre], столбец = выходы pre
    Gin = sp.csr_matrix((np.ones(len(pre), np.int8), (post, pre)), shape=(N, N))  # строка = post
    strong = nsyn >= 5
    Gin5 = sp.csr_matrix((np.ones(strong.sum(), np.int8), (post[strong], pre[strong])), shape=(N, N))
    rid = pd.Series(np.arange(N), index=a.index.values)

    tgt_idx = {}
    for k, (types, sd) in TARGETS.items():
        if isinstance(types[0], (int, np.integer)):
            tgt_idx[k] = [int(rid[x]) for x in types]
        else:
            tgt_idx[k] = [i for i in np.where(np.isin(ct, types))[0] if sd is None or side[i] == sd]
    groups = {}                                                         # MBON тип + сторона -> индексы
    for t in mbon_types:
        for sd in ("left", "right"):
            ii = [i for i in np.where(is_mbon & (ct == t))[0] if side[i] == sd]
            if ii:
                groups[f"{t}|{sd[0].upper()}"] = ii

    out = dict(method=("доля входа: A[post,pre] = синапсы / все входные синапсы post; сила пути = произведение A по "
                       "рёбрам; hop_k = сумма по всем путям длины k от всех клеток группы MBON к нейрону(ам) цели. "
                       "signed = то же с учётом знака синапсов (линейно, без порогов)."),
               targets={k: [dict(idx=int(i), root_id=int(a.index.values[i]), type=lab[i], side=side[i]) for i in v]
                        for k, v in tgt_idx.items()},
               per_target={}, steering={})
    infl = {}
    for k, ti in tgt_idx.items():
        e = np.zeros(N)
        e[ti] = 1.0
        r1 = AT @ e; r2 = AT @ r1; r3 = AT @ r2                        # r_k[m] = сила путей длины k от m
        s1 = ST @ e; s2 = ST @ s1; s3 = ST @ s2
        h1 = bfs(Gin, ti); h5 = bfs(Gin5, ti)                           # хопы от любого нейрона до цели
        rows = {}
        for g, ii in groups.items():
            rows[g] = dict(hops_any=int(h1[ii].min()), hops_ge5=int(h5[ii].min()),
                           hop1=float(r1[ii].sum()), hop2=float(r2[ii].sum()), hop3=float(r3[ii].sum()),
                           signed_1to3=float((s1 + s2 + s3)[ii].sum()))
        infl[k] = rows
        # лучшие цепочки по типам для самых сильных групп
        best = sorted(rows, key=lambda g: -(rows[g]["hop1"] + rows[g]["hop2"] + rows[g]["hop3"]))[:8]
        chains = {}
        for g in best:
            chains[g] = top_chains(groups[g], r1, r2, Acsc, lab, e)
        out["per_target"][k] = dict(
            n_mbon_groups_reach_hops_any={h: int(sum(1 for r in rows.values() if r["hops_any"] == h)) for h in range(1, 6)},
            n_mbon_groups_reach_hops_ge5={h: int(sum(1 for r in rows.values() if r["hops_ge5"] == h)) for h in range(1, 6)},
            rows=rows, top_chains=chains)
    # руль: влияние на левые DNa01/02 минус правые
    for g in groups:
        L, R = infl["steer_L"][g], infl["steer_R"][g]
        uL = L["hop1"] + L["hop2"] + L["hop3"]; uR = R["hop1"] + R["hop2"] + R["hop3"]
        out["steering"][g] = dict(unsigned_L=uL, unsigned_R=uR, lat_index=(uL - uR) / max(uL + uR, 1e-12),
                                  signed_L=L["signed_1to3"], signed_R=R["signed_1to3"],
                                  signed_L_minus_R=L["signed_1to3"] - R["signed_1to3"],
                                  hops_L=L["hops_any"], hops_R=R["hops_any"])
    print(f"пути MBON->DN: {time.time() - t0:.1f} с")
    return out


def top_chains(src, r1, r2, Acsc, lab, e_t, k=4):
    """Лучшие цепочки типов MBON -> X -> цель и MBON -> Y -> X -> цель (сумма по клеткам)."""
    c2, c3 = defaultdict(float), defaultdict(float)
    for m in src:
        s, en = Acsc.indptr[m], Acsc.indptr[m + 1]
        ys, wy = Acsc.indices[s:en], Acsc.data[s:en]                   # m -> y, доля входа y
        for y, w in zip(ys, wy):
            if r1[y] > 0:
                c2[lab[y]] += w * r1[y]
        # 3 хопа: берём 40 сильнейших y по w * r2[y], для них раскладываем по x
        sc = wy * r2[ys]
        for j in np.argsort(-sc)[:40]:
            y, w = ys[j], wy[j]
            if sc[j] <= 0:
                break
            s2, e2 = Acsc.indptr[y], Acsc.indptr[y + 1]
            xs, wx = Acsc.indices[s2:e2], Acsc.data[s2:e2]
            v = w * wx * r1[xs]
            for x, vv in zip(xs[v > 0], v[v > 0]):
                c3[(lab[y], lab[x])] += vv
    return dict(two_hop=[[x, float(v)] for x, v in sorted(c2.items(), key=lambda kv: -kv[1])[:k]],
                three_hop=[[f"{y} -> {x}", float(v)] for (y, x), v in sorted(c3.items(), key=lambda kv: -kv[1])[:k]])


# ============================ MARKDOWN ============================
def fmt(x, d=4):
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def gd(g):
    """'MBON27|L' -> 'MBON27 L' (вертикальная черта ломает таблицы markdown)."""
    return g.replace("|", " ")


def lr_summary(r, f):
    """Выводы «кто может сдвигать выбор лево/право» — из функциональной проверки + структуры."""
    out = []
    if not f or "all" not in f:
        return out
    A = f["all"]
    for t, v in r["mbon"].items():
        if f"{t}|L" not in A or f"{t}|R" not in A:
            continue
        L, R = A[f"{t}|L"]["none"], A[f"{t}|R"]["none"]
        if L["sig_steer"] and R["sig_steer"] and L["d_steer"] * R["d_steer"] < 0:
            kind = "контралатерально" if L["d_steer"] < 0 < R["d_steer"] else "ипсилатерально"
            ks = v["kc_input_by_subtype"]
            out.append(
                f"**{t}** ({v['compartment']}, {'атипичный' if not v['typical'] else 'типичный'}, {v['nt_flywire']}) — "
                f"единственный тип с чистым лево/правым эффектом: левые клетки дают руль {L['d_steer']:+.1f} Гц "
                f"(DNa_R {L['delta']['DNa_R']:+.1f}), правые {R['d_steer']:+.1f} Гц (DNa_L {R['delta']['DNa_L']:+.1f}) — "
                f"{kind}, без всякого контекста; на фоне LC10 {A[f'{t}|L']['steer']['d_steer']:+.1f} / "
                f"{A[f'{t}|R']['steer']['d_steer']:+.1f}. Вход от KC: {v['kc_input_synapses']} синапсов "
                f"({', '.join(f'{k} {x}' for k, x in sorted(ks.items(), key=lambda kv: -kv[1])[:3])}); "
                f"DAN на дендрит: {', '.join(f'{k} {x}' for k, x in list(v['dan_input_top'].items())[:3])}.")
    one = []
    for t in r["mbon"]:
        for sd in "LR":
            g = f"{t}|{sd}"
            if g not in A:
                continue
            for ctx in ("none", "steer"):
                x = A[g][ctx]
                if x["sig_steer"] and not any(o.startswith(f"**{t}**") for o in out):
                    one.append(f"{gd(g)} ({ctx}: {x['d_steer']:+.1f})")
    if one:
        out.append("Сдвиг руля только с одной стороны или одинаковый с обеих (не годится как «лево/право» сам по себе): "
                   + ", ".join(one) + ". MBON26 (β'2d) с обеих сторон усиливает левый DNa — это врождённый левый крен "
                   "коннектома, а не выбор стороны.")
    return out


def write_md(r):
    L = []
    w = L.append
    kc, mb, dn, e = r["kc"], r["mbon"], r["dan"], r["kc_mbon_edges"]
    ctot = r["connection_totals"]
    w("# Карта грибовидного тела (MB) — FlyWire v783\n")
    w(f"_Сгенерировано `learning/mb_map.py` ({r['meta']['made']}). Все числа посчитаны по "
      "`brain_model/Connectivity_783.parquet` и `data/neuron_annotations.tsv`. Компартменты и валентность — "
      "из литературы (источники ниже), компартменты перепроверены связями DAN->MBON._\n")
    w("## Коротко\n")
    w(f"- Клеток Кеньона (KC): **{kc['total']}** (γ {sum(v['n'] for s, v in kc['by_subtype'].items() if s.startswith('g'))}, "
      f"α/β {sum(v['n'] for s, v in kc['by_subtype'].items() if s.startswith('ab'))}, "
      f"α'/β' {sum(v['n'] for s, v in kc['by_subtype'].items() if s.startswith(chr(97) + chr(39)))}).")
    w(f"- MBON: **{sum(v['n'] for v in mb.values())}** нейронов, {len(mb)} типов FlyWire; "
      f"DAN (PAM/PPL1/PPL2): **{sum(v['n'] for v in dn.values())}** нейронов, {len(dn)} типов.")
    w(f"- Связей KC->MBON: **{e['rows']}** строк (пар нейронов), **{e['synapses']}** синапсов; "
      f"в среднем {e['mean_syn_per_row']} синапса на пару (медиана {e['median_syn_per_row']:.0f}, макс {e['max_syn_per_row']}); "
      f"пар с 1 синапсом {e['rows_with_1_syn']}. KC, у которых есть выход на MBON: {e['kc_with_any_mbon_output']}.")
    w(f"- DAN->KC {ctot['dan_to_kc']['rows']} строк / {ctot['dan_to_kc']['synapses']} синапсов; "
      f"KC->DAN {ctot['kc_to_dan']['rows']} / {ctot['kc_to_dan']['synapses']}; "
      f"DAN->MBON {ctot['dan_to_mbon']['rows']} / {ctot['dan_to_mbon']['synapses']}; "
      f"MBON->DAN {ctot['mbon_to_dan']['rows']} / {ctot['mbon_to_dan']['synapses']}; "
      f"MBON->MBON {ctot['mbon_to_mbon']['rows']} / {ctot['mbon_to_mbon']['synapses']}.")
    w(f"- KC->KC (аксо-аксональные): {kc['kc_to_kc_rows']} строк / {kc['kc_to_kc_synapses']} синапсов "
      "(их выкидывает `drop_kc_kc=True`; позиции рёбер тогда другие).")
    fr0 = [v["dan_input_same_compartment_frac"] for v in mb.values()]
    w(f"- Компартменты из литературы подтверждены связями DAN->MBON: у {sum(x >= 0.7 for x in fr0)} из {len(fr0)} "
      f"типов MBON >= 70% дофаминового входа на дендрит — от DAN своего компартмента (медиана {100 * float(np.median(fr0)):.0f}%).")
    f0 = load_func()
    if f0 and "edge_check" in f0:
        ec = f0["edge_check"]
        w(f"- Позиции всех {ec['n_checked']} связей KC->MBON в `b._post/b._wv` проверены на живом мозге: "
          f"совпали {ec['post_match']} (post) и {ec['w_match']} (вес).")
    if f0 and "hits" in f0:
        for s_ in lr_summary(r, f0)[:1]:
            w(f"- Лево/право: {s_}")
    w("")

    w("## Клетки Кеньона\n")
    w("| подтип (hemibrain) | доля | всего | слева | справа |\n|---|---|---|---|---|")
    for s, v in kc["by_subtype"].items():
        w(f"| {s} | {v['lobe']} | {v['n']} | {len(v['idx_left'])} | {len(v['idx_right'])} |")
    w("\n| cell_type FlyWire | всего | слева | справа |\n|---|---|---|---|")
    for s, v in kc["by_cell_type"].items():
        w(f"| {s} | {v['n']} | {v['n_left']} | {v['n_right']} |")
    w(f"\ntop_nt FlyWire у KC: {kc['top_nt_flywire']} — ошибка предсказателя (KC холинергические). "
      f"В модели выходных связей KC {kc['out_rows']}, из них со знаком +1: {kc['out_rows_excitatory_in_model']}. "
      "Индексы по сторонам — в `mb_map.json` -> `kc.by_subtype.*.idx_left/idx_right`.\n")

    w("## Компартменты: кто пишет (DAN) и кто читает (MBON)\n")
    w("Метка — компартмент дендритов MBON (вход от KC разделён по доле KC: γ-KC / α/β-KC / α'/β'-KC). "
      "Метки вида «γ1γ2» — один MBON на несколько компартментов одной доли; по связям нейрон-нейрон их не разделить.\n")
    w("| компартмент | MBON (валентность) | DAN-учителя: валентность; синапсов DAN->эти MBON | строк KC->MBON | синапсов |\n|---|---|---|---|---|")
    for lb, c in r["compartments"].items():
        ms = ", ".join(f"{m}({c['mbon_valence'][m]})" for m in c["mbons"])
        dd = sorted(c["dans"], key=lambda d: -c["dan_to_these_mbons_syn"][d])
        ds = ", ".join(f"{d}({c['dan_valence'][d]}; {c['dan_to_these_mbons_syn'][d]})" for d in dd) or "—"
        w(f"| {lb} | {ms} | {ds} | {c['kc_mbon_rows']} | {c['kc_mbon_synapses']} |")
    w("\nDAN-учителя — по литературным компартментам (грубо: 15 канонических + pedc + calyx; подобласти β'2a/m/p/d "
      "слиты в β'2). Число в скобках — синапсы этого DAN прямо на дендриты MBON компартмента: это проверка, что DAN и "
      "MBON действительно рядом (например, для β'2mp главный учитель PAM06, а PAM02 почти не контактирует). "
      "Валентность MBON: approach = активация притягивает, avoid = отталкивает (Aso 2014b); «?» = не проверено. "
      "DAN: reward/punish (Aso & Rubin 2016); «?» = не проверено или неясно.\n")

    w("## MBON: литература против связей\n")
    w("`DAN того же компартмента` = доля синапсов DAN->MBON от DAN, чей компартмент по литературе совпадает "
      "(15 канонических + pedc + calyx) — это и есть проверка компартмента связями.\n")
    w("| тип | n (L/R) | компартмент (лит.) | NT FlyWire | выход в модели, % возб. | валентность [статус] | KC-синапсов (% входа) | вход по долям KC | главные DAN на дендрит | DAN того же компартмента |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for t, v in mb.items():
        dans = ", ".join(f"{k}:{x}" for k, x in list(v["dan_input_top"].items())[:3]) or "—"
        lob = ", ".join(f"{k}:{x}" for k, x in v["kc_input_by_lobe"].items())
        w(f"| {t}{'' if v['typical'] else '*'} | {v['n']} ({len(v['idx_left'])}/{len(v['idx_right'])}) | {v['compartment']} | "
          f"{v['nt_flywire']} | {100 * v['out_excitatory_frac_in_model']:.0f} | {v['valence']} [{v['valence_status']}] | "
          f"{v['kc_input_synapses']} ({100 * v['kc_input_frac']:.0f}%) | {lob} | {dans} | "
          f"{100 * v['dan_input_same_compartment_frac']:.0f}% из {v['dan_input_total']} "
          f"{'ok' if v['dan_input_same_compartment_frac'] >= 0.7 else 'частично'} |")
    fr = sorted((v["dan_input_same_compartment_frac"], t) for t, v in mb.items())
    w(f"\nПроверка компартментов связями: у {sum(1 for f_, _ in fr if f_ >= 0.7)} из {len(fr)} типов MBON >= 70% "
      f"синапсов DAN->MBON идут от DAN «своего» компартмента (медиана {100 * float(np.median([f_ for f_, _ in fr])):.0f}%). "
      "Ниже 70%: " + ", ".join(f"{t} {100 * f_:.0f}%" for f_, t in fr if f_ < 0.7) +
      ". Главная доля KC-входа (γ / α/β / α'/β') у всех типов совпадает с литературной. Вход из «чужой» доли "
      "> 5% KC-входа: " + ", ".join(
          f"{t} {k.replace('вне', '')} {x['synapses']} ({100 * x['synapses'] / max(1, v['kc_input_synapses']):.0f}%)"
          for t, v in mb.items() for k, x in v["kc_input_by_compartment"].items()
          if k.startswith("вне") and x["synapses"] > 0.05 * v["kc_input_synapses"])
      + " — в модели эти синапсы есть; к какому компартменту они относятся, по связям нейрон-нейрон не определить "
      "(строки «вне(...)» таблицы компартментов).")
    w("\n`*` — атипичный MBON (Li 2020: дендриты и вне долей MB). MBON08 (γ3) в аннотациях FlyWire нет "
      "(в hemibrain его тоже не нашли). Доказательства валентности по каждому типу — `mb_map.json` -> `mbon.*.valence_evidence`.\n")

    w("## DAN\n")
    w("| тип | n (L/R) | кластер | компартмент (лит.) | валентность [статус] | синапсов на KC (по долям) | на MBON | главные MBON-мишени |")
    w("|---|---|---|---|---|---|---|---|")
    for t, v in dn.items():
        lob = ", ".join(f"{k}:{x}" for k, x in v["kc_output_by_lobe"].items()) or "—"
        tm = ", ".join(f"{k}:{x}" for k, x in list(v["mbon_targets_top"].items())[:3]) or "—"
        w(f"| {t} | {v['n']} ({len(v['idx_left'])}/{len(v['idx_right'])}) | {v['cluster']} | {v['compartment']} | "
          f"{v['valence']} [{v['valence_status']}] | {v['kc_output_synapses']} ({lob}) | {v['mbon_output_synapses']} | {tm} |")
    s = r["sugar_to_dan"]
    w(f"\nСахар -> DAN (минимум хопов от {s['n_sugar_grn']} сахарных GRN, любые синапсы / только пары >= 5 синапсов): "
      + ", ".join(f"{t} {s['hops_to_dan_any_syn'][t]}/{s['hops_to_dan_ge5syn'][t]}" for t in dn if t.startswith("PAM"))
      + f"; до MN9 {s['hops_to_mn9_any']}/{s['hops_to_mn9_ge5']}.\n")
    ps = s["path_strength_1to3"]
    pam_best = sorted(((v["per_cell"], t) for t, v in ps.items() if t != "MN9"), reverse=True)
    w(f"Сила путей сахар -> цель (сумма долей входа по путям 1-3 хопа, на одну клетку цели): MN9 {ps['MN9']['per_cell']:.2e}; "
      + ", ".join(f"{t} {v:.2e}" for v, t in pam_best[:5]) + f"; медиана по PAM-типам "
      f"{float(np.median([v for v, _ in pam_best])):.2e}. Короткие пути к PAM есть, но они слабые "
      f"(к самому «сахарному» {pam_best[0][1]} в ~{ps['MN9']['per_cell'] / max(pam_best[0][0], 1e-12):.0f} раз слабее, "
      f"чем к MN9; к медианному PAM-типу — в ~{ps['MN9']['per_cell'] / max(float(np.median([v for v, _ in pam_best])), 1e-12):.0f} раз) — это согласуется с тем, "
      "что в модели сахар до PAM не доходит (измерено раньше). Поэтому награду даём стимуляцией PAM напрямую "
      "(как оптогенетика в Aso & Rubin 2016) — это модельный выбор, не биология вкуса.\n")

    w("## Где лежат веса KC->MBON в быстром ядре brain.py\n")
    fp = e["fast_path"]
    for k in ("arrays", "weight", "find", "modify", "spike_value", "slow_path"):
        w(f"- **{k}**: {fp[k]}")
    w(f"\nФайл `{e['npz']}`: для каждой из {e['rows']} связей — `pre_idx, post_idx, n_syn, weight (мВ), csc_pos, "
      "comp_id/comp_names, mbon_type_id/mbon_types, kc_subtype`. Пример:\n")
    w("```python\nz = np.load('learning/mb_kc_mbon_edges.npz')\npos = torch.as_tensor(z['csc_pos'], device=b.dev)\n"
      "assert (b._post[pos].cpu().numpy() == z['post_idx']).all()      # проверка\nw0 = b._wv[pos].clone()                                       # исходные веса\n"
      "b._wv[pos] = w0 * factor                                          # на месте; граф пересобирать не нужно\n```\n")
    f = load_func()
    if f and "edge_check" in f:
        ec = f["edge_check"]
        w(f"Проверено на живом FlyBrain (GPU): {ec['n_checked']} рёбер, post совпал у {ec['post_match']}, "
          f"вес совпал у {ec['w_match']} (макс. |Δw| {ec['max_abs_dw']:.2e}); позиции внутри отрезка cptr: {ec['in_segment']}. "
          f"find_edges() по живым массивам дал те же позиции: {ec['find_edges_match']}.\n")
    if f and "inplace_check" in f:
        ic = f["inplace_check"]
        w(f"Правка на месте при включённом CUDA Graph (graph_used={ic['graph_used']}, граф НЕ пересобирали): "
          f"{ic['n_kc_driven']} случайных KC (5%) по {ic['kc_hz']:.0f} Гц -> средняя частота всех 96 MBON "
          f"{ic['mbon_hz_orig'][0]:.2f}±{ic['mbon_hz_orig'][1]:.2f} Гц; веса KC->MBON x0 -> {ic['mbon_hz_zeroed'][0]:.2f}"
          f"±{ic['mbon_hz_zeroed'][1]:.2f} (остаток — обходные пути, не KC->MBON); x2 -> {ic['mbon_hz_x2'][0]:.2f}"
          f"±{ic['mbon_hz_x2'][1]:.2f}; вернули -> {ic['mbon_hz_restored'][0]:.2f}±{ic['mbon_hz_restored'][1]:.2f} "
          f"(веса восстановлены побитно: {ic['restored_exact']}). По 4 прогона на точку.\n")

    w("## Пути MBON -> моторные выходы\n")
    d = r["downstream"]
    w(d["method"] + "\n")
    for k, v in d["per_target"].items():
        tg = ", ".join(f"{x['type']}({x['side'][0]})" for x in d["targets"][k])
        w(f"### {k}: {tg}\n")
        w(f"Групп MBON (тип x сторона) с кратчайшим путём 1/2/3/4/5 хопов (любые синапсы): "
          f"{list(v['n_mbon_groups_reach_hops_any'].values())}; только пары >= 5 синапсов: "
          f"{list(v['n_mbon_groups_reach_hops_ge5'].values())}.\n")
        rows = sorted(v["rows"].items(), key=lambda kv: -(kv[1]["hop1"] + kv[1]["hop2"] + kv[1]["hop3"]))[:10]
        w("| MBON | хопов (любые / >=5) | 1 хоп | 2 хопа | 3 хопа | знаковая сумма 1-3 |\n|---|---|---|---|---|---|")
        for g, x in rows:
            w(f"| {gd(g)} | {x['hops_any']} / {x['hops_ge5']} | {x['hop1']:.2e} | {x['hop2']:.2e} | {x['hop3']:.2e} | {x['signed_1to3']:+.2e} |")
        w("\nСильнейшие цепочки типов:\n")
        for g, c in list(v["top_chains"].items())[:4]:
            two = "; ".join(f"{x} ({s:.1e})" for x, s in c["two_hop"][:3]) or "—"
            thr = "; ".join(f"{x} ({s:.1e})" for x, s in c["three_hop"][:3]) or "—"
            w(f"- **{gd(g)}** — 2 хопа: {two}. 3 хопа: {thr}")
        w("")
    w("### Лево/право: кто может сдвигать выбор стороны\n")
    st = sorted(d["steering"].items(), key=lambda kv: -abs(kv[1]["signed_L_minus_R"]))[:16]
    w("Влияние на DNa01+DNa02 левые минус правые (сумма путей 1-3 хопа). lat_index = (L-R)/(L+R) по беззнаковой силе.\n")
    w("| MBON | беззнак. L | беззнак. R | lat_index | знак. L-R | хопов L/R |\n|---|---|---|---|---|---|")
    for g, x in st:
        w(f"| {gd(g)} | {x['unsigned_L']:.2e} | {x['unsigned_R']:.2e} | {x['lat_index']:+.2f} | {x['signed_L_minus_R']:+.2e} | {x['hops_L']}/{x['hops_R']} |")
    w("")
    if f and "hits" in f:
        write_func_md(w, f)
        w("### Кто может сдвигать выбор лево/право (итог)\n")
        for s_ in lr_summary(r, f):
            w(f"- {s_}")
        w("- Структурно самые сильные пути к DNa01/02 идут от атипичных MBON (26, 27, 31, 32, 35) через LAL "
          "(LAL051, LAL171/172, LAL170) — как и пишет Li 2020 («atypical MBONs provide a direct path to motor control»). "
          "Но ГАМК-ergic MBON31/MBON32 (есть прямые синапсы на DNa) в модели руль значимо не двигают: их доля во "
          "входе DNa ~0.1-0.3%.")
        neg = defaultdict(list)
        for h in f["hits"]:
            for k, s_ in h["sig"].items():
                if s_ and h["delta"][k] < 0:
                    neg[(k, h["ctx"])].append(f"{gd(h['group'])} {h['delta'][k]:+.1f}")
        w("- Остальные значимые эффекты — торможение уже идущего ответа (Гц к фону): "
          + "; ".join(f"{k} на фоне {c}: {', '.join(v)}" for (k, c), v in neg.items()) + ". "
          "Ходьбу DNp09 не включил ни один MBON (0 Гц во всех прогонах).")
        allw = [(x["n_active_max"], g, c) for g, gg in f["all"].items() for c, x in gg.items()]
        mx = max(allw)
        rest = sorted(allw)[-2]
        w(f"- Стабильность: максимум активных нейронов за окно — {mx[0]:.0f} ({gd(mx[1])}, {mx[2]}; единичный прогон), "
          f"следующий по величине {rest[0]:.0f}. Повтор {gd(mx[1])} без контекста 20 раз "
          "(learning/_scratch/mbon12_burst.py): 193-477 активных, всплесков > 2000 нет. Лавины нет, но редкие всплески "
          "возможны — модулю обучения стоит следить за n_active.\n")
    w("## Что биология, а что наш выбор\n")
    for c in r["meta"]["model_caveats"]:
        w(f"- {c}")
    w(f"- Правило обучения (биология): {r['meta']['learning_rule_biology']}")
    w("- Награда стимуляцией PAM (а не сахаром) — модельный выбор, аналог оптогенетики.\n")
    w("## Источники\n")
    for k, v in r["meta"]["sources"].items():
        w(f"- {k}: {v}")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("записал", OUT_MD)


def load_func():
    try:
        return json.loads(FUNC_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_func_md(w, f):
    """Функциональная проверка (mb_func_test.py): MBON включаем Пуассоном, смотрим DN."""
    w("## Функциональная проверка на модели (GPU)\n")
    w(f_desc(f) + "\n")
    base = f["baseline"]
    w("Фон (без активации MBON), среднее ± SD по повторам, Гц:\n")
    w("| контекст | " + " | ".join(f["readouts"]) + " | активных нейронов |\n|---" * 1 + "|---" * (len(f["readouts"]) + 1) + "|")
    for c, v in base.items():
        w(f"| {c} | " + " | ".join(f"{v['mean'][k]:.1f}±{v['sd'][k]:.1f}" for k in f["readouts"]) + f" | {v['n_active']:.0f} |")
    n_groups = len(f["all"])
    w(f"\nЭффекты активации: {len(f['hits'])} значимых строк из {n_groups * len(base)} (группа x контекст). "
      "Δ к фону, Гц; жирным — значимо:\n")
    w("| MBON | контекст | " + " | ".join(f"Δ{k}" for k in f["readouts"]) + " | Δ руль (L-R) | MBON Гц | активных |")
    w("|---|---|" + "---|" * (len(f["readouts"]) + 3))
    for row in f["hits"]:
        w(f"| {gd(row['group'])} | {row['ctx']} | " + " | ".join(
            (f"**{row['delta'][k]:+.1f}**" if row['sig'][k] else f"{row['delta'][k]:+.1f}") for k in f["readouts"])
          + (f" | **{row['d_steer']:+.1f}**" if row["sig_steer"] else f" | {row['d_steer']:+.1f}")
          + f" | {row['mbon_hz']:.0f} | {row['n_active']:.0f} |")
    w("\nРуль по сторонам (Δ руля L-R при активации левых / правых клеток типа; >0 = поворот влево). "
      "Показаны типы, где хоть одна сторона значима:\n")
    w("| MBON | без контекста: левые / правые | на фоне LC10: левые / правые |\n|---|---|---|")
    for t in f["steer_by_side"]:
        cells = []
        anys = False
        for ctx in ("none", "steer"):
            L, R = f["all"][f"{t}|L"][ctx], f["all"][f"{t}|R"][ctx]
            anys |= L["sig_steer"] or R["sig_steer"]
            cells.append(" / ".join((f"**{x['d_steer']:+.1f}**" if x["sig_steer"] else f"{x['d_steer']:+.1f}") for x in (L, R)))
        if anys:
            w(f"| {t} | {cells[0]} | {cells[1]} |")
    w("")


def f_desc(f):
    return f.get("desc", "")


if __name__ == "__main__":
    main()
