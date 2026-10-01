"""Задача C: безусловный стимул (US) и выходы грибовидного тела (MB) -> поведение.

Все числа ниже измерены на мозге с противоэпилептикой PRM (скрипты и логи: learning/_scratch/c1..c7,
a1, a2). Что биология, а что наш модельный выбор — помечено «БИО» / «МОДЕЛЬ».

1) US = прямое включение DAN (МОДЕЛЬ; в живых опытах так делают оптогенетикой, Aso & Rubin 2016).
   Сахар до PAM в модели НЕ доходит (c2): 20 LB3 200 Гц -> все PAM 0 Гц; лучший PAM получает
   15-30 синапс*Гц при нужных ~5100 (MN9 получает 11244). Кратчайший путь 2 синапса
   (LB3 -> CB0130 -> PAM11, LB3 -> CB3199 -> PAM01), но на звене <= 8 синапсов и < 1% входа.
   Поэтому награда = PAM, и так и подписываем.
   Безопасность (c1): любая группа DAN 10-100 Гц x 1 с -> эхо 0, максимум 381 активный нейрон / 18 мс;
   KC от дофамина не стреляют (0). Вместе с запахом риск не растёт (эхо 62-106 против 100 у запаха одного).
2) Артефакт модели: дофамин здесь — обычный быстрый возбуждающий синапс (в мухе — GPCR). PAM/PPL1
   напрямую раскачивают MBON своего отсека (PAM11 100 Гц -> MBON07 113-125 Гц; PPL101 -> MBON11 116;
   PAM08 -> MBON05 138-144). Чтобы US не пачкал оценку CS: DAN заглушить (brain.silence) — спайки DAN
   всё равно считаются (lif_steps считает cnt до проверки SIL) и годятся как обучающий сигнал.
3) MBON -> поведение (c3, c7, n=10): сами по себе MBON почти не трогают поведенческие DN.
   DNp09 / MDN / GF / MN9 — 0 от любого MBON. Надёжный руль только от MBON27 (γ5d, ACh), противоположная
   сторона: 50 Гц L -> руль -7.1 Гц (z -9), R -> +7.8 (z +14); 20 Гц ~ +-3 (z ~3.5); 10 Гц не значимо.
   Группа «приближения» (MBON11+09+12) одной стороной 50 Гц: +2.4 / -1.7 (слабо), с двух сторон — 0.
   Группа «избегания» (MBON01+03+04) — не значимо. Врождённый руль фона влево: +11.2 +- 1.2 Гц.
   Выученное изменение MBON (KC(CS+)->MBON сладких отсеков x0.3) до DN не доходит (c6): руль меняется
   на <= 1-2 Гц, как и у контрольного CS-. Значит «оценка приближения -> руль» — это МОДЕЛЬНЫЙ мост.
4) CS: запах через ORN сейчас непригоден (c4, c5): любой запах (5-30 Гц, ступень или рампа 300 мс)
   даёт вспышку в антеннальной доле — 71% PN, 75% LN, ~50% KC в первые ~100 мс, потом KC молчат;
   наборы PN у разных запахов совпадают (Жаккар 0.994), KC — 0.95-0.97, вектор MBON — корр. 0.99.
   Без KC->KC синапсов (40% входа KC) то же (c4b). Рабочий вариант — CS прямо на 5% KC (МОДЕЛЬ):
   20 Гц -> другие KC не подтягиваются (0), эхо 0, MBON отвечают; 5-10 Гц — MBON почти молчат.
"""
import numpy as np

PRM = dict(U=0.1, tau_d=300, a_inc=0.1, tau_a=3000)

# отсек каждого DAN — проверено по их синапсам на MBON (a1_anatomy.log), совпадает с Li et al. 2020
DAN_COMP = {
    "PAM01": "γ5", "PAM02": "β'2a", "PAM03": "β2β'2a", "PAM04": "β2", "PAM05": "β'2p", "PAM06": "β'2m",
    "PAM07": "γ4<γ1γ2", "PAM08": "γ4", "PAM09": "β1ped", "PAM10": "β1", "PAM11": "α1", "PAM12": "γ3",
    "PAM13": "β'1ap", "PAM14": "β'1m", "PAM15": "γ5β'2a",
    "PPL101": "γ1pedc", "PPL102": "γ1", "PPL103": "γ2α'1", "PPL104": "α'3", "PPL105": "α'2α2",
    "PPL106": "α3", "PPL107": "(85% синапсов вне MB)", "PPL108": "(87% вне MB)",
}
MBON_COMP = {
    "MBON01": "γ5β'2a", "MBON02": "β2β'2a", "MBON03": "β'2mp", "MBON04": "β'2mp_bil", "MBON05": "γ4>γ1γ2",
    "MBON06": "β1>α", "MBON07": "α1", "MBON09": "γ3β'1", "MBON10": "β'1", "MBON11": "γ1pedc>α/β",
    "MBON12": "γ2α'1", "MBON13": "α'2", "MBON14": "α3", "MBON15": "α'1", "MBON16": "α'3ap", "MBON17": "α'3m",
    "MBON18": "α2sc", "MBON19": "α2p3p", "MBON20": "γ1γ2", "MBON21": "γ4γ5", "MBON22": "calyx",
    "MBON23": "α2sp", "MBON24": "β2γ5", "MBON25,MBON34": "γ1γ2/γ2", "MBON26": "β'2d", "MBON27": "γ5d",
    "MBON28": "α'3", "MBON29": "γ4γ5", "MBON30": "γ1γ2γ3", "MBON31": "α'1a", "MBON32": "γ2",
    "MBON33": "γ2γ3", "MBON35": "γ2",
}
# БИО: Aso et al. 2014b (CsChrimson, выбор свет/темнота): все «отталкивающие» MBON глутаматные
# (MB011B: 01/03/04, MB434B: 05/06), «притягивающие» — ГАМК (MB112C: 11, MB083C: 09) и ACh (MB077B: 12, MB052B: V2 = 15-19)
MBON_VALENCE = {"avoid": ["MBON01", "MBON03", "MBON04", "MBON05", "MBON06"],
                "approach": ["MBON11", "MBON09", "MBON12", "MBON15", "MBON16", "MBON17", "MBON18", "MBON19"]}

# US-наборы (МОДЕЛЬ: включаем DAN напрямую). Сладкое по литературе связывают с PAM-β'2 / PAM-γ4,
# долгую «питательную» награду — с PAM-α1 (Aso & Rubin 2016: MB043C); наказание — PPL1-γ1pedc (MB320C), γ2α'1.
US = {
    "sweet": ["PAM02", "PAM03", "PAM08"],        # β'2a, β2β'2a, γ4 — 71 нейрон
    "nutrient": ["PAM11"],                        # α1 — 16 нейронов
    "punish": ["PPL101", "PPL103"],               # γ1pedc, γ2α'1 — 4 нейрона
}
US_HZ = 50.0      # 30-100 Гц безопасны (c1); Aso & Rubin 2016 давали 30 импульсов света по 1 с за 60 с запаха
# MBON в отсеках «сладких» DAN (по синапсам DAN->MBON): их ослабление на CS+ = выученное приближение
SWEET_MBON = ["MBON05", "MBON21", "MBON29", "MBON01", "MBON03", "MBON04", "MBON02", "MBON24"]
# порог «это US, а не фон»: запах сам зажигает DAN во вспышке (PPL101 до 9.6 Гц, PPL106 8.1, PAM <= 1.9;
# усреднено за 1 с стимула), значит обучающий сигнал = частота DAN выше ~15 Гц (МОДЕЛЬ)
DA_THRESHOLD_HZ = 15.0


def _ty(brain, ann, types, side=None):
    a = ann[ann.cell_type.isin(types if isinstance(types, (list, tuple)) else [types])]
    if side:
        a = a[a.side == side]
    return [brain.idx[r] for r in a.index if r in brain.idx]


def build_mb_groups(brain, ann):
    """-> dict: 'us_sweet'/'us_nutrient'/'us_punish', каждый DAN-тип, 'MBON05_L' и т.п., 'sweet_mbon',
    'KC', 'KC_L', 'KC_R', руль 'steer_L'/'steer_R' (DNa01+DNa02), 'mbon27_L'/'mbon27_R'."""
    g = {f"us_{k}": _ty(brain, ann, v) for k, v in US.items()}
    for t in DAN_COMP:
        g[t] = _ty(brain, ann, t)
    for t in MBON_COMP:
        for s in ("left", "right"):
            ix = _ty(brain, ann, t, s)
            if ix:
                g[f"{t}_{s[0].upper()}"] = ix
    g["sweet_mbon"] = [i for t in SWEET_MBON for s in "LR" for i in g.get(f"{t}_{s}", [])]
    kc = ann[ann.cell_class == "Kenyon_Cell"]
    g["KC"] = [brain.idx[r] for r in kc.index if r in brain.idx]
    for s in ("left", "right"):
        g[f"KC_{s[0].upper()}"] = [brain.idx[r] for r in kc.index[kc.side == s] if r in brain.idx]
        g[f"steer_{s[0].upper()}"] = _ty(brain, ann, ["DNa01", "DNa02"], s)
    return g


def kc_cs_sets(groups, n_cs=2, frac=0.05, seed=0):
    """CS прямо на KC (МОДЕЛЬ): n_cs непересекающихся случайных наборов по frac всех KC (обе стороны).
    5% x 20 Гц: другие KC не подтягиваются, эхо 0 (c6)."""
    kc = np.asarray(groups["KC"])
    perm = np.random.default_rng(seed).permutation(len(kc))
    n = int(frac * len(kc))
    return [kc[perm[k * n:(k + 1) * n]].tolist() for k in range(n_cs)]


def approach_score(sweet_hz_now, sweet_hz_naive, eps=0.2):
    """Оценка приближения к CS = насколько упал ответ MBON сладких отсеков против наивного (МОДЕЛЬ, по
    Hige 2015: награда ослабляет KC->MBON). 1 = MBON замолчали, 0 = как до обучения, < 0 = выросли.
    c7 (5% KC 20 Hz, окно 1 с, n=10): наивно 2.05 +- 0.14 Гц; после x0.3 на CS+ 0.31 +- 0.10 (score 0.85),
    CS- 1.94 -> 2.08 (score -0.07). Окно 200 мс шумнее: 2.21 +- 0.52 -> 0.69 +- 0.39."""
    return 1.0 - float(sweet_hz_now) / max(float(sweet_hz_naive), eps)


def choice_turn(score_L, score_R, gain=3.0):
    """Мост «выход MB -> руль» для игры лево/право (МОДЕЛЬ, в коннектоме такого провода нет):
    >0 = влево. Складывать с поворотом от DNa (senses.Behaviour.turn) ПОСЛЕ вычитания
    врождённого левого смещения DNa (фон +11.2 Гц)."""
    return float(np.tanh(gain * (score_L - score_R)))
