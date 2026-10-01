"""Органы чувств мухи для TikTok-реакций.

Какими НАСТОЯЩИМИ сенсорными нейронами коннектома FlyWire мы передаём мухе то, что
происходит в ролике, и с каких нейронов читаем её реакции. Все числа ниже измерены
в tiktok/_scratch/exp*.py (логи там же) на мозге с противоэпилептикой
PRM = dict(U=0.1, tau_d=300, a_inc=0.1, tau_a=3000). Ни один канал не даёт лавины:
эхо после стимула = 0 нейронов, максимум активных за 18 мс — сотни/2-3 тыс. (T4/T5).

Проход 1 (пример):
    import brain as B, senses as S
    ann = B.load_annotations(); brain = B.FlyBrain(params=S.PRM); brain.enable_graph()
    groups = S.build_groups(brain, ann)
    filt, beh = S.Filt(), S.Behaviour()
    feats["onset"] = S.onset_envelope(feats["onset"], 30)   # удар звучит ~100 мс, а не 1 кадр
    for k in range(n_frames):                         # кадры 30 fps
        ms = S.frame_ms(k, 30)                        # 32.4 или 34.2 мс (целые блоки по 1.8 мс)
        rates = S.inputs_from_frame(features[k], stim_state)
        S.apply_inputs(brain, groups, rates)
        hz = brain.run(ms) * (1000.0 / ms)            # numpy (N,), Гц на нейрон
        f = filt.update(S.readouts(hz, groups), ms / 1000)
        body = beh.update(f, k / 30)                  # поворот, голова, хоботок, чистка, взлёт…

Скорость (быстрое ядро brain.py): 150 с клипа ~6 с мозга+senses (1.3 мс/кадр: мозг ~1.1, остальное ~0.2);
apply_inputs пишет все входы одной операцией, если groups — из build_groups (иначе медленнее, ~2 мс/кадр).
Повторяемость: одинаковый прогон даёт только НОВЫЙ B.FlyBrain(params=S.PRM, seed=...);
brain.reset() не сбрасывает ГСЧ, поэтому второй прогон после reset() немного другой (5 vs 6 взлётов).
Руль: у коннектома своя асимметрия — одинаковые объекты с двух сторон дают DNa L 28 / R 11 Гц,
на клипе поворот влево (>0.3) 22% времени, вправо 13%. Это свойство модели, не баг кода.

Итог по каналам (подробности — в комментариях у SENSES/THRESH):
  LC10 (объект)      -> руль DNa02 ипсилатерально, 15 Гц уже насыщение; за 5 с гаснет 47 -> 21 (не до нуля)
  T4/T5+HS/H2 (поток) -> синдирекциональный поворот (картинка влево -> влево), |flow| 0.5 -> руль ~22,
                        голова DNp15 ~130; за 5 с непрерывного потока руль гаснет ~70%, голова ~45%
  LPLC2+LC4 (надвиг.) -> GF 50-120 за 16-34 мс, не привыкает (5 повторов раз в 1-4 с одинаковы),
                        руль ОТ угрозы (надвигание слева -> вправо)
  JO-A/B (звук)       -> слуховые нейроны hear2; сильные удары -> GF 26-32 = вздрагивание, взлёта нет
  щетинки (обдув)     -> DNg84/DNg35 80-215 Гц, чистка = длительность стимула + ~70 мс
  сахар               -> MN9 < 35 мс; постоянный сахар гаснет за ~0.5 с, пульсы 4-5 Гц держат 3 с
  ходьба вперёд/назад (DNp09, MDN) ни от одного канала не включается -> шар крутить только рулём.

Подписи для честности: «упрощённая сетчатка» (признаки кадра -> LC10/LPLC2/LC4/T4/T5/HS/H2),
«упрощённое ухо» (громкость полос -> Джонстонов орган), «полёт на месте — муха привязана».
"""
import numpy as np

PRM = dict(U=0.1, tau_d=300, a_inc=0.1, tau_a=3000)
BLOCK_MS = 1.8                     # синаптическая задержка = блок CUDA Graph
MN9_ID = 720575940660219265
# 21 сахарный вкусовой нейрон из Shiu et al. 2024 (в модели v783 есть 20; тип LB3)
SUGAR_IDS_630 = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
                 720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
                 720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
                 720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
                 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
                 720575940611875570]
LC10 = ["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"]
LOOM = ["LPLC2", "LC4"]
FTB_LPT = ["HSE", "HSN", "HSS", "VCH", "DCH"]      # тангенциальные клетки «спереди назад»
BTF_LPT = ["H1", "H2"]                             # тангенциальные клетки «сзади вперёд»

# ---------------------------------------------------------------------------
# ВХОДЫ. max_hz — частота при признаке = 1. Отбор: types (+side) по cell_type,
# prefix — cell_type начинается с…, ids — root_id FlyWire.
# ---------------------------------------------------------------------------
SENSES = {
    # --- зрение (упрощённая сетчатка) ---
    # LC10: насыщение уже к 15 Гц (руль L-R: 5 Гц -> 15, 10 -> 38, 15 -> 45, 30 -> 49) -> max 20
    "obj_L": dict(types=LC10, side="left", max_hz=20.0, ru="мелкий движущийся объект слева -> LC10"),
    "obj_R": dict(types=LC10, side="right", max_hz=20.0, ru="мелкий движущийся объект справа -> LC10"),
    # LPLC2+LC4: ОЧЕНЬ чувствительны — 12 Гц x 300 мс уже даёт GF > 45 (взлёт)
    "loom_L": dict(types=LOOM, side="left", max_hz=60.0, ru="надвигание слева -> LPLC2 + LC4"),
    "loom_R": dict(types=LOOM, side="right", max_hz=60.0, ru="надвигание справа -> LPLC2 + LC4"),
    # оптомоторика. Картинка едет ВЛЕВО = левый глаз видит «спереди назад» (T4a/T5a -> HS/CH),
    # правый — «сзади вперёд» (T4b/T5b -> H1/H2). По связям: T4a/T5a -> HS ~26 тыс. синапсов,
    # T4b/T5b -> H2 ~25 тыс. Только все 4 группы вместе дают сильный синдирекциональный поворот.
    "ftb_L": dict(types=["T4a", "T5a"], side="left", max_hz=40.0, ru="движение спереди назад, левый глаз -> T4a/T5a"),
    "ftb_R": dict(types=["T4a", "T5a"], side="right", max_hz=40.0, ru="движение спереди назад, правый глаз -> T4a/T5a"),
    "btf_L": dict(types=["T4b", "T5b"], side="left", max_hz=40.0, ru="движение сзади вперёд, левый глаз -> T4b/T5b"),
    "btf_R": dict(types=["T4b", "T5b"], side="right", max_hz=40.0, ru="движение сзади вперёд, правый глаз -> T4b/T5b"),
    "lpt_ftb_L": dict(types=FTB_LPT, side="left", max_hz=100.0, ru="клетки HS/CH левой доли"),
    "lpt_ftb_R": dict(types=FTB_LPT, side="right", max_hz=100.0, ru="клетки HS/CH правой доли"),
    "lpt_btf_L": dict(types=BTF_LPT, side="left", max_hz=100.0, ru="клетки H1/H2 левой доли"),
    "lpt_btf_R": dict(types=BTF_LPT, side="right", max_hz=100.0, ru="клетки H1/H2 правой доли"),
    # --- слух: Джонстонов орган антенны (упрощённое ухо) ---
    # JO-A сильнее бьёт в гигантское волокно (вздрагивание), JO-B — в слуховые нейроны AMMC/WED.
    # JO-A 80 Гц: GF 11, hear2 4;  JO-B 80 Гц: GF 3.5, hear2 13;  оба 80: GF 22, hear2 16.
    "hear_A": dict(prefix=["JO-A"], max_hz=30.0, ru="Джонстонов орган, зона A (звук/вибрация)"),
    "hear_B": dict(prefix=["JO-B"], max_hz=80.0, ru="Джонстонов орган, зона B (звук, низкие частоты)"),
    # --- обдув: щетинки головы (они и запускают чистку) + отклонение антенн ветром ---
    "puff": dict(types=["BM_Ant", "BM_Fr", "BM_FrOr"], max_hz=60.0, ru="механорецепторы-щетинки антенн и лба"),
    "wind": dict(prefix=["JO-C", "JO-E"], max_hz=40.0, ru="Джонстонов орган, зоны C/E (ветер)"),
    # --- вкус ---
    "sugar": dict(ids=SUGAR_IDS_630, max_hz=300.0, ru="сахарные вкусовые нейроны хоботка"),
    # горькое: 65 горьких GRN (cell_sub_class bitter). Проверка 30.09: хоботок (MN9) НЕ гасит,
    # зато 100 Гц -> гиг. волокно ~35 Гц (вздрагивание), 200 Гц -> ~55 (рывок) — это и показываем
    "bitter": dict(sub_class="bitter", max_hz=100.0, ru="горькие вкусовые нейроны"),
}
FLOW_DEADZONE = 0.12    # |flow_x| меньше — тряска камеры, не стимул
LOOM_MIN = 0.3          # loom ниже — шум, не надвигание (LPLC2 очень избирательны, а GF срабатывает
                        # уже от 12 Гц x 300 мс). На клипе со шкалой analyze.py v3: взлётов за 150 с
                        # LOOM_MIN 0 -> 12, 0.3 -> 6, 0.6 -> 4 (со старым eye v1 было 46 / 37 / 28)
ONSET_A, ONSET_B = 100.0, 40.0  # добавка к JO-A/JO-B на ударе (onset = 1); onset — огибающая, см. onset_envelope
HEAR_CAP = 130.0
PUFF_DUR, PUFF_TAU = 1.5, 0.8   # с: обдув 60*exp(-t/0.8) -> чистка 1.1-1.3 с (окно 0.6 с -> чистка ~0.65 с)
SUGAR_DUR = 3.0                 # с: капля (пульсы 4 Гц, как касания при питье)
SUGAR_PULSE = (0.25, 0.10)      # период, длительность «касания», с
BITTER_DUR = 1.5                # с: горькая капля (те же касания)

# ---------------------------------------------------------------------------
# ВЫХОДЫ (средняя частота группы, Гц)
# ---------------------------------------------------------------------------
READOUTS = {
    "steer_L": dict(types=["DNa01", "DNa02"], side="left", ru="руль влево (DNa01/DNa02)"),
    "steer_R": dict(types=["DNa01", "DNa02"], side="right", ru="руль вправо"),
    "head_L": dict(types=["DNp15"], side="left", ru="голова влево (DNp15 = DNHS1, вход от HS)"),
    "head_R": dict(types=["DNp15"], side="right", ru="голова вправо"),
    "gf": dict(types=["DNp01"], ru="гигантское волокно DNp01 (взлёт / вздрагивание)"),
    "fear": dict(types=["DNp01", "DNp02", "DNp04", "DNp11"], ru="нейроны побега DNp"),
    "mn9": dict(ids=[MN9_ID], ru="мотонейрон хоботка MN9"),
    "groom": dict(types=["DNg84", "DNg35"], ru="команда чистки антенн DNg84/DNg35"),
    "walk": dict(types=["DNp09"], ru="ходьба вперёд (P9) — ни один канал её не включает"),
    "back": dict(types=["MDN"], ru="ход назад (moonwalker) — ни один канал её не включает"),
    "hear2": dict(auto="jo_ab_top60", ru="слуховые нейроны мозга (AMMC/WED, 2-й порядок)"),
    "hear_jo": dict(prefix=["JO-A", "JO-B"], ru="сами рецепторы уха"),
}

# ---------------------------------------------------------------------------
# ПОРОГИ (по сглаженным Filt: tau 60 мс; mn9 — 200 мс; hear2 — 150 мс)
# ---------------------------------------------------------------------------
THRESH = dict(
    takeoff=45.0,      # gf > 45 -> взлёт (на месте). Надвигание 12-80 Гц x 300 мс даёт 49-119; звук — нет
    startle=25.0,      # 25 < gf <= 45 -> вздрагивание (удары с onset_envelope: пики 26-32, ~8-15 на 150 с)
    eat_on=8.0,        # mn9 > 8 -> хоботок; пульсы сахара держат > 8 99% времени
    eat_keep=3.0,
    groom_on=35.0,     # щетинки 30-120 Гц дают 80-215, старт через 16-52 мс
    groom_keep=12.0,
    turn_gain=0.04,    # turn = tanh(0.04*(steer_L-steer_R)); >0 = влево. 0.12 из show.py насыщается уже при 20
    head_gain=0.008,   # head = tanh(0.008*(head_L-head_R)); DNp15 до ~200 Гц
    hear_full=14.0,    # hear2 при «громко» (шкала «Слух»; на клипе hear2 ср ~6, 95% ~11.5)
)


def _sel(brain, ann, d):
    if "ids" in d:
        return [brain.idx[r] for r in d["ids"] if r in brain.idx]
    if "sub_class" in d:
        a = ann[ann.cell_sub_class.astype(str) == d["sub_class"]]
        return [brain.idx[r] for r in a.index if r in brain.idx]
    ct = ann.cell_type.astype(str)
    if "prefix" in d:
        m = np.zeros(len(ann), bool)
        for p in d["prefix"]:
            m |= ct.str.startswith(p).values
    else:
        m = ct.isin(d["types"]).values
    a = ann[m]
    if d.get("side"):
        a = a[a.side == d["side"]]
    return [brain.idx[r] for r in a.index if r in brain.idx]


def _hear2(brain, ann, k=60):
    """Слуховые нейроны 2-го порядка: топ-k по возбуждающему входу от JO-A/B (по матрице
    связей), без сенсорных и нисходящих. 45 из 60 совпадают с реальными ответчиками."""
    import torch
    jo = _sel(brain, ann, dict(prefix=["JO-A", "JO-B"]))
    x = torch.zeros(brain.N, device=brain.dev)
    x[jo] = 1.0
    w_in = torch.mv(brain.W, x).cpu().numpy()
    bad = ann.index[ann.super_class.isin(["sensory", "sensory_ascending", "descending"])]
    w_in[[brain.idx[r] for r in bad if r in brain.idx]] = -1
    return [int(i) for i in np.argsort(-w_in)[:k]]


class Groups(dict):
    """Обычный dict имя -> список индексов; плюс место под кэш индексов на GPU для apply_inputs."""
    _inp = None

    def __getstate__(self):          # кэш с GPU-тензорами не пиклим (его пересоберёт apply_inputs)
        return {}


def build_groups(brain, ann):
    """-> dict имя -> список индексов нейронов (входы SENSES + выходы READOUTS)."""
    g = Groups({k: _sel(brain, ann, d) for k, d in SENSES.items()})
    for k, d in READOUTS.items():
        g[k] = _hear2(brain, ann) if d.get("auto") == "jo_ab_top60" else _sel(brain, ann, d)
    empty = [k for k, v in g.items() if not v]
    assert not empty, f"пустые группы: {empty}"
    return g


def frame_ms(k, fps=30):
    """Длительность шага мозга для кадра k: целое число блоков по 1.8 мс (граф),
    в среднем ровно 1000/fps (30 fps -> 18 или 19 блоков)."""
    per = 1000.0 / fps / BLOCK_MS
    return (round((k + 1) * per) - round(k * per)) * BLOCK_MS


# ---------------------------------------------------------------------------
def _t(v):
    """stim_state[...]: число (с от начала), dict {'t': с, ...} или None."""
    if v is None or v is False:
        return None, {}
    if isinstance(v, (bool, np.bool_)):
        # True/np.True_ — не время: float(True) = 1 с давал бы каждый кадр одну и ту же фазу стимула
        if not v:
            return None, {}
        raise TypeError("stim_state: нужно время с начала стимула (с) или dict {'t': ...}, а не True")
    if isinstance(v, dict):
        return (None if v.get("t") is None else float(v["t"])), v
    return float(v), {}


def onset_envelope(onset, fps=30, tau=0.1):
    """Покадровые пики onset -> огибающая «пик + затухание tau с» (удар звучит ~100 мс).
    Одиночный кадр (33 мс) почти не доходит до гигантского волокна; с огибающей
    сильные удары дают GF 26-32 -> вздрагивание. Применять ко всей колонке до прохода 1."""
    x = np.nan_to_num(np.asarray(onset, dtype=float))     # NaN = нет удара (иначе рвёт хвост огибающей)
    y = np.empty_like(x)
    k, acc = np.exp(-1.0 / (fps * tau)), 0.0
    for i, v in enumerate(x):
        acc = max(float(v), acc * k)
        y[i] = acc
    return y


def sugar_rate(t):
    """Частота сахарных нейронов через t с после начала кормления: ритмичные касания
    4 Гц по 100 мс (как при питье). Постоянные 150-350 Гц дают MN9 только первые ~0.5 с,
    потом он гаснет (адаптация); пульсы держат MN9 ~25 Гц все 3 с."""
    per, on = SUGAR_PULSE
    return SENSES["sugar"]["max_hz"] if (t % per) < on else 0.0


def puff_rate(t):
    return SENSES["puff"]["max_hz"] * np.exp(-t / PUFF_TAU)


def inputs_from_frame(f, stim_state=None):
    """Признаки кадра/звука -> частоты сенсорных групп (Гц), ключи = SENSES.

    f (dict / pandas Series); признаки 0..1, кроме flow_x (-1..1). Нет ключа = 0.
      obj_L, obj_R   — мелкий движущийся объект в левой/правой половине экрана
      loom_L, loom_R — надвигание (быстро растущее тёмное пятно)
      flow_x         — общий горизонтальный сдвиг картинки; < 0 = картинка едет ВЛЕВО
                       (камера панорамирует вправо) -> муха поворачивает ВЛЕВО (за картинкой)
      hear           — энергия 100-500 Гц (analyze.py: упор на 150-250 Гц, песня ухаживания), 0..1 по клипу
      bass           — энергия басов 40-160 Гц (analyze.py), 0..1
      onset          — сила удара, 0..1, лучше после onset_envelope (иначе удары почти не слышны GF)
    stim_state: {'sugar': t | {'t': t, 'dur': с, 'strength': 0..1},
                 'puff':  t | {'t': t, 'dur': с, 'strength': 0..1}}
      t — сколько секунд идёт стимул (0 = только что начался); None/нет ключа — нет стимула.
    """
    def get(k):
        v = f.get(k, 0.0) if f is not None and hasattr(f, "get") else 0.0
        v = float(v) if v is not None else 0.0
        return v if np.isfinite(v) else 0.0          # NaN из таблицы признаков = нет сигнала

    S = SENSES
    r = {k: 0.0 for k in S}
    # --- зрение ---
    for side in "LR":
        r["obj_" + side] = S["obj_" + side]["max_hz"] * float(np.clip(get("obj_" + side), 0, 1))
        lv = (get("loom_" + side) - LOOM_MIN) / (1.0 - LOOM_MIN)
        r["loom_" + side] = S["loom_" + side]["max_hz"] * float(np.clip(lv, 0, 1))
    fx = float(np.clip(get("flow_x"), -1, 1))
    m = max(0.0, (abs(fx) - FLOW_DEADZONE) / (1 - FLOW_DEADZONE))
    if m > 0:
        ftb, btf = ("L", "R") if fx < 0 else ("R", "L")      # картинка влево: левый глаз FtB
        for k in ("ftb_" + ftb, "lpt_ftb_" + ftb, "btf_" + btf, "lpt_btf_" + btf):
            r[k] = S[k]["max_hz"] * m
    # --- слух ---
    on = float(np.clip(get("onset"), 0, 1))
    r["hear_A"] = min(HEAR_CAP, S["hear_A"]["max_hz"] * float(np.clip(get("hear"), 0, 1)) + ONSET_A * on)
    r["hear_B"] = min(HEAR_CAP, S["hear_B"]["max_hz"] * float(np.clip(get("bass"), 0, 1)) + ONSET_B * on)
    # --- стимулы-награды ---
    st = stim_state or {}
    tp, dp = _t(st.get("puff"))
    if tp is not None and 0 <= tp < dp.get("dur", PUFF_DUR):
        a = float(dp.get("strength", 1.0))
        r["puff"] = float(puff_rate(tp) * a)
        r["wind"] = float(S["wind"]["max_hz"] * np.exp(-tp / PUFF_TAU) * a)
    ts, ds = _t(st.get("sugar"))
    if ts is not None and 0 <= ts < ds.get("dur", SUGAR_DUR):
        r["sugar"] = float(sugar_rate(ts) * float(ds.get("strength", 1.0)))
    tb, db = _t(st.get("bitter"))
    if tb is not None and 0 <= tb < db.get("dur", BITTER_DUR):
        per, on = SUGAR_PULSE
        r["bitter"] = S["bitter"]["max_hz"] * float(db.get("strength", 1.0)) if (tb % per) < on else 0.0
    return r


def _input_plan(brain, groups):
    """Кэш для apply_inputs: все индексы входов одним тензором на устройстве мозга +
    номер группы для каждого индекса. None -> медленный путь (set_input по группам)."""
    import torch
    rate_in = getattr(brain, "rate_in", None)
    if not isinstance(groups, Groups) or not isinstance(rate_in, torch.Tensor):
        return None
    key = (str(rate_in.device), tuple(id(groups[k]) for k in SENSES), tuple(len(groups[k]) for k in SENSES))
    if groups._inp is not None and groups._inp[0] == key:
        return groups._inp[1]
    flat = [i for k in SENSES for i in groups[k]]
    plan = None
    if len(flat) == len(set(flat)):          # группы не пересекаются -> порядок записи не важен
        rep = [j for j, k in enumerate(SENSES) for _ in groups[k]]
        plan = (torch.tensor(flat, dtype=torch.long, device=rate_in.device),
                torch.tensor(rep, dtype=torch.long, device=rate_in.device))
    groups._inp = (key, plan)
    return plan


def apply_inputs(brain, groups, rates):
    """Выставить частоты всем группам SENSES (отсутствующие -> 0).
    Для групп из build_groups — одна запись в brain.rate_in (~0.1 мс вместо ~2 мс на кадр
    из 17 вызовов set_input со списками индексов)."""
    plan = _input_plan(brain, groups)
    if plan is None:
        for k in SENSES:
            brain.set_input(groups[k], float(rates.get(k, 0.0)))
        return
    import torch
    idx, rep = plan
    vals = torch.tensor([float(rates.get(k, 0.0)) for k in SENSES], dtype=brain.rate_in.dtype)
    brain.rate_in[idx] = vals.to(idx.device)[rep]


def readouts(hz, groups):
    """hz: numpy (N,) частоты всех нейронов (Гц) за последнее окно.
    -> dict средних частот групп READOUTS + steer (L-R), head (L-R), n_active."""
    out = {k: float(hz[groups[k]].mean()) for k in READOUTS}
    out["steer"] = out["steer_L"] - out["steer_R"]
    out["head"] = out["head_L"] - out["head_R"]
    out["n_active"] = int((hz > 0).sum())
    return out


class Filt:
    """Экспоненциальное сглаживание readouts: tau 60 мс (как show.py);
    MN9 — один нейрон, ему 200 мс, иначе хоботок дёргается; hear2 — 150 мс."""

    def __init__(self, tau=0.06, tau_over=None):
        self.tau = tau
        self.tau_over = dict(mn9=0.2, hear2=0.15, **(tau_over or {}))
        self.v = {}

    def update(self, r, dt):
        for k, x in r.items():
            if k == "n_active":
                self.v[k] = x
                continue
            a = min(1.0, dt / self.tau_over.get(k, self.tau))
            self.v[k] = self.v.get(k, 0.0) + a * (float(x) - self.v.get(k, 0.0))
        return dict(self.v)


class Behaviour:
    """Сглаженные частоты -> команды телу. Только пороги, никаких «правил реакции».
    update(f, t) -> dict(turn -1..1 (>0 влево), head -1..1 (>0 влево, как head_turn в stage.py),
                         proboscis 0..1, groom bool,
                         flight bool, hear 0..1, startle_age с,
                         takeoff/startle/eat_start/groom_start bool (события кадра)).
    GF растёт за 1-2 кадра, поэтому перед взлётом часто есть startle на кадр раньше:
    проход 1 офлайн — можно убрать startle, если в следующие 0.15 с был takeoff."""

    def __init__(self, th=None, flight_s=0.9, gf_cool=1.5, startle_cool=0.4):
        self.th = dict(THRESH, **(th or {}))
        self.flight_s, self.gf_cool, self.startle_cool = flight_s, gf_cool, startle_cool
        self.t_flight, self.t_startle = -99.0, -99.0
        self.eating = self.grooming = False

    def update(self, f, t):
        th = self.th
        ev = dict(takeoff=False, startle=False, eat_start=False, groom_start=False)
        gf = f.get("gf", 0.0)
        if gf > th["takeoff"] and t - self.t_flight > self.flight_s + self.gf_cool:
            self.t_flight, ev["takeoff"] = t, True
        elif th["startle"] < gf <= th["takeoff"] and t - self.t_startle > self.startle_cool \
                and t - self.t_flight > self.flight_s:
            self.t_startle, ev["startle"] = t, True
        flight = t - self.t_flight < self.flight_s
        mn9 = f.get("mn9", 0.0)
        eat = mn9 > th["eat_on"] or (self.eating and mn9 > th["eat_keep"])
        ev["eat_start"] = eat and not self.eating
        self.eating = eat
        g = f.get("groom", 0.0)
        groom = (not flight) and (g > th["groom_on"] or (self.grooming and g > th["groom_keep"]))
        ev["groom_start"] = groom and not self.grooming
        self.grooming = groom
        return dict(turn=float(np.tanh(th["turn_gain"] * f.get("steer", 0.0))),
                    head=float(np.tanh(th["head_gain"] * f.get("head", 0.0))),
                    proboscis=float(np.clip(mn9 / 25.0, 0, 1)) if eat else 0.0,
                    groom=groom, flight=flight, startle_age=t - self.t_startle,
                    hear=float(np.clip(f.get("hear2", 0.0) / th["hear_full"], 0, 1)), **ev)
