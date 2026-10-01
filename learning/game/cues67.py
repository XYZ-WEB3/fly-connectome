"""Зрительные метки для игры 67COIN: «следующий тап — ЛЕВОЙ» / «— ПРАВОЙ» (задача B).

Метка = предмет с одной стороны поля зрения. Его «видят» два набора НАСТОЯЩИХ зрительных
проекционных нейронов (VPN) FlyWire v783 той же стороны:
  1. LC10 (LC10a, c-1, c-2, d, e; ~360 клеток на сторону) — детекторы мелких объектов. Родной путь
     к рулю DNa01/02: поворот К метке (ипсилатерально). Наивная муха отвечает сразу.
  2. VPN дополнительной чашечки MB (24 типа, 33-34 клетки на сторону: aMe12, MTe32, MTe30, LTe25, ...) —
     родной зрительный вход в клетки Кеньона KCg-d (γd). Даёт KC-код метки, к которому может
     «прилипнуть» память (KC->MBON). Руля сами почти не дают (±1-3 Гц).
  3. (выкл. по умолчанию) kc_boost — прямое включение KCg-d своей стороны. НЕ родное («как оптогенетика KC»).

Почему два набора: в коннектоме нет VPN, который делал бы и то и другое (b1_anat.log): у LC10a/d
прямых синапсов на KC 0, путь 1-3 хопа ~1e-5; у VPN чашечки путь к DNa ~1e-4 против 1e-2 у LC10.

Что РОДНОЕ и что НАШ ВЫБОР:
  * родное: сами нейроны (типы и сторона из аннотаций), все синапсы и путь метка -> KC -> MBON -> руль,
    путь LC10 -> DNa. Мост «память -> поведение» не нужен: руль читается с DNa.
  * наш выбор: (а) что метка включает ВСЕ 24 типа VPN чашечки своей стороны (их зрительная настройка
    не известна — это «упрощённая сетчатка», а не утверждение, что так ответит реальный предмет);
    (б) частоты: LC10 = strength x lc10_hz, VPN чашечки = cal_hz (постоянно, пока метка горит, чтобы
    KC-код не зависел от трудности); 150 Гц — верх физиологического диапазона;
    (в) kc_boost_hz > 0 — не родное.

ИЗМЕРЕНО (learning/game/_scratch/b1-b9, логи там же; PRM, выход DAN заглушен как в MBPlasticity, обучение выкл.,
метка 504 мс от покоя; «к метке» = DNa01+02 (L-R) для L-метки и (R-L) для R-метки; 5 seed x 3, если не сказано):
  LC10 без чашечки, к метке L / R, Гц (доля проб с верным знаком):
      2 Гц  +2.2 / +0.1 (1.00 / 0.07);  3 Гц  +5.5 / +0.6 (1.00 / 0.40);  4 Гц +12.5 / +4.6 (1.00 / 1.00);
      5 Гц +18.5 / +10.0 (1.00 / 1.00); 10 Гц +38 / +26; 20 Гц +48 / +33 (3 seed). Врождённый левый крен:
      R-метка слабее; ниже ~4 Гц R-метка часто не поворачивает.
  VPN чашечки (одна сторона), KC с >=1 спайком за 504 мс / MBON27 своей стороны / DA отсека calyx (порог 20):
      40 Гц 17 KC (0.3%) / 0 / 7;  60 Гц 47 (0.9%) / 0.4 / 12;  80 Гц 70 (1.4%) / 3.5-6 / 14;
      100 Гц 88 (1.7%) / 13 / 18-19;  120 Гц 106 (2.0%) / 23 / 26;  150 Гц 122 (2.4%) / 39 / 31-37.
      Код: KCg-d (+ несколько KCg-s1), 100% своей стороны; перекрытие кодов L и R (Жаккар) 0.00-0.03;
      разреженный среди всех KC (2.4% при 150 Гц), но плотный внутри KCg-d своей стороны (~120 из 147);
      повторяемость (Жаккар повторов) 0.8 (80 Гц) - 0.9 (150 Гц). Дофамин выше 10 Гц — ТОЛЬКО в отсеке calyx
      (учителя PPL201/202, MBON22); отсеки награды (α1, β'2a, β1, γ5, γ5d) — нет.
  Эпилепсия: эхо (активные нейроны через 0.5-1.5 с после метки) = 0 во всех ~3000 пробах (b2-b7); активных за метку
      300-1100 из 138 639; 12 с чередования L/R по 306 мс без reset — эхо 0, макс. ~400 активных за 18 мс.
  Запас обучения (память насыщена: все обучаемые KC->MBON отсеков награды у KC-кода метки -> 0.1·w0):
      LC10 3 Гц + VPN чашечки 150 Гц: наивно R-метка к метке -1.5 Гц, верно 27%; память на коде R-метки:
      +4.3 ± 0.8 Гц (p = 0.006), верно 87%; на кодах обеих: +4.7 ± 1.0 (p = 0.010), верно 93%;
      память только на коде L-метки R-метку не чинит (+1.2, p = 0.41). Нокаут MBON27 даёт то же (+2.6..+5.0) —
      рычаг = MBON27 (γ5d, учит PAM01): KCg-d -> MBON27 толкает руль ОТ метки, награда это ослабляет.
      LC10 2 Гц: R +6.3 (p = 0.002); LC10 4 Гц: R +4.6 (p = 0.005). LC10 без KC-кода (контроль): Δ от -0.9 до +0.7,
      значимо 1 из 9 тестов (-0.9 Гц, p = 0.009) — на уровне случайности (у LC10 нет KC-кода, механизма нет).
      При VPN чашечки 80-100 Гц рычага нет (Δ от -0.7 до +0.9, p >= 0.07); при LC10 >= 5 Гц наивная муха и так
      права в 100% — учиться нечему. (b3, b5, b6: LC10 3 Гц + чашечка 150 Гц — b6.)
  Игра без reset (b7, b8): метка чередуется L/R каждые 306 мс, 12 с, 5 seed. Прокси тапа: руль сглажен 60 мс,
      тап = вход в состояние стороны (триггер Шмитта ±T, возврат при |f| < T/2); доля меток с верным первым тапом:
          strength   только LC10   LC10 + VPN чашечки (эта метка)   то же, память насыщена (верх. граница)
          0.3           0.60              0.59                          0.72      (T = 3-5 Гц; ±0.01-0.04)
          0.4           0.80              0.75-0.80                     0.92
          0.5           0.98              0.91-0.97                     0.95      <- учиться нечему
          0.7           1.00              0.98-0.99                     1.00
      Руль к метке, память − наивно (strength 0.3): L +1.7 ± 0.4 Гц (p = 0.016), R +2.1 ± 0.3 (p = 0.0015).
      Руль за 12 с чередования гаснет на ~35% (адаптация), знак держится; KC-код стабилен (Жаккар нач./кон. 0.67).
  САМООБУЧЕНИЕ без награды: при cal_hz >= 120 сама метка гонит дофамин отсека calyx (PPL201/202 -> MBON22)
      выше порога 20 Гц (120 Гц: 26; 150 Гц: 31-37). За 12 с чередования с включённым обучением и без US
      память у KC метки: calyx 0.002-0.004, отсеки награды 0.000. Даже насыщенная память в calyx руль значимо
      не меняет (L +1.5, p = 0.36; R +0.2, p = 0.84). Для чистых контролей — freeze_compartments(pl, ["calyx"])
      (выбор модели: отсек calyx не учится) или cal_hz <= 100 (но тогда нет рычага обучения, см. выше).

Пример:
    import brain as B; from plasticity import MBPlasticity
    from cues67 import CueSet, freeze_compartments, GAME_STRENGTH
    b = B.FlyBrain(params=PRM, seed=0); b.enable_graph(); pl = MBPlasticity(b)
    cs = CueSet(b, B.load_annotations())           # LC10 x strength·10 Гц + VPN чашечки 150 Гц
    freeze_compartments(pl, ["calyx"])             # по желанию: метка сама не учит отсек calyx
    cs.apply("L", strength=GAME_STRENGTH)          # метка слева; 0.4 -> LC10 4 Гц: наивно ~0.8 верных тапов
    c = b.run(18, tensor=True); steer = cs.steer(c, 18)   # DNa01+02 L-R, Гц
    cs.off()
После b.reset() метку ставить заново (reset обнуляет rate_in).
"""
import numpy as np
import torch

LC10_TYPES = ["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"]          # как tiktok/senses.py
# VPN дополнительной чашечки: >= 100 прямых синапсов на KC (обе стороны вместе) и >= 70% — на KC своей
# стороны (b1_anat.py по Connectivity_783; MTe07 выпал: 74% его синапсов на KC чужой стороны)
CAL_TYPES = ["LTe02", "LTe04", "LTe16", "LTe25", "LTe40", "LTe50", "LTe51", "LTe72", "LTe73",
             "MTe12", "MTe22", "MTe26", "MTe28", "MTe30", "MTe32", "MTe34", "MTe35", "MTe37", "MTe38", "MTe40",
             "aMe12", "aMe20", "aMe26", "aMe9"]
KC_BOOST_TYPES = ["KCg-d"]                                                   # не родное (как оптогенетика)
DNA_TYPES = ["DNa01", "DNa02"]
GAME_STRENGTH = 0.4      # трудность для игры: наивно ~0.75-0.80 верных тапов, верхняя граница после обучения ~0.92 (b8)

NATIVE = {
    "LC10": "родное: VPN детекторы мелких объектов -> руль DNa к метке",
    "CAL": "родное: VPN дополнительной чашечки -> KCg-d своей стороны (KC-код метки); частоты — наш выбор",
    "KCgd": "НЕ родное: прямое включение KCg-d своей стороны («как оптогенетика KC»)",
}
SIDES = {"L": "left", "R": "right", "left": "left", "right": "right"}


class CueSet:
    """Метки «тап ЛЕВОЙ / ПРАВОЙ». Группы строятся по cell_type + side из аннотаций FlyWire.

    lc10_hz      — частота LC10 при strength = 1 (strength 0.3 -> 3 Гц: трудный режим, см. шапку);
    cal_hz       — частота VPN чашечки, пока метка горит (не зависит от strength, если cal_follows=False);
    kc_boost_hz  — прямое включение KCg-d своей стороны (0 = выкл.; НЕ родное; > 20 Гц зажигает PPL1 α'1/γ2
                   выше порога — метка начнёт сама «наказывать», поэтому не больше 20).
    """

    def __init__(self, brain, ann, lc10_hz=10.0, cal_hz=150.0, kc_boost_hz=0.0, cal_follows=False,
                 lc10_types=LC10_TYPES, cal_types=CAL_TYPES):
        assert 0 <= kc_boost_hz <= 20, "kc_boost_hz > 20 Гц: KCg-d зажигают PPL1 выше порога (b4)"
        self.b = brain
        self.lc10_hz, self.cal_hz, self.kc_boost_hz, self.cal_follows = float(lc10_hz), float(cal_hz), \
            float(kc_boost_hz), bool(cal_follows)
        types = {"LC10": list(lc10_types), "CAL": list(cal_types), "KCgd": KC_BOOST_TYPES, "DNa": DNA_TYPES}
        self.groups = {}
        for g, tl in types.items():
            for s in "LR":
                self.groups[f"{g}_{s}"] = self._sel(ann, tl, SIDES[s])
        empty = [k for k, v in self.groups.items() if len(v) == 0]
        assert not empty, f"пустые группы: {empty}"
        self.cue_groups = [k for k in self.groups if not k.startswith("DNa")]
        self._all = torch.cat([self.groups[k] for k in self.cue_groups])     # все входы меток (для off)
        self.side = None
        self.strength = 0.0

    def _sel(self, ann, types, side):
        a = ann[ann.cell_type.isin(types) & (ann.side == side)]
        ix = sorted(self.b.idx[r] for r in a.index if r in self.b.idx)
        return torch.tensor(ix, dtype=torch.long, device=self.b.dev)

    # ---------------- входы ----------------
    def cue_rates(self, side, strength=1.0):
        """-> dict группа -> Гц для метки на стороне side ('L'/'R'); группы другой стороны = 0.
        strength 0..1 меняет только LC10 (и VPN чашечки, если cal_follows=True). side=None или strength=0 -> всё 0."""
        out = {k: 0.0 for k in self.cue_groups}
        s = float(np.clip(strength, 0.0, 1.0))
        if side is None or s == 0.0:
            return out
        S = "L" if SIDES[side] == "left" else "R"
        out[f"LC10_{S}"] = self.lc10_hz * s
        out[f"CAL_{S}"] = self.cal_hz * (s if self.cal_follows else 1.0)
        out[f"KCgd_{S}"] = self.kc_boost_hz
        return out

    @torch.no_grad()
    def apply(self, side, strength=1.0):
        """Поставить метку в brain.rate_in (другие входы мозга не трогает)."""
        r = self.b.rate_in
        for k, hz in self.cue_rates(side, strength).items():
            r[self.groups[k]] = float(hz)
        self.side, self.strength = side, float(strength)

    def off(self):
        self.b.rate_in[self._all] = 0.0
        self.side, self.strength = None, 0.0

    # ---------------- чтение ----------------
    @torch.no_grad()
    def steer(self, counts, ms):
        """Руль (родной): средняя частота DNa01+02 левых минус правых, Гц; > 0 = влево."""
        return float(counts[self.groups["DNa_L"]].float().mean() - counts[self.groups["DNa_R"]].float().mean()) * 1000.0 / ms

    def describe(self):
        """Таблица групп: имя, клеток, родное или нет."""
        rows = []
        for k, v in self.groups.items():
            g = k.split("_")[0]
            rows.append(dict(group=k, n=len(v), native=g != "KCgd",
                             role=NATIVE.get(g, "выход: руль DNa01+02 (родной)")))
        return rows


def freeze_compartments(pl, comps=("calyx",)):
    """Выбор модели: эти отсеки не учатся (дофамин их учителей обнуляется в правиле MBPlasticity).
    Нужен, чтобы метка VPN чашечки 120-150 Гц сама (без награды) не ослабляла KC->MBON22 (отсек calyx, PPL2).
    -> список замороженных отсеков."""
    done = []
    with torch.no_grad():
        for c in comps:
            if c in pl.comp_names:
                k = pl.comp_names.index(c)
                pl.T[2 * k:2 * k + 2] = 0.0
                done.append(c)
    return done
