# Результаты и сырые данные опытов

Здесь — логи и таблицы первых опытов, кадры шоу. Главные исследования с выводами лежат рядом с кодом:

| тема | документ | графики | данные |
|---|---|---|---|
| память в грибовидном теле, опыт Павлова | [learning/README.md](../learning/README.md) | [learning/figures/](../learning/figures/) | — |
| карта грибовидного тела: отсеки, учителя, валентность | [learning/MB_MAP.md](../learning/MB_MAP.md) | — | `learning/mb_map.json`, `learning/mb_func.json` |
| игра 67COIN: наивная муха | [learning/game/BASELINE.md](../learning/game/BASELINE.md) | `learning/game/baseline.png` | `learning/game/logs/baseline_*.json` — 35 партий |
| игра 67COIN: обучение, контроли, носитель памяти | [learning/game/TRAINING.md](../learning/game/TRAINING.md) | [learning/game/figures/](../learning/game/figures/) | `learning/game/memory/`, `learning/game/logs/play_*.json` |
| обученная муха | [learning/game/TRAINING.md](../learning/game/TRAINING.md) | — | `learning/game/official/fly67_v1.npz` (+ описание `.json`, контрольные партии `check_*.json`) |
| игра без подсказки, быстрый декодер | шапки `learning/career/nohint.py`, `speed67.py` | — | — |
| идеи на будущее | [learning/game/IDEAS.md](../learning/game/IDEAS.md) | — | — |

Ссылки в документах на `_scratch/…` — это черновые скрипты и логи прогонов, из которых взяты числа.
В публичную версию они не вошли (их сотни, и это рабочие черновики); итоговые числа и графики — здесь.

## `results/` — первые опыты (мозг без тела и в теле)

| файл | что |
|---|---|
| `check_brian2.log` | исходная модель на Brian2: запах 5 Гц на DM1 -> 8 162 активных нейрона («эпилепсия» есть и в оригинале) |
| `results_epilepsy.csv` | подбор «противоэпилептики»: торможение, адаптация, утомление синапсов -> хоботок, взлёт, руль, лавина от запаха |
| `results_exp1_dn.csv` | какие нисходящие нейроны отвечают на какие входы (без тела) |
| `results_sense_scan.log` | скан органов чувств: какие сенсорные группы вызывают лавину |
| `results_vpn_scan.csv`, `.log` | скан зрительных проекционных нейронов (VPN): что они включают |
| `run1.log`, `run2.log` | замкнутый цикл «мозг -> тело» (`exp2_closed_loop.py`) |
| `show_v1..3.log`, `fly_002..004.log` | шоу «муха на кухне»: события по времени. В `fly_004` мозг уже с «противоэпилептикой» (строка «параметры мозга» в начале лога) |

Как читать `results_epilepsy.csv`: строка — набор параметров (`inh` — множитель торможения, `a_inc` — адаптация,
`U`/`tau_d` — утомление синапсов). Дальше: частота MN9 на сахар, активных нейронов во время и **после** стимула
(`*_after` — остаток, который не погас), частота гигантского волокна на надвигание, руль на объект, лавина от запаха.

## `figures/show/` — кадры шоу

`show_frame.png` — общий вид; `v1_flight.png` — взлёт от шара; `groom_frames.png` — чистка; `proboscis_test.png` — хоботок;
`v3_grid.png`, `f1.png` — раскадровки.
