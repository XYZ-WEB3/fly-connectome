# Откуда что взято

Этот проект стоит на работе других людей. Здесь — что именно мы взяли у каждого и где это в коде.
Их данные и код в репозиторий **не скопированы**: `setup/get_data.py` скачивает закреплённые версии из первоисточников.

## Данные и модели

### Коннектом FlyWire (версия 783)
Полная карта связей мозга взрослой самки дрозофилы: кто с кем связан и сколькими синапсами.
- Dorkenwald S. et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138.
- Schlegel P. et al. (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634, 139–152.
- Matsliah A. et al. (2024). Neuronal parts list and wiring diagram for a visual system. *Nature* 634, 166–180.
- Сайт: [flywire.ai](https://flywire.ai). Условия использования данных — на сайте FlyWire.

### Модель мозга Shiu et al.
LIF-уравнения, все параметры нейронов и синапсов, таблица связей с знаками (`Connectivity_783.parquet`,
`Completeness_783.csv`). Наш `brain.py` — та же модель, переписанная на GPU; сверка с оригиналом — `check_brian2.py`.
- Shiu P.K. et al. (2024). A *Drosophila* computational brain model reveals sensorimotor processing. *Nature* 634, 210–219.
- Код: [github.com/philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model), коммит `91bdd1e`, лицензия MIT.

### Аннотации нейронов FlyWire
Тип клетки, класс, сторона, медиатор, координаты для каждого нейрона (`neuron_annotations.tsv`).
По ним мы находим LC10, DNa01/02, MN9, клетки Кеньона, MBON, PAM и т.д.
- [github.com/flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations), коммит `17fc577`.
- Авторы просят цитировать: Schlegel et al. (2024), Matsliah et al. (2024), Dorkenwald et al. (2024),
  Berg et al. (2026) — полные ссылки в README их репозитория.

### Тело мухи и физика
3D-модель NeuroMechFly, суставы, контроллер ходьбы (CPG + рефлексы), модель сложного глаза (721 фасетка),
окружение MuJoCo. Наши добавки (хоботок, крылья, чистка, сцены) — в [body_and_world.md](body_and_world.md).
- Wang-Chen S., Stimpfling V.A., Lam T.K.C., Özdil P.G., Genoud L., Hurtak F., Ramdya P. (2024).
  NeuroMechFly v2: simulating embodied sensorimotor control in adult *Drosophila*. *Nature Methods* 21, 2353–2362.
- Lobato-Rios V. et al. (2022). NeuroMechFly, a neuromechanical model of adult *Drosophila melanogaster*.
  *Nature Methods* 19, 620–627.
- Код: [github.com/NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym), лицензия Apache-2.0.
- Todorov E., Erez T., Tassa Y. (2012). MuJoCo: A physics engine for model-based control. *IROS*.
  [MuJoCo](https://github.com/google-deepmind/mujoco) и dm_control — Apache-2.0.

### Распознавание речи
Режим «по смыслу речи»: текст ролика по словам с временем.
- Radford A. et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML*. Модель Whisper large-v3 — MIT.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (SYSTRAN) — MIT.

## Биология обучения, на которую мы опирались

Правило памяти и выбор отсеков в `learning/` следуют этим работам. Что взято из литературы, а что наш
модельный выбор, — таблица в [learning/README.md](../learning/README.md).

- Aso Y. et al. (2014a). The neuronal architecture of the mushroom body provides a logic for associative learning. *eLife* 3, e04577.
- Aso Y. et al. (2014b). Mushroom body output neurons encode valence and guide memory-based action selection in *Drosophila*. *eLife* 3, e04580.
- Hige T., Aso Y., Modi M.N., Rubin G.M., Turner G.C. (2015). Heterosynaptic plasticity underlies aversive olfactory learning in *Drosophila*. *Neuron* 88, 985–998.
- Cohn R., Morantte I., Ruta V. (2015). Coordinated and compartmentalized neuromodulation shapes sensory processing in *Drosophila*. *Cell* 163, 1742–1755.
- Aso Y., Rubin G.M. (2016). Dopaminergic neurons write and update memories with cell-type-specific rules. *eLife* 5, e16135.
- Handler A. et al. (2019). Distinct dopamine receptor pathways underlie the temporal sensitivity of associative learning. *Cell* 178, 60–75.
- Li F. et al. (2020). The connectome of the adult *Drosophila* mushroom body provides insights into function. *eLife* 9, e62576.
- Brembs B., Plendl W. (2008). Double dissociation of PKC and AC manipulations on operant and classical learning in *Drosophila*. *Current Biology* 18, 1168–1171.

## Библиотеки

PyTorch, NumPy, pandas, PyArrow, SciPy, OpenCV, Pillow, PyAV, imageio и imageio-ffmpeg (FFmpeg), Matplotlib,
fontTools, psutil, tqdm — каждая под своей лицензией.

## Что сделали мы

- быстрая GPU-реализация модели (CUDA-ядра через NVRTC, событийная доставка, CUDA Graph) и проверка, что она
  считает побитно так же, как простой путь;
- «противоэпилептика» (утомление синапсов + адаптация) и её подбор;
- привязка органов чувств к нейронам коннектома и чтение поведения с нейронов;
- сцены, «кукла», хоботок, крылья, чистка, рендер и вёрстка видео, окно утилиты;
- правило обучения в грибовидном теле в коде, опыт Павлова с контролями, игра 67COIN, обучение мухи,
  проверки и повторы на новых «мухах», все графики, таблицы и выводы.
