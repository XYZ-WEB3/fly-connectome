"""Муха реагирует на ролик -> вертикальное видео для TikTok.

    python tiktok/react.py "видео на реакцию/клип.mp4" [--watermark @ник] [--title "..."] [--workers 5]
        [--format base|streamer] [--variant N|random] [--flyeye on|off]      (форматы: tiktok/FORMATS.md)
        [--meaning on|off]      (смысл речи -> сахар/горькое; off — только зрение и звук)
    превью окна из готового лога (без мозга):
    python tiktok/react.py "<клип>" --skip_pass1 --t0 12 --secs 8 --fly_no 999 --workers 1 --lock метка

Проход 1 (последовательно, GPU): анализ ролика -> план стимулов (сахар / обдув) ->
    органы чувств -> мозг (коннектом FlyWire, 139k нейронов) -> команды телу -> «кукла» на шаре
    -> лог состояний по кадрам.
Проход 2 (параллельно): рендер 3D-сцены по логу + вёрстка 1080x1920 + оригинальный звук.

Правдивость: каждая видимая реакция (взлёт, хоботок, чистка, поворот, «слушает») — только
по частотам нейронов модели. Мы решаем лишь КОГДА дать сахар/обдув (по острым моментам ролика);
зрение и слух — непрерывные входы.
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import formats as FM                                              # noqa: E402  (форматы: FORMATS.md)

COUNTER = ROOT / "data" / "fly_counter.json"
WATERMARK_LOCK = None             # сборка .exe: знак проекта зашит, --watermark не действует
VIS_KEYS = ("obj_L", "obj_R", "loom_L", "loom_R", "flow_x")      # = formats.VIS_KEYS (признаки зрения в лог)


class StageFactory:
    """Пиклится в воркеры рендера: создаёт сцену формата с нужной вотермаркой —
    stage.Stage (Базовый) или stage_streamer.StreamerStage (Стример, комната по RoomStyle(вариант))."""

    def __init__(self, watermark, fmt=None):
        self.watermark = watermark
        self.fmt = dict(fmt or {})                 # formats.fmt_cfg(...)

    def __call__(self):
        if self.fmt.get("format") == "streamer":
            import stage_streamer as SS
            import style as STY
            return SS.StreamerStage(STY.RoomStyle(self.fmt.get("variant") or 0), watermark=self.watermark)
        import stage as SG
        SG.WATERMARK = self.watermark
        return SG.Stage()


def next_fly_no(dry=False):
    n = (json.load(open(COUNTER)) if COUNTER.exists() else {"n": 0})["n"] + 1
    if not dry:
        json.dump({"n": n}, open(COUNTER, "w"))
    return n


# ---------------------------------------------------------------- режиссёр (ракурсы)
class Director:
    """Меняет ракурсы: по сильным моментам ролика и по реакциям мухи.
    Кадр держится 3..7 с, переход 0.7 с. Реакции показываем там, где муха крупнее."""
    REACT_SHOT = dict(takeoff="behind", eat_start="low_hero", groom_start="pov", startle="behind")

    def __init__(self, order, rng):
        self.order, self.rng = list(order), rng
        self.cur, self.prev = "behind", None
        self.t_switch, self.i = -10.0, 0

    def update(self, t, strong_moment, body):
        want = None
        for ev, shot in self.REACT_SHOT.items():
            if body.get(ev) and shot != self.cur and t - self.t_switch > 1.5:
                want = shot
        held = t - self.t_switch
        if want is None and ((strong_moment and held > 3.0) or held > 7.0):
            self.i = (self.i + 1) % len(self.order)
            want = self.order[self.i]
            if want == self.cur:
                self.i = (self.i + 1) % len(self.order); want = self.order[self.i]
        cut = False
        if want is not None and want != self.cur:
            self.prev, self.cur, self.t_switch, cut = self.cur, want, t, True
        blend = min(1.0, (t - self.t_switch) / 0.7)
        return self.prev, self.cur, blend, cut


# ---------------------------------------------------------------- проход 1
def _free_gpu():
    """После прохода 1 отдаём кэш torch на видеокарте (иначе он висит до конца рендера)."""
    import gc
    gc.collect()
    if "torch" in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


SEIZ_ODOR_HZ, SEIZ_ODOR_S = 50.0, 1.5      # резкий запах без защиты: 50 Гц на все ORN -> ~11 тыс. нейронов, не гаснет
SEIZ_SHOW_S, DEATH_S, RESPAWN_S = 4.0, 5.0, 3.0


def sad_moment(mean_ev, dur, win=8.0):
    """Грустный момент: окно win с, где больше всего «плохих» слов (не меньше 2). None — не нашли."""
    neg = sorted(e["t"] for e in mean_ev if e["kind"] == "neg")
    best, bt = 1, None
    for i, t in enumerate(neg):
        c = sum(1 for u in neg[i:] if u - t <= win)
        if c > best and t + SEIZ_SHOW_S + DEATH_S + RESPAWN_S < dur - 1:
            best, bt = c, t
    return bt


def pass1(video, log_dir, title, fly_no, seed=0, secs=None, verbose=True, seizure="off", meaning=True):
    import torch
    import brain as B
    import analyze as A
    import stimuli as ST
    import senses as S
    import stage as SG
    from render_parallel import StateLogWriter

    T0 = time.perf_counter()
    tl = A.analyze(video, fps_out=30, verbose=verbose)
    meta = tl["meta"]
    fps = 30
    n = meta["n_frames_out"] if secs is None else min(meta["n_frames_out"], int(secs * fps))
    mean_ev = []
    if not meaning:                                          # --meaning off: только зрение и звук
        print("[0] смысл речи выключен (--meaning off)", flush=True)
    else:
        try:                                                 # смысл речи -> сахар / горькое (meaning.py)
            import meaning as MN
            mean_ev = MN.meaning_events(video, verbose=verbose)
        except Exception as e:                               # нет речи / сбой распознавания — как раньше
            print(f"[0] смысл речи пропущен: {e}", flush=True)
    plan = ST.plan_stimuli(tl, seed=seed, max_gap=10, puff_gap=10, meaning=mean_ev)
    ticker = ST.ticker(tl, plan)
    t_seiz = None                                            # приступ: мы отключаем защиту и даём резкий запах
    seiz_why = "грустный момент"
    if seizure == "auto":
        t_seiz = sad_moment(mean_ev, meta["duration"])
        if t_seiz is None:                                   # слов нет (музыка) — самый резкий бас
            tail = SEIZ_SHOW_S + DEATH_S + RESPAWN_S + 1
            ev = [e for e in tl["events"] if e["kind"] in ("drop", "peak") and 3 < e["t"] < meta["duration"] - tail]
            if ev:
                e = max(ev, key=lambda e: (e["kind"] == "drop", e["strength"]))
                t_seiz, seiz_why = float(e["t"]), "самый резкий бас"
    elif seizure not in (None, "off"):
        t_seiz, seiz_why = None, "выбранный момент"
    # несколько приступов: «30, 75» — каждый со своей смертью и новой мухой; внахлёст не ставим
    if seizure in (None, "off"):
        seiz_list = []
    elif seizure == "auto":
        seiz_list = [] if t_seiz is None else [t_seiz]
    else:
        seiz_list = sorted(float(x) for x in str(seizure).replace(";", ",").split(",") if x.strip())
    kept = []
    for ts in seiz_list:
        if ts >= 0 and (not kept or ts >= kept[-1] + SEIZ_SHOW_S + DEATH_S + RESPAWN_S):
            kept.append(ts)
    seiz_list = kept
    for ts in seiz_list:
        ticker = [it for it in ticker if not (ts - 1 <= it["t"] <= ts + SEIZ_SHOW_S + DEATH_S + RESPAWN_S)]
        ticker.append(dict(t=round(ts, 3), dur=3.0, kind="seizure",
                           text=f"{seiz_why}: отключаем мухе защиту и даём резкий запах"))
    ticker.sort(key=lambda x: x["t"])
    if seiz_list:
        print("[1] приступы в " + ", ".join(f"{ts:.1f}" for ts in seiz_list) + " с", flush=True)
    onset = S.onset_envelope(np.asarray(tl["onset"], np.float32), fps=fps)
    keys = ["obj_L", "obj_R", "loom_L", "loom_R", "flow_x", "hear", "bass", "onset"]
    feats = {k: np.asarray(tl[k], np.float32) for k in keys}
    feats["onset"] = onset
    strong = np.zeros(n, bool)
    for e in tl["events"]:
        if e["kind"] in ("drop", "zoom", "peak") and e["strength"] > 0.6 and e["frame"] < n:
            strong[e["frame"]] = True
    ev_text = {}
    for it in ticker:
        k = int(round(it["t"] * fps))
        if k < n:
            ev_text[k] = it["text"]
    if verbose:
        print(f"[1] ролик: {meta['w']}x{meta['h']} {meta['orientation']}, {meta['duration']:.1f} с, "
              f"кадров: {n}; стимулов в плане: {len(plan)} "
              f"(сахар {sum(p['type'] == 'sugar' for p in plan)}, горькое {sum(p['type'] == 'bitter' for p in plan)}, обдув {sum(p['type'] == 'puff' for p in plan)})")

    # --- мозг ---
    ann = B.load_annotations()
    brain = B.FlyBrain(params=S.PRM, seed=seed)
    groups = S.build_groups(brain, ann)
    brain.enable_graph()
    filt, beh = S.Filt(), S.Behaviour()
    orn = np.where((ann["cell_class"].astype(str) == "olfactory").values)[0].tolist()
    life, fly_cur, respawned = "alive", fly_no, 0

    # --- тело и сцена (без рендера) ---
    SG.prepare_assets()
    st = SG.Stage()
    st.set_orientation(meta["orientation"], 0)
    director = Director(SG.Stage.PRESET_ORDER, np.random.default_rng(seed))

    writer = StateLogWriter(log_dir, meta=dict(fps=fps, title=title, fly_no=fly_no, source=str(video),
                                               src_t0=0.0, plan=plan, seizure_t=seiz_list))
    DUR = dict(sugar=S.SUGAR_DUR, puff=S.PUFF_DUR, bitter=S.BITTER_DUR)
    hunger, drop, groom_t0, dsign = 0.75, 0.0, None, 1.0
    life_prev = "alive"
    stats = dict(takeoff=0, startle=0, eat=0, groom=0)
    for k in range(n):
        t = k / fps
        # стимулы по плану (длительности — из опытов senses)
        stim = {}
        for p in plan:
            if p["t"] <= t < p["t"] + DUR[p["type"]]:
                stim[p["type"]] = dict(t=t - p["t"], dur=DUR[p["type"]], strength=p["strength"])
        row = {kk: float(feats[kk][k]) for kk in keys}
        # --- жизнь мухи: приступ -> смерть -> новая муха ---
        life_t = 0.0
        past = [(i, ts) for i, ts in enumerate(seiz_list) if t >= ts]
        if past:
            i_s, ts = past[-1]
            dt_s = t - ts
            if dt_s < SEIZ_SHOW_S:
                life, life_t = "seizure", dt_s
            elif dt_s < SEIZ_SHOW_S + DEATH_S:
                life, life_t = "dead", dt_s - SEIZ_SHOW_S
            elif dt_s < SEIZ_SHOW_S + DEATH_S + RESPAWN_S:
                if life != "respawn":                        # новая муха: мозг с нуля, защита на месте
                    brain.reset()
                    brain.P.update(S.PRM)
                    brain._check_consts()
                    brain.enable_graph()
                    filt, beh = S.Filt(), S.Behaviour()
                    fly_cur, respawned = fly_no + i_s + 1, i_s + 1
                life, life_t = "respawn", dt_s - SEIZ_SHOW_S - DEATH_S
            else:
                life = "alive"
        if life == "seizure" and life_prev != "seizure":
            brain.P.update(dict(U=0.0, a_inc=0.0))              # защиты больше нет (исходная модель)
            brain._check_consts()
            brain.enable_graph()
        life_prev = life
        if life in ("dead", "respawn"):                      # мёртвой / новой мухе стимулы не даём
            stim = {}
        S.apply_inputs(brain, groups, S.inputs_from_frame(row, stim))
        if life == "seizure":
            brain.set_input(orn, SEIZ_ODOR_HZ if life_t < SEIZ_ODOR_S else 0.0)
        ms = S.frame_ms(k, fps)
        hz = (brain.run(ms, tensor=True) * (1000.0 / ms)).cpu().numpy()
        f = filt.update(S.readouts(hz, groups), ms / 1000)
        body = beh.update(f, t)
        for key, ev in (("takeoff", "takeoff"), ("startle", "startle"), ("eat", "eat_start"), ("groom", "groom_start")):
            stats[key] += int(body[ev])

        hunger = min(1.0, hunger + 0.006 / fps)          # голод растёт (шкала игры, время ускорено)
        # --- капля сахара: пипетка подносит заранее, хоботок выпивает ---
        next_sugar = [p for p in plan if p["type"] in ("sugar", "bitter") and p["t"] - 0.6 <= t < p["t"] + DUR[p["type"]]]
        if next_sugar:
            p = next_sugar[0]
            dsign = -1.0 if p["type"] == "bitter" else 1.0     # знак капли: минус = горькая (зелёная)
            if t < p["t"]:
                drop = max(drop, (t - (p["t"] - 0.6)) / 0.6)
            elif body["proboscis"] > 0.5:
                drop = max(0.0, drop - (1 / fps) / 2.2)
                hunger = max(0.0, hunger - 0.01)
        else:
            drop = max(0.0, drop - (1 / fps) / 0.3)
        # --- тело по командам мозга ---
        busy = body["flight"] or body["proboscis"] > 0.3 or body["groom"]
        walk = float(np.clip(f.get("walk", 0.0) / 30.0, 0, 1))
        base = 0.0 if busy else 0.3 + 0.5 * walk
        turn = body["turn"]
        if body["groom"]:
            groom_t0 = t if groom_t0 is None else groom_t0
        else:
            groom_t0 = None
        startle = body["startle_age"] < 0.15
        spread = 1.0 if body["flight"] else (0.35 if startle else 0.0)
        flap = 1.0 if body["flight"] else (0.4 if startle else 0.0)
        st.set_audio_level(float(feats["bass"][k]))
        prev, cur, blend, cut = director.update(t, bool(strong[k]), body)
        st.set_camera(prev or cur, cur if prev else None, blend if prev else 0.0, shake=0.6)
        st.step_puppet(1 / fps, drive_L=base * (1 - max(turn, 0)), drive_R=base * (1 + min(turn, 0)),
                       head_turn=body["head"], proboscis=body["proboscis"], wing_spread=spread, wing_flap=flap,
                       groom_t=None if groom_t0 is None else t - groom_t0, drop=dsign * drop)
        hud = dict(fear_hz=f.get("fear", 0.0), interest=abs(f.get("steer", 0.0)), hearing_hz=f.get("hear2", 0.0),
                   hunger=hunger, steer=turn, gf_hz=f.get("gf", 0.0), mn9_hz=f.get("mn9", 0.0),
                   groom_hz=f.get("groom", 0.0), flying=bool(body["flight"]), proboscis=body["proboscis"],
                   grooming=bool(body["groom"]), n_active=int(f.get("n_active", 0)), cut=False,
                   takeoff=bool(body["takeoff"]), startle=bool(body["startle"]),
                   eat_start=bool(body["eat_start"]), groom_start=bool(body["groom_start"]),
                   head=float(body["head"]), vis={kk: row[kk] for kk in VIS_KEYS},
                   life=life, life_t=round(life_t, 3), fly_no=fly_cur)           # приступ/смерть/новая муха   # для Стримера и «как видит муха»
        if k in ev_text:
            hud["event"] = ev_text[k]
        state = st.get_state()
        state["cut"] = False            # смены ракурса плавные — без вспышки
        writer.add(stage=state, hud=hud, hz=hz)
        if k % 30 == 0:
            print(f"PROGRESS 1 {k} {n}", flush=True)
        if verbose and k % 300 == 0:
            print(f"    кадр {k}/{n}  t={t:5.1f} с  горит {int(f.get('n_active', 0))} нейронов  "
                  f"взлётов {stats['takeoff']}, еда {stats['eat']}, чистка {stats['groom']}", flush=True)
    print(f"PROGRESS 1 {n} {n}", flush=True)          # окну: проход 1 закончен (полоса до 35%)
    writer.close()
    if verbose:
        print(f"[1] готово за {time.perf_counter() - T0:.1f} с: {stats}")
    return dict(n=n, stats=stats, plan=plan, meta=meta, respawned=respawned)


def main():
    ap = argparse.ArgumentParser(description="Муха реагирует на ролик -> вертикальное видео")
    ap.add_argument("video")
    ap.add_argument("--watermark", default="@xyz_web3")
    ap.add_argument("--title", default=None)
    ap.add_argument("--fly_no", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--secs", type=float, default=None,
                    help="только N секунд (для проб): окно [t0, t0+N); проход 1 считает с начала до t0+N")
    ap.add_argument("--t0", type=float, default=0.0, help="превью: с какой секунды ролика рендерить")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_pass1", action="store_true", help="не считать мозг: рендер из готового лога")
    ap.add_argument("--log", default=None, help="папка лога прохода 1 (по умолчанию tiktok/_cache/log_<имя>)")
    ap.add_argument("--crf", type=int, default=24, help="качество: меньше = лучше и тяжелее (20..28)")
    ap.add_argument("--format", default="base", choices=FM.FORMATS,
                    help="base — Базовый (муха на шаре у монитора), streamer — Стример (кресло, наушники)")
    ap.add_argument("--variant", default="random",
                    help="вариант комнаты Стримера: число 0..999999 или random (у Базового не используется)")
    ap.add_argument("--flyeye", default="off", choices=("on", "off"), help="врезка «как видит муха»")
    ap.add_argument("--seizure", default="auto",
                    help="приступ: off | auto (грустный момент по словам / резкий бас) | секунды через запятую: 30,75")
    ap.add_argument("--meaning", default="on", choices=("on", "off"),
                    help="смысл речи (Whisper + словарь) -> сахар/горькое; off — только зрение и звук")
    ap.add_argument("--lock", default=None, metavar="МЕТКА",
                    help="один рендер на машине: ждать и взять tiktok/_render.lock, потом ждать > 3 ГБ свободной памяти")
    args = ap.parse_args()

    video = Path(args.video).resolve()
    title = args.title or video.stem.replace("_", " ")
    wm = WATERMARK_LOCK or (args.watermark or "").strip()                        # "" или " " (так шлёт окно) — без знака
    t0 = max(0.0, float(args.t0 or 0.0))
    preview = args.secs is not None or t0 > 0
    variant = FM.resolve_variant(args.variant) if args.format == "streamer" else None
    fmt = FM.fmt_cfg(args.format, variant, flyeye=args.flyeye == "on", watermark=wm)
    fly_no = args.fly_no or next_fly_no(dry=preview)          # пробный прогон не тратит номер мухи
    tag = FM.name_tag(fmt)
    name = f"{'проба' if preview else 'муха'}_{fly_no:03d}_{tag + '_' if tag else ''}{video.stem[:40].strip(" .")}.mp4"
    out = Path(args.out).resolve() if args.out else ROOT / "готово" / name   # ИТОГ — всегда полный путь
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log).resolve() if args.log else HERE / "_cache" / f"log_{video.stem[:40].strip(" .")}"
    print(f"МУХА №{fly_no} смотрит: {title}\n  выход: {out}\n  вотермарка: {wm or '(нет)'}")
    if not FM.is_default(fmt):
        print(f"  {FM.describe(fmt)}")
    if variant is not None:
        print(f"ВАРИАНТ: {variant}", flush=True)                  # окну: какой номер выпал при «случайный»
    fps = 30
    k0 = int(round(t0 * fps))
    n_win = None if args.secs is None else max(1, int(round(args.secs * fps)))
    with (FM.RenderLock(args.lock) if args.lock else contextlib.nullcontext()):
        if not args.skip_pass1:
            r1 = pass1(video, log_dir, title, fly_no, seed=args.seed,
                       secs=None if args.secs is None else t0 + args.secs, seizure=args.seizure,
                       meaning=args.meaning == "on")
            for _ in range(r1.get("respawned", 0) if not preview else 0):
                next_fly_no()                                     # новая муха после каждого приступа — свой номер
        _free_gpu()                                                # мозг больше не нужен: память видеокарты — воркерам и кодировщику
        from render_parallel import run_pass2
        if args.format == "base":
            import stage as SG
            SG.prepare_assets()
        else:                                                     # текстуры комнаты варианта — до воркеров
            import stage_streamer as SS
            import style as STY
            SS.prepare_assets(STY.RoomStyle(variant or 0))
        print("PROGRESS 2 0 1", flush=True)
        stats = run_pass2(log_dir, out, video, StageFactory(wm, fmt), n_workers=args.workers, crf=args.crf,
                          start=k0, frames=n_win, fmt=fmt)
    print("[2] готово:", stats)
    print("ИТОГ:", out, flush=True)


if __name__ == "__main__":
    main()
