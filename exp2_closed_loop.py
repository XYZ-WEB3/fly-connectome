"""Опыт 2: мозг мухи (коннектом FlyWire, 139k нейронов) управляет виртуальным телом.

Задача: на арене стоит «еда» (капля сахара, оранжевый шар).
  глаз   -> простой детектор «объект слева/справа» -> зрительные проекционные
            нейроны (VPN) левого/правого полушария
  мозг   -> LIF-модель всего коннектома, считает сама
  выходы -> DNa01/DNa02 (руль: разница левых и правых = поворот),
            DNp09 (P9, «иди вперёд»),
            MN9 (мотонейрон хоботка = «ем»)
  вкус   -> когда голова у еды, включаются сахарные вкусовые нейроны

Никаких правил «поверни к еде» в коде нет: поворот считается только
из активности нисходящих нейронов, которую выдаёт коннектом.
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import trange

from flygym import Fly
from flygym.arena import OdorArena
from flygym.camera import YawOnlyCamera
from flygym.examples.locomotion import HybridTurningController

import brain as B

ap = argparse.ArgumentParser()
ap.add_argument("--time", type=float, default=8.0, help="сек. жизни мухи")
ap.add_argument("--foods", type=float, nargs="+", default=[14.0, 9.0], help="x1 y1 x2 y2 ... капли еды, мм")
ap.add_argument("--eat_time", type=float, default=1.5, help="сек. еды, после чего капля съедена")
ap.add_argument("--heading", type=float, default=0.0, help="начальный курс мухи, град")
ap.add_argument("--vpn", type=str, default="config_vpn.json", help="какие VPN = «глаз»")
ap.add_argument("--silence", type=str, nargs="*", default=[], help="типы нейронов, которые выключить")
ap.add_argument("--out", type=str, default="videos/fly_brain.mp4")
args = ap.parse_args()

SYNC_MS = 15.0          # как у Eon: мозг и тело синхронизируются каждые 15 мс
PHYS_DT = 1e-4
FOODS = np.array(args.foods).reshape(-1, 2)
alive = np.ones(len(FOODS), bool)   # какие капли ещё не съедены
EAT_RADIUS = 1.6        # мм, голова ближе -> хоботок касается сахара
MN9_ID = 720575940660219265
SUGAR_IDS_630 = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
                 720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
                 720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
                 720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
                 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
                 720575940611875570]

# ---------------- мозг ----------------
ann = B.load_annotations()
brain = B.FlyBrain()
def by_type(t, side=None):
    a = ann[ann.cell_type == t]
    if side:
        a = a[a.side == side]
    return [brain.idx[r] for r in a.index if r in brain.idx]

cfg = json.load(open(args.vpn, encoding="utf8"))
eye_L = sum((by_type(t, "left") for t in cfg["vpn_types"]), [])
eye_R = sum((by_type(t, "right") for t in cfg["vpn_types"]), [])
steer_L = by_type("DNa01", "left") + by_type("DNa02", "left")
steer_R = by_type("DNa01", "right") + by_type("DNa02", "right")
walk = by_type("DNp09")
sugar = [brain.idx[r] for r in SUGAR_IDS_630 if r in brain.idx]
mn9 = [brain.idx[MN9_ID]]
for t in args.silence:
    brain.silence(by_type(t))
print(f"глаз: {len(eye_L)}+{len(eye_R)} VPN ({cfg['vpn_types']}), руль: {len(steer_L)}+{len(steer_R)}, "
      f"ход: {len(walk)}, сахар: {len(sugar)}, выключено типов: {args.silence}")

# ---------------- тело и мир ----------------
arena = OdorArena(odor_source=np.array([[*f, 0.6] for f in FOODS]), peak_odor_intensity=np.array([[1, 0]] * len(FOODS)),
                  marker_colors=[(1.0, 0.55, 0.1, 1)] * len(FOODS), marker_size=0.6)
fly = Fly(spawn_pos=(0, 0, 0.2), spawn_orientation=(0, 0, np.deg2rad(args.heading)),
          enable_olfaction=True, enable_adhesion=True, draw_adhesion=False,
          contact_sensor_placements=[f"{l}{s}" for l in ["LF", "LM", "LH", "RF", "RM", "RH"]
                                     for s in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]])
W, H = 1280, 720
cam = YawOnlyCamera(attachment_point=fly.model.worldbody, camera_name="camera_back_track_game",
                    targeted_fly_names=fly.name, window_size=(W, H), play_speed=0.25, fps=30,
                    play_speed_text=False)
sim = HybridTurningController(fly=fly, cameras=[cam], arena=arena, timestep=PHYS_DT)
obs, _ = sim.reset()


def eye(obs):
    """«Сетчатка»: где капли еды в поле зрения. Муха видит ~330°, кроме зоны сзади.
    Сумма по всем видимым каплям -> интенсивность для левого и правого глаза (0..1)."""
    pos, heading = obs["fly"][0, :2], obs["fly_orientation"][:2]
    left = right = 0.0
    best = (1e9, 0.0)
    for f in FOODS[alive]:
        to = f - pos
        dist = np.linalg.norm(to)
        ang = np.arctan2(heading[0] * to[1] - heading[1] * to[0], heading @ to)   # >0 — слева
        best = min(best, (dist, np.rad2deg(ang)))
        if abs(ang) > np.deg2rad(165):
            continue
        size = np.clip(3.0 / max(dist, 0.5), 0, 1)          # угловой размер ~ 1/расстояние
        left += size * np.clip(np.sin(ang) * 1.2 + 0.25, 0, 1)
        right += size * np.clip(-np.sin(ang) * 1.2 + 0.25, 0, 1)
    return min(left, 1.0), min(right, 1.0), best[0], best[1]


def eat_food(i):
    alive[i] = False
    body = arena.root_element.find("body", f"odor_source_marker_{i}")
    sim.physics.bind(body).mocap_pos = [0, 0, -50]      # капля съедена -> убираем под пол


# ---------------- видео: HUD ----------------
def hud(img, st, traj):
    img = img.copy()
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (W, 58), (15, 15, 20), -1)
    cv2.rectangle(ov, (W - 330, 70), (W - 10, 390), (15, 15, 20), -1)
    cv2.rectangle(ov, (10, H - 250), (350, H - 10), (15, 15, 20), -1)
    img = cv2.addWeighted(ov, 0.72, img, 0.28, 0)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "FlyWire connectome brain  |  139k neurons  54M synapses  |  no hand-coded steering",
                (16, 25), f, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(img, f"t = {st['t']:.2f} s   active neurons (15 ms): {st['n_active']:,}   food eaten: {st['eaten']}/{len(FOODS)}   state: {st['state']}",
                (16, 48), f, 0.58, (140, 220, 255), 1, cv2.LINE_AA)
    # панель нейронов
    x0, y = W - 318, 96
    cv2.putText(img, "BRAIN OUTPUT (Hz)", (x0, y), f, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    bars = [("eye L  (VPN)", st["eyeL"] * 100, (120, 200, 120)), ("eye R  (VPN)", st["eyeR"] * 100, (120, 200, 120)),
            ("DNa01/02 L", st["sL"], (255, 170, 80)), ("DNa01/02 R", st["sR"], (255, 170, 80)),
            ("DNp09 walk", st["walk"], (120, 180, 255)), ("sugar GRN", st["sugar"], (80, 220, 255)),
            ("MN9 proboscis", st["mn9"], (60, 90, 255))]
    for name, val, col in bars:
        y += 40
        cv2.putText(img, name, (x0, y - 8), f, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y - 2), (x0 + 290, y + 10), (60, 60, 60), -1)
        cv2.rectangle(img, (x0, y - 2), (x0 + int(290 * min(val, 150) / 150), y + 10), col, -1)
        cv2.putText(img, f"{val:.0f}", (x0 + 250, y - 8), f, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    # мини-карта
    mx, my, sc = 180, H - 130, 9.0
    cv2.putText(img, "top view", (20, H - 228), f, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cx, cy = FOODS.mean(0) / 2
    to_px = lambda p: (int(mx + (p[0] - cx) * sc), int(my - (p[1] - cy) * sc))
    for f, a in zip(FOODS, alive):
        cv2.circle(img, to_px(f), int(EAT_RADIUS * sc), (255, 140, 40) if a else (90, 90, 90), -1 if a else 1)
    for a, b_ in zip(traj[:-1], traj[1:]):
        cv2.line(img, to_px(a), to_px(b_), (255, 255, 255), 2, cv2.LINE_AA)
    if traj:
        cv2.circle(img, to_px(traj[-1]), 5, (80, 255, 80), -1)
    return img


# ---------------- главный цикл ----------------
n_sync = int(args.time * 1000 / SYNC_MS)
steps_brain = int(SYNC_MS / brain.dt)
steps_phys = int(SYNC_MS * 1e-3 / PHYS_DT)
frames, traj, log = [], [], []
action = np.array([0.0, 0.0])
st = dict(t=0, n_active=0, state="start", eyeL=0, eyeR=0, sL=0, sR=0, walk=0, sugar=0, mn9=0, eaten=0)
sL_f = sR_f = walk_f = mn9_f = 0.0   # сглаженные частоты (~ мышцы/инерция), тау 60 мс
alpha = SYNC_MS / 60.0
eat_clock, eaten = 0.0, 0

for k in trange(n_sync):
    # 1) органы чувств -> сенсорные нейроны
    eL, eR, dist, ang = eye(obs)
    head = obs["fly"][0, :2] + obs["fly_orientation"][:2] * 1.2
    near = [i for i in np.nonzero(alive)[0] if np.linalg.norm(head - FOODS[i]) < EAT_RADIUS]
    touching = bool(near)
    eat_clock = eat_clock + SYNC_MS / 1000 if (touching and mn9_f > cfg["mn9_eat_hz"]) else (eat_clock if touching else 0.0)
    if touching and eat_clock > args.eat_time:
        eat_food(near[0]); eaten += 1; eat_clock = 0.0; touching = False
    brain.set_input(eye_L, cfg["max_rate"] * eL)
    brain.set_input(eye_R, cfg["max_rate"] * eR)
    brain.set_input(sugar, 150.0 if touching else 0.0)

    # 2) мозг думает 15 мс
    cnt = torch.zeros(brain.N, device=brain.dev)
    for _ in range(steps_brain):
        cnt += brain.step()
    hz = cnt.cpu().numpy() * (1000.0 / SYNC_MS)
    sL, sR = hz[steer_L].mean(), hz[steer_R].mean()
    wk, m9 = hz[walk].mean(), hz[mn9].mean()
    sL_f += alpha * (sL - sL_f); sR_f += alpha * (sR - sR_f)
    walk_f += alpha * (wk - walk_f); mn9_f += alpha * (m9 - mn9_f)

    # 3) нисходящие нейроны -> команды ногам (фиксированное простое отображение)
    if touching and mn9_f > cfg["mn9_eat_hz"]:
        action = np.array([0.0, 0.0]); state = "EATING (MN9 active)"
    else:
        fwd = cfg["base_drive"] + cfg["walk_gain"] * walk_f
        turn = np.tanh(cfg["turn_gain"] * (sL_f - sR_f))          # >0: левые DN сильнее -> налево
        action = np.clip(np.array([fwd * (1 - max(turn, 0)), fwd * (1 + min(turn, 0))]), 0, 1.2)
        state = "turn LEFT" if turn > 0.15 else "turn RIGHT" if turn < -0.15 else "walking"

    # 4) тело двигается 15 мс
    for _ in range(steps_phys):
        obs, *_ = sim.step(action)
        img = sim.render()[0]
        if img is not None:
            traj.append(obs["fly"][0, :2].copy())
            st = dict(t=sim.curr_time, n_active=int((cnt > 0).sum()), state=state, eyeL=eL, eyeR=eR,
                      sL=sL_f, sR=sR_f, walk=walk_f, sugar=150.0 if touching else 0.0, mn9=mn9_f, eaten=eaten)
            frames.append(hud(img, st, traj))
    log.append(dict(t=float(sim.curr_time), x=float(obs["fly"][0, 0]), y=float(obs["fly"][0, 1]), dist=float(dist),
                    ang=float(ang), eyeL=float(eL), eyeR=float(eR), dnL=float(sL), dnR=float(sR), walk=float(wk),
                    mn9=float(m9), n_active=int((cnt > 0).sum()), state=state))

Path(args.out).parent.mkdir(exist_ok=True, parents=True)
imageio.mimsave(args.out, frames, fps=30, macro_block_size=8)
json.dump(log, open(Path(args.out).with_suffix(".json"), "w"))
print(f"видео: {args.out}  ({len(frames)} кадров); съедено капель: {eaten}/{len(FOODS)}")
