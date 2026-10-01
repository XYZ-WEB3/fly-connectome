"""Шоу: мозг мухи (коннектом FlyWire) живёт в виртуальном теле.

Сюжет: муха голодна, видит еду -> идёт к ней -> ест (хоботок!) -> идёт к следующей;
на пути на неё катится тёмный шар -> детекторы угрозы LPLC2/LC4 -> гигантское
волокно DNp01 -> ВЗЛЁТ -> посадка -> продолжает есть.

Что решает мозг (коннектом, 139k нейронов):  куда поворачивать (DNa01/02),
когда есть (MN9), когда взлетать (DNp01), «эмоции» — это частоты настоящих нейронов.
Что задано вручную: «сетчатка» (где объект -> сигнал в LC10/LPLC2/LC4),
перевод команд DN в движение ног (готовый контроллер ходьбы flygym),
сам полёт (анимация — в этой физике нет аэродинамики).
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as Rot
from tqdm import trange

from flygym import Fly
from flygym.examples.locomotion import HybridTurningController
from flygym.simulation import SingleFlySimulation
from dm_control.rl.control import PhysicsError

import brain as B
import hud
from scene import KitchenArena, CineCamera, FlyPoser

ap = argparse.ArgumentParser()
ap.add_argument("--time", type=float, default=16.0)
ap.add_argument("--out", default="videos/show.mp4")
ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
ap.add_argument("--no_threat", action="store_true")
ap.add_argument("--silence", nargs="*", default=[], help="выключить типы нейронов (нейрохирургия)")
ap.add_argument("--seed", type=int, default=None, help="зерно мира (по умолчанию = номер мухи)")
ap.add_argument("--brain", default="U=0.1,tau_d=300,a_inc=0.1,tau_a=3000", help="параметры мозга; '' = оригинальная модель")
ap.add_argument("--turn_gain", type=float, default=0.12)
args = ap.parse_args()

W, H = args.res
_cnt_path = Path("data/fly_counter.json")
FLY_NO = (json.load(open(_cnt_path))["n"] if _cnt_path.exists() else 0) + 1
json.dump({"n": FLY_NO}, open(_cnt_path, "w"))
rng = np.random.default_rng(args.seed if args.seed is not None else FLY_NO * 7919)
FOODS = []
_n_food = int(rng.integers(3, 5))
while len(FOODS) < _n_food:
    r, a = rng.uniform(6, 19), rng.uniform(-np.pi, np.pi)
    xy = np.array([r * np.cos(a), r * np.sin(a)])
    if all(np.linalg.norm(xy - np.array(f[1:3])) > 7 for f in FOODS):
        FOODS.append((str(rng.choice(["banana", "syrup", "berry"])), float(xy[0]), float(xy[1]), float(rng.uniform(0, 6.28))))
print("еда:", [(k, round(x, 1), round(y, 1)) for k, x, y, _ in FOODS])
SYNC_MS, PHYS_DT = 14.4, 1e-4          # 14.4 мс = 8 блоков по 1.8 мс
PLAY = 0.25                     # замедление x4
FPS = 30
FRAME_DT = PLAY / FPS           # шаг кадра во времени мухи
EAT_R, EAT_TIME = 1.7, 1.4
FOOD_RU = {"banana": "банан", "syrup": "сироп", "berry": "ягода"}
MN9_ID = 720575940660219265
SUGAR_IDS_630 = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
                 720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
                 720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
                 720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
                 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
                 720575940611875570]

# ======================= мозг =======================
ann = B.load_annotations()
BRAIN_PRM = {k: float(v) for k, v in (kv.split("=") for kv in args.brain.split(",") if kv)}
brain = B.FlyBrain(params=BRAIN_PRM)
print("параметры мозга:", BRAIN_PRM or "оригинал")
def ty(types, side=None):
    a = ann[ann.cell_type.isin(types if isinstance(types, list) else [types])]
    if side:
        a = a[a.side == side]
    return [brain.idx[r] for r in a.index if r in brain.idx]

LC10 = ["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"]
LOOM = ["LPLC2", "LC4"]
N = dict(food_L=ty(LC10, "left"), food_R=ty(LC10, "right"),
         loom_L=ty(LOOM, "left"), loom_R=ty(LOOM, "right"),
         steer_L=ty(["DNa01", "DNa02"], "left"), steer_R=ty(["DNa01", "DNa02"], "right"),
         gf=ty("DNp01"), fear=ty(["DNp01", "DNp02", "DNp04", "DNp11"]),
         pam=[brain.idx[r] for r in ann.index[ann.cell_type.astype(str).str.startswith("PAM")] if r in brain.idx],
         sugar=[brain.idx[r] for r in SUGAR_IDS_630 if r in brain.idx], mn9=[brain.idx[MN9_ID]],
         bristle=ty(["BM_Ant", "BM_Fr", "BM_FrOr"]), groom=ty(["DNg84", "DNg35"]))
# «вкусовой путь»: центральные нейроны, которые отвечают на сахар (короткая калибровка)
brain.reset(); brain.set_input(N["sugar"], 180)
_c = brain.run(300)
_sens = set(ty(list(ann.cell_type[ann.super_class.isin(["sensory", "motor"])].dropna().unique())))
N["taste"] = [i for i in np.nonzero(_c > 3)[0] if i not in _sens][:400]
brain.reset()
for t in args.silence:
    brain.silence(ty(t))
brain.enable_graph()
print({k: len(v) for k, v in N.items()}, "silenced:", args.silence)
cloud = hud.BrainCloud(brain.ids)

# ======================= мир и тело =======================
arena = KitchenArena(FOODS)
fly = Fly(spawn_pos=(0, 0, 0.2), spawn_orientation=(0, 0, np.deg2rad(-15)), enable_adhesion=True,
          draw_adhesion=False, contact_sensor_placements=[f"{l}{s}" for l in ["LF", "LM", "LH", "RF", "RM", "RH"]
                                                          for s in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]])
arena.add_fly_light(fly)
arena.add_extra_joints(fly)
sim = HybridTurningController(fly=fly, cameras=[], arena=arena, timestep=PHYS_DT)
obs, _ = sim.reset()
phys = sim.physics
m, d = phys.model.ptr, phys.data.ptr
pose = FlyPoser(phys, fly.name)
cam = CineCamera(phys)

root_j = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
ROOT_Q, ROOT_V = m.jnt_qposadr[root_j], m.jnt_dofadr[root_j]
head_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{fly.name}/Head")

def body_id(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

food_state = []
for i, f in enumerate(arena.foods):
    bid = body_id(f"food_{i}")
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    food_state.append(dict(kind=f["kind"], pos=f["pos"].copy(), alive=True, left=1.0, bid=bid,
                           mocap=m.body_mocapid[bid], geoms=geoms,
                           size0=m.geom_size[geoms].copy(), gpos0=m.geom_pos[geoms].copy(),
                           rgba0=m.geom_rgba[geoms].copy(), seen=False))
trail_mocap = [m.body_mocapid[body_id(f"trail_{k}")] for k in range(len(arena.trail))]
trail_geom = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"trail_{k}_sensorfx") for k in range(len(arena.trail))]
trail_t = np.full(len(trail_mocap), -99.0)
threat_mocap = m.body_mocapid[body_id("threat")]
THREAT_R = 2.0


def fly_xy():
    return obs["fly"][0, :2].copy()


def heading():
    h = obs["fly_orientation"][:2]
    return h / (np.linalg.norm(h) + 1e-9)


def side_split(to, h, strength, back=np.deg2rad(165)):
    """Объект в направлении `to` -> сигнал левому/правому глазу."""
    ang = np.arctan2(h[0] * to[1] - h[1] * to[0], h @ to)
    if abs(ang) > back:
        return 0.0, 0.0, ang
    return (strength * np.clip(np.sin(ang) * 1.2 + 0.25, 0, 1),
            strength * np.clip(-np.sin(ang) * 1.2 + 0.25, 0, 1), ang)


# ======================= состояние =======================
st = dict(mn9_last=-9.0, dead=False, death_t=0.0, cause="", starve=0.0, sticky=0.0, groom_t0=0.0, end_t=None, state="ищет еду", hunger=float(rng.uniform(0.55, 0.85)), eaten=0, eat_clock=0.0, flying=False, fly_t0=0, fly_from=None,
          fly_to=None, fly_q0=None, fly_dpsi=0, fly_z0=0, gf_cool=0.0, threat_on=False, threat_done=args.no_threat,
          threat_pos=None, threat_vel=None, threat_t0=0, theta_prev=0.0, trail_k=0, last_trail=-1)
filt = dict(sL=0, sR=0, mn9=0, gf=0, fear=0, pam=0, groom=0)
alpha = SYNC_MS / 60.0
popups = hud.Popups()
traj, log = [], []
crashes = []
next_frame = 0.0
writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                            ffmpeg_params=["-pix_fmt", "yuv420p"])
brain_hz = np.zeros(brain.N, np.float32)
proboscis = 0.0

def die(cause, t):
    st["dead"], st["death_t"], st["cause"], st["state"] = True, t, cause, "погибла"
    st["die_q0"] = d.qpos[ROOT_Q + 3:ROOT_Q + 7].copy()
    st["die_p0"] = d.qpos[ROOT_Q:ROOT_Q + 3].copy()
    st["die_axis"] = np.array([*heading(), 0.0])
    st["end_t"] = t + 1.4
    popups.add(f"муха №{FLY_NO} погибла", (255, 70, 60), t, 2.5, key="dead")


# ======================= чистка (движение передних лап) =======================
PS = sim.preprogrammed_steps
BASE_POSE = PS.default_pose.copy().ravel()
DOF = {n: i for i, n in enumerate(PS.dofs_per_leg)}


def groom_angles(tg):
    """Две чередующиеся фазы, как у живой мухи: провести лапами по голове
    (синхронно, от морды до макушки), потом потереть лапки (в противофазе)."""
    j = BASE_POSE.copy()
    ramp = min(1.0, tg / 0.2)
    w = 2 * np.pi * 4.0
    head_phase = (tg % 1.2) < 0.6
    P_FACE = dict(Femur=-1.6, Tibia=1.0, Coxa=-0.6, Coxa_roll=0.0)
    P_TOP = dict(Femur=-1.4, Tibia=0.6, Coxa=-0.9, Coxa_roll=0.0)
    for li, leg in enumerate(PS.legs):
        if leg[1] != "F":
            continue
        sgn = 1 if leg[0] == "L" else -1
        anti = 0 if leg[0] == "L" else np.pi
        if head_phase:
            a = 0.5 + 0.5 * np.sin(w * tg)
            dq = {k: P_FACE[k] + (P_TOP[k] - P_FACE[k]) * a for k in P_FACE}
        else:
            o = np.sin(w * 1.5 * tg + anti)
            dq = dict(Femur=-1.5 + 0.1 * o, Tibia=1.1 + 0.35 * o, Coxa=-0.5, Coxa_roll=(-0.6 + 0.35 * o))
        for dof, v in dq.items():
            j[li * 7 + DOF[dof]] += ramp * v * (sgn if dof in ("Coxa_roll", "Coxa_yaw") else 1)
    return j


# ======================= кадр =======================
def shot_for_state():
    s = st["state"]
    if sim.curr_time < 1.2:
        return dict(dist=16 - 8 * sim.curr_time / 1.2, height=14 - 9.5 * sim.curr_time / 1.2, side=4, lead=2, ease=0.3)
    if s == "ест":
        return dict(dist=-3.4, height=1.5, side=3.8, lead=0.9, look_z=0.35, ease=0.45)
    if s == "чистится":
        return dict(dist=-3.8, height=0.9, side=1.6, lead=0.9, look_z=0.75, ease=0.45)
    if s in ("УГРОЗА!",) and st["threat_on"]:
        return dict(dist=13, height=8, side=-7, lead=3, ease=0.5)
    if s in ("ВЗЛЁТ!",):
        return dict(dist=9, height=5 + obs["fly"][0, 2], side=-2, lead=1, look_z=obs["fly"][0, 2], ease=0.25)
    return dict(dist=7.5, height=3.8, side=1.8, lead=2.2, ease=0.35)


def compose_frame(eye, t):
    img = phys.render(height=H, width=W, camera_id="cine")
    base = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    sc = W / 1920
    # --- заголовок ---
    hud.panel(dr, (24, 22, 24 + (1010 if BRAIN_PRM else 700) * sc, 22 + 120 * sc))
    dr.text((44, 32), f"МУХА №{FLY_NO}  ·  мозг в виртуальном теле", font=hud.F[30], fill=(255, 255, 255, 255))
    dr.text((44, 72), "коннектом FlyWire · 139 тыс. нейронов · 54 млн синапсов", font=hud.F[18], fill=(170, 200, 255, 255))
    dr.text((44, 100), "поведение не запрограммировано: решают связи нейронов"
            + ("  ·  + адаптация и утомление синапсов (без приступов)" if BRAIN_PRM else ""),
            font=hud.F[16], fill=(210, 210, 215, 230))
    # --- время + мини-карта ---
    tx = f"t = {t:4.2f} с   ·   замедление ×{1 / PLAY:.0f}"
    tw = dr.textlength(tx, font=hud.F[20])
    hud.panel(dr, (W - tw - 64, 22, W - 24, 66))
    dr.text((W - tw - 44, 32), tx, font=hud.F[20], fill=(255, 255, 255, 255))
    mx0, my0, mw, mh = W - 284, 80, 260, 190
    hud.panel(dr, (mx0, my0, mx0 + mw, my0 + mh))
    dr.text((mx0 + 14, my0 + 8), "карта", font=hud.F[16], fill=(200, 200, 210, 255))
    pts = np.array([[0, 0]] + [f[1:3] for f in FOODS])
    cxy = (pts.min(0) + pts.max(0)) / 2
    s_ = min((mw - 30) / (np.ptp(pts[:, 0]) + 6), (mh - 40) / (np.ptp(pts[:, 1]) + 6))
    to_px = lambda p: (mx0 + mw / 2 + (p[0] - cxy[0]) * s_, my0 + mh / 2 + 10 - (p[1] - cxy[1]) * s_)
    for fs in food_state:
        if fs["alive"]:
            x, y = to_px(fs["pos"]); rr = 6 * max(fs["left"], 0.3)
            dr.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(255, 170, 60, 255))
    if len(traj) > 1:
        dr.line([to_px(p) for p in traj[-900:]], fill=(90, 230, 255, 220), width=2)
    if st["threat_on"]:
        x, y = to_px(st["threat_pos"]); dr.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(20, 20, 25, 255), outline=(255, 60, 40, 255), width=2)
    x, y = to_px(fly_xy()); dr.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(120, 255, 120, 255))
    # --- мозг ---
    bimg, n_act = cloud.render()
    bx, by = W - cloud.w - 24, H - cloud.h - 96
    hud.panel(dr, (bx - 12, by - 44, W - 12, H - 16))
    dr.text((bx, by - 36), f"АКТИВНОСТЬ МОЗГА  ·  сейчас горят {n_act:,} нейронов".replace(",", " "),
            font=hud.F[18], fill=(255, 255, 255, 255))
    lx, ly = bx, H - 76
    for name, col in hud.GROUPS.items():
        wlen = 22 + dr.textlength(name, font=hud.F[14])
        if lx + wlen > W - 24:
            lx, ly = bx, ly + 22
        dr.ellipse((lx, ly + 5, lx + 10, ly + 15), fill=(*col, 255))
        dr.text((lx + 14, ly), name, font=hud.F[14], fill=(210, 210, 220, 255))
        lx += wlen
    # --- эмоции и решения ---
    px0, py0, pw = 24, H - 514, 380
    hud.panel(dr, (px0, py0, px0 + pw, H - 16))
    dr.text((px0 + 18, py0 + 12), "ЭМОЦИИ (читаем с нейронов)", font=hud.F[18], fill=(255, 255, 255, 255))
    hud.gauge(dr, px0 + 18, py0 + 44, pw - 36, "Голод" + ("  ·  ИСТОЩЕНИЕ" if st["hunger"] >= 1 else ""), st["hunger"] * 100,
              (255, 80, 50) if st["hunger"] > 0.9 else (255, 190, 60), 100, f"{st['hunger'] * 100:.0f}%")
    hud.gauge(dr, px0 + 18, py0 + 88, pw - 36, "Страх · нейроны побега DNp", filt["fear"], (255, 70, 70), 120)
    hud.gauge(dr, px0 + 18, py0 + 132, pw - 36, "Вкус · сахарный путь мозга", filt["pam"], (255, 110, 220), 40)
    dr.text((px0 + 18, py0 + 190), "РЕШЕНИЯ МОЗГА", font=hud.F[18], fill=(255, 255, 255, 255))
    hud.steer_gauge(dr, px0 + 18, py0 + 222, pw - 36, np.tanh(args.turn_gain * (filt["sL"] - filt["sR"])))
    hud.gauge(dr, px0 + 18, py0 + 266, pw - 36, "Хоботок · MN9", filt["mn9"], (255, 150, 50), 120)
    hud.gauge(dr, px0 + 18, py0 + 310, pw - 36, "Взлёт · гигантское волокно DNp01", filt["gf"], (120, 200, 255), 150)
    hud.gauge(dr, px0 + 18, py0 + 354, pw - 36, "Чистка · DNg84 / DNg35", filt["groom"], (140, 255, 170), 150)
    dr.text((px0 + 18, py0 + 404), f"Глаз видит еду:  L {eye[0]:.2f}   R {eye[1]:.2f}", font=hud.F[16], fill=(170, 230, 170, 255))
    dr.text((px0 + 18, py0 + 428), f"Глаз видит угрозу:  L {eye[2]:.2f}   R {eye[3]:.2f}", font=hud.F[16], fill=(255, 150, 140, 255))
    dr.text((px0 + 18, py0 + 458), f"Съедено: {st['eaten']} из {len(FOODS)}", font=hud.F[20], fill=(255, 210, 120, 255))
    # --- 3D подписи ---
    for fs in food_state:
        if fs["alive"]:
            hud.label3d(dr, cam.project([*fs["pos"], 2.6], W, H), FOOD_RU[fs["kind"]], (255, 215, 140))
    col = {"погибла": (255, 80, 70), "чистится": (170, 255, 190), "ест": (255, 190, 90), "УГРОЗА!": (255, 90, 80), "ВЗЛЁТ!": (130, 210, 255)}.get(st["state"], (180, 255, 180))
    hud.label3d(dr, cam.project([*fly_xy(), obs["fly"][0, 2] + 1.9], W, H), st["state"], col)
    popups.draw(dr, t, W, H)
    base.alpha_composite(ov)
    base.alpha_composite(bimg.convert("RGBA"), (bx, by))
    if st["end_t"] is not None and t > st["end_t"]:
        fade = Image.new("RGBA", (W, H), (0, 0, 0, 0)); fdr = ImageDraw.Draw(fade)
        a = min(1.0, (t - st["end_t"]) / 0.6)
        fdr.rectangle((0, 0, W, H), fill=(0, 0, 0, int(200 * a)))
        n_to = sum(1 for a_, b_ in zip(log, log[1:]) if b_["state"] == "ВЗЛЁТ!" and a_["state"] != "ВЗЛЁТ!")
        head_txt = (f"муха №{FLY_NO} погибла", (255, 110, 90)) if st["dead"] else (f"муха №{FLY_NO} сыта", (255, 230, 150))
        sub = (f"{st['cause']}  ·  прожила {st['death_t']:.1f} с  ·  съедено {st['eaten']} из {len(FOODS)}" if st["dead"]
               else f"съедено {st['eaten']} из {len(FOODS)}  ·  побегов от угрозы: {n_to}")
        for i, (txt, fnt, colr) in enumerate([(head_txt[0], hud.F[64], head_txt[1]),
                                               (sub, hud.F[30], (230, 230, 240)),
                                               ("ни одного правила «иди к еде» или «убегай» в коде нет", hud.F[24], (170, 200, 255))]):
            tw = fdr.textlength(txt, font=fnt)
            fdr.text((W / 2 - tw / 2, H / 2 - 90 + i * 80), txt, font=fnt, fill=(*colr, int(255 * a)))
        base.alpha_composite(fade)

    return np.asarray(base.convert("RGB"))


# ======================= главный цикл =======================
steps_brain = int(SYNC_MS / brain.dt)
steps_phys = int(round(SYNC_MS * 1e-3 / PHYS_DT))
eye = (0, 0, 0, 0)
for k in trange(int(args.time * 1000 / SYNC_MS)):
    t = sim.curr_time
    if st["end_t"] is not None and t > st["end_t"] + 1.3:
        break
    pos, h = fly_xy(), heading()
    head = d.xpos[head_bid][:2].copy()

    # ---------- органы чувств ----------
    fL = fR = 0.0
    for fs in food_state:
        if not fs["alive"]:
            continue
        to = fs["pos"] - pos
        dist = np.linalg.norm(to)
        l_, r_, ang = side_split(to, h, np.clip(6.0 / (dist + 2.0), 0, 1) * fs["left"])
        fL += l_; fR += r_
        if (l_ + r_) > 0.3 and not fs["seen"] and st["state"] != "ест":
            fs["seen"] = True
            popups.add(f"замечена еда: {FOOD_RU[fs['kind']]}", (255, 215, 120), t, 1.5, key=f"seen{fs['kind']}")
    fL, fR = min(fL, 1), min(fR, 1)
    tL = tR = 0.0
    if st["threat_on"]:
        to = st["threat_pos"] - head
        dist = max(np.linalg.norm(to), 0.3)
        theta = 2 * np.arctan(THREAT_R / dist)
        dtheta = (theta - st["theta_prev"]) / (SYNC_MS / 1000)
        st["theta_prev"] = theta
        loom = np.clip(theta / 0.9, 0, 1) * (1.0 if dtheta > 0 else 0.3)
        tL, tR, _ = side_split(to, h, loom, back=np.deg2rad(175))
    r_eat = EAT_R + (0.7 if st["state"] == "ест" else 0.0)      # запас, чтобы не мигало
    near = [i for i, fs in enumerate(food_state) if fs["alive"] and np.linalg.norm(head - fs["pos"]) < r_eat]
    touching = bool(near) and not st["flying"]
    eye = (fL, fR, tL, tR)

    brain.set_input(N["food_L"], 60 * fL); brain.set_input(N["food_R"], 60 * fR)
    brain.set_input(N["loom_L"], 80 * tL); brain.set_input(N["loom_R"], 80 * tR)
    brain.set_input(N["sugar"], (80 + 120 * st["hunger"]) if touching else 0.0)
    st["sticky"] *= np.exp(-SYNC_MS / 1000 / 0.8)
    brain.set_input(N["bristle"], 90 * st["sticky"] if st["sticky"] > 0.05 else 0.0)

    # ---------- мозг думает 15 мс ----------
    cnt = torch.zeros(brain.N, device=brain.dev)
    if not st["dead"]:                       # мёртвый мозг гаснет
        cnt = brain.run(SYNC_MS, tensor=True).clone()
    brain_hz = cnt.cpu().numpy() * (1000.0 / SYNC_MS)
    cloud.update(brain_hz)
    for key, grp in (("sL", "steer_L"), ("sR", "steer_R"), ("mn9", "mn9"), ("gf", "gf"), ("fear", "fear"), ("pam", "taste"), ("groom", "groom")):
        filt[key] += alpha * (brain_hz[N[grp]].mean() - filt[key])

    # ---------- решения -> тело ----------
    st["hunger"] = min(1.0, st["hunger"] + 0.03 * SYNC_MS / 1000)      # время жизни ускорено
    st["starve"] = st["starve"] + SYNC_MS / 1000 if st["hunger"] >= 1.0 else 0.0
    if st["starve"] > 0.3:
        popups.add("истощение!", (255, 90, 60), t, 1.5, key="истощение")
    if not st["dead"] and st["starve"] > 2.5:
        die("умерла от голода", t)
    if not st["dead"] and st["threat_on"] and not st["flying"] and np.linalg.norm(st["threat_pos"] - pos) < THREAT_R + 0.2:
        die("раздавлена шаром: не успела взлететь", t)
    st["gf_cool"] -= SYNC_MS / 1000
    if not st["flying"] and filt["gf"] > 45 and st["gf_cool"] <= 0 and st["threat_on"]:
        # ВЗЛЁТ: команда гигантского волокна
        st["flying"], st["fly_t0"], st["state"] = True, t, "ВЗЛЁТ!"
        away = pos - st["threat_pos"]; away /= np.linalg.norm(away) + 1e-9
        side = np.array([-away[1], away[0]])
        st["fly_from"] = d.qpos[ROOT_Q:ROOT_Q + 3].copy()
        st["fly_to"] = np.array([*(pos + away * 6.5 + side * 2.0), st["fly_from"][2]])
        st["fly_q0"] = d.qpos[ROOT_Q + 3:ROOT_Q + 7].copy()
        want = np.arctan2(away[1], away[0]); cur = np.arctan2(h[1], h[0])
        st["fly_dpsi"] = (want - cur + np.pi) % (2 * np.pi) - np.pi
        popups.add("ВЗЛЁТ!  гигантское волокно DNp01", (130, 210, 255), t, 1.8, key="takeoff")
    if filt["mn9"] > 3:
        st["mn9_last"] = t
    grooming_now = (not st["dead"] and not st["flying"] and not touching and st["state"] != "УГРОЗА!"
                    and (filt["groom"] > 35 or (st["state"] == "чистится" and filt["groom"] > 12)))
    if st["flying"] or st["dead"]:
        action = np.zeros(2)
    elif grooming_now:
        action = np.zeros(2)
        if st["state"] != "чистится":
            st["groom_t0"] = t
            popups.add("чистка!  DNg84 / DNg35", (190, 255, 200), t, 1.5, key=f"groom{st['eaten']}")
        st["state"] = "чистится"
    # у еды: пока идут подходы хоботка (или пауза < 0.8 с) — стоим. Остановка при касании сахара
    # лапками у живых мух — рефлекс брюшной нервной цепочки (её в модели нет) -> это правило тела.
    elif touching and (filt["mn9"] > 8 or (st["state"] == "ест" and t - st["mn9_last"] < 0.8)):
        action = np.zeros(2)
        if st["state"] != "ест":
            popups.add("сахар!  MN9 включает хоботок", (255, 190, 90), t, 1.4, key=f"sugar{near[0]}")
        st["state"] = "ест"
    else:
        turn = np.tanh(args.turn_gain * (filt["sL"] - filt["sR"]))
        action = np.clip(np.array([1 - max(turn, 0), 1 + min(turn, 0)]), 0, 1.2)
        if st["state"] in ("ест", "ВЗЛЁТ!", "start", "чистится") and not st["dead"]:
            st["state"] = "ищет еду"
        if st["threat_on"] and max(tL, tR) > 0.15:
            if st["state"] != "УГРОЗА!":
                popups.add("УГРОЗА!", (255, 80, 70), t, 1.2, key="threat")
            st["state"] = "УГРОЗА!"
        elif st["state"] == "УГРОЗА!" and (not st["threat_on"] or max(tL, tR) < 0.05):
            st["state"] = "ищет еду"

    # еда съедается
    if st["state"] == "ест" and near and filt["mn9"] > 3:      # сосёт только когда хоботок работает
        fs = food_state[near[0]]
        fs["eat_t"] = fs.get("eat_t", 0.0) + SYNC_MS / 1000
        fs["left"] = max(0.0, 1 - fs["eat_t"] / EAT_TIME)
        sc_ = 0.35 + 0.65 * fs["left"]
        m.geom_size[fs["geoms"]] = fs["size0"] * sc_
        m.geom_pos[fs["geoms"]] = fs["gpos0"] * sc_
        st["hunger"] = max(0.0, st["hunger"] - 0.35 * SYNC_MS / 1000 / EAT_TIME)
        if fs["eat_t"] >= EAT_TIME:
            fs["alive"] = False
            d.mocap_pos[fs["mocap"]] = [0, 0, -50]
            st["eaten"] += 1; st["state"] = "ищет еду"
            if fs["kind"] in ("syrup", "banana"):
                st["sticky"] = 1.0
                popups.add("липко!  щетинки головы сигналят мозгу", (255, 225, 150), t, 1.6, key=f"sticky{st['eaten']}")
            popups.add(f"+1   {FOOD_RU[fs['kind']]} съеден(а)", (120, 255, 140), t, 1.8)
            if st["eaten"] == len(FOODS):
                st["end_t"] = t + 1.2
            if not st["threat_done"] and st["eaten"] == 1:
                # через полсекунды с левого бока выкатывается шар
                st["threat_t0"] = t + 0.5

    # запуск угрозы
    if not st["threat_done"] and not st["threat_on"] and st["eaten"] >= 1 and t >= st["threat_t0"] > 0:
        ang = np.deg2rad(65)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        st["threat_pos"] = pos + rot @ h * 17.0
        v = pos - st["threat_pos"]; st["threat_vel"] = v / np.linalg.norm(v) * 17.0
        st["threat_on"], st["theta_prev"] = True, 0.0

    # ---------- тело двигается 14.4 мс ----------
    snap = (d.qpos.copy(), d.qvel.copy(), d.act.copy())       # снимок на случай «взрыва» физики
    for _ in range(steps_phys):
        tt = sim.curr_time
        prob_target = float(np.clip(filt["mn9"] / 12, 0, 1)) if st["state"] == "ест" else 0.0
        proboscis += (prob_target - proboscis) * 0.004
        flap, spread = 0.0, 0.0
        if st["flying"]:
            s = (tt - st["fly_t0"]) / 0.75
            if s >= 1:
                st["flying"] = False; st["gf_cool"] = 1.5; st["state"] = "ищет еду"
                d.qvel[ROOT_V:ROOT_V + 6] = 0
                popups.add("посадка", (200, 230, 255), tt, 1.0, big=False,
                           pos=cam.project([*fly_xy(), 1.0], W, H))
            else:
                e = 0.5 - 0.5 * np.cos(np.pi * s)
                p = st["fly_from"] + (st["fly_to"] - st["fly_from"]) * e
                p[2] += 3.5 * np.sin(np.pi * s)
                q0 = Rot.from_quat(st["fly_q0"][[1, 2, 3, 0]])
                q = (Rot.from_euler("z", st["fly_dpsi"] * e) * q0).as_quat()
                d.qpos[ROOT_Q:ROOT_Q + 3] = p
                d.qpos[ROOT_Q + 3:ROOT_Q + 7] = q[[3, 0, 1, 2]]
                d.qvel[ROOT_V:ROOT_V + 6] = 0
                spread = min(1.0, s * 6, (1 - s) * 6)
                flap = 0.9 * np.sin(2 * np.pi * 30 * tt) * spread
        if st["dead"]:                       # «лапками вверх»
            s_d = min(1.0, (tt - st["death_t"]) / 0.45)
            q0 = Rot.from_quat(st["die_q0"][[1, 2, 3, 0]])
            q = (Rot.from_rotvec(st["die_axis"] * np.pi * (0.5 - 0.5 * np.cos(np.pi * s_d))) * q0).as_quat()
            p_ = st["die_p0"].copy()
            p_[2] = st["die_p0"][2] + 1.2 * np.sin(np.pi * s_d) - 0.35 * s_d
            d.qpos[ROOT_Q:ROOT_Q + 3] = p_
            d.qpos[ROOT_Q + 3:ROOT_Q + 7] = q[[3, 0, 1, 2]]
            d.qvel[ROOT_V:ROOT_V + 6] = 0
        pose.apply(proboscis, spread, flap)
        try:
            if st["state"] == "чистится":
                obs, *_ = SingleFlySimulation.step(sim, {"joints": groom_angles(tt - st["groom_t0"]),
                                                          "adhesion": np.array([0, 1, 1, 0, 1, 1])})
            else:
                obs, *_ = sim.step(action)
        except PhysicsError:
            # физика «взорвалась» -> откат к снимку, скорости в ноль, едем дальше
            crashes.append(dict(t=float(tt), state=st["state"], action=[float(a_) for a_ in action]))
            d.qpos[:], d.qvel[:], d.act[:] = snap[0], 0.0, snap[2]
            mujoco.mj_forward(m, d)
            if st["state"] == "чистится":
                st["state"] = "ищет еду"
            break

        # след и угроза — в мировом времени
        if tt - st["last_trail"] > 0.05 and not st["flying"]:
            st["last_trail"] = tt
            kk = st["trail_k"] % len(trail_mocap)
            d.mocap_pos[trail_mocap[kk]] = [*fly_xy(), 0]
            trail_t[kk] = tt; st["trail_k"] += 1
        if st["threat_on"]:
            st["threat_pos"] = st["threat_pos"] + st["threat_vel"] * PHYS_DT
            d.mocap_pos[threat_mocap] = [*st["threat_pos"], THREAT_R]
            axis = np.array([-st["threat_vel"][1], st["threat_vel"][0], 0]); axis /= np.linalg.norm(axis)
            roll = Rot.from_rotvec(axis * np.linalg.norm(st["threat_vel"]) * (tt - st["threat_t0"]) / THREAT_R).as_quat()
            d.mocap_quat[threat_mocap] = roll[[3, 0, 1, 2]]
            if np.linalg.norm(st["threat_pos"] - pos) > 30 and tt - st["threat_t0"] > 1.5:
                st["threat_on"], st["threat_done"] = False, True
                d.mocap_pos[threat_mocap] = [0, 0, -30]

        if tt >= next_frame:
            next_frame += FRAME_DT
            age = tt - trail_t
            m.geom_rgba[trail_geom, 3] = np.clip(0.75 * np.exp(-age / 2.5), 0, 0.75) * (trail_t > 0)
            for fs in food_state:
                if fs["alive"]:
                    m.geom_rgba[fs["geoms"][0], 3] = 0.10 + 0.07 * np.sin(tt * 7)
            traj.append(fly_xy())
            cam.update(fly_xy(), heading(), shot_for_state(), FRAME_DT)
            writer.append_data(compose_frame(eye, tt))

    log.append(dict(t=float(t), x=float(pos[0]), y=float(pos[1]), state=st["state"], eye=[float(e) for e in eye],
                    sL=float(filt["sL"]), sR=float(filt["sR"]), mn9=float(filt["mn9"]), gf=float(filt["gf"]),
                    fear=float(filt["fear"]), pam=float(filt["pam"]), hunger=float(st["hunger"]),
                    n_active=int((brain_hz > 0).sum()), eaten=st["eaten"]))

writer.close()
if crashes:
    print(f"откатов физики: {len(crashes)}: {crashes[:10]}")
json.dump(log, open(Path(args.out).with_suffix(".json"), "w"))
print(f"готово: {args.out}; съедено {st['eaten']}/{len(FOODS)}; взлётов: "
      f"{sum(1 for a, b_ in zip(log, log[1:]) if b_['state'] == 'ВЗЛЁТ!' and a['state'] != 'ВЗЛЁТ!')}")
