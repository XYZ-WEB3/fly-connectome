"""Прототип стрима: муха в игровом сетапе смотрит ролик.

Слои: фон (5 с в цикле) + ролик на экране + муха на шаре (рендер с альфой) + интерфейс.
Мозг — коннектом FlyWire с противоэпилептикой, считается блоками через CUDA Graph.
Донаты пока демо (по расписанию) — потом подключим настоящие.
"""
import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import brain as B          # noqa: E402
import hud                 # noqa: E402
from eye import Eye        # noqa: E402
from flylayer import FlyOnBall   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--clip", default=str(HERE / "testclip.mp4"))
ap.add_argument("--title", default="тестовый ролик: жук, полосы, угроза")
ap.add_argument("--out", default=str(HERE / "proto.mp4"))
ap.add_argument("--secs", type=float, default=20)
ap.add_argument("--fly_no", type=int, default=5)
args = ap.parse_args()

W, H, FPS = 1920, 1080, 30
DT = 1 / FPS
DONATIONS = [(13.0, "Вася_2007", 100), (16.5, "fly_enjoyer", 300)]     # демо: (время, ник, сумма)
CHAT = [(1.0, "neuro_kot", "она реально за жуком головой ведёт"), (4.0, "Вася_2007", "муха смотрит ютуб лол"),
        (8.0, "sci_girl", "полосы ей не страшны, а круг?"), (10.3, "neuro_kot", "ВЗЛЕТЕЛА"),
        (13.2, "fly_enjoyer", "покормите её!"), (17.0, "sci_girl", "хоботок видно!")]

# ---------- мозг ----------
ann = B.load_annotations()
brain = B.FlyBrain(params=dict(U=0.1, tau_d=300, a_inc=0.1, tau_a=3000))
ty = lambda ts, s=None: [brain.idx[r] for r in ann[ann.cell_type.isin(ts) & ((ann.side == s) if s else True)].index
                         if r in brain.idx]
LC10 = ["LC10a", "LC10c-1", "LC10c-2", "LC10d", "LC10e"]
SUGAR_IDS = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
             720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
             720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
             720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
             720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
             720575940611875570]
G = dict(obj_L=ty(LC10, "left"), obj_R=ty(LC10, "right"), loom_L=ty(["LPLC2", "LC4"], "left"),
         loom_R=ty(["LPLC2", "LC4"], "right"), sL=ty(["DNa01", "DNa02"], "left"), sR=ty(["DNa01", "DNa02"], "right"),
         gf=ty(["DNp01"]), fear=ty(["DNp01", "DNp02", "DNp04", "DNp11"]), mn9=[brain.idx[720575940660219265]],
         sugar=[brain.idx[r] for r in SUGAR_IDS if r in brain.idx])
brain.enable_graph()
cloud = hud.BrainCloud(brain.ids, w=500, h=270, annot_path=B.PATH_ANN)

# ---------- слои ----------
print("загружаю фон…")
bg = [f for f in iio.imiter(HERE / "bg_loop.mp4")]
corners = np.load(HERE / "screen_corners.npy")
fly = FlyOnBall(W, H)
eye = Eye()
clip = iio.imiter(args.clip)
clip_first = None
filt = {k: 0.0 for k in ("sL", "sR", "gf", "fear", "mn9")}
LOG = []
st = dict(flight_t=-9.0, drop=0.0, prob=0.0, fed=0, spread=0.0, hunger=0.7, events=[])
popups = hud.Popups()
wr = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, macro_block_size=8)


def warp_screen(frame, dst, glow=True):
    h, w = frame.shape[:2]
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
    scr = cv2.warpPerspective(frame, Hm, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Hm, (W, H))
    return scr, mask


n_frames = int(args.secs * FPS)
for i in tqdm(range(n_frames)):
    t = i * DT
    try:
        frame = next(clip)
    except StopIteration:
        clip = iio.imiter(args.clip); frame = next(clip)
    # ---------- глаз -> мозг ----------
    e = eye(frame, DT)
    brain.set_input(G["obj_L"], 60 * e["obj_L"]); brain.set_input(G["obj_R"], 60 * e["obj_R"])
    brain.set_input(G["loom_L"], 80 * e["loom_L"]); brain.set_input(G["loom_R"], 80 * e["loom_R"])
    for (td, nick, amount) in DONATIONS:
        if abs(t - td) < DT / 2:
            st["drop"] = 1.0
            popups.add(f"ДОНАТ {amount} руб от {nick}:  капля сахара!", (255, 210, 90), t, 2.5)
    touching = st["drop"] > 0.05 and st["prob"] > 0.5            # хоботок дотянулся до капли
    brain.set_input(G["sugar"], 160.0 if st["drop"] > 0.05 else 0.0)
    hz = brain.run(1000 * DT, tensor=True).cpu().numpy() * FPS
    cloud.update(hz)
    for k in filt:
        filt[k] += min(1, DT / 0.06) * (hz[G[k]].mean() - filt[k])
    # ---------- решения мозга -> тело ----------
    turn = float(np.tanh(0.12 * (filt["sL"] - filt["sR"])))
    if filt["gf"] > 40 and t - st["flight_t"] > 1.2:
        st["flight_t"] = t
        popups.add("ВЗЛЁТ!  DNp01  (привязана: летит на месте)", (130, 210, 255), t, 2.2)
    flying = t - st["flight_t"] < 0.9
    st["spread"] += (float(flying) - st["spread"]) * 0.35
    flap = 0.9 * np.sin(2 * np.pi * 7.5 * t * 4) * st["spread"]
    eating = st["drop"] > 0.05 and filt["mn9"] > 3
    st["prob"] += ((1.0 if eating else 0.0) - st["prob"]) * 0.3
    if touching and eating:
        st["drop"] = max(0.0, st["drop"] - DT / 1.6)                 # капля убывает, пока сосёт
        st["hunger"] = max(0.0, st["hunger"] - DT * 0.15)
        if st["drop"] <= 0.05:
            st["fed"] += 1
            popups.add("капля выпита  +1", (120, 255, 140), t, 1.6)
    base = 0.0 if flying or eating else 0.45
    dL, dR = base * (1 - max(turn, 0)), base * (1 + min(turn, 0))
    fly.update(DT, dL, dR, proboscis=st["prob"], head_turn=turn, wing_spread=st["spread"], wing_flap=flap)
    fly.set_drop(st["drop"])
    rgba = fly.render()
    # ---------- композитинг ----------
    img = bg[i % len(bg)].astype(np.float32)
    scr, mask = warp_screen(frame, corners)
    m_ = (mask[..., None] / 255.0)
    img = img * (1 - m_) + scr.astype(np.float32) * 0.95 * m_
    glow = cv2.GaussianBlur(scr, (0, 0), 40).astype(np.float32) * 0.35                  # свет экрана на сцене
    img = img + glow * (1 - m_)
    a = rgba[..., 3:4].astype(np.float32) / 255
    lit = rgba[..., :3].astype(np.float32) + cv2.resize(glow, (W, H)) * 0.15             # экран подсвечивает муху
    img = img * (1 - a) + lit * a
    base_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    # заголовок
    hud.panel(dr, (24, 22, 900, 128))
    dr.text((44, 30), f"МУХА №{args.fly_no}  ·  СМОТРИТ РОЛИКИ", font=hud.F[30], fill=(255, 255, 255, 255))
    dr.text((44, 72), f"сейчас: {args.title}", font=hud.F[18], fill=(170, 200, 255, 255))
    dr.text((44, 98), "мозг: коннектом FlyWire, 139 тыс. нейронов · глаз: упрощённая сетчатка", font=hud.F[16],
            fill=(200, 200, 205, 230))
    # эмоции и решения
    px0, py0, pw = 24, H - 420, 380
    hud.panel(dr, (px0, py0, px0 + pw, H - 16))
    dr.text((px0 + 18, py0 + 12), "ЧТО ЧУВСТВУЕТ (с нейронов)", font=hud.F[18], fill=(255, 255, 255, 255))
    hud.gauge(dr, px0 + 18, py0 + 44, pw - 36, "Страх · нейроны побега DNp", filt["fear"], (255, 70, 70), 120)
    hud.gauge(dr, px0 + 18, py0 + 88, pw - 36, "Интерес · следит за объектом",
              abs(filt["sL"] - filt["sR"]), (120, 220, 255), 40)
    hud.gauge(dr, px0 + 18, py0 + 132, pw - 36, "Голод", st["hunger"] * 100, (255, 190, 60), 100,
              f"{st['hunger'] * 100:.0f}%")
    dr.text((px0 + 18, py0 + 188), "РЕШЕНИЯ МОЗГА", font=hud.F[18], fill=(255, 255, 255, 255))
    hud.steer_gauge(dr, px0 + 18, py0 + 220, pw - 36, turn)
    hud.gauge(dr, px0 + 18, py0 + 264, pw - 36, "Взлёт · гигантское волокно DNp01", filt["gf"], (120, 200, 255), 150)
    hud.gauge(dr, px0 + 18, py0 + 308, pw - 36, "Хоботок · MN9", filt["mn9"], (255, 150, 50), 60)
    dr.text((px0 + 18, py0 + 356), f"глаз: объект L {e['obj_L']:.2f} R {e['obj_R']:.2f}   угроза L {e['loom_L']:.2f} "
            f"R {e['loom_R']:.2f}", font=hud.F[14], fill=(200, 230, 200, 255))
    dr.text((px0 + 18, py0 + 378), f"покормлена: {st['fed']} раз", font=hud.F[18], fill=(255, 210, 120, 255))
    # мозг
    bimg, n_act = cloud.render()
    bx, by = W - cloud.w - 24, 150
    hud.panel(dr, (bx - 12, by - 44, W - 12, by + cloud.h + 12))
    dr.text((bx, by - 36), f"МОЗГ СЕЙЧАС · горят {n_act} нейронов", font=hud.F[18], fill=(255, 255, 255, 255))
    # чат (демо)
    cx0, cy0 = W - 524, H - 300
    hud.panel(dr, (cx0, cy0, W - 12, H - 16))
    dr.text((cx0 + 16, cy0 + 10), "ЧАТ (демо)", font=hud.F[18], fill=(255, 255, 255, 255))
    msgs = [c for c in CHAT if c[0] <= t][-6:]
    for j, (_, nick, txt) in enumerate(msgs):
        dr.text((cx0 + 16, cy0 + 44 + j * 38), f"{nick}:", font=hud.F[16], fill=(150, 200, 255, 255))
        dr.text((cx0 + 26 + dr.textlength(nick + ':', font=hud.F[16]), cy0 + 44 + j * 38), txt, font=hud.F[16],
                fill=(235, 235, 240, 255))
    base_img.alpha_composite(ov)
    base_img.alpha_composite(bimg.convert("RGBA"), (bx, by))
    top = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    popups.draw(ImageDraw.Draw(top), t, W, H)
    base_img.alpha_composite(top)
    wr.append_data(np.asarray(base_img.convert("RGB")))
    LOG.append(dict(t=round(t, 3), **{k: round(float(v), 2) for k, v in e.items()},
                    **{f"f_{k}": round(float(v), 1) for k, v in filt.items()}, drop=round(st["drop"], 2),
                    prob=round(st["prob"], 2), flying=bool(flying)))
wr.close()
import json
json.dump(LOG, open(Path(args.out).with_suffix(".json"), "w"))
print("готово:", args.out)
