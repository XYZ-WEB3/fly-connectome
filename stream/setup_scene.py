"""Игровой сетап мухи для стрима (масштаб — миллиметры, муха ~2.5 мм).

Координаты: муха в (0,0,0), шар под ней — как в арене Ball из flygym
(центр (-0.1,-0.05,-5.2), радиус 5.39). Тогда камера и свет одинаковы
для фона и для слоя мухи.

Фон (стол, монитор, RGB-подсветка, кормушка) рендерится ОДИН раз: 5 с в цикле.
Муха на шаре и картинка на экране — отдельные слои поверх (см. compositor).
"""
from pathlib import Path

import cv2
import numpy as np
from dm_control import mjcf

ROOT = Path(__file__).parent
BALL_C = np.array([-0.0987, -0.0544, -5.2031])     # центр шара (как в flygym Ball)
BALL_R = 5.3909
SCREEN = dict(center=(14.0, 0.0, 1.8), w=16.0, h=9.0)   # экран перед мухой (x — вперёд)
PIPETTE_TIP = (1.95, 0.4, -0.35)                  # кончик кормушки — у хоботка
CAM = dict(pos=(-13.0, -10.5, 5.2), look=(5.5, 0.5, 0.2), fovy=40)
LIGHTS = [dict(name="key", pos=(-10, -14, 24.8), dir=(0.3, 0.45, -1), castshadow=True, cutoff=45, exponent=3,
               diffuse=(0.9, 0.85, 0.8), specular=(0.3, 0.3, 0.3)),
          dict(name="rim", pos=(25, 10, 8.8), dir=(-1, -0.3, -0.4), castshadow=False, cutoff=50, exponent=2,
               diffuse=(0.25, 0.35, 0.9), specular=(0.2, 0.2, 0.4))]
LOOP_S, FPS = 5.0, 30


def carbon_texture(path, size=512):
    """Тёмный «карбон» для стола."""
    y, x = np.mgrid[0:size, 0:size]
    weave = ((x // 8 + y // 8) % 2).astype(np.float32)
    n = np.random.default_rng(1).random((size, size)).astype(np.float32)
    v = 0.10 + 0.05 * weave + 0.02 * n
    img = np.clip(np.stack([v * 1.05, v, v * 0.95], -1), 0, 1)
    cv2.imencode(".png", (img * 255).astype(np.uint8))[1].tofile(str(path))


def add_camera_and_lights(root):
    """Одинаковые камера и свет — и в фоне, и в слое мухи."""
    wb = root.worldbody
    for L in LIGHTS:
        wb.add("light", directional=False, **L)
    wb.add("body", name="cam_target", pos=CAM["look"])
    wb.add("camera", name="stream", pos=CAM["pos"], fovy=CAM["fovy"], mode="targetbody", target="cam_target")


def set_quality(root):
    v = root.visual
    v.quality.shadowsize = 4096
    v.quality.offsamples = 8
    getattr(v, "global").offwidth = 1920
    getattr(v, "global").offheight = 1080
    v.map.znear = 0.01
    v.map.zfar = 200
    v.map.haze = 0.3
    v.headlight.ambient = (0.12, 0.12, 0.14)
    v.headlight.diffuse = (0.2, 0.2, 0.22)
    v.headlight.specular = (0.05, 0.05, 0.05)


def build_bg():
    r = mjcf.RootElement(model="setup")
    set_quality(r)
    r.asset.add("texture", type="skybox", builtin="gradient", rgb1=(0.05, 0.05, 0.08), rgb2=(0.0, 0.0, 0.0),
                width=256, height=256)
    tex = ROOT / "carbon.png"
    if not tex.exists():
        carbon_texture(tex)
    t_desk = r.asset.add("texture", name="carbon", type="2d", file=str(tex))
    m_desk = r.asset.add("material", name="desk", texture=t_desk, texrepeat=(30, 30), reflectance=0.25,
                         specular=0.6, shininess=0.8)
    m_glow = r.asset.add("material", name="glow", emission=1.0, specular=0)
    m_black = r.asset.add("material", name="black", specular=0.8, shininess=0.9, reflectance=0.1)
    m_metal = r.asset.add("material", name="metal", specular=1.0, shininess=1.0, reflectance=0.4)
    add_camera_and_lights(r)

    # всё «железо» — в системе координат центра шара
    st = r.worldbody.add("body", name="stage", pos=tuple(BALL_C))
    R = BALL_R
    st.add("geom", name="desk", type="plane", size=(120, 120, 1), pos=(0, 0, -R - 3.0), material=m_desk)
    st.add("geom", name="wall", type="box", size=(0.5, 80, 40), pos=(34, 0, 20), rgba=(0.06, 0.06, 0.09, 1))
    # монитор
    sc = np.array(SCREEN["center"]) - BALL_C
    w, h = SCREEN["w"], SCREEN["h"]
    mon = st.add("body", name="monitor", pos=tuple(sc))
    mon.add("geom", name="bezel", type="box", size=(0.35, w / 2 + 0.5, h / 2 + 0.5), pos=(0.4, 0, 0), material=m_black,
            rgba=(0.04, 0.04, 0.05, 1))
    mon.add("geom", name="screen", type="box", size=(0.05, w / 2, h / 2), rgba=(0.02, 0.02, 0.03, 1))
    bottom = sc[2] - h / 2
    st.add("geom", name="stand", type="cylinder", size=(0.5, (bottom + R + 3.0) / 2),
           pos=(sc[0] + 1.0, 0, (bottom - R - 3.0) / 2), material=m_metal, rgba=(0.3, 0.3, 0.33, 1))
    st.add("geom", name="foot", type="cylinder", size=(3.0, 0.2), pos=(sc[0] + 1.0, 0, -R - 2.8),
           material=m_metal, rgba=(0.25, 0.25, 0.28, 1))
    for k in range(24):                      # RGB-лента под экраном
        yy = -w / 2 - 1 + (w + 2) * k / 23
        st.add("geom", name=f"led_{k}", type="box", size=(0.1, 0.35, 0.12), pos=(sc[0] + 1.2, yy, bottom - 0.9),
               material=m_glow, rgba=(1, 0, 0, 1))
    for k in range(40):                      # RGB-лента на стене
        yy = -40 + 80 * k / 39
        st.add("geom", name=f"wled_{k}", type="box", size=(0.1, 1.0, 0.25), pos=(33.4, yy, 2.0),
               material=m_glow, rgba=(1, 0, 0, 1))
    # чашка с «воздушной подушкой» под шаром + кольцо подсветки + стойка
    st.add("geom", name="cup", type="cylinder", size=(R * 0.8, 1.2), pos=(0, 0, -R + 0.2),
           material=m_black, rgba=(0.07, 0.07, 0.08, 1))
    for k in range(36):
        a = 2 * np.pi * k / 36
        st.add("geom", name=f"ring_{k}", type="box", size=(0.12, 0.38, 0.1),
               pos=(R * 0.84 * np.cos(a), R * 0.84 * np.sin(a), -R + 1.45),
               euler=(0, 0, a), material=m_glow, rgba=(0.2, 0.9, 1.0, 1))
    st.add("geom", name="post", type="cylinder", size=(0.9, 1.5), pos=(0, 0, -R - 1.5), material=m_metal,
           rgba=(0.2, 0.2, 0.22, 1))
    # кормушка-пипетка: из неё на донат капает сахар к хоботку
    tip = np.array(PIPETTE_TIP) - BALL_C
    st.add("geom", name="pipette", type="capsule", fromto=(*(tip + [0.6, 0.9, 7.0]), *(tip + [0, 0, 0.35])), size=(0.16,),
           rgba=(0.8, 0.9, 1.0, 0.22), material=m_metal)
    st.add("geom", name="pipette_tip", type="sphere", size=(0.13,), pos=tuple(tip), rgba=(1.0, 0.7, 0.2, 0.9),
           material=m_glow)
    return r


def project(physics, p3, W, H):
    import mujoco
    m, d = physics.model.ptr, physics.data.ptr
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "stream")
    pos, mat = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3)
    fy = (H / 2) / np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)
    v = mat.T @ (np.asarray(p3, float) - pos)
    return np.array([W / 2 + fy * v[0] / -v[2], H / 2 - fy * v[1] / -v[2]])


def screen_corners_px(physics, W, H):
    """Где на кадре углы экрана (TL, TR, BR, BL — как видит муха)."""
    cx, cy, cz = SCREEN["center"]
    w, h = SCREEN["w"], SCREEN["h"]
    return np.array([project(physics, (cx - 0.06, cy + y, cz + z), W, H)
                     for y, z in [(w / 2, h / 2), (-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2)]], np.float32)


def render_bg_loop(out=str(ROOT / "bg_loop.mp4"), W=1920, H=1080):
    import colorsys
    import imageio.v2 as imageio
    import mujoco
    phys = mjcf.Physics.from_mjcf_model(build_bg())
    m = phys.model.ptr
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    leds = [gid(f"led_{k}") for k in range(24)]
    wleds = [gid(f"wled_{k}") for k in range(40)]
    ring = [gid(f"ring_{k}") for k in range(36)]
    wr = imageio.get_writer(out, fps=FPS, codec="libx264", quality=9, macro_block_size=8)
    n = int(LOOP_S * FPS)
    for f in range(n):
        ph = f / n                                            # фаза 0..1 -> идеальная петля
        for i, g in enumerate(leds):
            m.geom_rgba[g, :3] = colorsys.hsv_to_rgb((ph + i / 24) % 1, 1, 1)
        for i, g in enumerate(wleds):
            m.geom_rgba[g, :3] = colorsys.hsv_to_rgb((ph + i / 40 * 0.5) % 1, 0.9, 0.8)
        for i, g in enumerate(ring):
            m.geom_rgba[g, :3] = colorsys.hsv_to_rgb((ph * 2 + i / 36) % 1, 0.8, 1)
        phys.forward()
        wr.append_data(phys.render(height=H, width=W, camera_id="stream"))
    wr.close()
    corners = screen_corners_px(phys, W, H)
    np.save(ROOT / "screen_corners.npy", corners)
    print("фон:", out, "кадров:", n, "углы экрана:", corners.round(0).tolist())


if __name__ == "__main__":
    render_bg_loop()
