"""Красивая сцена для мухи: деревянный стол, небо, свет с тенями,
еда (банан, капля сиропа, ягода), светящийся след, «хищник» и кинокамера."""
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from flygym.arena.base import BaseArena

ROOT = Path(__file__).parent
NO_COLLIDE = dict(contype=0, conaffinity=0, group=1)
_fx_id = [0]


def fx(**kw):
    """Декоративный geom: без коллизий. В имени 'sensor' — иначе flygym
    считает его полом и добавляет пары контактов с ногами."""
    _fx_id[0] += 1
    kw.setdefault("name", f"fx{_fx_id[0]}")
    kw["name"] += "_sensorfx"
    return {**NO_COLLIDE, **kw}


def make_wood_texture(path, size=1024, seed=3):
    """Процедурная текстура светлого дерева (доски + волокна)."""
    rng = np.random.default_rng(seed)
    def noise(res):
        n = rng.random((res, res)).astype(np.float32)
        return cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    warp = noise(6) * 0.08 + noise(24) * 0.015
    plank = np.floor(x * 4)                                  # 4 доски
    grain = np.sin((x * 4 - plank) * 90 + warp * 120 + plank * 7.3)
    rings = np.sin((y + noise(4) * 0.3) * 18 + (x * 4 - plank) * 3) * 0.5 + 0.5
    v = 0.55 + 0.05 * grain + 0.06 * rings + 0.05 * (noise(64) - 0.5) + 0.04 * np.sin(plank * 12.1)
    gap = np.abs((x * 4) - np.round(x * 4)) < 0.0015                      # щели между досками
    v = np.where(gap, 0.25, v)
    base = np.array([135, 175, 210], np.float32) / 255                 # BGR светлый дуб
    img = np.clip(v[..., None] * base[None, None] * 1.25, 0, 1)
    cv2.imencode(".png", (img * 255).astype(np.uint8))[1].tofile(str(path))   # imwrite не любит кириллицу в пути


class KitchenArena(BaseArena):
    def __init__(self, foods, n_trail=160, size=(300, 300)):
        """foods: список (kind, x, y, yaw), kind in {banana, syrup, berry}."""
        super().__init__()
        r = self.root_element
        tex_path = ROOT / "data" / "wood.png"
        if not tex_path.exists():
            make_wood_texture(tex_path)

        # --- небо и стол ---
        r.asset.add("texture", type="skybox", builtin="gradient", rgb1=(0.98, 0.93, 0.85),
                    rgb2=(0.35, 0.28, 0.22), width=512, height=512)
        wood = r.asset.add("texture", name="wood", type="2d", file=str(tex_path))
        mat = r.asset.add("material", name="table", texture=wood, texrepeat=(6, 6),
                          reflectance=0.08, specular=0.2, shininess=0.3)
        r.worldbody.add("geom", type="plane", name="ground", material=mat, size=(*size, 1),
                        friction=(1, 0.005, 0.0001), conaffinity=0)

        self.friction = (1, 0.005, 0.0001)   # иначе берётся 100 из BaseArena -> физика в 15 раз медленнее
        glow = r.asset.add("material", name="glow", emission=1.0, specular=0)
        glossy = r.asset.add("material", name="glossy", specular=0.9, shininess=0.9, reflectance=0.25)

        # --- еда ---
        self.foods = []
        for i, (kind, x, y, yaw) in enumerate(foods):
            body = r.worldbody.add("body", name=f"food_{i}", pos=(x, y, 0), mocap=True,
                                   euler=(0, 0, yaw))
            body.add("geom", type="cylinder", size=(1.9, 0.005), pos=(0, 0, 0.012),
                     rgba=(1.0, 0.8, 0.35, 0.14), material=glow, **fx(name=f"food_{i}_glow"))
            if kind == "banana":
                # гладкая дуга из капсул, тёмные кончики
                n, R, span = 24, 3.2, np.deg2rad(85)
                pts = [np.array([R * np.sin(a), R * (1 - np.cos(a)) - 0.9, 0])
                       for a in np.linspace(-span / 2, span / 2, n)]
                rads = [0.28 + 0.34 * np.sin(np.pi * k / (n - 1)) ** 0.5 for k in range(n)]
                for k in range(n - 1):
                    rr = 0.5 * (rads[k] + rads[k + 1])
                    p1, p2 = pts[k] + [0, 0, rr], pts[k + 1] + [0, 0, rr]
                    tip = k < 2 or k > n - 4
                    body.add("geom", type="capsule", fromto=(*p1, *p2), size=(rr,),
                             rgba=(0.42, 0.3, 0.12, 1) if tip else (0.97, 0.84, 0.28, 1),
                             material=glossy, **fx())
            elif kind == "syrup":
                body.add("geom", type="ellipsoid", size=(1.3, 1.1, 0.38), pos=(0, 0, 0.2),
                         rgba=(0.78, 0.33, 0.02, 0.78), material=glossy, **fx())
                body.add("geom", type="ellipsoid", size=(0.35, 0.22, 0.08), pos=(-0.35, 0.3, 0.52),
                         rgba=(1, 0.95, 0.85, 0.7), material=glow, **fx())
            elif kind == "berry":
                body.add("geom", type="sphere", size=(1.0,), pos=(0, 0, 1.0),
                         rgba=(0.78, 0.05, 0.12, 1), material=glossy, **fx())
                for k in range(5):
                    a = 2 * np.pi * k / 5
                    body.add("geom", type="ellipsoid", size=(0.45, 0.16, 0.06),
                             pos=(0.3 * np.cos(a), 0.3 * np.sin(a), 1.93), euler=(0, 0.3, a),
                             rgba=(0.2, 0.55, 0.15, 1), **fx())
                for k in range(14):
                    th, ph = np.arccos(1 - 2 * (k + 0.5) / 14), np.pi * (1 + 5 ** 0.5) * k
                    if th < 0.5:
                        continue
                    body.add("geom", type="sphere", size=(0.07,),
                             pos=(1.0 * np.sin(th) * np.cos(ph), 1.0 * np.sin(th) * np.sin(ph), 1.0 + np.cos(th)),
                             rgba=(0.95, 0.85, 0.3, 1), **fx())
            self.foods.append(dict(kind=kind, pos=np.array([x, y]), body=body))

        # --- светящийся след ---
        self.trail = []
        for k in range(n_trail):
            b = r.worldbody.add("body", name=f"trail_{k}", pos=(0, 0, -5), mocap=True)
            b.add("geom", type="cylinder", size=(0.08, 0.004), pos=(0, 0, 0.011),
                  rgba=(0.3, 0.9, 1.0, 0.0), material=glow, **fx(name=f"trail_{k}"))
            self.trail.append(b)

        # --- «хищник»: тёмный шар с красным поясом ---
        self.threat = r.worldbody.add("body", name="threat", pos=(0, 0, -20), mocap=True)
        self.threat.add("geom", type="sphere", size=(2.0,), rgba=(0.06, 0.06, 0.08, 1), material=glossy, **fx())
        self.threat.add("geom", type="cylinder", size=(2.03, 0.18), rgba=(0.9, 0.08, 0.05, 1), material=glow, **fx())

        # --- кинокамера (двигаем сами) ---
        r.worldbody.add("camera", name="cine", pos=(-10, 0, 6), fovy=48, mode="fixed")

    def init_lights(self):
        wb = self.root_element.worldbody
        wb.add("light", name="sun", directional=True, castshadow=False, pos=(0, 0, 50), dir=(0.3, 0.2, -1),
               diffuse=(0.38, 0.36, 0.33), specular=(0.15, 0.15, 0.15), ambient=(0.14, 0.13, 0.12))

    def add_fly_light(self, fly):
        """Прожектор над мухой — даёт чёткие тени от ног. Плюс настройки качества
        (их держит модель мухи, иначе конфликт при сборке)."""
        v = fly.model.visual
        v.quality.shadowsize = 2048
        v.quality.offsamples = 4
        getattr(v, "global").offwidth = 1920
        getattr(v, "global").offheight = 1080
        v.map.haze = 0.4
        v.map.znear = 0.005
        v.map.zfar = 60
        v.headlight.ambient = (0.25, 0.25, 0.25)
        v.headlight.diffuse = (0.25, 0.25, 0.25)
        v.headlight.specular = (0.05, 0.05, 0.05)
        thorax = fly.model.find("body", "Thorax")
        thorax.add("light", name="key", mode="track", directional=False, castshadow=True,
                   pos=(-6, 5, 22), dir=(0.25, -0.2, -1), cutoff=40, exponent=4,
                   diffuse=(0.75, 0.7, 0.62), specular=(0.3, 0.3, 0.3), attenuation=(1, 0, 0))

    @staticmethod
    def add_extra_joints(fly):
        """Суставы, которых нет в стандартной модели: хоботок и крылья.
        Двигаем их кинематически (выставляем угол), это «косметика»."""
        m = fly.model
        m.find("body", "Rostrum").add("joint", name="joint_Rostrum", type="hinge", axis=(0, 1, 0),
                                      range=(-3, 3), damping=1e-4, armature=1e-6)
        hau = m.find("body", "Haustellum")
        hau.add("joint", name="joint_Haustellum", type="hinge", axis=(0, 1, 0),
                range=(-3, 3), damping=1e-4, armature=1e-6)
        # визуальное продолжение хоботка (у меша он очень короткий) + лабеллум
        col = (0.62, 0.42, 0.18, 0.0)          # альфа 0 = спрятан, включаем при выдвижении
        ext = hau.add("body", name="ProbExt")
        ext.add("geom", type="capsule", fromto=(-0.05, 0, -0.05, 0.05, 0, -0.42),
                size=(0.055,), rgba=col, mass=1e-9, **fx(name="prob_tube"))
        for k in (-1, 1):
            ext.add("geom", type="ellipsoid", size=(0.09, 0.07, 0.05),
                    pos=(0.07, 0.055 * k, -0.46), rgba=col, mass=1e-9, **fx(name=f"prob_lobe{k}"))
        for side in "LR":
            w = m.find("body", f"{side}Wing")
            w.add("joint", name=f"joint_{side}Wing_yaw", type="hinge", axis=(0, 0, 1), range=(-3, 3),
                  damping=1e-4, armature=1e-6)
            w.add("joint", name=f"joint_{side}Wing_roll", type="hinge", axis=(1, 0, 0), range=(-3, 3),
                  damping=1e-4, armature=1e-6)

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

    def _get_max_floor_height(self):
        return 0.0


class CineCamera:
    """Плавная камера: следует за мухой, на событиях меняет план."""
    def __init__(self, physics, name="cine"):
        self.p = physics
        self.m, self.d = physics.model.ptr, physics.data.ptr
        self.cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        self.pos = None
        self.look = None
        self.fovy = float(self.m.cam_fovy[self.cid])

    def update(self, target, heading, shot, dt_frame):
        """shot = dict(dist, height, side, lead, ease)."""
        h = heading / (np.linalg.norm(heading) + 1e-9)
        side = np.array([-h[1], h[0]])
        want_xy = target - h * shot["dist"] + side * shot.get("side", 0)
        want = np.array([*want_xy, shot["height"]])
        look = np.array([*(target + h * shot.get("lead", 1.5)), shot.get("look_z", 0.6)])
        k = 1 - np.exp(-dt_frame / shot.get("ease", 0.35))
        self.pos = want if self.pos is None else self.pos + (want - self.pos) * k
        self.look = look if self.look is None else self.look + (look - self.look) * k
        f = self.look - self.pos
        f /= np.linalg.norm(f)
        right = np.cross(f, [0, 0, 1]); right /= np.linalg.norm(right)
        up = np.cross(right, f)
        mat = np.stack([right, up, -f], axis=1)
        q = Rot.from_matrix(mat).as_quat()              # x y z w
        self.m.cam_pos[self.cid] = self.pos
        self.m.cam_quat[self.cid] = [q[3], q[0], q[1], q[2]]
        mujoco.mj_camlight(self.m, self.d)

    def project(self, p3, W, H):
        """3D -> пиксели (для подписей над объектами)."""
        pos = self.d.cam_xpos[self.cid]
        mat = self.d.cam_xmat[self.cid].reshape(3, 3)
        v = mat.T @ (np.asarray(p3) - pos)
        if v[2] >= -1e-6:
            return None
        f = (H / 2) / np.tan(np.deg2rad(self.fovy) / 2)
        return int(W / 2 + f * v[0] / -v[2]), int(H / 2 - f * v[1] / -v[2])


class FlyPoser:
    """Кинематика «косметических» суставов: хоботок (0..1), крылья."""
    def __init__(self, physics, fly_name):
        self.p = physics
        j = lambda n: physics.named.data.qpos.axes.row.names.index(f"{fly_name}/{n}") if False else None
        self.m, self.d = physics.model.ptr, physics.data.ptr
        def adr(n):
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{fly_name}/{n}")
            return self.m.jnt_qposadr[jid], self.m.jnt_dofadr[jid]
        self.ros, self.hau = adr("joint_Rostrum"), adr("joint_Haustellum")
        self.wings = {s: (adr(f"joint_{s}Wing_yaw"), adr(f"joint_{s}Wing_roll")) for s in "LR"}
        self.prob_geoms = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"{fly_name}/{n}")
                           for n in ("prob_tube_sensorfx", "prob_lobe-1_sensorfx", "prob_lobe1_sensorfx")]

    def _set(self, a, val):
        self.d.qpos[a[0]] = val
        self.d.qvel[a[1]] = 0.0

    def apply(self, proboscis=0.0, wing_spread=0.0, wing_flap=0.0):
        self._set(self.ros, -1.0 * proboscis)
        self._set(self.hau, 1.0 * proboscis)
        for g in self.prob_geoms:
            self.m.geom_rgba[g, 3] = float(np.clip((proboscis - 0.25) / 0.5, 0, 1))
        for s, k in (("L", 1), ("R", -1)):
            yaw, roll = self.wings[s]
            self._set(yaw, k * 1.2 * wing_spread)
            self._set(roll, k * wing_flap)
