"""Слой «муха на шаре» для стрима — кукольный режим (без физики ног).

Мозг выдаёт команды (руль, ход, хоботок, чистка, взлёт) -> ноги анимирует тот же
генератор шага (CPG), что и в обычной ходьбе flygym, шар вращается как у мух
в лабораторных установках. Рендер с маской (альфа), чтобы положить поверх фона.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).parent.parent))
from flygym import Fly                                              # noqa: E402
from flygym.arena.tethered import Ball                                        # noqa: E402
from flygym.simulation import SingleFlySimulation                    # noqa: E402
from flygym.examples.locomotion import CPGNetwork, PreprogrammedSteps  # noqa: E402
from flygym.examples.locomotion.turning_controller import _tripod_phase_biases, _tripod_coupling_weights  # noqa: E402

import setup_scene as S                                              # noqa: E402
from scene import KitchenArena, FlyPoser                             # noqa: E402


class FlyOnBall:
    def __init__(self, W=1920, H=1080):
        self.W, self.H = W, H
        arena = Ball()
        S.add_camera_and_lights(arena.root_element)
        # шар — белый «пенопласт» с лёгким узором (как в лаборатории)
        tex = arena.root_element.asset.add("texture", name="dots", type="2d",
                                           file=str(Path(__file__).parent / "ball_dots.png"))
        mat = arena.root_element.asset.add("material", name="foam", texture=tex, texrepeat=(3, 3),
                                           specular=0.15, shininess=0.2)
        for g in arena.root_element.find_all("geom"):
            g.rgba = (0.85, 0.83, 0.8, 1)
            g.material = mat
        drop = arena.root_element.worldbody.add("body", name="drop", pos=S.PIPETTE_TIP, mocap=True)
        drop.add("geom", name="drop_sensorfx", type="sphere", size=(0.28,), rgba=(1.0, 0.72, 0.15, 0.0),
                 contype=0, conaffinity=0, group=1)
        self.fly = Fly(spawn_pos=(0, 0, 0), enable_adhesion=False)
        KitchenArena.add_extra_joints(self.fly)                      # хоботок + крылья
        S.set_quality(self.fly.model)
        self.sim = SingleFlySimulation(fly=self.fly, arena=arena, cameras=[], timestep=1e-4)
        self.sim.reset()
        self.p = self.sim.physics
        self.m, self.d = self.p.model.ptr, self.p.data.ptr
        self.pose = FlyPoser(self.p, self.fly.name)
        self.steps = PreprogrammedSteps()
        self.cpg = CPGNetwork(timestep=1e-3, intrinsic_freqs=np.ones(6) * 12, intrinsic_amps=np.ones(6),
                              coupling_weights=_tripod_coupling_weights, phase_biases=_tripod_phase_biases,
                              convergence_coefs=np.ones(6) * 20)
        jadr = lambda n: self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        self.leg_adr = [jadr(f"{self.fly.name}/{j}") for j in self.fly.actuated_joints]
        self.head_yaw = jadr(f"{self.fly.name}/joint_Head_yaw")
        ball_j = next(j for j in range(self.m.njnt) if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL)
        self.ball_adr = self.m.jnt_qposadr[ball_j]
        self.ball_q = Rot.identity()
        self.fly_geoms = set(g for g in range(self.m.ngeom)
                             if mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or True)
        self.head_angle = 0.0

    def update(self, dt, drive_L, drive_R, proboscis=0.0, head_turn=0.0, wing_spread=0.0, wing_flap=0.0,
               groom_angles=None):
        """drive_L/R — как у контроллера ходьбы (0 = стоит, 1 = бежит)."""
        # --- ноги: CPG как у HybridTurningController ---
        action = np.array([drive_L, drive_R])
        self.cpg.intrinsic_amps = np.repeat(np.abs(action), 3)
        freqs = np.ones(6) * 12
        self.cpg.intrinsic_freqs = freqs
        for _ in range(max(1, int(round(dt / 1e-3)))):
            self.cpg.step()
        if groom_angles is not None:
            angles = groom_angles
        else:
            angles = np.concatenate([self.steps.get_joint_angles(leg, self.cpg.curr_phases[i],
                                                                 self.cpg.curr_magnitudes[i])
                                     for i, leg in enumerate(self.steps.legs)])
        self.d.qpos[self.leg_adr] = angles
        # --- шар крутится под ногами: вперёд + поворот ---
        fwd = 9.0 * (drive_L + drive_R) / 2           # мм/с по поверхности шара
        yaw = 2.2 * (drive_R - drive_L)               # рад/с (правые ноги сильнее -> поворот влево)
        dq = Rot.from_rotvec(np.array([0.0, -fwd / S.BALL_R, yaw]) * dt)
        self.ball_q = dq * self.ball_q
        q = self.ball_q.as_quat()
        self.d.qpos[self.ball_adr:self.ball_adr + 4] = q[[3, 0, 1, 2]]
        # --- голова смотрит туда, куда тянет руль ---
        self.head_angle += (np.clip(head_turn, -1, 1) * 0.35 - self.head_angle) * min(1, dt / 0.08)
        self.d.qpos[self.head_yaw] = self.head_angle
        self.pose.apply(proboscis, wing_spread, wing_flap)
        mujoco.mj_forward(self.m, self.d)

    def set_drop(self, amount):
        """Капля сахара у пипетки: 0 — нет, 1 — полная."""
        gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "drop_sensorfx")
        self.m.geom_rgba[gid, 3] = 0.85 if amount > 0.05 else 0.0
        self.m.geom_size[gid, 0] = 0.1 + 0.2 * amount

    def render(self):
        """RGBA: цвет + альфа по маске (муха и шар)."""
        rgb = self.p.render(height=self.H, width=self.W, camera_id="stream")
        seg = self.p.render(height=self.H, width=self.W, camera_id="stream", segmentation=True)[..., 0]
        alpha = (seg >= 0).astype(np.uint8) * 255
        return np.dstack([rgb, alpha])


if __name__ == "__main__":
    import imageio.v2 as imageio
    f = FlyOnBall()
    for i in range(30):
        f.update(1 / 30, 1.0, 0.7, proboscis=0.0, head_turn=0.5)
    im = f.render()
    imageio.imwrite(str(Path(__file__).parent / "fly_layer_test.png"), im)
    print("ok", im.shape, "alpha px:", int((im[..., 3] > 0).sum()))
