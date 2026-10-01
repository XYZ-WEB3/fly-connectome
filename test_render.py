"""Проверка тела: муха идёт по кругу, камера от 3-го лица едет за ней."""
import numpy as np
from tqdm import trange

from flygym import Fly
from flygym.camera import YawOnlyCamera
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController

arena = OdorArena(odor_source=np.array([[15, 5, 1.5]]), peak_odor_intensity=np.array([[1, 0]]),
                  marker_colors=[(1, 0.5, 0.1, 1)], marker_size=0.4)
fly = Fly(spawn_pos=(0, 0, 0.2), enable_olfaction=True, enable_adhesion=True, draw_adhesion=False,
          contact_sensor_placements=[f"{l}{s}" for l in ["LF", "LM", "LH", "RF", "RM", "RH"]
                                     for s in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]])
cam = YawOnlyCamera(attachment_point=fly.model.worldbody, camera_name="camera_back_track_game",
                    targeted_fly_names=fly.name, window_size=(960, 540), play_speed=0.2, fps=30,
                    play_speed_text=False)
sim = HybridTurningController(fly=fly, cameras=[cam], arena=arena, timestep=1e-4)
obs, _ = sim.reset()
for i in trange(int(1.0 / 1e-4)):
    action = np.array([1.0, 0.6]) if i > 4000 else np.array([1.0, 1.0])
    obs, *_ = sim.step(action)
    sim.render()
cam.save_video("videos/test_walk.mp4")
print("pos", obs["fly"][0])
