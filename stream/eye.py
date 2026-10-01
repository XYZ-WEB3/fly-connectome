"""«Глаз» v1: кадр с экрана -> сигналы для зрительных нейронов мухи.

Упрощённая сетчатка (честно подписываем на экране):
  - мелкий движущийся объект слева/справа  -> LC10 (детекторы мелких объектов)
  - надвигание (быстро растущее тёмное пятно) слева/справа -> LPLC2 + LC4 (угроза)
  - общее движение поля влево/вправо -> пока только показываем (оптомоторика — следующий шаг)
Левая половина кадра = левый глаз мухи (она смотрит в экран).
"""
import cv2
import numpy as np


class Eye:
    def __init__(self, w=160, h=90):
        self.w, self.h = w, h
        self.prev = None
        self.hist = []
        self.out = dict(obj_L=0.0, obj_R=0.0, loom_L=0.0, loom_R=0.0, flow=0.0)

    def __call__(self, frame_rgb, dt):
        g = cv2.cvtColor(cv2.resize(frame_rgb, (self.w, self.h), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        halves = (slice(0, self.w // 2), slice(self.w // 2, self.w))
        o = self.out
        if self.prev is None:
            self.prev = g
            return o
        diff = np.abs(g - self.prev)
        moving = diff > 0.12
        # мелкие движущиеся объекты: немного движущихся пикселей в половине кадра
        obj = []
        for hs in halves:
            frac = moving[:, hs].mean()
            obj.append(float(np.clip(frac / 0.01, 0, 1)) if frac < 0.08 else 0.0)   # большое поле — не «объект»
        # надвигание: самое большое тёмное пятно РАСТЁТ (относительно своего размера) несколько кадров подряд
        dark_mask = (g < 0.2).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        area, cx = 0.0, self.w / 2
        if n > 1:
            k_big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            area, cx = float(stats[k_big, cv2.CC_STAT_AREA]), float(cent[k_big][0])
        self.hist.append(area)
        self.hist = self.hist[-6:]
        loom_val = 0.0
        if len(self.hist) == 6 and self.hist[0] > 0.01 * self.w * self.h:
            rel = (self.hist[-1] / self.hist[0] - 1) / (5 * dt)            # относительный рост в секунду
            steady = all(b >= a * 0.98 for a, b in zip(self.hist, self.hist[1:]))
            loom_val = float(np.clip((rel - 0.8) / 2.5, 0, 1)) if steady else 0.0
        side = np.clip((cx / self.w - 0.5) * 3 + 0.5, 0, 1)                # 0 — слева, 1 — справа
        loom = np.array([loom_val * (1 - side * 0.8), loom_val * (0.2 + side * 0.8)])
        # общее движение поля (горизонтальный поток)
        flow = cv2.calcOpticalFlowFarneback(self.prev, g, None, 0.5, 2, 9, 2, 5, 1.1, 0)
        fx = float(flow[..., 0].mean())
        k = min(1.0, dt / 0.06)                                    # сглаживание ~60 мс
        o["obj_L"] += k * (obj[0] - o["obj_L"]); o["obj_R"] += k * (obj[1] - o["obj_R"])
        o["loom_L"] += k * (loom[0] - o["loom_L"]); o["loom_R"] += k * (loom[1] - o["loom_R"])
        o["flow"] += k * (np.clip(fx / 2.0, -1, 1) - o["flow"])
        self.prev = g
        return o
