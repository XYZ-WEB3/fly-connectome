"""Тестовый «ролик» для мухи (свой, без авторских прав): 20 с, 30 кадров/с.
0-6 с: маленький тёмный «жук» ползает по экрану;
6-9 с: полосы бегут слева направо (всё поле движется);
9-12 с: чёрный круг надвигается (угроза!);
12-20 с: снова «жук», ходит туда-сюда."""
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

W, H, FPS, T = 640, 360, 30, 20
out = Path(__file__).parent / "testclip.mp4"
wr = imageio.get_writer(str(out), fps=FPS, codec="libx264", quality=8, macro_block_size=8)
for i in range(T * FPS):
    t = i / FPS
    img = np.full((H, W, 3), (215, 225, 235), np.uint8)
    cv2.rectangle(img, (0, int(H * 0.72)), (W, H), (180, 200, 170), -1)          # «пол»
    if t < 6 or t >= 12:
        x = W / 2 + W * 0.38 * np.sin(t * 0.9)
        y = H * 0.6 + 25 * np.sin(t * 2.3)
        cv2.ellipse(img, (int(x), int(y)), (13, 8), 0, 0, 360, (30, 25, 20), -1, cv2.LINE_AA)
        for k in (-1, 1):                                                        # лапки
            cv2.line(img, (int(x) - 8, int(y) + 4 * k), (int(x) - 16, int(y) + 10 * k), (30, 25, 20), 2)
            cv2.line(img, (int(x) + 8, int(y) + 4 * k), (int(x) + 16, int(y) + 10 * k), (30, 25, 20), 2)
    elif t < 9:
        ph = (t - 6) * 160
        for xs in range(-80, W + 80, 80):
            x0 = int((xs + ph) % (W + 80)) - 40
            cv2.rectangle(img, (x0, 0), (x0 + 40, H), (40, 40, 45), -1)
    else:
        s = (t - 9) / 3
        r = int(8 + 330 * s ** 3)                                                 # ускоряющееся надвигание
        cv2.circle(img, (int(W * 0.35), int(H * 0.45)), r, (10, 10, 12), -1, cv2.LINE_AA)
    wr.append_data(img)
wr.close()
print("ролик:", out)
