import cv2
import numpy as np


#build motion field to hsv color transcription image in hsv_colors.tif file in save directory

n = 255
dir = np.ndarray((n,n,2))
for i in range(n):
    for j in range(n):
        dir[i,j] = [j - (n - 1) // 2, i - (n - 1) // 2]
hsv_map = np.zeros((n,n,3)).astype(np.uint8)


mag, ang = cv2.cartToPolar(dir[..., 0], dir[..., 1])
mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
mag = 255 * np.ones(np.shape(mag)) / (np.ones(np.shape(mag)) + np.exp(0.03 * (-mag + 127.5 * np.ones(np.shape(mag)))))
print(np.mod((ang * 180 / np.pi / 2) , 180)[127,0])
hsv_map[..., 0] = np.mod(-(ang * 180 / np.pi / 2) - 45, 180)
hsv_map[..., 1] = mag
hsv_map[..., 2] = 255
bgr =  cv2.cvtColor(hsv_map, cv2.COLOR_HSV2RGB)

cv2.imwrite('save/hsv_colors.tif', bgr)
