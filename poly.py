import random
import cv2
import os
import numpy as np

img_list = os.listdir('./avatars')
img_count = len(img_list)

width = 528
height = 560

pix_dict = {}

for i in range(height):
    for j in range(width):
        pix_dict[f'{i}_{j}'] = {'r': [], 'g': [], 'b': []}
        pix_dict[f'{i}_{j}']['r'] = [0 for k in range(256)]
        pix_dict[f'{i}_{j}']['g'] = [0 for k in range(256)]
        pix_dict[f'{i}_{j}']['b'] = [0 for k in range(256)]

for item in img_list:
    img = cv2.imread('./avatars/' + item)

    for i in range(height):
        for j in range(width):
            pix_dict[f'{i}_{j}']['r'][img[i, j, 0]] += 1
            pix_dict[f'{i}_{j}']['g'][img[i, j, 1]] += 1
            pix_dict[f'{i}_{j}']['b'][img[i, j, 2]] += 1

output_image = np.zeros((height, width, 3), dtype=int)
pixel_range = np.arange(256)

for i in range(5):
    for i in range(height):
        for j in range(width):
            output_image[i, j, 0] = random.choices(pixel_range, pix_dict[f'{i}_{j}']['r'])[0]
            output_image[i, j, 1] = random.choices(pixel_range, pix_dict[f'{i}_{j}']['g'])[0]
            output_image[i, j, 2] = random.choices(pixel_range, pix_dict[f'{i}_{j}']['b'])[0]

    cv2.imwrite(f'gen_{i}.jpg', output_image)