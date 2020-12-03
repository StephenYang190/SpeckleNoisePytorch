import cv2
import torch
import numpy as np
from model.median_pooling import median_pool_2d
from dataset import load_image_test
from matplotlib import pyplot as plt

path = "./DataRoot/TEST/origin/breast.png"
img = load_image_test(path)
img = img[np.newaxis, :]
img = torch.Tensor(img)
denoise_img = median_pool_2d(img, 3, 1, 1, 1)

img = img[0, :, :, :]
denoise_img = denoise_img[0, :, :, :]

plt.subplot(1, 2, 1)
plt.title("Origin")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Median denoise")
plt.imshow(denoise_img, cmap='gray')

plt.show()
