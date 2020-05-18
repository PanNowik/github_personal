# Wczytanie bibliotek.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu jako czarno-białego.
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Rozmycie obrazu.
image_blurry = cv2.blur(image, (5,5))

# Wyświetlenie obrazu.
plt.imshow(image_blurry, cmap="gray"), plt.axis("off")
plt.show()
