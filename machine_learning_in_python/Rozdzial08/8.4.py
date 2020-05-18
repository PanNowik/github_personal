# Wczytanie bibliotek.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu jako czarno-białego.
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Pobranie połowy kolumn i wszystkich wierszy.
image_cropped = image[:,:128]

# Wyświetlenie obrazu.
plt.imshow(image_cropped, cmap="gray"), plt.axis("off")
plt.show()
