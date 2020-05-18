# Wczytanie bibliotek.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu.
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Poprawienie obrazu.
image_enhanced = cv2.equalizeHist(image)

# Wyświetlenie obrazu.
plt.imshow(image_enhanced, cmap="gray"), plt.axis("off")
plt.show()
