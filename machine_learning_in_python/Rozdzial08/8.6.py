# Wczytanie bibliotek.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu jako czarno-białego.
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Utworzenie jądra.
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

# Wyostrzenie obrazu.
image_sharp = cv2.filter2D(image, -1, kernel)

# Wyświetlenie obrazu.
plt.imshow(image_sharp, cmap="gray"), plt.axis("off")
plt.show()
