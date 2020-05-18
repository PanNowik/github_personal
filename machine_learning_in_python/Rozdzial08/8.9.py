# Wczytanie bibliotek.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu jako czarno-białego.
image_grey = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Zastosowanie progowania adaptacyjnego.
max_output_value = 255
neighborhood_size = 99
subtract_from_mean = 10
image_binarized = cv2.adaptiveThreshold(image_grey,
                                        max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        neighborhood_size,
                                        subtract_from_mean)

# Wyświetlenie obrazu.
plt.imshow(image_binarized, cmap="gray"), plt.axis("off")
plt.show()
