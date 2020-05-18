# Wczytanie biblioteki.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu jako czarno-białego.
image_gray = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Obliczenie mediany intensywności.
median_intensity = np.median(image_gray)

# Zdefiniowanie wartości progowych o jedno odchylenie standardowe więcej i jedno mniej niż mediana intensywności piksela.
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

# Zastosowanie techniki wykrywania krawędzi Canny.
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)

# Wyświetlenie obrazu.
plt.imshow(image_canny, cmap="gray"), plt.axis("off")
plt.show()
