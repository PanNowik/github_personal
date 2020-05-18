# Wczytanie bibliotek.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu jako czarno-białego.
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)

# Zapisanie obrazu.
cv2.imwrite("images/plane_new.jpg", image)
