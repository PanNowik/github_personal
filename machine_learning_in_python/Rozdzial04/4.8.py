# Wczytanie bibliotek.
import numpy as np
from sklearn.preprocessing import Binarizer

# Utworzenie cechy.
age = np.array([[6],
                [12],
                [20],
                [36],
                [65]])

# Utworzenie egzemplarza klasy Binarizer.
binarizer = Binarizer(18)

# Przekształcenie cechy.
binarizer.fit_transform(age)
