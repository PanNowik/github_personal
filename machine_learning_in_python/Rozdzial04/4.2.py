# Wczytanie bibliotek.
import numpy as np
from sklearn import preprocessing

# Utworzenie cechy.
x = np.array([[-1000.1],
              [-200.2],
              [500.5],
              [600.6],
              [9000.9]])

# Utworzenie przelicznika.
scaler = preprocessing.StandardScaler()

# Przekształcenie cechy.
standardized = scaler.fit_transform(x)

# Wyświetlenie cechy.
standardized
