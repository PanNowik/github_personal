# Wczytanie bibliotek.
from sklearn import preprocessing
import numpy as np

# Utworzenie cech.
features = np.array([[-100.1, 3240.1],
                     [-200.2, -234.1],
                     [5000.5, 150.1],
                     [6000.6, -125.1],
                     [9000.9, -673.1]])

# Utworzenie egzemplarza typu StandardScaler.
scaler = preprocessing.StandardScaler()

# Przekształcenie cech.
features_standardized = scaler.fit_transform(features)

# Wyświetlenie cech.
features_standardized
