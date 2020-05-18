# Wczytanie bibliotek.
import numpy as np
from sklearn import preprocessing

# Utworzenie cechy.
feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])

# Utworzenie przelicznika, czyli egzemplarza typu MinMaxScaler.
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Skalowanie cechy.
scaled_feature = minmax_scale.fit_transform(feature)

# Wyświetlenie cechy.
scaled_feature
