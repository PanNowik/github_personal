# Wczytanie biblioteki.
from sklearn.feature_selection import VarianceThreshold

# Utworzenie macierzy cech o następujących właściwościach:
# Cecha 0: 80% klasy 0.
# Cecha 1: 80% klasy 1.
# Cecha 2: 60% klasy 0, 40% klasy 1.
features = [[0, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0]]

# Progowanie według wariancji.
thresholder = VarianceThreshold(threshold=(.75 * (1 - .75)))
thresholder.fit_transform(features)
