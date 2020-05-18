# Wczytanie bibliotek.
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

# Utworzenie symulowanych danych.
features, _ = make_blobs(n_samples = 10,
                         n_features = 2,
                         centers = 1,
                         random_state = 1)

# Zastąpienie wartości pierwszej obserwacji wartościami odstającymi.
features[0,0] = 10000
features[0,1] = 10000

# Utworzenie detektora.
outlier_detector = EllipticEnvelope(contamination=.1)

# Wypełnienie detektora.
outlier_detector.fit(features)

# Wykrywanie elementów odstających.
outlier_detector.predict(features)
