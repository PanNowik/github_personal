# Wczytanie bibliotek.
import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Utworzenie symulowanej macierzy cech.
features, _ = make_blobs(n_samples = 1000,
                         n_features = 2,
                         random_state = 1)

# Standaryzacja cech.
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# Pozbycie się pierwszej wartości pierwszej cechy.
true_value = standardized_features[0,0]
standardized_features[0,0] = np.nan

# Prognozowanie brakującej wartości w macierzy cech.
features_knn_imputed = KNN(k=5, verbose=0).complete(standardized_features)

# Porównanie wartości rzeczywistej i prognozowanej.
print("Wartość rzeczywista:", true_value)
print("Wartość prognozowana:", features_knn_imputed[0,0])
