import numpy as np
from sklearn.metrics import silhouette_score
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Wygenerowanie macierzy cech.
features, _ = make_blobs(n_samples = 1000,
                         n_features = 10,
                         centers = 2,
                         cluster_std = 0.5,
                         shuffle = True,
                         random_state = 1)

# Klastrowanie danych za pomocą k-krotności, aby prognozować klasy.
model = KMeans(n_clusters=2, random_state=1).fit(features)

# Pobranie prognozowanych klas.
target_predicted = model.labels_

# Ocena modelu.
silhouette_score(features, target_predicted)
