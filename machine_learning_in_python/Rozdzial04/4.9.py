# Wczytanie bibliotek.
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Utworzenie symulowanej macierzy cech.
features, _ = make_blobs(n_samples = 50,
                         n_features = 2,
                         centers = 3,
                         random_state = 1)

# Utworzenie obiektu typu DataFrame.
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])

# Utworzenie klastra za pomocą algorytmu k-średnich.
clusterer = KMeans(3, random_state=0)

# Wypełnienie klastra.
clusterer.fit(features)

# Prognozowanie wartości.
dataframe["group"] = clusterer.predict(features)

# Wyświetlenie kilku pierwszych obserwacji.
dataframe.head(5)
