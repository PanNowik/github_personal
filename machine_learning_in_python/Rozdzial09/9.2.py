# Wczytanie bibliotek.
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

# Utworzenie liniowo nierozłącznych danych.
features, _ = make_circles(n_samples=1000, random_state=1, noise=0.1, factor=0.1)

# Zastosowanie analizy głównej składowej z radialną funkcją bazową.
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)
features_kpca = kpca.fit_transform(features)

print("Początkowa liczba cech:", features.shape[1])
print("Liczba cech po redukcji:", features_kpca.shape[1])
