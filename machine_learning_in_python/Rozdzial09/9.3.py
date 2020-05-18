# Wczytanie bibliotek.
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Wczytanie zbioru danych Iris.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Zdefiniowanie i przeprowadzenie LDA, a następnie użycie jej do przekształcenia cech.
lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features, target).transform(features)

# Wyświetlenie liczby cech.
print("Początkowa liczba cech:", features.shape[1])
print("Liczba cech po redukcji:", features_lda.shape[1])
