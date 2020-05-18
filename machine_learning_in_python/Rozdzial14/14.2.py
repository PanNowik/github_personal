# Wczytanie bibliotek.
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets

# Wczytanie danych zawierających tylko dwie cechy.
boston = datasets.load_boston()
features = boston.data[:,0:2]
target = boston.target

# Utworzenie obiektu klasyfikatora drzewa decyzyjnego.
decisiontree = DecisionTreeRegressor(random_state=0)

# Wytrenowanie modelu.
model = decisiontree.fit(features, target)
