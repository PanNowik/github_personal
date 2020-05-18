# Wczytanie bibliotek.
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets

# Wczytanie danych zawierających tylko dwie cechy.
boston = datasets.load_boston()
features = boston.data[:,0:2]
target = boston.target

# Utworzenie obiektu klasyfikatora losowego lasu.
randomforest = RandomForestRegressor(random_state=0, n_jobs=-1)

# Wytrenowanie modelu.
model = randomforest.fit(features, target)
