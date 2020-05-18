# Wczytanie bibliotek.
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenie obiektu naiwnego klasyfikatora bayesowskiego Gaussa.
classifer = GaussianNB()

# Zdefiniowanie skalibrowanego sprawdzianu krzyżowego wraz z kalibracją esowatą.
classifer_sigmoid = CalibratedClassifierCV(classifer, cv=2, method='sigmoid')

# Kalibrowanie prawdopodobieństwa.
classifer_sigmoid.fit(features, target)

# Utworzenie nowej obserwacji.
new_observation = [[ 2.6,  2.6,  2.6,  0.4]]

# Wyświetlenie skalibrowanego prawdopodobieństwa.
classifer_sigmoid.predict_proba(new_observation)
