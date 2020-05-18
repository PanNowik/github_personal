# Wczytanie bibliotek.
import numpy as np
from sklearn.naive_bayes import BernoulliNB

# Utworzenie trzech cech binarnych.
features = np.random.randint(2, size=(100, 3))

# Utworzenie binarnego wektora docelowego.
target = np.random.randint(2, size=(100, 1)).ravel()

# Utworzenie obiektu naiwnego klasyfikatora bayesowskiego Bernoulliego z wartościami prawdopodobieństwa poszczególnych klas.
classifer = BernoulliNB(class_prior=[0.25, 0.5])

# Wytrenowanie modelu.
model = classifer.fit(features, target)
