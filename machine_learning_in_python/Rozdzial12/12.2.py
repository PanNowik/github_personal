# Wczytanie bibliotek.
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Zdefiniowanie regresji logistycznej.
logistic = linear_model.LogisticRegression()

# Zdefiniowanie zakresu wartości hiperparametru penalty.
penalty = ['l1', 'l2']

# Zdefiniowanie zakresu wartości hiperparametru regularyzacji.
C = uniform(loc=0, scale=4)

# Zdefiniowanie opcji hiperparametru.
hyperparameters = dict(C=C, penalty=penalty)

# Zdefiniowanie przeszukiwania losowego.
randomizedsearch = RandomizedSearchCV(
    logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0,
    n_jobs=-1)

# Dopasowanie przeszukiwania losowego.
best_model = randomizedsearch.fit(features, target)
