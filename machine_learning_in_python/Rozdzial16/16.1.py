# Wczytanie bibliotek.
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Wczytanie danych zawierających tylko dwie klasy.
iris = datasets.load_iris()
features = iris.data[:100,:]
target = iris.target[:100]

# Standaryzowanie cech.
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Utworzenie obiektu regresji logistycznej.
logistic_regression = LogisticRegression(random_state=0)

# Wytrenowanie modelu.
model = logistic_regression.fit(features_standardized, target)
