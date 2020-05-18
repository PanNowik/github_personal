# Wczytanie bibliotek.
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Wczytanie zbioru danych w postaci liczb.
digits = datasets.load_digits()

# Utworzenie macierzy cech.
features = digits.data

# Utworzenie wektora docelowego.
target = digits.target

# Utworzenie egzemplarza StandardScaler.
standardizer = StandardScaler()

# Utworzenie obiektu regresji logistycznej.
logit = LogisticRegression()

# Utworzenie potoku standaryzującego, a następnie przeprowadzającego regresję logistyczną.
pipeline = make_pipeline(standardizer, logit)

# Zdefiniowanie k-krotnego sprawdzianu krzyżowego.
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Przeprowadzenie k-krotnego sprawdzianu krzyżowego.
cv_results = cross_val_score(pipeline, # Potok.
                             features, # Macierz cech.
                             target, # Wektor docelowy.
                             cv=kf, # Technika sprawdzianu krzyżowego.
                             scoring="accuracy", # Funkcja straty.
                             n_jobs=-1) # Użycie wszystkich dostępnych rdzeni procesora.

# Obliczenie średniej.
cv_results.mean()
