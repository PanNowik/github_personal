# Wczytanie bibliotek.
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Utworzenie macierzy cech i wektora docelowego.
features, target = make_classification(n_samples=10000,
                                       n_features=10,
                                       n_classes=2,
                                       n_informative=3,
                                       random_state=3)

# Podział na zbiory uczący i testowy.
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1)

# Utworzenie klasyfikatora.
logit = LogisticRegression()

# Wytrenowanie modelu.
logit.fit(features_train, target_train)

# Pobranie prognozowanego prawdopodobieństwa.
target_probabilities = logit.predict_proba(features_test)[:,1]

# Utworzenie współczynników prawdziwie i fałszywie pozytywnych.
false_positive_rate, true_positive_rate, threshold = roc_curve(target_test,
                                                               target_probabilities)

# Wyświetlenie krzywej ROC.
plt.title("Krzywa ROC")
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel("Współczynnik wartości prawdziwie pozytywnych")
plt.xlabel("Współczynnik wartości fałszywie pozytywnych")
plt.show()
