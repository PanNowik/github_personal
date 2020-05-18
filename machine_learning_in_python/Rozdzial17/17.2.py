# Wczytanie bibliotek.
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# Zdefiniowanie wartości zalążka.
np.random.seed(0)

# Wygenerowanie dwóch cech.
features = np.random.randn(200, 2)

# Użycie bramki XOR (nie musisz wiedzieć, co to jest) do wygenerowania
# liniowo nierozdzielnych klas.
target_xor = np.logical_xor(features[:, 0] > 0, features[:, 1] > 0)
target = np.where(target_xor, 0, 1)

# Utworzenie maszyny wektora nośnego z jądrem radialnej funkcji bazowej.
svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1)

# Wytrenowanie klasyfikatora.
model = svc.fit(features, target)
