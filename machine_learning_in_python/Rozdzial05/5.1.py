# Wczytanie bibliotek.
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# Utworzenie cechy.
feature = np.array([["Teksas"],
                    ["Kalifornia"],
                    ["Teksas"],
                    ["Delaware"],
                    ["Teksas"]])

# Utworzenie kodera „gorącojedynkowego”.
one_hot = LabelBinarizer()

# Zakodowanie cechy za pomocą przygotowanego kodera.
one_hot.fit_transform(feature)
