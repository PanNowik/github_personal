# Wczytanie bibliotek.
import pandas as pd
import numpy as np

# Utworzenie daty.
time_index = pd.date_range("01/01/2010", periods=5, freq="M")

# Utworzenie obiektu typu DataFrame i zdefiniowanie indeksu.
dataframe = pd.DataFrame(index=time_index)

# Utworzenie cechy, w której brakuje wartości.
dataframe["Sales"] = [1.0,2.0,np.nan,np.nan,5.0]

# Interpolacja brakujących wartości.
dataframe.interpolate()
