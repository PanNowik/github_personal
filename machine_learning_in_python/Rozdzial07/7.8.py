# Wczytanie biblioteki.
import pandas as pd

# Utworzenie obiektów daty i godziny.
time_index = pd.date_range("01/01/2010", periods=5, freq="M")

# Utworzenie obiektu typu DataFrame i zdefiniowanie indeksu.
dataframe = pd.DataFrame(index=time_index)

# Utworzenie cechy.
dataframe["Stock_Price"] = [1,2,3,4,5]

# Obliczenie średniej.
dataframe.rolling(window=2).mean()
