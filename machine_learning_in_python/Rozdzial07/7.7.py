# Wczytanie biblioteki.
import pandas as pd

# Utworzenie obiektu typu DataFrame.
dataframe = pd.DataFrame()

# Utworzenie danych.
dataframe["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
dataframe["stock_price"] = [1.1,2.2,3.3,4.4,5.5]

# Opóźnienie wartości o jeden wiersz.
dataframe["previous_days_stock_price"] = dataframe["stock_price"].shift(1)

# Wyświetlenie danych.
dataframe
