# Wczytanie biblioteki.
import pandas as pd

# Utworzenie adresu URL.
url = 'https://tinyurl.com/simulated_excel'

# Wczytanie danych.
dataframe = pd.read_excel(url, sheetname=0, header=1)

# Wyświetlenie dwóch pierwszych wierszy wczytanych danych.
dataframe.head(2)
