# Wczytanie biblioteki.
import pandas as pd

# Utworzenie adresu URL.
url = 'https://tinyurl.com/titanic-csv'

# Wczytanie danych.
dataframe = pd.read_csv(url)

# Wyświetlenie dwóch pierwszych wierszy.
dataframe.head(2)
