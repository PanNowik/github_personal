# Wczytanie biblioteki.
import pandas as pd

# Utworzenie adresu URL.
url = 'https://tinyurl.com/simulated_data'

# Wczytanie zbioru danych.
dataframe = pd.read_csv(url)

# Wyświetlenie dwóch pierwszych wierszy wczytanych danych.
dataframe.head(2)
