# Wczytanie biblioteki.
import pandas as pd

# Utworzenie adresu URL.
url = 'https://tinyurl.com/simulated_json'

# Wczytanie danych.
dataframe = pd.read_json(url, orient='columns')

# Wyświetlenie dwóch pierwszych wierszy wczytanych danych.
dataframe.head(2)
