# Wczytanie biblioteki.
import pandas as pd

# Utworzenie adresu URL.
url = 'https://tinyurl.com/titanic-csv'

# Wczytanie danych.
dataframe = pd.read_csv(url)

# Zmiana nazwy kolumny, wyświetlenie dwóch pierwszych wierszy.
dataframe.rename(columns={'PClass': 'Passenger Class'}).head(2)
