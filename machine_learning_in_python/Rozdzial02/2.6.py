# Wczytanie bibliotek.
import pandas as pd
from sqlalchemy import create_engine

# Nawiązanie połączenia z bazą danych.
database_connection = create_engine('sqlite:///sample.db')

# Wczytanie danych.
dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)

# Wyświetlenie dwóch pierwszych rekordów z wczytanych danych.
dataframe.head(2)
