# Wczytanie biblioteki.
import pandas as pd

# Utworzenie obiektu typu DataFrame.
dataframe = pd.DataFrame()

# Utworzenie obiektów daty i godziny.
dataframe['date'] = pd.date_range('1/1/2001', periods=100000, freq='H')

# Pobranie obserwacji znajdujących się między dwoma datami.
dataframe[(dataframe['date'] > '2002-1-1 01:00:00') &
          (dataframe['date'] <= '2002-1-1 04:00:00')]
