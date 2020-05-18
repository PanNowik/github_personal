# Wczytanie biblioteki.
import pandas as pd

# Utworzenie dat.
dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))

# Wyświetlenie dni tygodnia.
dates.dt.weekday_name
