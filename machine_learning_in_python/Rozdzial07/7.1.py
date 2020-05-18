# Wczytanie bibliotek.
import numpy as np
import pandas as pd

# Utworzenie ciągów tekstowych.
date_strings = np.array(['03-04-2005 11:35 PM',
                         '23-05-2010 12:01 AM',
                         '04-09-2009 09:09 PM'])

# Konwersja na postać daty i godziny.
[pd.to_datetime(date, format='%d-%m-%Y %I:%M %p') for date in date_strings]
