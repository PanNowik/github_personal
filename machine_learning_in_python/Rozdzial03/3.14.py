# Wczytanie bibliotek.
import pandas as pd
import numpy as np

# Utworzenie przedziału czasu.
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')

# Utworzenie obiektu typu DataFrame.
dataframe = pd.DataFrame(index=time_index)

# Utworzenie kolumny losowo wybranych wartości.
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

# Grupowanie wierszy według tygodnia, obliczenie sumy dla danego tygodnia.
dataframe.resample('W').sum()
