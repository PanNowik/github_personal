# Wczytanie biblioteki.
import pandas as pd

# Utworzenie obiektu typu DataFrame.
dataframe = pd.DataFrame()

# Utworzenie pięciu dat.
dataframe['date'] = pd.date_range('1/1/2001', periods=150, freq='W')

# Utworzenie cech dla roku, miesiąca, dnia, godziny i minuty.
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute

# Wyświetlenie trzech wierszy.
dataframe.head(3)
