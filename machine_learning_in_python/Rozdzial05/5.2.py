# Wczytanie biblioteki.
import pandas as pd

# Utworzenie cech.
dataframe = pd.DataFrame({"Score": ["niski", "niski", "średni", "średni", "wysoki"]})

# Utworzenie mapowania.
scale_mapper = {"niski": 1,
                "średni": 2,
                "wysoki": 3}

# Zastąpienie wartości cechy skalą.
dataframe["Score"].replace(scale_mapper)
