# Wczytanie bibliotek.
from nltk import pos_tag
from nltk import word_tokenize

# Utworzenie tekstu.
text_data = "Chris loved outdoor running"

# Użycie wstępnie wytrenowanego algorytmu.
text_tagged = pos_tag(word_tokenize(text_data))

# Wyświetlenie części mowy.
text_tagged
