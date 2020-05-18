# Wczytanie biblioteki.
from nltk.stem.porter import PorterStemmer

# Utworzenie tokenów słów.
tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']

# Utworzenie egzemplarza klasy PorterStemmer.
porter = PorterStemmer()

# Użycie przygotowanego obiektu porter.
[porter.stem(word) for word in tokenized_words]
