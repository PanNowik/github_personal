# Wczytanie biblioteki.
from nltk.corpus import stopwords

# Po pierwszym zaimportowaniu pakietu NLTK
# konieczne jest pobranie listy słów o małym znaczeniu.
# nltk.download('stopwords')

# Utworzenie tokenów słów.
tokenized_words = ['i',
                   'am',
                   'going',
                   'to',
                   'go',
                   'to',
                   'the',
                   'store',
                   'and',
                   'park']

# Wczytanie listy słów o małym znaczeniu.
stop_words = stopwords.words('english')

# Usunięcie słów o małym znaczeniu.
[word for word in tokenized_words if word not in stop_words]
