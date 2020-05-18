# Wczytanie bibliotek.
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Utworzenie tekstu.
text_data = np.array(['I love Brazil. Brazil!',
                      'Brazil is best',
                      'Germany beats both'])

# Utworzenie worka słów.
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# Utworzenie macierzy cech.
features = bag_of_words.toarray()

# Utworzenie wektora docelowego.
target = np.array([0,0,1])

# Utworzenie obiektu wielomianowego naiwnego klasyfikatora bayesowskiego z wartościami prawdopodobieństwa poszczególnych klas.
classifer = MultinomialNB(class_prior=[0.25, 0.5])

# Wytrenowanie modelu.
model = classifer.fit(features, target)
