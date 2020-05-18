# Wczytanie bibliotek.
import unicodedata
import sys

# Utworzenie tekstu.
text_data = ['Cześć!!!! Po. Prostu. Kocham. Tę. Piosenkę....',
             'Zgadzam się w 10000%!!!! #KochamTO',
             'Racja?!?!']

# Utworzenie słownika znaków przestankowych.
punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))

# We wszystkich ciągach tekstowych mają zostać usunięte znaki przestankowe.
[string.translate(punctuation) for string in text_data]
