# Wczytanie biblioteki.
from bs4 import BeautifulSoup

# Utworzenie przykładowego kodu HTML
html = """
       <div class='full_name'><span style='font-weight:bold'>
       Masego</span> Azra</div>"
       """

# Przetworzenie kodu HTML.
soup = BeautifulSoup(html, "lxml")

# Odszukanie znacznika div wraz z klasą "full_name" i wyświetlenie tekstu.
soup.find("div", { "class" : "full_name" }).text
