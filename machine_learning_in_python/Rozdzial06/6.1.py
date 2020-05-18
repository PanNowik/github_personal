# Utworzenie tekstu.
text_data = ["   Wykrzyknik i znak zapytania, autor Aishwarya Henriette     ",
             "Techniki parkowania, autor Karl Gautier",
             "    Dzisiaj jest ta noc, autor Jarek Prakash   "]

# Usunięcie białych znaków.
strip_whitespace = [string.strip() for string in text_data]

# Wyświetlenie tekstu.
strip_whitespace

['Wykrzyknik i znak zapytania, autor Aishwarya Henriette',
 'Techniki parkowania, autor Karl Gautier',
 'Dzisiaj jest ta noc, autor Jarek Prakash']

# Usunięcie słowa „autor”.
remove_periods = [string.replace("autor", "") for string in strip_whitespace]

# Wyświetlenie tekstu.
remove_periods
