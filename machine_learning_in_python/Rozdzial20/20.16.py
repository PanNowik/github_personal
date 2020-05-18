# Wczytanie biblioteki.
from keras.preprocessing.image import ImageDataGenerator

# Zdefiniowanie zestawu modyfikacji obrazu do przeprowadzenia.
augmentation = ImageDataGenerator(featurewise_center=True, # Przekształcenie typu „ZCA whitening”.
                                  zoom_range=0.3, # Losowe przybliżenie obrazów.
                                  width_shift_range=0.2, # Losowe przesunięcie obrazów.
                                  horizontal_flip=True, # Losowe odbicie lustrzane obrazów.
                                  rotation_range=90) # Losowe obrócenie obrazów.

# Przetworzenie wszystkich obrazów znajdujących się w katalogu raw/images.
augment_images = augmentation.flow_from_directory("raw/images", # Katalog obrazów.
                                                  batch_size=32, # Wielkość paczki.
                                                  class_mode="binary", # Klasy.
                                                  save_to_dir="processed/images")
