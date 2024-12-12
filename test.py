import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle
import matplotlib.pyplot as plt

# Percorso al modello salvato
MODEL_PATH = 'C:/Users/Asus/Desktop/progetto/saved_model/modello.h5'

# Carica il modello salvato
model = load_model(MODEL_PATH)

# Carica class_indices dal file salvato
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)

# Funzione per caricare e preprocessare una nuova immagine
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Scala i valori dei pixel tra 0 e 1
    return img_array

# Percorso alla nuova immagine
new_image_path = 'C:/Users/Asus/Desktop/progetto/data_set/immagini/wikiart-target_style-class_6-keepgenre_True-merge_style_m1-flat_False/test/expressionism/expressionism_adam-baltatu_fantastic-landscape.jpg'

# Carica e preprocessa la nuova immagine
new_image = load_and_preprocess_image(new_image_path)

# Fai una previsione sulla nuova immagine
predictions = model.predict(new_image)

# Ottieni la classe con la probabilità più alta
predicted_class = np.argmax(predictions, axis=1)

predicted_probability = np.max(predictions, axis=1)

# Mappa l'indice della classe predetta al nome della classe
class_labels = list(class_indices.keys())
class_labels.sort(key=lambda x: class_indices[x])
predicted_label = class_labels[predicted_class[0]]

print(f'Predicted label: {predicted_label} with probability: {predicted_probability[0]:.2f}')

# Visualizza l'immagine e la previsione
plt.imshow(image.load_img(new_image_path))
plt.title(f'Predicted: {predicted_label}')
plt.axis('off')
plt.show()
