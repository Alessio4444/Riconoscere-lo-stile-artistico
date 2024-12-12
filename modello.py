import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import pickle

# Configura i percorsi
TRAIN_DIR = 'C:/Users/Asus/Desktop/progetto/output/immagini/wikiart-target_style-class_6-keepgenre_True-merge_style_m1-flat_False/train/'
VAL_DIR = 'C:/Users/Asus/Desktop/progetto/output/immagini/wikiart-target_style-class_6-keepgenre_True-merge_style_m1-flat_False/val/'
TEST_DIR = 'C:/Users/Asus/Desktop/progetto/output/immagini/wikiart-target_style-class_6-keepgenre_True-merge_style_m1-flat_False/test/'

# Configura i generatori di dati
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


with open('class_indices.pkl', 'wb') as f:
    pickle.dump(train_generator.class_indices, f)

# modello VGG16 pre-addestrato
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congela i pesi del modello base
for layer in base_model.layers:
    layer.trainable = False

#livelli di classificazione
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Addestra il modello
history = model.fit(
    train_generator,
    epochs=3,  
    validation_data=val_generator
)


test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')


model.save('C:/Users/Asus/Desktop/progetto/saved_model/modello.h5')