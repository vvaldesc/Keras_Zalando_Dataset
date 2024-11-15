import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split


#a) Hacer una RNA y entrenarla
#       (dividir el conjunto de entrenamiento y de test, escalar las características ...)
#b) Evaluar el modelo --> Gráficos, convergencia
#c) Fase de preprocesamiento (Que iría antes de hacer la RNA)
#d) Una de las imágenes de zalando , guardarlas (imload cargar, imsave guardar)
#e) Hacer una imagen en Paint o de internñet pero que se ajuste a nuestro modelo y ponerla a ver como funciona ('Predicción')

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


#Qué representan cada una de las variables que devuelve mnist.load_data(),
#¿Cuántos elementos y qué forma tiene cada uno de ellos?
aux = mnist.load_data()
print(len(training_images))
print(training_images[0])
print(training_images[0].shape)
#lista = [ x for x in training_images]
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# Crar un subplot
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 8))

# Definir los subplots

#random_plots = [[[axes[i][j].imshow(X_train[random.randint(0, X_train.shape[0] - 1)])]for j, col in enumerate(fila)] for i, fila in enumerate(axes)]
#plt.show()


for i in range(10):
    plt.imshow(training_images[i])
    plt.show()

