'''
Respuestas
1)
Cargamos los datos de fashion mnist de esta forma :
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
Tienen 2 elementos con una forma de (2,) cada uno
La forma se puede ver con  print(training_images[0].shape)


2) Visualiza el contenido  de training_images[0] con un print. 1.	¿Cuál es su forma? ¿Qué tipo de dato contiene?
Puedes visualizar una imagen en matplotlib.pyplot con el método imshow, que recibe como parámetro la imagen que quieres mostrar.

print(training_images[0])
print(training_images[0].shape)
Su forma es (28,28)

3) Hay 10 clases  de ropa diferentes. , las cuales son : Camiseta', 'Pantalón', 'Jersey', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Bota'
Mostramos los primeras 10 imágenes del dataset
for i in range(10):
    plt.imshow(training_images[i])
    plt.show()

4)

5) Para normalizar los datos simplemente debemos de dividir los datos de test y train por 255 para escalalarlos (0,1)
Sobre lo de la investigación de MinMaxScaler :
No es directamente posible usar MinMaxScaler para escalar imágenes en la misma forma que se haría
con columnas específicas de un DataFrame. Si tuvieramos un dataframe con columnas específicas podríamos hacer
algo parecido a este ejercicio : from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Crear un column transformer que nos ayude a normalizar/preprocesar nuestros datos
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]), # ajustar todos los valores entre 0 y 1
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)
# Creamos X e y
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

# Nos quedamos con un split de train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el column transformer sólo con los datos de entrenamiento (hacerlo en test produciría "data leakage")
ct.fit(X_train)

# Transform training and test data with normalization (MinMaxScalar) and one hot encoding (OneHotEncoder)
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test

6) Creamos el modelo de la siguiente manera :
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
¿Cómo debe ser la capa de salida? ¿Qué función de activación debe tener?
La capa de salida es de esta forma : tf.keras.layers.Dense(10, activation='softmax') , ya que necesitamos 10 neuronas
e investigando como solucionar el error de clasificación multiclase nos recomendaba usar la activación softmax

7) Compilamos el modelo de la siguiente manera :
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['Accuracy'])

8) Los resultados son muy malos, obtenemos muy poco aprendizaje: un acierto del 86% aproximadamente y una pérdida muy baja
de unos 0.3

9) Evaluamos el modelo de la siguiente forma :
loaded_model = tf.keras.models.load_model('zalando_fashion_mnist_model.h5')

print(loaded_model.evaluate(X_test_normalized, y_test)), obtenemos la siguiente salida :
    [0.3482398986816406, 0.8762000203132629]

10) Ejecutamos el siguiente código :
classifications = loaded_model.predict(X_test_normalized)
    print(classifications[0])

El resultado de la ejecución nos da estos números :
    [1.1162268e-04 1.1399360e-06 1.1661443e-06 1.0660240e-05 3.3630624e-07
    6.1134905e-02 3.0534788e-06 4.5179516e-02 3.5995683e-05 8.9352155e-01]
    creemos que esos números representan la probabilidad de cada clase
11) El resultado es la b)El décimo elemento de la lista es el más grande y la bota tiene la etiqueta 9

12) Cargamos la imágen a nuestro modelo
La imágen la hemos generado con paint a mano con una resolución de 28 x 28 píxeles
y la hemos cargado de la siguiente manerea :
import cv2
img=cv2.imread('miImagen.png',cv2.IMREAD_GRAYSCALE)
plt.imshow(img)

13) Al incrementar las neuronas de la capa densa a 512 y entrenar el mismo número de épocas = 5
  obtenemos que el % de acierto se incrementa un 1% pasa de 88.6 a 89.6 y la pérdida se reduce
  por ende , en la última época de 0.289 a 0.279

  Incrementamos la capa oculta (La que sigue a nuestra capa flatten a 1024 neuronas)
  respuesta : a) En nuestro caso el entrenamiento lleva más tiempo , es más preciso
  pero casi insignificante la subida 89.9% de precisión y 0.27 de pérdida

14) Si eliminamos la capa flatten , nos da un error de forma y no nos deja compilar ni entrenar
  el modelo . Ya que parece que tras buscar información hemos visto que las capas Densas, esperan
  datos de entrada unidimensionales, pero las imágenes de fashion MNIST son de (28,28)
  y la capa flatten de nuestro modelo transforma la matriz bidimensional en unidimensional.
'''

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
mnist_train = mnist.load_data()

#print(training_images)

lista = [ x for x in training_images]
(X_train, y_train), ( X_test, y_test)= mnist.load_data()

plt.imshow(X_train[0])


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

#Normalizamos los datos
X_train_normalized = X_train/255
X_test_normalized = X_test/255

y_train_normalized = y_train/10
y_test_normalized = y_test/10

print(X_train_normalized, X_test_normalized)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

print(y_train.shape)
y_train_oneHot = tf.one_hot(y_train, depth=10)


# model.fit(X_train_normalized, y_train, epochs=5)
# model.save('zalando_fashion_mnist_model.h5')
# Load the model (for future use)

loaded_model = tf.keras.models.load_model('zalando_fashion_mnist_model.h5')

print(loaded_model.evaluate(X_test_normalized, y_test))


classifications = loaded_model.predict(X_test_normalized)
plt.imshow(X_test[0])
plt.show()
print(classifications[0])


# Ruta de la imagen
image_path = './miImagen.png'

# Cargar la imagen
img = plt.imread(image_path)

# Mostrar la imagen
plt.imshow(img)
plt.axis('off')  # Opcional: quitar los ejes
plt.show()

import cv2
img=cv2.imread('miImagen.png',cv2.IMREAD_GRAYSCALE)
plt.imshow(img)

img=tf.reshape(img, (28, 28))
print(tf.Variable([img]))

classifications = loaded_model.predict(tf.Variable([img]))
print(classifications[0])