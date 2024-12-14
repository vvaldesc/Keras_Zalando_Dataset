# Keras Zalando Dataset

This repository contains a project using Keras to work with the Zalando Fashion MNIST dataset for image classification.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vvaldesc/Keras_Zalando_Dataset.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Keras_Zalando_Dataset
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Load and preprocess the dataset:
   ```python
   mnist = tf.keras.datasets.fashion_mnist
   (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
   training_images = training_images / 255.0
   test_images = test_images / 255.0
   ```

2. Build and compile the model:
   ```python
   model = tf.keras.models.Sequential([
       tf.keras.layers.Flatten(input_shape=(28, 28)),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10, activation='softmax')
   ])
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   ```

3. Train the model:
   ```python
   model.fit(training_images, training_labels, epochs=10)
   ```

4. Evaluate the model:
   ```python
   test_loss, test_acc = model.evaluate(test_images, test_labels)
   print('\nTest accuracy:', test_acc)
   ```

5. Make predictions:
   ```python
   predictions = model.predict(test_images)
   print(predictions[0])
   ```

## Contributors

- [vvaldesc](https://github.com/vvaldesc)
- [DaniGhr43](https://github.com/DaniGhr43)
- [PatriciaIA](https://github.com/pfernandezdi)

For more details on commits and project updates, visit the [commits page](https://github.com/vvaldesc/Keras_Zalando_Dataset/commits).

---

Please let me know if you need any additional sections or information included in the README.
