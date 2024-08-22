# Handwritten digit recognition for MNIST dataset using Convolutional Neural Networks

import sys
sys.stdout.reconfigure(encoding='utf-8')

# import keras 
import keras
from keras import layers
import numpy as np
from keras.src.utils import load_img
from keras.src.utils import img_to_array
from keras.src.saving import load_model

def load_dataset():
    input_shape = (28, 28, 1)

    # load and return training and test datasets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # reshape for X train and test vars
    x_train = x_train.astype('float32')
    x_test = x_test.astype("float32") 

    # normalize inputs from 0-255 to 0-1
    x_train = x_train / 255
    x_test = x_test/255


    # convert y_train and y_test to categorical classes
    # use 10 as it is categorical digits from 0-9
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # check if images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)

    # return your X_train, X_test, y_train, y_test
    print("x training samples: ", x_train.shape[0])
    print("x testing samples: ", x_test.shape[0])
    print("y training samples: ", y_train.shape[0])
    print("y testing samples: ", y_test.shape[0])
    return x_train, y_train, x_test, y_test, input_shape

# Define the model
def digit_recognition_cnn(x_train, y_train, x_test, y_test, input_shape):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model

x_train, y_train, x_test, y_test, input_shape = load_dataset()

# build the model
model = digit_recognition_cnn(x_train, y_train, x_test, y_test, input_shape)

# train the model
batch_size = 200
epochs = 20

# after the model has been built, compile it
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", score[1])

# save model
model.save("digitRecognizer.h5")

# Load the saved model
saved_model = load_model("digitRecognizer.h5")
saved_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# load and normalize new image
def load_new_image(path):
    # load new image
    img = load_img(path, color_mode="grayscale", target_size=(28, 28))

    # Convert image to array
    img_array = img_to_array(img)

    # reshape into a single sample with 1 channel 
    img_array = np.expand_dims(img_array, axis=0)

    # normalize image data
    img_array = img_array.astype('float32') / 255

    # return array
    return img_array

# load a new image and predict its class
def test_model_performance():
    for i in range(1,10):
        path = f"sample_images/digit{i}.png"
        img_array = load_new_image(path)
        # Make prediction
        prediction = saved_model.predict(img_array)
        imageClass = np.argmax(prediction)
        print(f"Predicted class for {path}: {imageClass}".encode('utf-8'))

test_model_performance()




