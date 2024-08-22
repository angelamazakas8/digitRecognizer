# this code just ensures the saved model can be used

import sys
sys.stdout.reconfigure(encoding='utf-8')

#import numpy as np
import keras
from keras import layers
import numpy as np
from keras.src.utils import load_img
from keras.src.utils import img_to_array
from keras.src.saving import load_model

def load_dataset():
    input_shape = (28, 28, 1)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype("float32") 

    x_train = x_train / 255
    x_test = x_test/255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)

    print("x training samples: ", x_train.shape[0])
    print("x testing samples: ", x_test.shape[0])
    print("y training samples: ", y_train.shape[0])
    print("y testing samples: ", y_test.shape[0])

    return x_train, y_train, x_test, y_test, input_shape


x_train, y_train, x_test, y_test, input_shape = load_dataset()


# load the saved model
model = load_model("digitRecognizer.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# evaluate your model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", score[1])

# load the saved model
saved_model = load_model("digitRecognizer.h5")
saved_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# load and normalize new image
def load_new_image(path):
    # load new image
    img = load_img(path, color_mode="grayscale", target_size=(28, 28))

    # convert image to array
    img_array = img_to_array(img)

    # reshape into a single sample with 1 channel 
    img_array = np.expand_dims(img_array, axis=0)

    # normalize image data
    img_array = img_array.astype('float32') / 255

    return img_array

# load a new image and predict its class
def test_model_performance():
    path = "sample_images/digit1.png"
    img_array = load_new_image(path)
    # Make prediction
    prediction = saved_model.predict(img_array)
    imageClass = np.argmax(prediction)
    return print("Predicted class:", imageClass)

test_model_performance()




