import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
input_dim = (28,28,1)
output_dim = 10
batch_size = 128

x_train, x_test = x_train.reshape((x_train.shape[0], 28, 28, 1)), x_test.reshape((x_test.shape[0], 28, 28, 1))
x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
y_train, y_test = to_categorical(y_train, output_dim), to_categorical(y_test, output_dim)


def create_model(input_shape, output_shape):
    i = Input(shape = input_shape)
    x = Conv2D(32, (3,3), padding='same')(i)
    x = MaxPool2D((2,2))(x)

    x = Conv2D(32, (3,3), padding='same')(x)
    x = MaxPool2D((2,2))(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(output_shape, activation='softmax')(x)

    model = Model(i, x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = create_model(input_dim, output_dim)
r = model.fit(x_train,
            y_train,
            batch_size= batch_size,
            steps_per_epoch= len(x_train) // batch_size,
            epochs=10,
            validation_data=(x_test, y_test))

model.save('mnist_model.h5')