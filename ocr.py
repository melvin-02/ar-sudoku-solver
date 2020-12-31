import cv2 
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

PATH = "digitImages"
BATCH_SIZE = 32
EPOCHS = 22


images = []    
classNo = []    
digits = os.listdir(PATH)

print("Total Classes Detected:",len(digits))
noOfClasses = len(digits)
print("Importing Classes .......")
for x in range (1, noOfClasses + 1):
    num_list = os.listdir(PATH + "/" + str(x))
    for y in num_list:
        curImg = cv2.imread(PATH + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg, (32,32), cv2.INTER_AREA)
        images.append(curImg)
        classNo.append(x)
    print(x,end= " ")
print(" ")
print("Total Images in Images List = ",len(images))
print("Total labels in classNo List= ",len(classNo))

images = np.array(images)
classNo = np.array(classNo)

x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size = 0.2, random_state = 420)
encoder = LabelEncoder()
encoder.fit(y_train)
y_train, y_test = encoder.transform(y_train), encoder.transform(y_test)
y_train, y_test = to_categorical(y_train, noOfClasses), to_categorical(y_test, noOfClasses)
with open('model/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    img = img.astype('float32')
    return img

x_train= np.array( list( map( preProcessing, x_train )))
x_test= np.array( list( map( preProcessing, x_test )))

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

data_gen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             )


def create_model(input_shape, output_shape):

    model = Sequential()
    model.add(Input(shape = input_shape))
    model.add( Conv2D(64, (3,3), padding='same' ))
    model.add( MaxPool2D((2,2)))

    model.add( Conv2D(64, (3,3), padding='same'))
    model.add( MaxPool2D((2,2)))

    model.add( Flatten())
    model.add( Dense(256, activation='relu'))
    model.add( Dropout(0.5))
    model.add( Dense(256, activation='relu'))
    model.add( Dropout(0.5))

    model.add( Dense(output_shape, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def scheduler(epoch, lr):
    if epoch < 15:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lrcallback = LearningRateScheduler(scheduler)

model = create_model(input_shape=(x_train.shape[1], x_train.shape[2], 1), output_shape=noOfClasses)

r = model.fit( data_gen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True), 
            batch_size= BATCH_SIZE,
            epochs= EPOCHS,
            steps_per_epoch= x_train.shape[0] // BATCH_SIZE,
            validation_data= data_gen.flow(x_test, y_test, batch_size=BATCH_SIZE),
            validation_steps= x_test.shape[0] // BATCH_SIZE,
            callbacks = [lrcallback]
            )

model.save('models/digitOCR_seq.h5')