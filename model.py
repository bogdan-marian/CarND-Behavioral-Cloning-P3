import os
import csv

samples = []

#my root folder for the data
dataFolder = "/home/bogdan/colected_data/workspace"

with open(dataFolder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

#define the generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = dataFolder+'/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

#define the model
def define_network():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    model.add( Cropping2D(cropping=((70,25), (0,0))))
    model.add(Dropout(0.1))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))

    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

#train the model
model = define_network()
history_object = model.fit_generator(train_generator,
                        samples_per_epoch = len(train_samples),
                        validation_data = validation_generator,
                        nb_val_samples = len(validation_samples), nb_epoch=3)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()

#just a list of shortcuts used by me when working with the model
# python3 clone.py
# python3 drive.py model.h5 run1
# python3 video.py run1 --fps 120
# totem run1.mp4 &
