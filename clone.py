import csv
import cv2
import numpy as np

lines =[]
dataFolder = "/home/bogdan/colected_data/session3"
with open (dataFolder+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images =[]
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = dataFolder+'/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image);
    measurement = float(line[3])
    measurements.append(measurement)

augumented_images, augumented_measurements = [], []
for image,measurement in zip(images,measurements):
    augumented_images.append(image)
    augumented_measurements.append(measurement)
    augumented_images.append(cv2.flip(image,1))
    augumented_measurements.append(measurement * -1.0)

x_train = np.array(augumented_images)
y_train = np.array(augumented_measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add( Cropping2D(cropping=((70,25), (0,0))))
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
history_object = model.fit(x_train, y_train,
    validation_split=0.2,
    shuffle=True, nb_epoch=5, 
    verbose=1)

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
