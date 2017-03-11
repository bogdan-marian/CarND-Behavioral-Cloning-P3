import csv
import cv2
import numpy as np

lines =[]
with open ('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images =[]
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
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

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add( Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=5)

model.save('model.h5')
exit()
