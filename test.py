from __future__ import print_function
import sys
import heapq
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from utils import *

batch_size = 128
nb_filter = 32          # Number of convolutional filters to use
pool_size = (2, 2)      # Size of poolig area
kernel_size = (3, 3)    # Convolution kernel size

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
          'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
          'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
          'y', 'z']

"""
Load test dataset here.
"""
test_folder = 'test_imgs'
min_num_images = 10
x_test = load_letter(test_folder, min_num_images, is_invert = True)

nb_classes = 36

img_width = x_test.shape[1]
img_height = x_test.shape[2]

if K.image_dim_ordering() == 'th':
    X_test = x_test.reshape(x_test.shape[0], 1, img_height, img_width)
    input_shape = (1, img_height, img_width)
else:
    X_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)
    input_shape = (img_height, img_width, 1)

""" Change type and normalize. """
X_test = X_test.astype('float32')
X_test /= 255
nb_samples = X_test.shape[0]
print(nb_samples, 'test samples.')

""" Create a sequential model. """
model = Sequential()

model.add(Convolution2D(nb_filter, kernel_size[0], kernel_size[1],
                        border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filter, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

""" Let's look at the summary of the model. """
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

weights_file = os.path.join('weights/', 'chars74k_weights.h5')
print(weights_file)

if os.path.exists(weights_file):
    """ Load pre-computed weights """
    print('Loading weights...')
    model.load_weights(weights_file)

    """ Predict for test dataset """
    predictions = model.predict(X_test)

    fig = plt.figure()
    gs = gridspec.GridSpec(1, nb_samples)

    for i in np.arange(nb_samples):
        sample_img = x_test[i, :, :]
        fig.add_subplot(gs[i])
        plt.imshow(sample_img, cmap='gray')

        indxs = heapq.nlargest(3, range(len(predictions[i])), predictions[i].take)
        scores = predictions[i][indxs]
        out_test = str('{:s}={:.3f}\n{:s}={:.3f}\n{:s}={:.3f}'.format(
            str(labels[indxs[0]]), scores[0],
            str(labels[indxs[1]]), scores[1],
            str(labels[indxs[2]]), scores[2]))

        plt.text(0, -10, out_test, color='brown')

    plt.show()
else:
    print('No pre-trained weights found! Exiting.')
