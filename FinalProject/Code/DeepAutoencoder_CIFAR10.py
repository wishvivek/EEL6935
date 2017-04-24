
# coding: utf-8

# In[5]:

from __future__ import print_function
import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta, RMSprop

batch_size = 32
num_classes = 10
epochs = 200
data_augmentation = True

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#print (x_train)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

'''
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''

encoding_dim = 32

input_img = Input(shape=(3072, ))

#encoded = Dense(encoding_dim, activation = 'relu')(input_img)
encoded = Dense(128, activation='tanh')(input_img)
encoded = Dense(64, activation='tanh')(encoded)
encoded = Dense(32, activation='tanh')(encoded)

decoded = Dense(64, activation='tanh')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(3072, activation='sigmoid')(decoded)

autoencoder = Model(input=input_img, output=decoded)

encoder = Model(input=input_img, output=encoded)
encoded_input = Input(shape=(encoding_dim,))

decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]

# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

#decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

optimizer = RMSprop(lr = 0.03)
autoencoder.compile(optimizer = optimizer, loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, nb_epoch=100, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

encoded_images = encoder.predict(x_test)
print (encoded_images.shape)
decoded_images = decoder.predict(encoded_images)



# In[6]:

import matplotlib.pyplot as plt

print ('Here is the visualization of the CIFAR10 DeepAutoencoder Predictions:-')
decoded_imgs = autoencoder.predict(x_test)
n = 100

plt.figure(figsize=(10, 10))
for i in range(n):
    ax = plt.subplot(10, 10, i + 1)
    image = decoded_imgs[i]
    plt.imshow(np.rollaxis(decoded_imgs[i].reshape((32, 32, 3)), 0 , 1))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:



