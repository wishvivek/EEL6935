
# coding: utf-8

# In[4]:

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist, cifar10
from keras.optimizers import RMSprop
import numpy as np

(x_train, _), (x_test, _) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

input_img = Input(shape=(32, 32, 3))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((4, 4))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
optimizer = RMSprop(lr = 0.001)
autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs = 50, batch_size = 128, shuffle = True, validation_data = (x_test, x_test))


# In[5]:

import matplotlib.pyplot as plt

print ('Here is the visualization of the CIFAR10 DeepConvAutoencoder Predictions:-')
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

'''
import matplotlib.pyplot as plt

print ('Here is the visualization of the CIFAR10 DeepConvAutoencoder Predictions:-')
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
'''

