
# coding: utf-8

# In[8]:

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adadelta, RMSprop

encoding_dim = 32

input_img = Input(shape=(784, ))

#encoded = Dense(encoding_dim, activation = 'relu')(input_img)
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input=input_img, output=decoded)

encoder = Model(input=input_img, output=encoded)
encoded_input = Input(shape=(encoding_dim,))

#decoder_layer = autoencoder.layers[-1]

decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]

# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

#decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

optimizer = RMSprop(lr = 0.00004)
autoencoder.compile(optimizer = optimizer, loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape

autoencoder.fit(x_train, x_train, nb_epoch=100, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

encoded_images = encoder.predict(x_test)
print encoded_images.shape
decoded_images = decoder.predict(encoded_images)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20,4))

for i in range(n):
	ax = plt.subplot(2, n, i+1)
	plt.imshow(x_test[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(2, n, i + 1 + n)
	plt.imshow(decoded_images[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()
