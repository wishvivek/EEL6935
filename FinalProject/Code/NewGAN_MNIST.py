
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')
import os, random
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["THEANO_FLAGS"]  = "device=gpu%d, lib.cnmem=0"%(random.randint(0,3))
import numpy as np
#import theano as th
#import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,MaxPooling2D,merge
from keras.models import Sequential
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, UpSampling1D, Convolution2D, UpSampling2D
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist, cifar10
from sklearn import preprocessing
from sklearn.utils import shuffle
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cPickle, random, sys, keras
from keras.models import Model
from IPython import display

sys.path.append("../common")
from keras.utils import np_utils
from tqdm import tqdm

K.set_image_dim_ordering('tf')


# In[2]:

img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[:7000]
X_test = X_test[:7000]
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print np.min(X_train), np.max(X_train)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[3]:

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


# In[4]:

shp = X_train.shape[1:]
print shp

dropout_rate = 0.25
# Optim

opt = Adam(lr=1e-3)
dopt = Adam(lr=1e-4)
#opt = Adam(lr=1e-3)
#opt = Adamax(lr=1e-4)
#opt = Adam(lr=0.0002)
#opt = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

# Build Generative model ...
input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
generator = Model(input_img, decoded)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()


# Build Discriminative model ...
d_input = Input(shape=shp)
H = Conv2D(256, (5, 5), strides=(2, 2), padding = 'same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Conv2D(512, (5, 5), strides=(2, 2), padding = 'same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2,activation='softmax')(H)
discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()

# Freeze weights in the discriminator for stacked training
make_trainable(discriminator, False)
# Build stacked GAN model
gan_input = Input(shape=(28, 28, 1))
H = generator(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()


# In[5]:

def plot_loss(losses):
        #display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.figure(figsize=(10,8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        plt.show()


# In[6]:

def plot_gen(n_ex=16,dim=(4,4), figsize=(10,10) ):
    noise = np.random.uniform(0,1,size=[n_ex,784])
    noise = noise.reshape(n_ex, 28, 28, 1)
    generated_images = generator.predict(noise)
    #print 'the shape of generated_images: ', generated_images.shape #the shape of generated_images:  (16, 28, 28, 1)
    #print 'the rockafeller shape: ', generated_images[1,0,:,:].shape #the rockafeller shape:  (28, 1)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        img = generated_images[i,0,:,:]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# In[7]:

#print X_train.shape[0] = 7000

ntrain = 7000
trainidx = random.sample(range(0, X_train.shape[0]), ntrain)
#print 'trainidx: ', trainidx.shape
XT = X_train[trainidx,:,:,:]
print 'XT: ', XT.shape

# Pre-train the discriminator network ...
noise_gen = np.random.uniform(0, 1, size=5488000)
noise_gen = noise_gen.reshape(ntrain, 28, 28, 1)
generated_images = generator.predict(noise_gen)
print 'generated_images: ', generated_images.shape
X = np.concatenate((XT, generated_images))
n = XT.shape[0]
y = np.zeros([2*n,2])
y[:n,1] = 1
y[n:,0] = 1

make_trainable(discriminator,True)
discriminator.fit(X, y, epochs=1, batch_size=32)
y_hat = discriminator.predict(X)


# In[8]:

y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
diff = y_idx-y_hat_idx
n_tot = y.shape[0]
n_rig = (diff==0).sum()
acc = n_rig*100.0/n_tot
print "Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot)


# In[9]:

# set up loss storage vector
losses = {"d":[], "g":[]}


# In[10]:

def train_for_n(nb_epoch=1000, plt_frq=25,BATCH_SIZE=32):

    for e in tqdm(range(nb_epoch)):  
        
        # Make generative images
        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]
        noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 784])
        noise_gen = noise_gen.reshape(BATCH_SIZE, 28, 28, 1)
        generated_images = generator.predict(noise_gen)
        
        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        make_trainable(discriminator,True)
        d_loss  = discriminator.train_on_batch(X,y)
        losses["d"].append(d_loss)
    
        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE,784])
        noise_tr = noise_tr.reshape(BATCH_SIZE, 28, 28, 1)
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:,1] = 1
        
        make_trainable(discriminator,False)
        g_loss = GAN.train_on_batch(noise_tr, y2)
        losses["g"].append(g_loss)
        
        # Updates plots
        if e%plt_frq==plt_frq-1:
            plot_loss(losses)
            plot_gen()


# In[16]:

train_for_n(nb_epoch=10000, plt_frq=500,BATCH_SIZE=128)


# In[14]:

K.set_value(opt.lr, 1e-4)
K.set_value(dopt.lr, 1e-5)
train_for_n(nb_epoch=100, plt_frq=10,BATCH_SIZE=128)


# In[15]:

K.set_value(opt.lr, 1e-5)
K.set_value(dopt.lr, 1e-6)
train_for_n(nb_epoch=100, plt_frq=10,BATCH_SIZE=256)


# In[ ]:



