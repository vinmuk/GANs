from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.initializers import RandomNormal

import sys

import numpy as np

class RaGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
    
        self.optimizer = Adam(0.0002, 0.5)

        # Relavastic average gan
        self.avg = True

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)

        
    def los(self, yt, yp):
        return K.mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=yp, labels=yt))

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=3000):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1)).astype(np.float32)
        fake = np.zeros((batch_size, 1)).astype(np.float32)
        
        
        #place holders
        real_img = Input(shape=(self.img_rows, self.img_cols, self.channels), dtype='float32')
        noise = Input(shape=(self.latent_dim,) ,dtype='float32')

        # output of discriminator and combined model.
        d_out = self.discriminator(real_img) 
        c_out = self.combined(noise)

        # loss
        if(self.avg):
            #loss_dis = self.los(d_out - K.mean(c_out), valid)+ self.los(c_out - K.mean(d_out), fake)
            #loss_gen = self.los(c_out - K.mean(d_out), valid)+ self.los(d_out - K.mean(c_out), fake)
            loss_dis = (K.mean((d_out-K.mean(c_out) - valid) ** 2)
                    +K.mean((c_out-K.mean(d_out) + valid) ** 2))/2.
            loss_gen = (K.mean((c_out-K.mean(d_out) - valid) ** 2)
                    +K.mean((d_out-K.mean(c_out) + valid) ** 2))/2.
        else:
            loss_dis = self.los(d_out - c_out, valid)+ self.los(c_out - d_out, fake)
            loss_gen = self.los(c_out - d_out, valid)+ self.los(d_out - c_out, fake)

        # discriminator training
        training_updates = self.optimizer.get_updates(params=self.discriminator.trainable_weights, loss=loss_dis)
        dis_train = K.function([real_img, noise], [loss_dis], training_updates)

        # generator training
        training_updates = self.optimizer.get_updates(params=self.generator.trainable_weights, loss=loss_gen)
        gen_train = K.function([real_img, noise], [loss_gen], training_updates)
        

        for epoch in range(epochs):

            # Select a random real images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the models
            d_loss = dis_train([imgs, noise])
            g_loss = gen_train([imgs, noise])

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss[0]))


            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    ragan = RaGAN()
    ragan.train(epochs=4000, batch_size=128, save_interval=50)