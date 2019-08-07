#sagan
from __future__ import print_function, division
from sn import DenseSN, ConvSN2D
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Permute
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Lambda, Add, multiply
from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.initializers import RandomNormal

import sys

import numpy as np

class SAGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
    
        self.d_optimizer = Adam(0.0002, 0.5)
        self.g_optimizer = Adam(0.0001, 0.5)

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

        
    def los(self, yp, yt):
        return K.mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=yp,labels=yt))
    
    def up_block(self, x, n_channels):
        x = UpSampling2D()(x)
        x = Conv2D(n_channels, kernel_size=3, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("relu")(x)
        return x
    
    def down_block(self, x, n_channels):
        x = Conv2D(n_channels, kernel_size=3, strides=2, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        return x
    
    def selfAttention(self, x, n_channels, k=8):
        x_shape = K.int_shape(x)
        f = Reshape((-1,x_shape[-1]))(ConvSN2D(n_channels//k, 1, padding='same')(x)) # [b, n, c/k]
        g = Reshape((-1,x_shape[-1]))(ConvSN2D(n_channels//k, 1, padding='same')(x)) # [b, n, c/k]
        g = Permute((1,2))(g) # [b,c/k,n]
        s = multiply([f,g])
        s = Softmax(axis=-1)(s) #[b,n,n]
        h = Reshape((-1,x_shape[-1]))(ConvSN2D(n_channels//k, 1, padding='same')(x)) #[b,n,c/k]
        v  = multiply([s,h])#[b,n,c/k]
        v = Reshape((x_shape[1],x_shape[2],n_channels//k))(v)
        o = ConvSN2D(n_channels, 1, padding='same')(v) #[b,h,w,c]
        gamma = K.variable(value=0)
        ret = Lambda(lambda x: x*gamma)(o)
        ret = Add()([ret, x])
        return ret

    def build_generator(self):
        noise = Input(shape=(self.latent_dim,))
        x = Dense(128 * 7 * 7, input_dim=self.latent_dim)(noise)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((7, 7, 128))(x)
        x = self.up_block(x, 128)
        #x = ZeroPadding2D(1)(x)
        x = self.up_block(x, 64)
        x = self.selfAttention(x, 64)
       # x = self.up_block(x, 32)
        x = Conv2D(self.channels, kernel_size=3, padding="same")(x)
        x = Activation("tanh")(x)
        return Model(noise, x)

    def build_discriminator(self):
        img = Input(shape=self.img_shape)
        x = self.down_block(img, 32)
        x = self.down_block(x, 64)
        x = self.down_block(x, 128)
        x = self.selfAttention(x, 128)
        x = self.down_block(x, 256)
        x = Flatten()(x)
        x = Dense(1)(x)
        return Model(img, x)

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
        loss_dis_real = self.los(d_out, valid)
        loss_dis_fake = self.los(c_out, fake)
        loss_gen = self.los(c_out, valid)


        # discriminator training
        training_updates = self.d_optimizer.get_updates(params=self.discriminator.trainable_weights, loss=loss_dis_real)
        dis_train_real = K.function([real_img, noise], [loss_dis_real], training_updates)
        training_updates = self.d_optimizer.get_updates(params=self.discriminator.trainable_weights, loss=loss_dis_fake)
        dis_train_fake = K.function([real_img, noise], [loss_dis_fake], training_updates)

        # generator training
        training_updates = self.g_optimizer.get_updates(params=self.generator.trainable_weights, loss=loss_gen)
        gen_train = K.function([real_img, noise], [loss_gen], training_updates)
        

        for epoch in range(epochs):

            # Select a random real images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the models
            d_loss = (dis_train_real([imgs, noise])[0]+dis_train_fake([imgs, noise])[0])/2
            g_loss = gen_train([imgs, noise])

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss[0]))

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
    sagan = SAGAN()
    sagan.train(epochs=40000, batch_size=128, save_interval=50)
