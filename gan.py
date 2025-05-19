# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:33:51 2025

@author: HP
"""
# Install library “tqdm” from Anaconda Navigator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers import LeakyReLU
from keras.optimizers import Adam


def load_data():
    # Load MNIST dataset and normalize the images
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    # Reshape images from (num_samples, 28, 28) to (num_samples, 784)
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    return (x_train, y_train, x_test, y_test)

def adam_optimizer():
    return Adam(learning_rate=0.0002, beta_1=0.5)

def create_generator():
    generator = Sequential()
    generator.add(Dense(units=128, input_dim=100))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Dense(units=128))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Dense(units=784, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator

def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(units=128, input_dim=784))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dense(units=1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer(), metrics=['accuracy'])
    return discriminator

def create_gan(discriminator, generator):
    # Freeze discriminator weights in the GAN model
    discriminator.trainable = False
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return gan

def plot_generated_images(epoch, generator, examples=16, dim=(4, 4), figsize=(6, 6)):
    noise = np.random.normal(0, 1, [examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("gan_generated_image_epoch_%d.png" % epoch)
    plt.close()

def training(epochs=1, batch_size=128):
    # Load data
    (X_train, y_train, X_test, y_test) = load_data()
    batch_count = X_train.shape[0] // batch_size

    # Build GAN components
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(discriminator, generator)

    for e in range(1, epochs + 1):
        print("Epoch %d" % e)
        for _ in tqdm(range(batch_count)):
            # Generate fake images
            noise = np.random.normal(0, 1, [batch_size, 100])
            generated_images = generator.predict(noise)
            # Select a random batch of real images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            image_batch = X_train[idx]
            # Combine real and fake images
            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 1

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train GAN: try to trick the discriminator
            noise = np.random.normal(0, 1, [batch_size, 100])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        # Periodically plot generated images
        if e == 1 or e % 20 == 0:
            plot_generated_images(e, generator)

training(epochs=200, batch_size=128)