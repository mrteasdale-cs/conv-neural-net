# -*- coding: utf-8 -*-
"""
Author: Myran
Date: 27/2/25
Description: CNN using the Cifar Image Dataset - using OOP principles and Leaky ReLU
"""
#import libraries
import tensorflow as tf
import pandas as pd
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization, LeakyReLU
from keras.utils import to_categorical
#from keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator # used in data augmentation
from datetime import datetime
import os
import matplotlib.pyplot as plt

class CNN:
    def __init__(self, epochs, batch_size):
        #cnn attributes for when a new model is instantiated
        self.model = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_shape = (32, 32, 3)
        self.num_classes = 10
        self.history = None
        
    def loadAndPreprocessData(self):
        # import the dataset
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        # onehot encoding - converts the training y output values to a binary representation
        # using the keras.utils.to_catagorical function (e.g. 10000000, 010000000 etc)
        y_train_onehot = to_categorical(y_train, 10)
        y_test_onehot = to_categorical(y_test, 10)

        # convert pixels to normalised values 0-1
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255
        
        return X_train, y_train_onehot, X_test, y_test_onehot
    
    def augment_data(self, X_train):
        # Data Augmentation to genreate slightly difference images by manipulating the dataset
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
        )
        datagen.fit(X_train)
               
        return datagen
    
    #function uses several parameters to make modification of hyperparameters more siplified
    def build_model(self, filters=32, kernel_size=(3,3), dropout_rate=0.25, learning_rate=0.001, decay_rate=0.01, alpha=0.3):
        # create a sequential model
        model = Sequential([
            # add several layers to the CNN including the convelutional layers (32 = 32 depth/neurons)
            # max pooling layers with 2x2 filters, and dropout layers with 0.25 / 25% probability
            Input(shape=(32, 32, 3)),
            Conv2D(filters, kernel_size=kernel_size, padding='same'),
            LeakyReLU(alpha=alpha),
            Conv2D(filters, kernel_size=kernel_size, padding='same'),
            LeakyReLU(alpha=alpha),
            BatchNormalization(), # normalise each batch
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout_rate), # dropout (default 0.5) to drop a specified number of connections during training
            Conv2D(filters * 2, kernel_size=kernel_size, padding='same'),
            LeakyReLU(alpha=alpha),
            Conv2D(filters * 2, kernel_size=kernel_size, padding='same'), # filters * 2  doubles the orignal layer filters
            LeakyReLU(alpha=alpha),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout_rate),
            # at this point, our neurons are spatially arranged in a cube-like format rather
            # than in just one row. To make this cube-like format of neurons into one row,
            # we have to first flatten it.
            Flatten(),
            # add a dense layer with 128 neurons
            Dense(128),
            LeakyReLU(alpha=alpha),
            Dropout(dropout_rate),
            Dense(10, activation='softmax')
        ])
        self.model = model
        # set learning rate of optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=cnn.learning_rate_schedule(learning_rate,decay_rate))
        # compile the model using Adam optimizer and LR of 0.0001
        model.compile(optimizer=optimizer, 
                           loss="categorical_crossentropy", 
                           metrics=['accuracy'])
        
    
    def learning_rate_schedule(self, lr, dr):
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = lr, # learning rate passed as parameter from its parent function (build_model)
            decay_steps = 10000,
            decay_rate = dr # decay rate passed as parameter from its parent function (build_model)
            )
    
    def save_model(self, path):
        self.model.save(path)
        
        
    #load the model
    def load_model(self, path):
        return tf.keras.models.load_model(path)


    def train(self, datagen, X_train, y_train_onehot, X_test, y_test_onehot, use_augmentation=False):
        
        # callback = [CSVLogger('training_log.csv', append=True)]
        callback = []
        
        if use_augmentation:
            # Train with augmented data
            training_data = datagen.flow(X_train, y_train_onehot, batch_size=self.batch_size)
        else:
            # Train without augmentation
            training_data = (X_train, y_train_onehot)
        
        # Fit the model
        if use_augmentation:
            history = self.model.fit(
                training_data,
                epochs=self.epochs,
                validation_data=(X_test, y_test_onehot),
                callbacks=callback
            )
        else: #if augmentation is not used, apply these parameters to the fit function
            history = self.model.fit(
                X_train,
                y_train_onehot,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_test, y_test_onehot),
                callbacks=callback
            )
        
        history_dframe = pd.DataFrame(history.history)
        history_path = 'training_history.xlsx'
        sheet_name = f"Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if os.path.exists(history_path):
            with pd.ExcelWriter(history_path, engine='openpyxl', mode="a", if_sheet_exists='overlay') as writer:
                history_dframe.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            print("Error locating training history file.")
        
        return history

        
    def plot_metrics(self, trained_model):
        # Plot training and validation loss
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1) #create smaller subplot for better viewing side by side
        plt.plot(trained_model.history['loss'], label='Train Loss')
        plt.plot(trained_model.history['val_loss'], label='Validation Loss')
        plt.title("Model Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend(['Train', 'Val'])
        
        plt.subplot(1, 2, 2) #create smaller subplot for better viewing side by side
        plt.plot(trained_model.history['accuracy'])
        plt.plot(trained_model.history['val_accuracy'])
        plt.title("Accuracy (CNN 10 Class image set)")
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'])
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.show()
           
    
# Main execution using the CNN class
if __name__ == "__main__":
    cnn = CNN(18, 32) # create new instance of CNN class with 22 epochs and batch_size of 24
    
    # Load and preprocess data
    X_train, y_train_onehot, X_test, y_test_onehot = cnn.loadAndPreprocessData()
    print(f"X Train Shape: {X_train.shape}") # check the set has loaded
    print(f"Y Train Shape: {y_train_onehot.shape}")
    # augment data with additional generated images
    datagen = cnn.augment_data(X_train)
    
    # Build and compile the CNN model
    cnn.build_model()
    
    # Train the model
    # Augmentation ON - set use_augmentation to True
    trained_model = cnn.train(datagen, X_train, y_train_onehot, X_test, y_test_onehot, use_augmentation=True)
    # Save the trained CNN model to file
    cnn.save_model("models/cnn_model_14_removedcnnlayer_lrelu.keras")
    # Plot training metrics (loss and accuracy)
    cnn.plot_metrics(trained_model)