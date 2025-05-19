"""
Author: Myran
Date: 27/2/25
Description: CNN using the Cifar Image Dataset - using GridSearch
"""
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import os

TF_ENABLE_ONEDNN_OPTS=0

# Define the model-building function
def build_model(filters=32, kernel_size=(3, 3), dropout_rate=0.5, learning_rate=0.001, activation='relu'):
    model = Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(filters, kernel_size=kernel_size, activation=activation, padding='same'),
        BatchNormalization(), # normalise each batch
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),

        Conv2D(filters * 2, kernel_size=kernel_size, activation=activation, padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        
        Conv2D(filters * 4, kernel_size=kernel_size, activation=activation, padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),

        Flatten(),
        Dense(128, activation=activation),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    return model

# Load cifar dataset
def load_dataset(name):
    (X_train, y_train), (X_test, y_test) = name.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return X_train, X_test, y_train, y_test

# Wrap the model with KerasClassifier
model = KerasClassifier(
    model=build_model,
    verbose=0,
)

def param_grid():
    # Define hyperparameter grid with proper routing using "model__" prefix
    return {
        'model__filters': [32, 48],
        'model__kernel_size': [(3, 3)],
        'model__dropout_rate': [0.3, 0.5],
        'model__learning_rate': [0.001, 0.002, 0.0001],
        'batch_size': [16, 24],  # Batch size for memory optimization
        'epochs': [18, 20, 22]
    }

def grid_search():
    grid = param_grid()    
    # Use GridSearchCV for hyperparameter tuning
    gridcv = GridSearchCV(estimator=model,
                          param_grid=grid,
                          scoring='accuracy',
                          cv=2, # 2 cross validation folds
                          n_jobs=1,
                          verbose=2, #verbose 2 to see output
                          error_score="raise",
                          pre_dispatch='2*n_jobs')
    return gridcv


# Main execution using the CNN class
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_dataset(cifar10)
    
    gridcv = grid_search()
    
    try:
        grid_result = gridcv.fit(X_train, y_train)
        # output the accuracy and params of the best model
        print(f"Best Accuracy: {grid_result.best_score_ * 100:.2f}%")
        print(f"Best Params: {grid_result.best_params_}")
    
        # train best model on test set
        best_model = grid_result.best_estimator_
        test_loss, test_accuracy = best_model.model_.evaluate(X_test, y_test, verbose=0)
        
        # output loss and accuracy of the best model's evaluation
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    except Exception as e:
        print(f"Error during GridSearchCV: {e}")
