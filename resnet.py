import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib as plt

def load_and_preprocess_data(size=(32, 32)):
    # Load dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Normalise pixels to range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    
    return (x_train, y_train), (x_test, y_test)

def res_block(x, filters, stride=1):
    shortcut = x
    
    # First convolutional layer
    x = tf.keras.layers.Conv2D(filters // 4, kernel_size=1, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Second convolutional layer
    x = tf.keras.layers.Conv2D(filters // 4, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Third convolutional layer to match dimensions with shortcut
    x = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # skip connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    
    return x


# Define the ResNet Model
def build_resnet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    # Initial Conv2D Layer 
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Residual blocks to reduce spatial dimensions
    x = res_block(x, filters=64)
    x = res_block(x, filters=128, stride=2)
    x = res_block(x, filters=256, stride=2)
    
    # Global Average Pooling and Fully Connected Layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # output layer uses a softmax activation
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    
    return model

def plot_metrics(trained_model):
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1) #create smaller subplot for better viewing side by side
    plt.plot(trained_model.history['loss'], label='Train Loss')
    plt.plot(trained_model.history['val_loss'], label='Validation Loss')
    plt.title("ResNet Model Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend(['Train', 'Val'])
    
    plt.subplot(1, 2, 2) #create smaller subplot for better viewing side by side
    plt.plot(trained_model.history['accuracy'])
    plt.plot(trained_model.history['val_accuracy'])
    plt.title("ResNet Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'])
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()
        
(x_train, y_train), (x_test, y_test) = load_and_preprocess_data(size=(32, 32))
# Build the model
model = build_resnet(input_shape=(32, 32, 3), num_classes=10)
# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), #adam optimser and 0.001 learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model using default settings - 20 epochs and 128 batch size
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=20,
    validation_data=(x_test, y_test),
    verbose=1
)

model.summary() # return a summary of the model
# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
#save the model
model.save("resnet2.keras")
plot_metrics(history)