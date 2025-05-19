# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 14:31:23 2025

@author: myran
"""
#""
# lets test the model out with a picture or two
import matplotlib.pyplot as plt
import numpy as np
from oopCNN import CNN

model_name = "models/cnn_model_withaugmentation.keras"

# create an instance of the CNN class to load the model
cnn = CNN(22, 24)
# load the previosuly trained model
cnn_model = cnn.load_model(model_name)
# Generate a summary to validate model
cnn_model.summary()

# Data
indices = np.arange(10)  # 0 to 9
best_percent = np.array([77.41, 76.94, 82.70, 75.86, 82.12, 48.40, 45.24, 85.34, 73.47, 73.03])
overall_percent = np.array([72.41, 75.86, 76.45, 72.41, 78.31, 44.82, 36.23, 79.31, 59.62, 51.72])


# Plotting
plt.figure(figsize=(10, 6))
plt.bar(indices, best_percent, label='Best %', color='blue')
plt.plot(indices, overall_percent, marker='o', linestyle='-', label='Overall %', color='orange')

plt.title('Best % and Overall % for CIFAR-10 CNN Models')
plt.xlabel('Model Index')
plt.ylabel('Percentage')
plt.xticks(indices)  # Set xticks to match indices
plt.legend()

plt.tight_layout()  # Ensure labels fit within plot area
plt.show()
