# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:11:10 2025

@author: myran
"""
import numpy as np
import pandas as pd

# List of predictions with their top-ranked class and whether it matches the correct class
predictions = [

    
    {"image": "bird.jpg", "top_class": "bird", "correct": True},
    {"image": "bird2.jpg", "top_class": "bird", "correct": True},
    {"image": "bird3.jpg", "top_class": "bird", "correct": True},
    {"image": "car.jpg", "top_class": "automobile", "correct": True},
    {"image": "car2.jpg", "top_class": "automobile", "correct": True},
    {"image": "car3.jpg", "top_class": "automobile", "correct": True},
    {"image": "cat.jpg", "top_class": "dog", "correct": False},
    {"image": "cat2.jpg", "top_class": "dog", "correct": False},
    {"image": "cat3.jpg", "top_class": "horse", "correct": False},
    {"image": "deer.jpg", "top_class": "deer", "correct": True},
    {"image": "deer3.jpg", "top_class": "cat", "correct": False},
    {"image": "dog.jpg", "top_class": "dog", "correct": True},
    {"image": "dog2.jpg", "top_class": "dog", "correct": True},
    {"image": "dog3.jpg", "top_class": "bird", "correct": False},
    {"image": "frog.jpg", "top_class": "frog", "correct": True},
    {"image": "frog2.jpg", "top_class": "deer", "correct": False},
    {"image": "frog3.jpg", "top_class": "frog", "correct": True},
    {"image": "horse.jpg", "top_class": "horse", "correct": True},
    {"image": "horse2.jpg", "top_class": "deer", "correct": False},
    {"image": "horse3.jpg", "top_class": "horse", "correct": True},
    {"image": "plane.jpg", "top_class": "airplane", "correct": True},
    {"image": "plane2.jpg", "top_class": "airplane", "correct": True},
    {"image": "plane3.jpg", "top_class": "airplane", "correct": True},
    {"image": "ship.jpg", "top_class": "ship", "correct": True},
    {"image": "ship2.jpg", "top_class": "ship", "correct": True},
    {"image": "ship3.jpg", "top_class": "ship", "correct": True},
    {"image": "truck.jpg", "top_class": "truck", "correct": True},
    {"image": "truck2.jpg", "top_class": "truck", "correct": True},
    {"image": "truck3.jpg", "top_class": "airplane", "correct": False}


]

# Calculate accuracy
correct_predictions = sum(1 for p in predictions if p["correct"])
total_predictions = len(predictions)
accuracy_percentage = (correct_predictions / total_predictions) * 100

print(f"Accuracy: {accuracy_percentage}%")


def checkModelAccuracy(data):
    
    # Convert to numpy array for easier calculations
    data_array = np.array(data)
    train_accuracy = data_array[:, 1]  # Training Accuracy
    train_loss = data_array[:, 2]      # Training Loss
    val_accuracy = data_array[:, 3]    # Validation Accuracy
    val_loss = data_array[:, 4]         # Validation Loss
    
    # Calculate overall accuracy and loss percentage
    avg_train_accuracy = np.mean(train_accuracy) * 100
    avg_train_loss = np.mean(train_loss)
    avg_val_accuracy = np.mean(val_accuracy) * 100
    avg_val_loss = np.mean(val_loss)
    
    # Calculate overall accuracy and loss percentages
    avg_train_accuracy = np.mean(train_accuracy) * 100
    avg_train_loss = np.mean(train_loss)
    avg_val_accuracy = np.mean(val_accuracy) * 100
    avg_val_loss = np.mean(val_loss)
    
    print(f"Average Training Accuracy: {avg_train_accuracy:.2f}%")
    print(f"Average Training Loss: {avg_train_loss:.2f}")
    print(f"Average Validation Accuracy: {avg_val_accuracy:.2f}%")
    print(f"Average Validation Loss: {avg_val_loss:.2f}")
        

# filename
filename = 'training_history.csv'

try:
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)
    
    # Convert the DataFrame to a numpy array
    dataset = df.to_numpy()
    
except FileNotFoundError:
    print("File not found. Please ensure the CSV file is in the correct location.")
    

checkModelAccuracy(dataset)
