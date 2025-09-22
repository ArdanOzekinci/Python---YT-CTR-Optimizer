# Python---YT-CTR-Optimizer
Designing an AI-powered YouTube title and thumbnail analyzing software that predicts click-through rate (CTR) using NLP and computer vision. It will provide an analysis of similar youtube content and based on that data, it will provide actionable recommendations to boost engagement, powered by user analytics and continuous feedback learning. CURRENTLY IN DEVELOPMENT

# Roadmap
- [x] CTR prediction prototype (basic neural net)
- [ ] Expand on CTR prediction model with deeper neural networks and additional input features
- [ ] Thumbnail CV model integration
- [ ] YouTube Data API feedback loop
- [ ] Real-time dashboard
- [ ] Beta release (July 2025)

To demonstrate the CTR classifying logic found whithin this project here is a small sample:

# Modules
import os
import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Load Data
df = pd.read_csv("click_data.csv")

# Preprocessing
scaler = MinMaxScaler()
inputs = scaler.fit_transform(df[["TitleLength", "NumWordsInThumbnail", "Brightness"]])
labels = df["Clicked"].values.reshape(-1, 1)

# Activation function
def sigmoid(x):

    return 1 / (1 + np.exp(-x))

# Initialize weights and biases with random values
np.random.seed(42)
weights_hidden = np.random.randn(3, 4) * 0.1
bias_hidden = np.random.randn(1, 4) * 0.1
weights_output = np.random.randn(4, 1) * 0.1
bias_output = np.random.randn(1, 1) * 0.1

# Forward pass
def forwardpass(x):

    weighted_sum_hidden = np.dot(x, weights_hidden) + bias_hidden
    hidden = sigmoid(weighted_sum_hidden)
    
    weighted_sum_output = np.dot(hidden, weights_output) + bias_output
    output = sigmoid(weighted_sum_output)  
    return output, hidden

# Define the loss function
def binary_cross_entropy(y_true, y_pred):

    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Training parameters
learning_rate = 0.1
epochs = 1000

# Training loop
for epoch in range(epochs):

    prediction, hidden = forwardpass(inputs)
    loss = binary_cross_entropy(labels, prediction)
    
    # Backpropagation
    error_output = prediction - labels
    gradient_weights_output = np.dot(hidden.T, error_output)
    gradient_bias_output = np.sum(error_output, axis=0, keepdims=True)
    error_hidden = np.dot(error_output, weights_output.T) * hidden * (1 - hidden)
    gradient_weights_hidden = np.dot(inputs.T, error_hidden)
    gradient_bias_hidden = np.sum(error_hidden, axis=0, keepdims=True)
    
    # Update weights and biases
    weights_output -= learning_rate * gradient_weights_output
    bias_output -= learning_rate * gradient_bias_output
    weights_hidden -= learning_rate * gradient_weights_hidden
    bias_hidden -= learning_rate * gradient_bias_hidden
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final prediction
prediction, hidden = forwardpass(inputs)
print("\nFinal Prediction:\n", prediction.round())
print("\nFinal Labels:\n", labels)

# Final accuracy
accuracy = np.mean(prediction.round() == labels)
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")

DISCLAIMER:
This is just a small experimental sample and not representative of the full system’s complexity or final design!
