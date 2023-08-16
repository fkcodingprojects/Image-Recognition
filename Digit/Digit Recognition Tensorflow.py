#!/usr/bin/env python
# coding: utf-8

# ## Load and Preprocess Data 

# In[60]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load data from CSV files
data_df = pd.read_csv('train.csv')

# Split data into features (X) and labels (y)
X = data_df.drop('label', axis=1).values
y = data_df['label'].values

# Split data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## Create a Custom Neural Network Model 

# In[71]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a simple neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(105, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ## Train the Model 

# In[80]:


batch_size = 64
num_epochs = 20

# Train the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_data=(X_test, y_test))


# ## Plot Training Progress

# In[81]:


import matplotlib.pyplot as plt

# Plot the training loss and accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()


# In[82]:


plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# ## Evaluate the Model 

# In[84]:


# Evaluate the model
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Accuracy on test data: {accuracy * 100:.2f}%")

