#!/usr/bin/env python
# coding: utf-8

# In[15]:


#Abhishek R S-EEE20005
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# import the TensorFlow library, the Keras module from TensorFlow, the NumPy library, and the pyplot module from the matplotlib library
model=tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])#sequential model is defined using Keras. It consists of a single dense (fully connected) layer with one unit/neuron.
# The input shape is specified as [1], indicating that the model expects one input feature.

for layer in model.layers:
    weights = layer.get_weights() # This code retrieves the weights of each layer in the model. 
    # In this case, it retrieves the weights of the single dense layer.


model.compile(optimizer='sgd', loss='mse') # The model is compiled using the stochastic gradient descent (SGD) optimizer and the mean squared error (MSE) loss function.
# The optimizer determines how the model is updated based on the calculated loss.

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32) # create numpy arrays that contains input values
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float32) # create numpy arrays that contains corresponding target/output values.

plt.scatter(xs, ys) # creates a scatter plot

model.fit(xs, ys, epochs=500) # train the model such that it takes the input data (xs) and the target/output data (ys) as input 
# performs training for a specified number of epochs (500).

print(model.predict([10.0])) # predict the output for a new input value of 10.0 using the trained model and prints the result.


# In[ ]:




