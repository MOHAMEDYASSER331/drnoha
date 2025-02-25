#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
def tanh(x):
    return np.tanh(x)

def neural_network(inputs, weights1, weights2, b1, b2):
    hidden_layer_input = np.dot(inputs, weights1) + b1
    hidden_layer_output = tanh(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights2) + b2
    output = tanh(output_layer_input)
    
    return output

inputs = np.array([1.0, 0.5, -0.5])

weights1 = np.random.uniform(-0.5, 0.5, (3, 4))  
weights2 = np.random.uniform(-0.5, 0.5, (4, 1))  

b1 = 0.5
b2 = 0.7

output = neural_network(inputs, weights1, weights2, b1, b2)

print("Output of the network:", output)


# In[ ]:




