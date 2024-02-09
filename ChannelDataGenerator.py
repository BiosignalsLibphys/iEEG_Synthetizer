# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:58:29 2024

@author: User
"""
import numpy as np
import json
from ReadSort import structure_data
import numpy as np


simplified_dict = structure_data('classifier')


# Function to divide a signal into segments
def divide_into_segments(signal, segment_length, sampling_frequency):
    segment_size = int(segment_length * sampling_frequency)
    # Check if the signal is shorter than the segment size
    if len(signal) < segment_size:
        # If the signal is shorter, just return the original signal as a single segment
        return [signal]
    num_segments = len(signal) // segment_size
    segments = np.array_split(signal[:num_segments * segment_size], num_segments)
    return segments


# Define the segment length in seconds
segment_length = 10#######################
sampling_frequency = 2048

# Create a new dictionary to store the segmented data
segmented_dict = {'healthy': [], 'injured': []}

# Iterate through the simplified_dict and divide each channel's data into segments
for condition, subjects in simplified_dict.items():
    segmented_subjects = []
    for subject_data in subjects:
        segmented_channels = []
        for channel_data in subject_data:
            # Flatten the channel_data to get the 1D signal
            flat_signal = np.concatenate(channel_data)
            # Divide the channel signal into segments
            segments = divide_into_segments(flat_signal, segment_length, sampling_frequency)
            segmented_channels.append(segments)
        
        segmented_subjects.append(segmented_channels)
    
    segmented_dict[condition] = segmented_subjects

simplified_dict = [] #space saver


#THIS DROPS SEGMENTS THAT AREN'T THE SAME SIZE - ONLY NECESSARY WHEN SUBDIVIDING THE SIGNAL INTO SHORTER SEGMENTS################
first_element_size = len(segmented_dict["healthy"][0][0][0])#This assumes that the first of all segments is the correct length, in the future if we use entire eeg channels as input this won't be the case

# Iterate through the dictionary and remove elements with different sizes
for X in range(len(segmented_dict["healthy"])):
    for Y in range(len(segmented_dict["healthy"][X])):
        for Z in range(len(segmented_dict["healthy"][X][Y])):
            current_element = segmented_dict["healthy"][X][Y][Z]
            # Check if the current element has a different size
            if len(current_element) != first_element_size:
                # Drop the element by setting it to None
                segmented_dict["healthy"][X][Y][Z] = None

# Remove the elements set to None
segmented_dict["healthy"] = [
    [
        [
            element for element in Z_list if element is not None
        ] for Z_list in Y_list
    ] for Y_list in segmented_dict["healthy"]
]

# Remove empty lists
segmented_dict["healthy"] = [
    Y_list for Y_list in segmented_dict["healthy"] if any(Y_list)
]
#THIS DROPS SEGMENTS THAT AREN'T THE SAME SIZE - ONLY NECESSARY WHEN SUBDIVIDING THE SIGNAL INTO SHORTER SEGMENTS################



# Iterate through X, Y, and Z indices and normalize them
normalized_arrays =[]
for X in range(len(segmented_dict["healthy"])):
    for Y in range(len(segmented_dict["healthy"][X])):
        for Z in range(int(len(segmented_dict["healthy"][X][Y])/5)):##Reducing the number of files looked at here for testing purposes
            # Grab the original array
            original_array = segmented_dict["healthy"][X][Y][Z]
            
            # Add 0.2 to the original array
            original_array += 0.2
            
            # Calculate the normalized array
            normalized_array = (original_array - np.min(original_array)) / (np.max(original_array) - np.min(original_array))
            
            # Append the normalized array to the list
            normalized_arrays.append(normalized_array)
            #normalized_arrays.extend([normalized_array])

# Convert the list to a NumPy array
normalized_arrays = np.array(normalized_arrays)



segmented_dict = normalized_arrays
normalized_arrays = []
print(segmented_dict.shape)
print(type(segmented_dict))
UsedSize = len(segmented_dict[0])#This is the length of the signals that will be used for training, so for example 10 seconds or 100 seconds, or the length of each channel 


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the VAE class
class VAE(tf.keras.Model):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc21 = tf.keras.layers.Dense(latent_size) # mean
        self.fc22 = tf.keras.layers.Dense(latent_size) # log variance
        # Decoder
        self.fc3 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc4 = tf.keras.layers.Dense(input_size, activation='sigmoid')

    def encode(self, x):
        # Encode the input x into the latent space
        h1 = self.fc1(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # Reparameterize the latent space using the reparameterization trick
        std = tf.exp(0.5*logvar)
        eps = tf.random.normal(shape=std.shape)
        return mu + eps*std
       #the addition of temperature to this only makes sen if we are using a softmax activation function

    def decode(self, z):
        # Decode the latent vector z into the output x
        h3 = self.fc3(z)
        return self.fc4(h3)

    def call(self, x):
        # Forward pass of the VAE
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Define the loss function
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss: binary cross entropy
    BCE = tf.keras.losses.binary_crossentropy(x, recon_x, from_logits=False)
    BCE = tf.reduce_sum(BCE)
    # Regularization loss: KL divergence
    KLD = -0.5 * tf.reduce_sum(1 + logvar - mu**2 - tf.exp(logvar))
    return BCE + KLD

"""
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss: mean squared error
    MSE = tf.keras.losses.mean_squared_error(x, recon_x)
    MSE = tf.reduce_sum(MSE)
    # Regularization loss: KL divergence
    KLD = -0.5 * tf.reduce_sum(1 + logvar - mu**2 - tf.exp(logvar))
    return MSE + KLD
"""

# Define the hyperparameters
input_size = UsedSize # number of points in a sine wave
hidden_size = 200 # size of the hidden layer
latent_size = UsedSize # size of the latent space
batch_size = 1#00 # size of the mini-batch, lowering this leads to an increase in performance/overfit, but slows down training dramatically (almost directly proportionally)
num_epochs = 1250 # number of epochs
learning_rate = 0.001 # learning rate

# Create the VAE model
model = VAE(input_size, hidden_size, latent_size)

# Create the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)




#converts the segments of data to 
#dataset = tf.convert_to_tensor(np.repeat(segmented_dict, 50, axis= 0))
dataset = tf.convert_to_tensor(segmented_dict)
print(type(dataset))




# Create an empty list to store the loss values
loss_history = []

# Define some variables to store the best loss and the patience
best_loss = float('inf')
patience = int(num_epochs/1)
no_improvement = 0

# Train the model
for epoch in range(num_epochs):
    # Shuffle the dataset
    permutation = tf.random.shuffle(tf.range(dataset.shape[0]))
    dataset = tf.gather(dataset, permutation)
    # Loop over the mini-batches
    for i in range(0, dataset.shape[0], batch_size):
        # Get the current batch
        batch = dataset[i:i+batch_size] ## if batch_size = dataset.shape[0] this means that every training epoch is with a set of all of the signals inside the dataset
        # Compute the gradients
        with tf.GradientTape() as tape:
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)
        gradients = tape.gradient(loss, model.trainable_variables)
        # Update the parameters
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
    # Print the loss
    print(f'Epoch {epoch+1}, Loss: {loss.numpy():.4f}')
    loss_history.append(loss.numpy())
    if loss.numpy() < best_loss and epoch >= 0:## change the epoch term as necessary, this is merely used as a time saving modification
        # Save the best loss and the model's weights
        best_loss = loss.numpy()
        #model.save_weights('best_weights.h5')
        best_weights = model.get_weights()
        print("new weights")
        # Reset the no improvement counter
        no_improvement = 0
    else:
        # Increment the no improvement counter
        no_improvement += 1
        # Check if the no improvement counter exceeds the patience
        if no_improvement >= patience:
            # Stop the training
            print(f'Training stopped after {epoch+1} epochs. No improvement for {patience} epochs.')
            break


if best_weights is not None:
    model.set_weights(best_weights)
    print("best generator recovered successfully")


# Generate a sine wave from the model
output, _, _ = model(tf.random.normal(shape=(1, latent_size)))
output = output.numpy().reshape(-1)

# Plot the original and generated sine waves
#plt.plot(x, y, label='Original')
plt.figure(70)
plt.plot(segmented_dict[0], label='Original')
plt.plot(output, label='Generated')
plt.title('EEG Generation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

plt.figure(80)
plt.plot(segmented_dict[1], label='Original')
plt.plot(output, label='Generated')
plt.title('EEG Generation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

plt.figure(41)
# Plot the loss history
plt.plot(range(1, epoch+2), loss_history) 
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

import pandas as pd

# Define the smoothing function
def smooth(data, window_size):
    # Convert the data to a pandas series
    data = pd.Series(data)
    # Apply the rolling method with the SMA function
    smoothed_data = data.rolling(window_size, min_periods=1).mean()
    # Return the smoothed data as a numpy array
    return smoothed_data.to_numpy()

# Apply the smoothing function to the loss history
window_size = 25 # you can change this value to adjust the smoothness
smoothed_loss = smooth(loss_history, window_size)
plt.figure(30)
# Plot the original and smoothed loss history
#plt.plot(range(1, num_epochs+1), loss_history, label='Original')
plt.plot(range(1, epoch+2), smoothed_loss, label='Smoothed') 
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""
# Save the model 
model.save('VAEmodel.h5')
"""



#############################TEST FOR MODEL VARIABILITY
# Generate 100 random samples from the latent space
random_latent_samples = tf.random.normal(shape=(100, latent_size))

# Generate signals using the model
outputs, _, _ = model(random_latent_samples)
generated_signals = outputs.numpy()

import matplotlib.pyplot as plt
import numpy as np



# Choose 10 random indices from the generated signals
indices_to_plot = np.random.choice(len(generated_signals), size=10, replace=False)

# Create a common x-axis (assuming all signals have the same length)
x_axis = np.arange(len(generated_signals[0]))

# Plot the selected signals on separate subplots with shared x-axis
plt.figure(figsize=(12, 8))
for i, index in enumerate(indices_to_plot, 1):
    plt.subplot(5, 2, i, sharex=plt.gca())
    plt.plot(x_axis, generated_signals[index])
    plt.title(f'Signal {index + 1}')

plt.tight_layout()
plt.show()




