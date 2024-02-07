# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:51:21 2128

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
segment_length = 10
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

"""
# Function to calculate relative power of a frequency band
def relative_power(signal, frequency_band, sampling_frequency):

    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1 / sampling_frequency)
    
    # Find indices corresponding to the frequency band
    band_indices = np.where((frequencies >= frequency_band[0]) & (frequencies <= frequency_band[1]))[0]
    
    # Calculate the power within the frequency band
    band_power = np.sum(np.abs(fft_result[band_indices])**2)
    
    # Calculate the total power
    total_power = np.sum(np.abs(fft_result)**2)
    
    # Calculate relative power
    relative_power = band_power / total_power

    return relative_power
"""
import numpy as np
from scipy.signal import welch

def relative_power(signal, frequency_band, sampling_frequency):
    # Compute the Power Spectral Density of the signal
    frequencies, psd = welch(signal, fs=sampling_frequency)

    # Find indices where frequencies are within the specified band
    freq_indices = np.where((frequencies >= frequency_band[0]) & (frequencies <= frequency_band[1]))[0]

    # Calculate the power within the specified frequency band
    band_power = np.sum(psd[freq_indices])

    # Calculate the total power of the signal
    total_power = np.sum(psd)

    # Calculate and return the relative power
    return band_power / total_power if total_power > 0 else 0


# Function to calculate absolute power of a frequency band
def absolute_power(signal, frequency_band, sampling_frequency):

    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1 / sampling_frequency)
    
    # Find indices corresponding to the frequency band
    band_indices = np.where((frequencies >= frequency_band[0]) & (frequencies <= frequency_band[1]))[0]
    
    # Calculate the power within the frequency band
    band_power = np.sum(np.abs(fft_result[band_indices])**2)

    return band_power

# Calculate features for each segment and organize them into X and Y vectors
X_train = []
X_test = []
Y_train = []
Y_test = []

print("feature extraction start")
for condition, subjects in segmented_dict.items():
    for subject_data in subjects:
        for channel_data in subject_data:
            for i, segment in enumerate(channel_data):
                # Features for each segment
                features = [
                    relative_power(segment, (0.5, 4), sampling_frequency),
                    absolute_power(segment, (0.5, 4), sampling_frequency),
                    relative_power(segment, (4, 8), sampling_frequency),
                    absolute_power(segment, (4, 8), sampling_frequency),
                    relative_power(segment, (8, 13), sampling_frequency),
                    absolute_power(segment, (8, 13), sampling_frequency),
                    relative_power(segment, (13, 30), sampling_frequency),
                    absolute_power(segment, (13, 30), sampling_frequency),
                    relative_power(segment, (30, 80), sampling_frequency),
                    absolute_power(segment, (30, 80), sampling_frequency),
                    relative_power(segment, (80, 250), sampling_frequency),
                    absolute_power(segment, (80, 250), sampling_frequency),
                    relative_power(segment, (250, 500), sampling_frequency),
                    absolute_power(segment, (250, 500), sampling_frequency),
                ]
                
                # Append features to X_train or X_test
                if i % 5 == 0:  # Assuming 20% (1/5) for the test set this is quite a sloppy way of doing this but will have to be changed anyway to include options like rotating the test set later
                    X_test.append(features)
                    Y_test.append(1 if condition == 'injured' else 0)
                else:
                    X_train.append(features)
                    Y_train.append(1 if condition == 'injured' else 0)

# Convert X_train, X_test, Y_train, and Y_test to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# Print the shapes of X_train, X_test, Y_train, and Y_test for verification
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of Y_train:", Y_train.shape)
print("Shape of Y_test:", Y_test.shape)

segmented_dict = []


import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Define the EarlyStopping and ModelCheckpoint callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
model_checkpoint = callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)


# Define the CNN model
model = models.Sequential()

# Add a Convolutional layer with ReLU activation and max pooling
model.add(layers.Conv1D(256, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(layers.MaxPooling1D(2))

# Add another Convolutional layer with ReLU activation and max pooling
model.add(layers.Conv1D(512, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(2))


# Flatten the output before passing to the dense layer
model.add(layers.Flatten())

# Add a dense hidden layer with ReLU activation
model.add(layers.Dense(512, activation='relu'))

# Add a dense hidden layer with ReLU activation
model.add(layers.Dense(256, activation='relu'))

# Add the output layer with a single neuron and sigmoid activation for binary classification
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape X_train and X_test to include the channel dimension
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# Train the model with callbacks
model.fit(X_train_reshaped, Y_train, epochs=100, batch_size=16, validation_data=(X_test_reshaped, Y_test),
          callbacks=[early_stopping, model_checkpoint])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_reshaped, Y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


# Save the entire model to a HDF5 file
model.save('cnn_model_example.h5')

print("Model has been saved.")

import shap # Reminder shap uses pytorch to do its thing, so if we can't use pytorch in the platform...

# Load the best model
best_model = models.load_model('best_model.h5')
# Use SHAP to explain the model predictions
explainer = shap.DeepExplainer(best_model, X_test_reshaped)

from shap import Explainer
shap_values = explainer.shap_values(X_test_reshaped)#[0:300])#new_array)#X_train_reshaped)

# Visualize Shapley values
feature_names = ['rel_delta', 'abs_delta','rel_theta', 'abs_theta' ,'rel_alpha', 'abs_alpha' ,'rel_beta', 'abs_beta' ,'rel_gamma', 'abs_gamma' ,'rel_ripple', 'abs_ripple' ,'rel_fastripple', 'abs_fastripple']
shap_values_squeezed = np.squeeze(shap_values[0])
X_test_reshaped_squeezed = np.squeeze(X_test_reshaped)
shap.summary_plot(shap_values_squeezed, X_test_reshaped_squeezed, feature_names=feature_names)


"""
#load and single sample test
testing_model = models.load_model('cnn_model_example.h5')

sample = X_test_reshaped[0]
sample = np.expand_dims(sample, axis=0)
a = testing_model.predict(sample)
print(Y_test[0], a)
"""