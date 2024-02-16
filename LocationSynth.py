import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import tensorflow as tf
# Enable eager execution
tf.config.run_functions_eagerly(True)
tf.config.experimental_run_functions_eagerly(True)

from ReadSort import structure_data
simplified_dict = structure_data('loc_gen')
# Assume max_length is the fixed length of each sequence in simplified_dict
def max_signal_length(simplified_dict):
    max_length = 0

    for key, signal in simplified_dict.items():
        signal_length = len(signal)
        if signal_length > max_length:
            max_length = signal_length

    return max_length

max_length = max_signal_length(simplified_dict)

# Create a new dictionary with padded values
padded_dict = {}
for key, value in simplified_dict.items():
    # Copy the value list
    padded_value = value[:]
    # Calculate the number of -1s to add
    padding = max_length - len(value)
    # Extend the value list with -1s
    padded_value.extend([-1] * padding)
    padded_dict[key] = padded_value

simplified_dict_original= simplified_dict
simplified_dict = padded_dict


# Convert simplified_dict to a numpy array
data = np.array(list(simplified_dict.values()))

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras import layers, models
# Define the Generator model
def build_generator(latent_dim):
    
    model = Sequential()
    model.add(Dense(64, input_dim=latent_dim))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(Dense(64))
    #model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(max_length, activation='tanh'))
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(50,input_dim=latent_dim))
    
    # Dense layers with batch normalization
    model.add(Dense(128))
    model.add(layers.BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(256))
    model.add(layers.BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # Output layer with rounding
    model.add(layers.Dense(max_length, activation='sigmoid'))
    """
    
    return model

# Define the Discriminator model
def build_discriminator(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,)))
    #model.add(LeakyReLU(alpha=0.2))
    #model.add(Dense(128))
    #model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Compile the models
latent_dim = max_length  # You can adjust the latent dimension as needed
generator = build_generator(latent_dim)
discriminator = build_discriminator(max_length)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Training the GAN
epochs = 1000  # You can adjust the number of epochs as needed
batch_size = len(data)

best_g_loss = float('inf')  # Initialize with a large value
best_weights = None

# Generate fake samples with class labels
def generate_fake_samples(generator, latent_dim, num_samples, threshold=0.5):
    latent_points = generate_latent_points(latent_dim, num_samples)
    X = generator.predict(latent_points)
    
    # Apply threshold to get binary values
    X_binary = (X > threshold).astype(np.int)
    
    y = np.zeros((num_samples, 1))
    return X_binary, y
# Generate latent points as input for the generator
def generate_latent_points(latent_dim, num_samples):
    return np.random.randn(num_samples, latent_dim)

val_g_loss_vector = []
for epoch in range(epochs):
    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
    generated_data = generator.predict(noise)

    real_labels = np.ones((batch_size, 1))  # Use real labels for the discriminator
    #fake_labels = np.zeros((batch_size, 1))
    
    d_loss_real = discriminator.train_on_batch(data, real_labels)
    d_loss_fake = discriminator.train_on_batch(generated_data, 1 - real_labels)  # Use inverted labels for generated data

    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
    valid_labels = np.ones((batch_size, 1))

    g_loss = gan.train_on_batch(noise, valid_labels)

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
    
    
    # Validation loss calculation (using a small validation set)
    val_fake_X, val_fake_y = generate_fake_samples(generator, latent_dim, batch_size)
    val_g_loss = gan.evaluate(generate_latent_points(latent_dim, batch_size), np.ones((batch_size, 1)), verbose=0)
    val_g_loss_vector.append(val_g_loss)
    """
    # Print validation loss with limited decimal places
    print(f"Validation G loss: {val_g_loss:.2f}")
    if val_g_loss <= best_g_loss:
        best_g_loss = val_g_loss
        death_counter = 0
        best_weights = generator.get_weights()
    elif death_counter > 100:
        print("\n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n ")
        break
    else: death_counter += 1
    """
"""
# Set the best weights to the generator
if best_weights is not None:
    generator.set_weights(best_weights)
    print("best generator recovered successfully")
"""
# Generate new sequences
num_generated_sequences = 10
generated_sequences = []
for _ in range(num_generated_sequences):
    noise = np.random.normal(0, 1, size=(1, latent_dim))
    generated_sequence = generator.predict(noise)[0]
    generated_sequences.append(np.round(generated_sequence).astype(int).tolist())

# Display the generated sequences
print("\nGenerated Sequences:")
for i, sequence in enumerate(generated_sequences):
    print(f"Sequence {i}: {sequence}")



import numpy as np
import matplotlib.pyplot as plt

def calculate_similarity(real_sample, synthetic_samples):
    """
    Calculate the similarity scores between a real sample and a list of synthetic samples.
    This is just a very simple similarity measure used for testing purposes,
    in the future more and higher complexity metrics will be used here.
    Parameters:
    - real_sample: A single real data sample.
    - synthetic_samples: List of synthetic samples.

    Returns:
    - similarity_scores: List of similarity scores for each synthetic sample.
    """
    similarity_scores = [np.sum(np.abs(np.subtract(np.abs(synthetic_sample), np.abs(real_sample))))
                         for synthetic_sample in synthetic_samples]
    return similarity_scores

# Generate 1000 synthetic samples
num_synthetic_samples = 1000
synthetic_samples = []
for _ in range(num_synthetic_samples):
    noise = np.random.normal(0, 1, size=(1, latent_dim))
    generated_sequence = generator.predict(noise)[0]
    synthetic_samples.append(np.round(generated_sequence).astype(int).tolist())

# Compare real data samples with synthetic samples sequentially and select the closest ones
selected_synthetic_samples = []

for real_sample in simplified_dict.values():
    similarity_scores = calculate_similarity(real_sample, synthetic_samples)
    closest_index = np.argmin(similarity_scores)
    selected_synthetic_samples.append(synthetic_samples[closest_index])

# Display the selected synthetic samples
print("Selected Synthetic Samples:")
for i, synthetic_sample in enumerate(selected_synthetic_samples):
    print(f"Sample {i + 1}: {synthetic_sample}")

# Plot real data and selected synthetic samples side by side
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Real Data")
plt.imshow(np.array([real_data_sample for real_data_sample in simplified_dict.values()]),
           cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.xlabel("Channels")
plt.ylabel("Real Signals")
plt.colorbar(ticks=[-1, 0, 1], label='Signal Value').set_ticklabels(['Missing', 'Healthy', 'Injured'])

plt.subplot(1, 2, 2)
plt.title("Selected Synthetic Samples")
plt.imshow(np.array(selected_synthetic_samples), cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.xlabel("Channels")
plt.ylabel("Synthetic Signals")
plt.colorbar(ticks=[-1, 0, 1], label='Signal Value').set_ticklabels(['Missing', 'Healthy', 'Injured'])

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
plt.figure(5)
def plot_gan_losses(g_losses):
    plt.plot(g_losses, label='Generator Loss')
    plt.title('GAN Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Generator Loss')
    plt.legend()
    plt.show()


plot_gan_losses(val_g_loss_vector)


"""
# Create input and target data for the model
X, y = [], []
for key, sequence in simplified_dict.items():
    for i in range(0, len(sequence)):# - max_length):
        X.append(sequence[0:i])
        y.append(sequence[i+max_length])

X = np.array(X)
y = np.array(y)

# Define the model
model = Sequential()
model.add(LSTM(50, input_shape=(max_length, 1)))
model.add(Dense(1, activation='linear'))  # Assuming output range is -1, 0, 1

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', run_eagerly=True)
# Set run_eagerly after model compilation
model.run_eagerly = True

# Train the model
model.fit(X, y, epochs=50, batch_size=1)
# Reshape X for LSTM input (samples, time steps, features)
#X = np.reshape(X, (X.shape[0], max_length, 1))

# Train the model
#model.fit(X, y, epochs=50, batch_size=32)

# Generate new sequences
num_sequences_to_generate = 5
for _ in range(num_sequences_to_generate):
    # Randomly choose a starting point from existing data
    start_index = np.random.randint(0, len(X)-1)
    seed_sequence = X[start_index]

    generated_sequence = []

    # Generate a new sequence based on the learned model
    for _ in range(max_length):
        predicted_value = model.predict(np.reshape(seed_sequence, (1, max_length, 1)))[0][0]
        generated_sequence.append(np.round(predicted_value))

        # Update seed sequence for the next iteration
        seed_sequence = np.append(seed_sequence[1:], [predicted_value])

    print("Generated Sequence:", generated_sequence)
"""