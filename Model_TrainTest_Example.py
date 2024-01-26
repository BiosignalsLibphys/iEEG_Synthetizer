# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, LSTM, Dense, GRU
import matplotlib.pyplot as plt
import numpy as np

from extract_eeg_data_tools.tools import *
from extract_eeg_data_tools.TimeGan_functions import *
import tensorflow as tf


# Main execution
def main():
    json_file = 'structured_eeg.json'  # Update this path
    main_folder = r"E:\Code snippets\IntraOp Data"  # Update this path

    # Reading channel classifications for tissue
    json_classifications = read_channel_classifications(json_file)

    # Extracting EEG data
    eeg_data = extract_eeg_data(main_folder, json_classifications)

    # Trimming parameters
    start_index = 100000
    length = 120000 # adjust these parameters as needed

    # Apply trimming
    trimmed_result = trim_time_series(eeg_data, start_index, length)
    print(trimmed_result)

    # Quality analysis
    quality_results = process_eeg_data_quality(trimmed_result)
    print(quality_results)

    # Remove the channels that do not have quality to be used as input to the model
    input_data = remove_bad_quality(trimmed_result, quality_results)
    print(input_data)

    # Transform the data
    good_data_list, bad_data_list = transform_data_to_list_of_arrays(input_data)
    len(good_data_list), len(bad_data_list), good_data_list[0], bad_data_list[0]

    # Decide for true data or testing performing
    signals = bad_data_list  # Decide if is to generate good or bad tissue through the input
    data_type = 'false'
    if data_type == 'false':
        signals = np.zeros_like(signals)

        # Constants
        fs = 2048  # Sampling frequency in Hz
        N = signals.shape[1]  # Number of samples

        # Time array
        t = np.arange(N) / fs

        # Generating 16 sine waves
        frequencies = np.linspace(1, 16, 16)  # Example frequencies from 1 to 16 Hz
        sine_waves = np.array([np.sin(2 * np.pi * t) for f in frequencies]) # Do not forget f
        print('Sine waves shape: ', sine_waves.shape)
        signals = sine_waves

    else:
        signals = bad_data_list  # Decide if is to generate good or bad tissue through the input


    # Parameters
    n = 10 # Correspond to seconds
    seq_len = 2048 * n  # Length of each segment (in samples)
    overlap = seq_len // 4  # 2 - 50% overlap, 4 - 25% we can change so that we can decide the


    # Segmenting the signals and constructing the 3D array
    segmented_signals = [segment_signal(signal, seq_len, overlap) for signal in signals]
    ori_time, max_seq_len = extract_time(signals, seq_len, overlap)
    n_signals = len(signals)

    # Constructing the 3D array (X tensor)
    X_tensor = np.zeros((n_signals, max_seq_len, seq_len))
    for ii, segments in enumerate(segmented_signals):
        for jj, segment in enumerate(segments):
            X_tensor[ii, jj, :] = segment

    # Print the shape of the X_tensor data to confirm
    print(X_tensor.shape)

    # Normalize signals
    normalized_X = normalize_tensor_3d(X_tensor, method='standardize')

    # Print the shape of the normalized data to confirm
    print(normalized_X.shape)

    # Define hyperparameters
    parameters = {
        'z_dim': 50,
        'num_layers': 5,
        'hidden_dim': 128*3,
        'rnn_type': 'GRU',  # Can be 'LSTM', 'GRU', or 'SimpleRNN'
        'activation_fn': 'sigmoid',  # Can be 'sigmoid' or 'relu'
        'num_epochs': 200,  # number of epochs
        'batch_size': n_signals,
        'max_seq_len': max_seq_len
    }

    # Create the TimeGan
    # Here we will start with the embedder, using the function that is stored in TimeGan_function.py

    # Shape of a single segment
    input_shape = (normalized_X.shape[1], normalized_X.shape[2])

    # Assuming we have 10 samples, each with 119 time steps, and we choose a noise dimension (z_dim)
    num_samples = normalized_X.shape[0]
    time_steps = normalized_X.shape[1]
    z_dim = parameters['z_dim']

    # Generate random noise Z
    Z = np.random.normal(size=(num_samples, time_steps, z_dim))

    # Define time information T (simple range of integers for each time step)
    T = np.arange(time_steps)

    # Here we build the models fro the TimeGan
    # Create models for the embedder and recovery part
    embedder_model = create_embedder(input_shape, normalized_X.shape[1], parameters)

    # Assuming the latent space has the same time steps and hidden dimension as the embedder
    latent_shape = (normalized_X.shape[1], parameters['hidden_dim'])

    # Create the recovery model
    # recovery_model = create_recovery(latent_shape, normalized_X.shape[1], parameters)
    # Recreate the recovery model
    # Now let's recreate the recovery model with the correct feature size
    feature_size = normalized_X.shape[2]  # This should be the number of features in your original data X
    recovery_model = create_recovery((normalized_X.shape[1], parameters['hidden_dim']), normalized_X.shape[1],
                                     parameters,
                                     feature_size)
    # Model summary
    embedder_model.summary()
    recovery_model.summary()

    # Create the models for the generative part
    generator_model = create_generator((normalized_X.shape[1], parameters['z_dim']), normalized_X.shape[1], parameters)
    supervisor_model = create_supervisor((normalized_X.shape[1], parameters['hidden_dim']), normalized_X.shape[1],
                                         parameters)

    # Model summaries
    generator_model.summary()
    supervisor_model.summary()

    # Discriminator part:

    # Create the discriminator model
    discriminator_model = create_discriminator((normalized_X.shape[1], parameters['hidden_dim']), normalized_X.shape[1],
                                               parameters)

    # Model summary
    discriminator_model.summary()

    # Run all these thinks

    X = normalized_X
    no, seq_len, dim = np.asarray(X).shape

    # Embedder: Transforming real data to latent space
    H = embedder_model([X, T])

    # Recovery: Transforming latent representation back to original space
    X_tilde = recovery_model([H, T])

    # Generator: Creating latent representations from noise
    E_hat = generator_model([Z, T])

    # Supervisor: Refining the generated embeddings
    H_hat = supervisor_model([E_hat, T])

    # Also running supervisor with real embeddings
    H_hat_supervise = supervisor_model([H, T])

    # Synthetic data
    X_hat = recovery_model([H_hat, T])

    # Discriminator part: H_hat is the latent representation from the generator,
    # H is from the real data (embedder), and E_hat is the output of the generator.
    # The discriminator tries to classify each of these as real or synthetic.
    Y_fake = discriminator_model([H_hat, T])
    Y_real = discriminator_model([H, T])
    Y_fake_e = discriminator_model([E_hat, T])

    # Now it is time to calculate the losses of the models

    gamma = 1.0  # We can Adjust gamma as needed or check literature to see what makes sense

    # Discriminator Loss
    D_loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    # Generator Loss
    # 1. Adversarial loss
    G_loss_U = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(Y_fake_e), Y_fake_e)

    # 2. Supervised loss
    G_loss_S = tf.keras.losses.MeanSquaredError()(H[:, 1:, :], H_hat_supervise[:, :-1, :])

    # 3. Two Moments
    X_hat_float32 = tf.cast(X_hat, tf.float32)
    X_float32 = tf.cast(X, tf.float32)

    G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.math.reduce_variance(X_hat_float32, axis=0) + 1e-6) -
                                      tf.sqrt(tf.math.reduce_variance(X_float32, axis=0) + 1e-6)))
    G_loss_V2 = tf.reduce_mean(tf.abs(tf.math.reduce_mean(X_hat_float32, axis=0) -
                                      tf.math.reduce_mean(X_float32, axis=0)))

    G_loss_V = G_loss_V1 + G_loss_V2

    # 4. Summation
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V

    # Embedder Loss
    E_loss_T0 = tf.keras.losses.MeanSquaredError()(X, X_tilde)
    E_loss = 10 * tf.sqrt(E_loss_T0) + 0.1 * G_loss_S

    # Finally, let's train the model and get our syntehtic data

    # Define optimizers for each model component
    optimizer_E = tf.keras.optimizers.Adam()
    optimizer_D = tf.keras.optimizers.Adam()
    optimizer_G = tf.keras.optimizers.Adam()

    # Training loop
    for epoch in range(parameters['num_epochs']):
        print('Epoch: ', epoch)
        for X_batch, T_batch in batch_generator(X, ori_time, parameters['batch_size']):

            # Embedding network training
            with tf.GradientTape() as tape:
                H = embedder_model([X_batch, T_batch])  # Forward pass through the embedder
                X_tilde = recovery_model([H, T_batch])  # Forward pass through recovery
                E_loss0 = 10 * tf.sqrt(tf.keras.losses.MeanSquaredError()(X_batch, X_tilde))
                print('Embedder Loss ----------', E_loss)
                gradients = tape.gradient(E_loss0,
                                          embedder_model.trainable_variables + recovery_model.trainable_variables)
                optimizer_E.apply_gradients(
                    zip(gradients, embedder_model.trainable_variables + recovery_model.trainable_variables))

            # Supervised training for generator and supervisor
            with tf.GradientTape() as tape:
                # Forward pass through generator and supervisor
                Z_batch = random_generator(parameters['batch_size'], parameters['z_dim'], T_batch,
                                           parameters['max_seq_len'])
                E_hat = generator_model([Z_batch, T_batch])
                H_hat = supervisor_model([E_hat, T_batch])

                # Calculate supervised loss (adjust according to loss function)
                G_loss_S = tf.keras.losses.MeanSquaredError()(H[:, 1:, :], H_hat[:, :-1, :])
                print('Generator Loss ----------', G_loss_S)

                gradients = tape.gradient(G_loss_S,
                                          generator_model.trainable_variables + supervisor_model.trainable_variables)
                optimizer_G.apply_gradients(
                    zip(gradients, generator_model.trainable_variables + supervisor_model.trainable_variables))

            # Joint training
            for _ in range(2):  # Generator training
                with tf.GradientTape() as tape:
                    # Forward pass through generator and supervisor
                    Z_batch = random_generator(parameters['batch_size'], parameters['z_dim'], T_batch,
                                               parameters['max_seq_len'])
                    E_hat = generator_model([Z_batch, T_batch])
                    H_hat = supervisor_model([E_hat, T_batch])

                    # Forward pass through discriminator (for fake data)
                    Y_fake = discriminator_model([H_hat, T_batch])

                    # Calculate generator loss (adjust according to your loss function)
                    G_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(Y_fake), Y_fake)
                    print('Generator Loss ----------', G_loss)

                    gradients = tape.gradient(G_loss,
                                              generator_model.trainable_variables + supervisor_model.trainable_variables)
                    optimizer_G.apply_gradients(
                        zip(gradients, generator_model.trainable_variables + supervisor_model.trainable_variables))

            with tf.GradientTape() as tape:  # Discriminator training
                # Forward pass through discriminator for real and fake data
                H = embedder_model([X_batch, T_batch])
                Y_real = discriminator_model([H, T_batch])
                Y_fake = discriminator_model([H_hat, T_batch])

                # Calculate discriminator loss (adjust according to the loss function)
                D_loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(Y_real), Y_real)
                D_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(Y_fake), Y_fake)
                D_loss = D_loss_real + D_loss_fake
                print('Discriminator Loss ---------- ', D_loss)

                gradients = tape.gradient(D_loss, discriminator_model.trainable_variables)
                optimizer_D.apply_gradients(zip(gradients, discriminator_model.trainable_variables))

    # Synthetic data generation
    # no = 10
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)

    generated_data = []
    for ii in range(no):
        generated_sample = recovery_model(
            [supervisor_model([generator_model([Z_mb[ii:ii + 1], T_batch]), T_batch]), T_batch])
        generated_data.append(generated_sample[0, :ori_time[ii], :])

    # # Renormalization
    # max_val = np.max(signals[10])
    # min_val = np.min(signals[10])
    # generated_data = np.array(generated_data) * max_val
    # generated_data += min_val  # Here we are trying to recover the true values of the EEG

    _, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(normalized_X[0][2])
    ax[1].plot(generated_data[0][0])
    ax[2].plot(generated_data[5][0])

    # TODO: We have to add code here to handle the extracted data

    # Next will be the concatenate or flatten to have the length defined by the user

if __name__ == "__main__":
    main()

