import numpy as np
import json
from ReadSort import structure_data
import numpy as np
from scipy.linalg import sqrtm

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

# Load the entire saved model
saved_model = load_model('cnn_model_example.h5')

model_without_last_layer = Model(inputs=saved_model.input,
                                 outputs=saved_model.layers[-2].output)



######################################importing data###############################
synthetic_data_tester = True
if synthetic_data_tester:
    import pickle
    # Load the data from the file
    with open('healthy_synthetic_data.pkl', 'rb') as f:
        healthy_data = pickle.load(f)
    
    healthy_data = healthy_data.reshape(1000, -1)

    # Load the data from the file
    with open('injured_synthetic_data.pkl', 'rb') as f:
        injured_data = pickle.load(f)

    injured_data = injured_data.reshape(1000, -1)
    
    

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
segment_length = 15
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

###########################################calculating the features for each segment
# Calculate features for each segment and organize them into X and Y vectors
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


real_X_healthy = []
real_X_injured = []



testing_break = 0 ######{#´#´#´#´#´#´#{#´#´#´#´#{#{#´#´#´##´#´#{#{##{}}}}}}}
print("real feature extraction start")
for condition, subjects in segmented_dict.items():
    for subject_data in subjects:
        for channel_data in subject_data:
            for i, segment in enumerate(channel_data):
                # Features for each segment
                original_array = segment
                original_array += 1#min(segmented_dict["healthy"][X][Y][Z])
                
                # Calculate the normalized array
                segment =  (original_array - np.min(original_array)) / (np.max(original_array) - np.min(original_array))

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
                if condition == 'injured':
                    real_X_injured.append(features)
                else:
                    real_X_healthy.append(features)
                testing_break += 1
                if testing_break == 3000:
                    break
                
                

# Convert X_train, X_test, Y_train, and Y_test to numpy arrays
real_X_healthy= np.array(real_X_healthy)
real_X_injured= np.array(real_X_injured)


synth_X_healthy = []
synth_X_injured = []

print("synth feature extraction start")

for segment in healthy_data:
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
                
                synth_X_healthy.append(features)

                
for segment in injured_data:
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
                
                synth_X_injured.append(features)


# Convert X_train, X_test, Y_train, and Y_test to numpy arrays
synth_X_healthy = np.array(synth_X_healthy)
synth_X_injured= np.array(synth_X_injured)
########################################################




def extract_features_in_batches(model, data, batch_size=1):
    features = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_features = model.predict(batch)
        features.append(batch_features)
    return np.vstack(features)



def calculate_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Calculate the Fréchet Distance between two distributions."""
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

###############################################HEALTHY


##################FID FOR HEALTHY DATA
real_features_healthy = extract_features_in_batches(model_without_last_layer, real_X_healthy)
generated_features_healthy = extract_features_in_batches(model_without_last_layer, synth_X_healthy)


# Calculate statistics
mu_real_healthy, sigma_real_healthy = calculate_statistics(real_features_healthy)
mu_gen_healthy, sigma_gen_healthy = calculate_statistics(generated_features_healthy)

# Compute FID
fid_score_healthy = calculate_fid(mu_real_healthy, sigma_real_healthy, mu_gen_healthy, sigma_gen_healthy)
print(f"FID score: {fid_score_healthy}")

#################FID FOR INJURED DATA
real_features_injured = extract_features_in_batches(model_without_last_layer, real_X_injured)
generated_features_injured = extract_features_in_batches(model_without_last_layer, synth_X_injured)

# Calculate statistics
mu_real_injured, sigma_real_injured = calculate_statistics(real_features_injured)
mu_gen_injured, sigma_gen_injured = calculate_statistics(generated_features_injured)

# Compute FID
fid_score_injured = calculate_fid(mu_real_injured, sigma_real_injured, mu_gen_injured, sigma_gen_injured)
print(f"FID score: {fid_score_injured}")
######################################################


###############################################INJURED

"""
real_images = flat_list_real_injured
generated_images = injured_data
# Extract features
real_features = extract_features(model_without_last_layer, real_images)
generated_features = extract_features(model_without_last_layer, generated_images)

# Calculate statistics
mu_real, sigma_real = calculate_statistics(real_features)
mu_gen, sigma_gen = calculate_statistics(generated_features)

# Compute FID
fid_score = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
print(f"FID score: {fid_score}")
"""
######################################################

"""
The Fréchet Inception Distance (FID) is a measure used to evaluate the quality of images generated by generative models, such as GANs (Generative Adversarial Networks). It compares the similarity between two sets of images, usually real images and generated images, by looking at the difference in feature vectors calculated by an Inception model pre-trained on ImageNet. The lower the FID score, the more similar the two sets of images are, implying better quality of the generated images.
Steps to Calculate FID

    Prepare your datasets: You need two sets of images - one set of real images and one set of generated images. Ensure both sets are preprocessed in the same way (e.g., resized to the same dimensions, normalized).

    Load a pre-trained Inception Model: Load an InceptionV3 model pre-trained on ImageNet, excluding the top (final classification layer) since you're interested in the feature representations, not the classification output.

    Calculate feature vectors: Pass both sets of images through the Inception model to get their respective feature vectors. This typically involves using the output of one of the last layers of the model.

    Compute the statistics: For each set of feature vectors, calculate the mean and covariance matrix. These statistics summarize the distribution of features in each set of images.

    Calculate the Fréchet distance: Finally, use the mean and covariance of the real and generated images to calculate the Fréchet distance using the following formula:

FID=∣∣μreal−μgen∣∣2+Tr(Σreal+Σgen−2(ΣrealΣgen)1/2)
FID=∣∣μreal​−μgen​∣∣2+Tr(Σreal​+Σgen​−2(Σreal​Σgen​)1/2)

where:

    μreal,μgenμreal​,μgen​ are the means of the feature vectors for the real and generated images, respectively.
    Σreal,ΣgenΣreal​,Σgen​ are the covariance matrices for the real and generated images, respectively.
    TrTr is the trace of a matrix (sum of elements on the main diagonal).
    The square root of the product of the covariance matrices (ΣrealΣgen)1/2(Σreal​Σgen​)1/2 needs to be computed in a way that ensures the result is a positive semidefinite matrix (e.g., using the eigenvalue decomposition).

Python Code Example

While I can't run deep learning models or load large datasets directly here, I can provide a basic code outline using PyTorch and the scipy library for calculating the FID score.
"""


"""
import torch
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm

# Load pre-trained Inception model
model = inception_v3(pretrained=True)
model.fc = torch.nn.Identity()  # Modify to return feature vectors

def get_features(images, model):
    # Assuming images is a batch of images (as tensors)
    with torch.no_grad():
        features = model(images)
    return features.cpu().numpy()

def calculate_fid(real_features, gen_features):
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    ssdiff = np.sum((mu_real - mu_gen) ** 2.0)
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
    return fid
"""

"""
In this example, you'd need to:

    Replace images in get_features with your actual data (preprocessed appropriately).
    Ensure real_features and gen_features are arrays of feature vectors extracted from your real and generated images, respectively.

Keep in mind that calculating FID for large datasets can be computationally intensive, and you might need to batch the process if working with very large sets of images or limited resources.
"""