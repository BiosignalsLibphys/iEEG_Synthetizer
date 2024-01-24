# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:24:19 2024

@author: User
"""

import mne
import os
import pandas as pd


# Directory containing your EEG files
eeg_directory = r"E:\IntraOp Data"

# Initialize an empty dictionary to store EEG data
eeg_data_dict_injured = {}
eeg_data_dict_healthy = {}


# Iterate through all files in the directory and its subdirectories
for foldername, subfolders, filenames in os.walk(eeg_directory):
    for filename in filenames:
        # Check if the file is an EEG file (you may want to add more conditions)
        if filename.endswith('_ieeg.vhdr'):
            # Extract patient and situation information from the file path
            path_parts = foldername.split(os.sep)
            patient_id = None
            situation_id = None
            
            
            vhdr_file = os.path.join(foldername, filename)
            filename_channel_quality = vhdr_file.replace('acute_ieeg.vhdr', 'acute_channels.tsv')
            filename_channel_resect = vhdr_file.replace('task-acute_ieeg.vhdr', 'electrodes.tsv')
            for part in path_parts:
                if part.startswith('sub-'):
                    patient_id = part
                elif part.startswith('ses-'):
                    situation_id = part

            # Check if patient and situation information is found
            if patient_id and situation_id:
                # Create the dictionary key
                key = f'{patient_id}_{situation_id}'
                
                # Assuming you want to store the file path, you can modify this part accordingly
                file_path = os.path.join(foldername, filename)
                
                # Add the EEG information to the dictionary
                #eeg_data_dict[key] = {'file_path': file_path}                
                
                # Read the channels.tsv file
                channels_info = pd.read_csv(filename_channel_quality, delimiter='\t')
                # Extract good channels based on the "sampling_frequency" column
                good_channels = channels_info.loc[channels_info['status_description'] == 'included', 'name'].tolist()
                
                # Read the channels.tsv file
                channels_info = pd.read_csv(filename_channel_resect, delimiter='\t')
                # Extract good channels based on the "sampling_frequency" column
                injured_channels = channels_info.loc[(channels_info['resected'] == 'yes') & (channels_info['edge'] == 'no'), 'name'].tolist()
                
                common_channels_injured = list(set(good_channels).intersection(injured_channels))
                if common_channels_injured != []:
                    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
                    raw_injured = raw.pick_channels(ch_names=common_channels_injured)#good_channels) #apparently pick_channels is a legacy function and should use inst.pick or something
                    # Store the raw data in the dictionary with the constructed key
                    eeg_data_dict_injured[key] = raw_injured
                
                # Read the channels.tsv file
                channels_info = pd.read_csv(filename_channel_resect, delimiter='\t')
                # Extract good channels based on the "sampling_frequency" column
                healthy_channels = channels_info.loc[(channels_info['resected'] == 'no') & (channels_info['edge'] == 'no'), 'name'].tolist()
                
                common_channels_healthy = list(set(good_channels).intersection(healthy_channels))
                # Store the raw data in the dictionary with the constructed key
                #raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
                
                if common_channels_healthy != []:
                    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
                    raw_healthy = raw.pick_channels(ch_names=common_channels_healthy)
                    eeg_data_dict_healthy[key] = raw_healthy
                
                
# Create a new dictionary to store the combined data
combined_dict = {'healthy': list(eeg_data_dict_healthy.values()), 'injured': list(eeg_data_dict_injured.values())}


# Example function to extract channels from raw data
def extract_channels(raw_object):
    channels = []
    for channel_idx in range(raw_object.info['nchan']):
        channel_data = raw_object.get_data(picks=channel_idx)
        channels.append(channel_data)
    return channels

# Create a new dictionary to store the simplified data
simplified_dict = {'healthy': [], 'injured': []}

# Iterate through the combined_dict and extract channels
for condition, raw_objects in combined_dict.items():
    for raw_object in raw_objects:
        channels = extract_channels(raw_object)
        simplified_dict[condition].extend(channels)

# The result is a simplified dictionary that has 2 keys "healthy" and "injured" and all channels as an array (unidentified in terms of both subject and channel) of each type under the correct key




