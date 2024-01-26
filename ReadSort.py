# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:24:19 2024

@author: User
"""

import mne
import os
import pandas as pd
import numpy as np

def structure_data():
    # Directory containing your EEG files
    eeg_directory = r"E:\Code snippets\IntraOp Data"
    
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
                    if common_channels_healthy != []:
                        raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
                        raw_healthy = raw.pick_channels(ch_names=common_channels_healthy)
                        eeg_data_dict_healthy[key] = raw_healthy
                    
    
    # Create a new dictionary to store the combined data
    combined_dict = {'healthy': list(eeg_data_dict_healthy.values()), 'injured': list(eeg_data_dict_injured.values())}
    print("combined dict complete")
    
    # Example function to extract channels from raw data
    def extract_channels(raw_object):
        channels = []
        for channel_idx in range(raw_object.info['nchan']):
            channel_data = raw_object.get_data(picks=channel_idx)
            channels.append(channel_data)
        return channels
    
    # Create a new dictionary to store the simplified data
    simplified_dict = {'healthy': [], 'injured': []}
    
    # Dictionary to keep track of subjects and their indices in simplified_dict
    subject_indices = {'healthy': {}, 'injured': {}}
    
    # Iterate through the combined_dict and extract channels
    for condition, raw_objects in combined_dict.items():
        for raw_object in raw_objects:
            # Extract subject information from the raw_object string representation
            subject_info_start = raw_object.__str__().find('sub-')
            subject_info_end = raw_object.__str__().find('_ses-')
            subject_key = raw_object.__str__()[subject_info_start:subject_info_end]
    
            # Check if the subject already exists in the simplified_dict
            if subject_key not in subject_indices[condition]:
                subject_indices[condition][subject_key] = len(simplified_dict[condition])
                simplified_dict[condition].append([])
    
            # Extract channels and add them to the corresponding subject's entry
            channels = extract_channels(raw_object)
            simplified_dict[condition][subject_indices[condition][subject_key]].extend(channels)
    

    """
    #If it is necessary to use a json for something. Note: This is unecessary for the training process
    import json
    
    print("started converting")
    # Function to convert NumPy arrays to lists
    def convert_numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_numpy_to_list(value) for key, value in obj.items()}
        else:
            return obj
    
    # Convert NumPy arrays to lists in the simplified_dict
    converted_dict = convert_numpy_to_list(simplified_dict)
    
    # Specify the file path where you want to save the JSON file
    json_file_path = 'simplified_dict.json'
    print("finished converting")
    # Save the converted_dict to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(converted_dict, json_file)
    
    print(f"The simplified_dict has been saved to {json_file_path}")
    """
    
    return simplified_dict






