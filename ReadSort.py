import mne
import os
import pandas as pd
import numpy as np

# for key stuff
import re

def structure_data(model_type, eeg_directory=r"E:\Code snippets\IntraOp Data"): #change this to take arguments of type (classifier, location generator, data generator, etc)
    # Directory containing your EEG files
    #eeg_directory = r"E:\Code snippets\IntraOp Data"
    if not os.path.exists(eeg_directory):
        model_type = "oopsie_daisy"
        print("directory not found")
    
    if model_type == 'classifier' or model_type == 'chan_gen':
        #eeg_directory = r"E:\Code snippets\IntraOp Data"
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
                        print(common_channels_injured)
                        
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
                        print(common_channels_healthy)
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
        
        return simplified_dict

    elif model_type == 'loc_gen':
        binary_channel_dict = {}

        # Iterate through all files in the directory and its subdirectories
        for foldername, subfolders, filenames in os.walk(eeg_directory):
            for filename in filenames:
                # Check if the file is an EEG file (you may want to add more conditions)
                if filename.endswith('_ieeg.vhdr'):
                    # Extract patient and situation information from the file path
                    path_parts = foldername.split(os.sep)
                    patient_id = None
                    situation_id = None
    
                    for part in path_parts:
                        if part.startswith('sub-'):
                            patient_id = part
                        elif part.startswith('ses-'):
                            situation_id = part
    
                    # Check if patient and situation information is found
                    if patient_id and situation_id:
                        # Create the dictionary key
                        key = f'{patient_id}_{situation_id}'
    
                        vhdr_file = os.path.join(foldername, filename)
                        filename_channel_resect = vhdr_file.replace('task-acute_ieeg.vhdr', 'electrodes.tsv')
    
                        # Read the channels.tsv file
                        channels_info = pd.read_csv(filename_channel_resect, delimiter='\t')
                        
                        # Extract binary channel information based on the "resected" column
                        binary_channels = [1 if status == 'yes' else 0 for status in channels_info['resected']]
    
                        # Add the binary channel information to the dictionary
                        if key not in binary_channel_dict:
                            binary_channel_dict[key] = binary_channels
                        else:
                            binary_channel_dict[key].extend(binary_channels)
        
        # Create a new dictionary with numerical keys
        new_dict = {}
        for i, (key, value) in enumerate(binary_channel_dict.items()):
            new_dict[i] = value # Add 1 to the index to start from 1
        return new_dict#binary_channel_dict
    
    elif model_type == "breakdown":
        #eeg_directory = r"E:\Code snippets\IntraOp Data"
        # Initialize an empty dictionary to store EEG data
        eeg_data_dict_injured = {}
        eeg_data_dict_healthy = {}
        hitorie = {"healthy": {}, "injured": {}}
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
 
                        # Example function to extract channels from raw data
                        def extract_channels(raw_object):
                            channels_dict = {}
                            ch_names = raw_object.ch_names  # Get the channel names

                            for channel_idx, ch_name in enumerate(ch_names):
                                channel_data = raw_object.get_data(picks=channel_idx)
                                channels_dict[ch_name] = channel_data

                            return channels_dict
                        
                        if patient_id not in hitorie and common_channels_healthy != []:
                            # If the patient ID doesn't exist, create a new entry
                            channels = extract_channels(raw_healthy)
                            hitorie["healthy"][patient_id] = {situation_id: channels}
                        elif common_channels_healthy != []:
                            # If the patient ID already exists, add the new situation code and data
                            channels = extract_channels(raw_healthy)
                            hitorie["healthy"][patient_id][situation_id] = channels
                            
                        if patient_id not in hitorie and common_channels_injured != []:
                            # If the patient ID doesn't exist, create a new entry
                            channels = extract_channels(raw_injured)
                            hitorie["injured"][patient_id] = {situation_id: channels}
                        elif common_channels_injured != []:
                            # If the patient ID already exists, add the new situation code and data
                            channels = extract_channels(raw_injured)
                            hitorie["injured"][patient_id][situation_id] = channels
        return hitorie
    
    elif model_type == 'utsu':
        #eeg_directory = r"E:\Code snippets\IntraOp Data"
        eeg_data_dict_injured = {}
        eeg_data_dict_healthy = {}
        
        condition_mapping = {}
        counter = 0
        for cavity in ['no', 'yes', 'nan']:
            for edge in ['no', 'yes', 'nan']:
                for num in range(1, 6):
                    for letter in 'ABCDEFG':
                        situation_code = f"ses-SITUATION{num}{letter}"
                        condition_key = (cavity, edge, situation_code)
                        condition_mapping[condition_key] = counter
                        counter += 1
        
        for foldername, subfolders, filenames in os.walk(eeg_directory):
            for filename in filenames:
                if filename.endswith('_ieeg.vhdr'):
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
        
                    if patient_id and situation_id:        
                        key = f'{patient_id}_{situation_id}'
                        file_path = os.path.join(foldername, filename)
                        
                        # Read the channels.tsv file for quality information
                        channels_info_quality = pd.read_csv(filename_channel_quality, delimiter='\t')
                        good_channels = channels_info_quality.loc[channels_info_quality['status_description'] == 'included', 'name'].tolist()
                        
                        # Read the channels.tsv file for resection information
                        channels_info_resect = pd.read_csv(filename_channel_resect, delimiter='\t')
                        
                        #injured_channels = channels_info_resect.loc[(channels_info_resect['resected'] == 'yes') & (channels_info_resect['edge'] == 'no'), 'name'].tolist()
                        #healthy_channels = channels_info_resect.loc[(channels_info_resect['resected'] == 'no') & (channels_info_resect['edge'] == 'no'), 'name'].tolist()
                        injured_channels = channels_info_resect.loc[(channels_info_resect['resected'] == 'yes'), 'name'].tolist()
                        healthy_channels = channels_info_resect.loc[(channels_info_resect['resected'] == 'no'), 'name'].tolist()
                        
                        # Now, we'll add the condition values to these channels
                        injured_channels_with_conditions = []
                        healthy_channels_with_conditions = []
                        channels_info = pd.read_csv(filename_channel_resect, delimiter='\t')
                        
                        for _, row in channels_info.iterrows():
                            channel_name = row['name']
                            resected = row['resected']
                            # Check if 'cavity' column exists, if not, set to 'nan'
                            cavity = 'nan' if 'cavity' not in row.index else ('nan' if pd.isna(row['cavity']) else row['cavity'])
                            
                            # Check if 'edge' column exists, if not, set to 'nan'
                            edge = 'nan' if 'edge' not in row.index else ('nan' if pd.isna(row['edge']) else row['edge'])
                            
                            #cavity = 'nan' if pd.isna(row['cavity']) else row['cavity']
                            #edge = 'nan' if pd.isna(row['edge']) else row['edge']
                            
                            condition_key = (cavity, edge, situation_id)
                            print("\n\n\n")
                            print(condition_key)
                            condition_value = condition_mapping.get(condition_key, -1)
                            print(condition_value)

                            channel_name = row['name']
                            resected = row['resected']
                            if resected == 'yes': #and edge == 'no':
                                injured_channels.append((channel_name, condition_value))
                            elif resected == 'no': #and edge == 'no':
                                healthy_channels.append((channel_name, condition_value))
                        
                        print("just before intersection")
                        common_channels_injured = list(set([ch[0] for ch in injured_channels]).intersection(good_channels))
                        
                        if common_channels_injured:
                            raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
                            raw_injured = raw.pick_channels(ch_names=common_channels_injured)
                            print("just before making the eeg_data_dict")
                            eeg_data_dict_injured[key] = {
                                'raw': raw_injured,
                                'condition_values': [ch[1] for ch in injured_channels if ch[0] in common_channels_injured]
                            }
                        
                        common_channels_healthy = list(set([ch[0] for ch in healthy_channels]).intersection(good_channels))
                        if common_channels_healthy:
                            raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
                            raw_healthy = raw.pick_channels(ch_names=common_channels_healthy)
                            eeg_data_dict_healthy[key] = {
                                'raw': raw_healthy,
                                'condition_values': [ch[1] for ch in healthy_channels if ch[0] in common_channels_healthy]
                            }
        print("\n\n\n")
        print("just before combined_dict")
        combined_dict = {'healthy': list(eeg_data_dict_healthy.values()), 'injured': list(eeg_data_dict_injured.values())}
    
        print("combined dict complete")
        
        # Make sure your extract_channels function is defined like this:
        def extract_channels(raw_object, condition_values):
            channels = []
            for channel_idx in range(raw_object.info['nchan']):
                channel_data = raw_object.get_data(picks=channel_idx)
                channels.append((channel_data, condition_values[channel_idx]))
            return channels
        
        simplified_dict = {'healthy': [], 'injured': []}
        subject_indices = {'healthy': {}, 'injured': {}}
        
        # Iterate through the combined_dict and extract channels
        for condition, raw_objects in combined_dict.items():
            for raw_data in raw_objects:
                # Extract subject information from the raw_object string representation
                # Note: We're using raw_data['raw'] because the raw object is nested in the dictionary
                subject_info_start = raw_data['raw'].__str__().find('sub-')
                subject_info_end = raw_data['raw'].__str__().find('_ses-')
                subject_key = raw_data['raw'].__str__()[subject_info_start:subject_info_end]
        
                # Check if the subject already exists in the simplified_dict
                if subject_key not in subject_indices[condition]:
                    subject_indices[condition][subject_key] = len(simplified_dict[condition])
                    simplified_dict[condition].append([])
        
                # Extract channels and add them to the corresponding subject's entry
                # Note: We're passing raw_data['raw'] and raw_data['condition_values'] separately
                channels = extract_channels(raw_data['raw'], raw_data['condition_values'])
                simplified_dict[condition][subject_indices[condition][subject_key]].extend(channels)
        
        return simplified_dict
    elif model_type == 'oopsie_daisy':
        oopsie_daisy = "you need to pass a path or your path is wrong yo"
        return oopsie_daisy
     




def unwind_conditions(data): 
    condition_mapping = {}
    counter = 0
    for cavity in ['no', 'yes', 'nan']:
        for edge in ['no', 'yes', 'nan']:
            for num in range(1, 6):
                for letter in 'ABCDEFG':
                    situation_code = f"ses-SITUATION{num}{letter}"
                    condition_key = (cavity, edge, situation_code)
                    condition_mapping[condition_key] = counter
                    counter += 1
    # Create a reverse mapping
    reverse_mapping = {v: k for k, v in condition_mapping.items()}
    
    def unwind_single_condition(condition_value, is_healthy):
        if condition_value == -1:
            return "Unknown condition"
        cavity, edge, situation = reverse_mapping[condition_value]
        return f"cavity:{cavity}, edge:{edge}, situation:{situation}, status:{'healthy' if is_healthy else 'injured'}"
    
    unwound_data = {}
    for condition in ['healthy', 'injured']:
        unwound_data[condition] = []
        for subject in data[condition]:
            unwound_subject = []
            for channel in subject:
                raw_data, condition_value = channel
                unwound_condition = unwind_single_condition(condition_value, condition == 'healthy')
                unwound_subject.append((raw_data, unwound_condition))
            unwound_data[condition].append(unwound_subject)
    
    return unwound_data