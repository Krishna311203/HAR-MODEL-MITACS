import socket
import time
import pandas as pd
import numpy as np
import pickle

# Function to filter predictions to only valid classes
def filter_valid_classes(probs, valid_classes):
    valid_probs = probs[:, valid_classes]
    valid_class_index = np.argmax(valid_probs, axis=1)
    return np.array(valid_classes)[valid_class_index]

# Load the models and scaler
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
    
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('meta_model.pkl', 'rb') as f:
    meta_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the server address and port
server_address = ('127.0.0.1', 28000)

# Create a TCP/IP socket for E4 server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(server_address)

# Device connect command
device_id = 'CF13C6'
client_socket.sendall(f'device_connect {device_id}\n'.encode('utf-8'))
time.sleep(1)

# Subscribe to a channel (e.g., accelerometer)
channel = 'acc'
client_socket.sendall(f'device_subscribe {channel} ON\n'.encode('utf-8'))

# Unity server address and port
unity_server_address = ('127.0.0.1', 5000)
unity_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
unity_socket.connect(unity_server_address)

# Define the window size and sampling frequency
window_size = 0.5  # seconds
sampling_frequency = 32  # Hz
window_length = window_size * sampling_frequency

# Function to calculate features for each window
def calculate_features(window):
    features = {}

    # Calculate binned distribution
    for axis in ['X', 'Y', 'Z']:
        min_val = window[axis].min()
        max_val = window[axis].max()
        bins = np.linspace(min_val, max_val, num=11)  # 10 bins
        binned_counts, _ = np.histogram(window[axis], bins=bins)
        binned_distribution = binned_counts / len(window)
        for i in range(10):
            features[f'{axis}{i}'] = binned_distribution[i]

    # Average
    features['XAVG'] = window['X'].mean()
    features['YAVG'] = window['Y'].mean()
    features['ZAVG'] = window['Z'].mean()

    # Peak detection
    features['XPEAK'] = (window['X'].diff().abs() > 0.015).sum()
    features['YPEAK'] = (window['Y'].diff().abs() > 0.015).sum()
    features['ZPEAK'] = (window['Z'].diff().abs() > 0.015).sum()

    # Absolute deviation
    features['XABSOLDEV'] = np.abs(window['X'] - features['XAVG']).mean()
    features['YABSOLDEV'] = np.abs(window['Y'] - features['YAVG']).mean()
    features['ZABSOLDEV'] = np.abs(window['Z'] - features['ZAVG']).mean()

    # Standard deviation
    features['XSTANDDEV'] = window['X'].std()
    features['YSTANDDEV'] = window['Y'].std()
    features['ZSTANDDEV'] = window['Z'].std()

    # Resultant
    resultant = np.sqrt(window['X']*2 + window['Y']2 + window['Z']*2)
    features['RESULTANT'] = resultant.mean()

    return features

# Buffer to store incoming data
data_buffer = []

# Receive and process the data
try:
    while True:
        data = client_socket.recv(1024)
        if data:
            lines = data.decode('utf-8').strip().split('\n')
            for line in lines:
                if line.startswith('E4_Acc'):
                    _, timestamp, x, y, z = line.split()
                    data_buffer.append([float(timestamp), float(x), float(y), float(z)])
                    if len(data_buffer) >= window_length:
                        window_df = pd.DataFrame(data_buffer, columns=['Timestamp', 'X', 'Y', 'Z'])
                        features = calculate_features(window_df)
                        features_df = pd.DataFrame([features])
                        
                        # Standardize the features
                        standardized_features = scaler.transform(features_df)
                        
                        # Generate prediction probabilities using base models
                        rf_preds = rf_model.predict_proba(standardized_features)
                        svm_preds = svm_model.predict_proba(standardized_features)
                        
                        # Stack the prediction probabilities
                        stacked_features = np.hstack((rf_preds, svm_preds))
                        
                        # Predict using the meta-model
                        predictions_proba = meta_model.predict_proba(stacked_features)
                        
                        # Filter the predictions to only include valid classes
                        valid_classes = [0, 1, 3, 4]
                        final_predictions = filter_valid_classes(predictions_proba, valid_classes)
                        
                        if final_predictions[0] == 1 or final_predictions[0] == 0:
                            print(f'Predicted Activity: {final_predictions[0]}: walking')
                            unity_socket.sendall(b'walk\n')
                        else:
                            print(f'Predicted Activity: {final_predictions[0]}: sitting or standing')
                            unity_socket.sendall(b'stop\n')
                        
                        data_buffer = []  # Clear the buffer after processing the window
        else:
            break
except KeyboardInterrupt:
    print("Connection closed.")
finally:
    client_socket.close()
    unity_socket.close()