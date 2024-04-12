import streamlit as st
import numpy as np
import pickle
import pandas as pd
import time
import socket

#server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) (Not needed if needed to check in other systems)

#server.bind(('0.0.0.0', 9999)) (Not needed if needed to check in other systems)

# Load the dataframe
with open('.\df.pkl', 'rb') as file:
    df = pd.read_pickle(file)

# Load the trained model
with open('.\pipe.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Laptop Price Predictor')

# Inputs
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['CPU Brand'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu'].unique())
os = st.selectbox('OS', df['OS'].unique())
clock_speed = st.number_input('Clock Speed')

if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    # Preprocess input
    ppi = ((int(resolution.split('x')[0])**2) + (int(resolution.split('x')[1])**2))**0.5 / screen_size
    input_data = np.array([company, type, ram, gpu, weight, touchscreen, ips, ppi, cpu, clock_speed, ssd, 0, os])
    input_df = pd.DataFrame([input_data], columns=['Company', 'TypeName', 'Ram', 'Gpu', 'Weight', 'Touch', 'IPS', 'ppi', 'CPU Brand', 'Clock Speed', 'ssd', 'hybrid', 'OS'])

    # Predict
    predicted_price = model.predict(input_df)
    st.title("The predicted price of this configuration is " + str(int(np.exp(predicted_price[0]))))
