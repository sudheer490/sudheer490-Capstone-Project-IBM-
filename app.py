import pandas as pd
import numpy as np
from sklearn import *
import streamlit as st
import pickle
from datetime import datetime

import shap
import matplotlib
from IPython import get_ipython

# Load the trained model and preprocessing functions
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
import pandas as pd

def preprocess_data(data):
    # Encode categorical features
    data['ADDRTYPE'].replace(to_replace=['Intersection', 'Block', 'Alley', 'Unknown'], value=[1, 2, 3, 0], inplace=True)
    data['INATTENTIONIND'].replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
    data['PEDROWNOTGRNT'].replace(to_replace=['Yes','No'], value=[1,0], inplace=True)
    data['COLLISIONTYPE'].replace(to_replace=['Angles', 'Sideswipe', 'Parked Car', 'Other', 'Cycles',
                                              'Rear Ended', 'Head On', 'Left Turn', 'Pedestrian', 'Right Turn'],
                                  value=[1, 2, 3, 0, 4, 5, 6, 7, 8, 9], inplace=True)
    data['HITPARKEDCAR'].replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
    data['SPEEDING'].replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
    
    # Encode 'WEATHER' column
    data['WEATHER'].replace(to_replace=['Clear', 'Raining', 'Overcast', 'Unknown', 'Snowing', 'Other',
                                        'Fog/Smog/Smoke', 'Sleet/Hail/Freezing Rain', 'Blowing Sand/Dirt',
                                        'Severe Crosswind', 'Partly Cloudy'],
                            value=[1, 2, 3, 0, 4, 0, 5, 6, 7, 8, 9], inplace=True)
    
    # Encode 'ROADCOND' column
    data['ROADCOND'].replace(to_replace=['Wet', 'Dry', 'Unknown', 'Snow/Slush', 'Ice', 'Other',
                                         'Sand/Mud/Dirt', 'Standing Water', 'Oil'],
                             value=[1, 2, 0, 3, 4, 0, 5, 6, 7], inplace=True)
    
    # Encode 'LIGHTCOND' column
    data['LIGHTCOND'].replace(to_replace=['Daylight', 'Dark - Street Lights On', 'Unknown', 'Dusk', 'Dawn',
                                          'Dark - No Street Lights', 'Dark - Street Lights Off', 'Other',
                                          'Dark - Unknown Lighting'],
                              value=[0, 1, 2, 3, 4, 5, 6, 7, 8], inplace=True)
    
    # Encode 'SPEEDING' column
    data['SPEEDING'].replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)

    # For numeric features, do nothing (assuming they are already in the desired format)
    # You can add specific preprocessing steps here if needed in the future

    return data


# Streamlit UI elements for categorical features
addrtype = st.selectbox('Address Type', ['Intersection', 'Block', 'Alley', 'Unknown'])
inattention_ind = st.selectbox('Inattention Indicator', ['No', 'Yes'])
pedrownotgrnt = st.selectbox('Pedestrian Right of Way Not Granted', ['No', 'Yes'])
collision_type = st.selectbox('Collision Type', ['Angles', 'Sideswipe', 'Parked Car', 'Other', 'Cycles',
                                                  'Rear Ended', 'Head On', 'Left Turn', 'Pedestrian', 'Right Turn'])
hit_parked_car = st.selectbox('Hit Parked Car', ['No', 'Yes'])
weather = st.selectbox('Weather', ['Clear', 'Raining', 'Overcast', 'Unknown', 'Snowing', 'Other', 'Fog/Smog/Smoke',
                                   'Sleet/Hail/Freezing Rain', 'Blowing Sand/Dirt', 'Severe Crosswind', 'Partly Cloudy'])
road_cond = st.selectbox('Road Condition', ['Wet', 'Dry', 'Unknown', 'Snow/Slush', 'Ice', 'Other',
                                            'Sand/Mud/Dirt', 'Standing Water', 'Oil'])
light_cond = st.selectbox('Light Condition', ['Daylight', 'Dark - Street Lights On', 'Unknown', 'Dusk', 'Dawn',
                                              'Dark - No Street Lights', 'Dark - Street Lights Off', 'Other',
                                              'Dark - Unknown Lighting'])
speeding = st.selectbox('Speeding', ['No', 'Yes'])

# Streamlit UI elements for numeric features
person_count = st.number_input('Person Count', min_value=0)
ped_count = st.number_input('Pedestrian Count', min_value=0)
ped_cycle_count = st.number_input('Pedestrian Cyclist Count', min_value=0)
veh_count = st.number_input('Vehicle Count', min_value=0)
under_infl = st.slider('Under Influence', min_value=0, max_value=1, step=1)

# User input for date
date_input = st.date_input('Select a date', min_value=datetime(2004, 1, 1), max_value=datetime(2023, 12, 31))

# Splitting the date into year, month, day, and hour
year = date_input.year
month = date_input.month
day = date_input.day

# User input for hour
hour = st.slider('Hour', min_value=0, max_value=23, step=1)
# Preprocess user input
user_data = {
    'ADDRTYPE': [addrtype],
    'INATTENTIONIND': [inattention_ind],
    'PEDROWNOTGRNT': [pedrownotgrnt],
    'COLLISIONTYPE': [collision_type],
    'HITPARKEDCAR': [hit_parked_car],
    'WEATHER': [weather],
    'ROADCOND': [road_cond],
    'LIGHTCOND': [light_cond],
    'SPEEDING': [speeding],
    'PERSONCOUNT': [person_count],
    'PEDCOUNT': [ped_count],
    'PEDCYLCOUNT': [ped_cycle_count],
    'VEHCOUNT': [veh_count],
    'UNDERINFL': [under_infl],
    'Year': [year],
    'Month': [month],
    'Day': [day],
    'Hour': [hour]
}

df = pd.DataFrame(user_data)
preprocessed_df = preprocess_data(df)



# Make predictions using the loaded model
prediction = model.predict(preprocessed_df)

# Display prediction
st.write('Predicted Severity:', prediction)
