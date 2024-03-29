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
    data['UNDERINFL'].replace(to_replace=['Yes','No'],value=[0,1],inplace=True)
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

st.markdown("Accident Severity Prediction App 🚧", unsafe_allow_html=True)

# Streamlit UI elements for categorical features

col1, col2, col3 = st.columns(3)

# Streamlit UI elements for categorical features - Column 1
with col1:
    addrtype = st.selectbox('Address Type', ['Intersection', 'Block', 'Alley', 'Unknown'])
    inattention_ind = st.selectbox('Inattention Indicator', ['No', 'Yes'])
    pedrownotgrnt = st.selectbox('Pedestrian Right of Way Not Granted', ['No', 'Yes'])
    collision_type = st.selectbox('Collision Type', ['Angles', 'Sideswipe', 'Parked Car', 'Other', 'Cycles',
                                                      'Rear Ended', 'Head On', 'Left Turn', 'Pedestrian', 'Right Turn'])
    hit_parked_car = st.selectbox('Hit Parked Car', ['No', 'Yes'])

# Streamlit UI elements for categorical features - Column 2
with col2:
    weather = st.selectbox('Weather', ['Clear', 'Raining', 'Overcast', 'Unknown', 'Snowing', 'Other', 'Fog/Smog/Smoke',
                                       'Sleet/Hail/Freezing Rain', 'Blowing Sand/Dirt', 'Severe Crosswind', 'Partly Cloudy'])
    road_cond = st.selectbox('Road Condition', ['Wet', 'Dry', 'Unknown', 'Snow/Slush', 'Ice', 'Other',
                                                'Sand/Mud/Dirt', 'Standing Water', 'Oil'])
    light_cond = st.selectbox('Light Condition', ['Daylight', 'Dark - Street Lights On', 'Unknown', 'Dusk', 'Dawn',
                                                  'Dark - No Street Lights', 'Dark - Street Lights Off', 'Other',
                                                  'Dark - Unknown Lighting'])
    speeding = st.selectbox('Speeding', ['No', 'Yes'])

# Streamlit UI elements for numeric features - Column 3
with col3:
    person_count = st.number_input('Person Count', min_value=0)
    ped_count = st.number_input('Pedestrian Count', min_value=0)
    ped_cycle_count = st.number_input('Pedestrian Cyclist Count', min_value=0)
    veh_count = st.number_input('Vehicle Count', min_value=0)
    under_infl = st.selectbox('Under Drug Influence', ['No', 'Yes'])


# User input for date
date_input = st.date_input('Select a date', min_value=datetime(2004, 1, 1), max_value=datetime(2023, 12, 31))

# Splitting the date into year, month, day, and hour
year = date_input.year
month = date_input.month
day = date_input.day
weekday = date_input.weekday()

# User input for hour
hour = st.slider('Hour', min_value=0, max_value=23, step=1)
# Preprocess user input
user_data = {
    'ADDRTYPE': [addrtype],
    'PERSONCOUNT': [person_count],
    'PEDCOUNT': [ped_count],
    'PEDCYLCOUNT': [ped_cycle_count],
    'VEHCOUNT': [veh_count],
    'INATTENTIONIND': [inattention_ind],
    'UNDERINFL': [under_infl],
    'WEATHER': [weather],
    'ROADCOND': [road_cond],
    'LIGHTCOND': [light_cond],
    'PEDROWNOTGRNT': [pedrownotgrnt],
    'SPEEDING': [speeding],
    'COLLISIONTYPE': [collision_type],
    'HITPARKEDCAR': [hit_parked_car],
    'Year': [year],
    'Month': [month],
    'Day': [day],
    'Weekday': [weekday],
    'Hour': [hour]
}

df = pd.DataFrame(user_data)
preprocessed_df = preprocess_data(df)



# Make predictions using the loaded model
severity_prediction = model.predict(preprocessed_df)

# Define labels for severity levels
severity_labels = {
    1: 'Slight Injuries',
    2: 'Fatal'
}
# Contact Information
st.sidebar.header('Contact Information')
st.sidebar.subheader('Name:')
st.sidebar.write('Sai Sudheer Vishnumolakala')
st.sidebar.subheader('LinkedIn:')
st.sidebar.write('[LinkedIn Profile](https://www.linkedin.com/in/saisudheer-vishnumolakala/)')
st.sidebar.subheader('GitHub:')
st.sidebar.write('[GitHub Profile](https://github.com/sudheer490/)')
# Get the corresponding severity label based on the prediction
predicted_severity = severity_labels.get(severity_prediction[0], 'Unknown Severity')
st.header('Accident Severity Prediction')
st.subheader('User Input:')
st.table(pd.DataFrame(user_data))
# Display prediction
st.write('Predicted Severity:', predicted_severity)
st.markdown(f'<p style="font-size:50px;color:red;">{predicted_severity}</p>', unsafe_allow_html=True)

