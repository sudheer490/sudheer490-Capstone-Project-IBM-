import streamlit as st
import pandas as pd
import joblib  # For loading the trained model

# Load the trained model and preprocessing functions
model = joblib.load('decision_tree_model.pkl')  #Replace 'your_trained_model.pkl' with the actual filename of your trained model
# Load other preprocessing functions if you have any

# Function to preprocess input data
import pandas as pd

def preprocess_data(data):
    # Encode categorical features
    data['ADDRTYPE'].replace(to_replace=['Intersection', 'Block', 'Alley', 'Unknown'], value=[1, 2, 3, 0], inplace=True)
    data['INATTENTIONIND'].replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
    data['PEDROWNOTGRNT'].replace(to_replace=['Y'], value=[1], inplace=True)
    data['COLLISIONTYPE'].replace(to_replace=['Angles', 'Sideswipe', 'Parked Car', 'Other', 'Cycles',
                                              'Rear Ended', 'Head On', 'Left Turn', 'Pedestrian', 'Right Turn'],
                                  value=[1, 2, 3, 0, 4, 5, 6, 7, 8, 9], inplace=True)
    data['HITPARKEDCAR'].replace(to_replace=['N', 'Y'], value=[0, 1], inplace=True)
    data['SPEEDING'].replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)

    # For numeric features, do nothing (assuming they are already in the desired format)
    # You can add specific preprocessing steps here if needed in the future

    return data

import streamlit as st
import pandas as pd

# Load your model and preprocess_data function here
model = joblib.load('decision_tree_model.pkl') 
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
year = st.slider('Year', min_value=2004, max_value=2023, step=1)
month = st.slider('Month', min_value=1, max_value=12, step=1)
day = st.slider('Day', min_value=1, max_value=31, step=1)
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
