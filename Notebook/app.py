import torch
import streamlit as st
import numpy as np
from arch import CODPmodel
from sklearn.preprocessing import StandardScaler, LabelEncoder
input_dim = 10  # Number of input features
model = CODPmodel(input_dim)
model.load_state_dict(torch.load('CODPmodel.pth'))  # Load the saved model
model.eval()
label_encoders = {
    'Gender': LabelEncoder().fit(['Male', 'Female']),
    'Smoking_Status': LabelEncoder().fit(['Never', 'Former', 'Current']),
    'Location': LabelEncoder().fit(['Kathmandu', 'Pokhara', 'Lalitpur'])  # Adjust with your cities
}

scaler = StandardScaler()
st.title('COPD Diagnosis Prediction')
age = st.number_input('Age', min_value=1, max_value=100, value=30)
gender = st.selectbox('Gender', ['Male', 'Female'])
smoking_status = st.selectbox('Smoking Status', ['Never', 'Former', 'Current'])
biomass_exposure = st.selectbox('Biomass Fuel Exposure', [0, 1])
occupational_exposure = st.selectbox('Occupational Exposure', [0, 1])
family_history_copd = st.selectbox('Family History of COPD', [0, 1])
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
location = st.selectbox('Location', ['Kathmandu', 'Pokhara', 'Lalitpur'])
air_pollution_level = st.number_input('Air Pollution Level', min_value=0, max_value=500, value=100)
resp_infections_childhood = st.selectbox('Respiratory Infections in Childhood', [0, 1])
gender_encoded = label_encoders['Gender'].transform([gender])[0]
smoking_status_encoded = label_encoders['Smoking_Status'].transform([smoking_status])[0]
location_encoded = label_encoders['Location'].transform([location])[0]

# Combine inputs into a single numpy array
input_features = np.array([[age, gender_encoded, smoking_status_encoded, biomass_exposure,
                            occupational_exposure, family_history_copd, bmi, location_encoded,
                            air_pollution_level, resp_infections_childhood]])

# Scale continuous features (apply same scaling used during training)
input_features[:, [0, 6, 8]] = scaler.fit_transform(input_features[:, [0, 6, 8]])

# Convert input features to a PyTorch tensor
input_tensor = torch.tensor(input_features, dtype=torch.float32)

# Button to trigger prediction
if st.button('Predict'):
    with torch.no_grad():
        prediction = model(input_tensor)  # Get prediction
        prediction_class = (prediction > 0.5).float().item()  # Convert to binary class (0 or 1)

    # Display result
    if prediction_class == 1:
        st.write("The model predicts: **COPD diagnosed**")
    else:
        st.write("The model predicts: **No COPD**")