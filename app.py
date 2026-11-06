
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title='Breast Cancer Classification App', layout='wide')
st.title('⚕️ Breast Cancer Diagnosis Portal')

@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca_transformer.pkl')
    model = joblib.load('best_logistic_regression_model.pkl')
    return scaler, pca, model

try:
    scaler, pca, model = load_assets()
    st.success('Diagnostic models loaded successfully!')
except Exception as e:
    st.error(f'Error loading diagnostic model assets: {e}. Please ensure the .pkl files are in the same folder.')
    st.stop()

feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
    'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

st.subheader('Enter Patient Clinical Metrics:')
cols = st.columns(3)
inputs = {}

for idx, name in enumerate(feature_names):
    with cols[idx % 3]:
        inputs[name.replace('_', ' ')] = st.number_input(f'{name.replace("_", " ")}', value=0.0, step=0.1)

if st.button('Perform Diagnostic Run'):
    input_df = pd.DataFrame([inputs])
    # Align columns in correct order
    input_df.columns = feature_names
    
    scaled_features = scaler.transform(input_df)
    pca_features = pca.transform(scaled_features)
    
    pred = model.predict(pca_features)[0]
    probs = model.predict_proba(pca_features)[0]
    
    st.write('---')
    if pred == 'M':
        st.error(f'**Diagnosis Result:** Malignant (M) with {probs[1]*100:.2f}% probability.')
    else:
        st.success(f'**Diagnosis Result:** Benign (B) with {probs[0]*100:.2f}% probability.')
