import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
@st.cache_data
def load_data():
    # Assuming you have downloaded the dataset from the provided link
    df = pd.read_csv(r"C:\Users\WAQSA\Desktop\TTDS Course Project\heart_data.csv")
    return df

df = load_data()

# Sidebar with introduction and background
st.sidebar.title("Cardiopulse Guardian")
st.sidebar.markdown("A Machine Learning Model for Cardiovascular Risk Detection and Management")
st.sidebar.markdown("### Introduction")
st.sidebar.markdown(
    "Cardiovascular diseases (CVDs) are a leading cause of global mortality, accounting for approximately 17.9 million deaths annually. "
    "To address this critical issue, we present the Cardiopulse Guardian, a machine learning model designed for the early detection and management of individuals at high cardiovascular risk."
)

st.sidebar.markdown("### Background")
st.sidebar.markdown(
    "#### Cardiovascular Diseases (CVDs) Statistics\n"
    "- CVDs contribute to 31% of all deaths worldwide.\n"
    "- Four out of 5 CVD deaths result from heart attacks and strokes.\n"
    "- One-third of these deaths occur prematurely in individuals under 70 years of age."
)

# Main content
st.title("Cardiopulse Guardian Dashboard")

# Display dataset information
st.header("Dataset Information")
st.markdown(
    "The dataset used for training the Cardiopulse Guardian contains 11 features that can be utilized to predict the likelihood of heart disease."
)
st.markdown(
    "[Link to dataset](https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final)"
)
st.subheader("Features include:")
st.write(
    "- Age: age of the patient [years]\n"
    "- Sex: sex of the patient [M: Male, F: Female]\n"
    "- ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]\n"
    "- RestingBP: resting blood pressure [mm Hg]\n"
    "- Cholesterol: serum cholesterol [mm/dl]\n"
    "- FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]\n"
    "- RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality]\n"
    "- MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]\n"
    "- ExerciseAngina: exercise-induced angina [Y: Yes, N: No]\n"
    "- Oldpeak: oldpeak = ST [Numeric value measured in depression]\n"
    "- ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]\n"
    "- HeartDisease: output class [1: heart disease, 0: Normal]"
)

# Load your pre-trained model and pipeline
df_f = pd.read_csv(r"C:\Users\WAQSA\Desktop\TTDS Course Project\featured_data.csv")
#df_f = df.drop('Unnamed: 0', axis=1)

categorical_variable = ['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'RestBP_Category', 'CholesterolCategory', 'MaxHRCategory']
numerical_feature = ['Sex', 'FastingBS', 'Oldpeaksignificant']

X = df_f[categorical_variable + numerical_feature]
y = df_f['HeartDisease']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_feature),
        ('cat', OneHotEncoder(), categorical_variable)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=20))
])

pipeline.fit(X, y)

# Streamlit App
st.title("Heart Disease Prediction")

# Sidebar with user input
st.sidebar.header("User Input")
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", df_f['ChestPainType'].unique())
resting_ecg = st.sidebar.selectbox("Resting Electrocardiographic Result", df_f['RestingECG'].unique())
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", df_f['ExerciseAngina'].unique())
st_slope = st.sidebar.selectbox("ST Segment Slope", df_f['ST_Slope'].unique())
rest_bp_category = st.sidebar.selectbox("Resting Blood Pressure Category", df_f['RestBP_Category'].unique())
cholesterol_category = st.sidebar.selectbox("Cholesterol Category", df_f['CholesterolCategory'].unique())
max_hr_category = st.sidebar.selectbox("Maximum Heart Rate Category", df_f['MaxHRCategory'].unique())
sex = st.sidebar.selectbox("Gender", ['Female', 'Male'])
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar", ['Normal', 'Elevated'])
oldpeak_significant = st.sidebar.slider("Oldpeak Significant", min_value=0.0, max_value=6.0, step=0.1)

# Submit button to trigger prediction
submit_button = st.sidebar.button("Submit")

# Map categorical choices to numeric values
sex_mapping = {'Female': 0, 'Male': 1}
fasting_bs_mapping = {'Normal': 0, 'Elevated': 1}

# Transform user input
user_input = pd.DataFrame({
    'ChestPainType': [chest_pain_type],
    'RestingECG': [resting_ecg],
    'ExerciseAngina': [exercise_angina],
    'ST_Slope': [st_slope],
    'RestBP_Category': [rest_bp_category],
    'CholesterolCategory': [cholesterol_category],
    'MaxHRCategory': [max_hr_category],
    'Sex': [sex_mapping[sex]],
    'FastingBS': [fasting_bs_mapping[fasting_bs]],
    'Oldpeaksignificant': [oldpeak_significant]
})

# Make prediction when the "Submit" button is clicked
if submit_button:
    prediction = pipeline.predict(user_input)

    # Display prediction
    st.subheader("Prediction:")
    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")
#C:\Users\WAQSA\anaconda3\Scripts\streamlit.exe run "C:\Users\WAQSA\Desktop\TTDS Course Project\CoursePoject.py"