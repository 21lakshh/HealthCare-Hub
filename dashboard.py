import pickle 
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import numpy as np
from PIL import Image
from functions import make_prediction_for_brain_tumor_detection
from functions import preprocess_image_for_brain_tumor_detection
from functions import load_and_preprocess_image
from functions import make_prediction
import cv2

# loading the bone fracture model
working_dir = os.path.dirname(os.path.abspath(__file__)) 
model_path = f"{working_dir}/bonefracture_model.h5"
bonefracture_model = tf.keras.models.load_model(model_path)

# loading the diabetes model 
diabetes_model = pickle.load(open(f'{working_dir}/diabetes_trained_model.sav','rb'))
diabetes_scalar = pickle.load(open(f'{working_dir}/diabetes_scaler.sav','rb'))
breastcancer_scaler = pickle.load(open(f'{working_dir}/breastcancer_scaler.sav','rb'))

# loading the heart disease model 
heart_disease_model = pickle.load(open(f'{working_dir}/heart_disease_model.sav','rb'))

# loading the calories burnt model 
calories_burnt_model = pickle.load(open(f'{working_dir}/caloriesburnt_training_model.sav','rb'))

# loading the asthma model 
asthma_model = pickle.load(open(f'{working_dir}/asthma_model.sav','rb'))

# loading the medical insurance model 
medicalinsurance_model = pickle.load(open(f'{working_dir}/medicalinsurance_trained_model.sav','rb'))

# loading the breast cancer model 
model_path = f'{working_dir}/breastcancer_model.h5'
breastcancer_model = tf.keras.models.load_model(model_path)


brain_tumor_model_path = f'{working_dir}/braintumour_detection.h5'
brain_tumor_model = tf.keras.models.load_model(brain_tumor_model_path)

# sidebar for navigation 
with st.sidebar:
    selected = option_menu('Dashboard',
                           
                           ['Bone Fracture X-ray Scan',
                            'Brain Tumor Detection 🧠',  # Added brain emoji
                            'Breast Cancer Prediction',
                            'Asthma Prediction',
                            'Heart Disease Prediction',
                            'Diabetes Prediction',
                            'Calories Burn Prediction',
                            'Medical Insurance'],

                            icons=['shield', 'person-circle', 'person-standing-dress', 'lungs',
                                   'heart-pulse', 'activity', 'fire', 'hospital'],
                            default_index=0)  
 
if selected == 'Bone Fracture X-ray Scan':
    st.title('X-Ray Scan For Bone Fracture Detection for upper extremities:')

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)

        with col2:
            if st.button('Classify'):
                # Preprocess the uploaded image and predict the class
                processed_image = load_and_preprocess_image(uploaded_image)
                prediction = bonefracture_model.predict(processed_image)[0]

                if prediction[0] > 0.5:
                    st.success('Not Fractured')
                else:
                    st.success('Fractured')

if selected == 'Brain Tumor Detection 🧠':
    st.title('MRI Scan for Brain Tumor Detection 🧠:')

    uploaded_file_brain = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])


    if uploaded_file_brain is not None:
        image = Image.open(uploaded_file_brain)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)
        
        if col2.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                processed_image = preprocess_image_for_brain_tumor_detection(uploaded_file_brain)
                prediction = make_prediction_for_brain_tumor_detection(processed_image)
                st.session_state.prediction = prediction
                st.success(f"Prediction: {prediction}")  # Display result


if selected == 'Breast Cancer Prediction':

    st.title("Breast Cancer Prediction")

    # Grouping inputs into columns
    col1, col2, col3 = st.columns(3)

    with col1:
        mean_radius = st.number_input("Mean Radius", min_value=0.0, step=0.01)
        mean_compactness = st.number_input("Mean Compactness", min_value=0.0, step=0.01)
        mean_symmetry = st.number_input("Mean Symmetry", min_value=0.0, step=0.01)

    with col2:
        mean_texture = st.number_input("Mean Texture", min_value=0.0, step=0.01)
        mean_concavity = st.number_input("Mean Concavity", min_value=0.0, step=0.01)
        mean_fractal_dimension = st.number_input("Mean Fractal Dimension", min_value=0.0, step=0.01)

    with col3:
        mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, step=0.01)
        mean_area = st.number_input("Mean Area", min_value=0.0, step=0.01)
        mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, step=0.01)

    col4, col5, col6 = st.columns(3)

    with col4:
        mean_concave_points = st.number_input("Mean Concave Points", min_value=0.0, step=0.01)
        radius_error = st.number_input("Radius Error", min_value=0.0, step=0.01)
        compactness_error = st.number_input("Compactness Error", min_value=0.0, step=0.01)

    with col5:
        symmetry_error = st.number_input("Symmetry Error", min_value=0.0, step=0.01)
        texture_error = st.number_input("Texture Error", min_value=0.0, step=0.01)
        concavity_error = st.number_input("Concavity Error", min_value=0.0, step=0.01)

    with col6:
        fractal_dimension_error = st.number_input("Fractal Dimension Error", min_value=0.0, step=0.01)
        perimeter_error = st.number_input("Perimeter Error", min_value=0.0, step=0.01)
        area_error = st.number_input("Area Error", min_value=0.0, step=0.01)

    col7, col8, col9 = st.columns(3)

    with col7:
        smoothness_error = st.number_input("Smoothness Error", min_value=0.0, step=0.01)
        concave_points_error = st.number_input("Concave Points Error", min_value=0.0, step=0.01)
        worst_radius = st.number_input("Worst Radius", min_value=0.0, step=0.01)

    with col8:
        worst_compactness = st.number_input("Worst Compactness", min_value=0.0, step=0.01)
        worst_symmetry = st.number_input("Worst Symmetry", min_value=0.0, step=0.0)
        worst_texture = st.number_input("Worst Texture", min_value=0.0, step=0.01)

    with col9:
        worst_concavity = st.number_input("Worst Concavity", min_value=0.0, step=0.01)
        worst_fractal_dimension = st.number_input("Worst Fractal Dimension", min_value=0.0, step=0.01)
        worst_perimeter = st.number_input("Worst Perimeter", min_value=0.0, step=0.01)
    
    col10 = st.columns(1)[0]

    with col10:
        worst_area = st.number_input("Worst Area", min_value=0.0, step=0.01)
        worst_smoothness = st.number_input("Worst Smoothness", min_value=0.0, step=0.01)
        worst_concave_points = st.number_input("Worst Concave Points", min_value=0.0, step=0.01)

    if st.button('Result'):
        input_data = [mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension]

        # changing the input data to a numpy array
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        std_data = breastcancer_scaler.transform(input_data_reshaped)
        prediction = breastcancer_model.predict(std_data)
        prediction_label = np.argmax(prediction)

        if (prediction_label == 0):
            st.success('The tumor is Malignant')
        else:
            st.success('The tumor is Benign')


if selected == 'Asthma Prediction':
    st.title('Asthma Prediction')

    Age = st.number_input('Age of the person', min_value=0, step=1)
    Gender = st.selectbox('Gender of the person', ['None', 'Male', 'Female'])
    Ethnicity = st.number_input('Ethnicity of the person',min_value = 0,step = 1)
    BMI = st.number_input('BMI', min_value=0.0, step=0.1)
    Smoking = st.selectbox('Smoking Status', ['No', 'Yes'])
    PhysicalActivity = st.number_input('Physical Activity Level', min_value=0.0, step=0.1)
    DietQuality = st.number_input('Diet Quality', min_value=0.0, step=0.1)
    SleepQuality = st.number_input('Sleep Quality', min_value=0.0, step=0.1)
    PollutionExposure = st.number_input('Pollution Exposure',min_value=0.0, step=0.1)
    PollenExposure = st.number_input('Pollen Exposure', min_value=0.0, step=0.1)
    DustExposure = st.number_input('Dust Exposure', min_value=0.0, step=0.1)
    PetAllergy = st.selectbox('Pet Allergy', ['None', 'No', 'Yes'])
    FamilyHistoryAsthma = st.selectbox('Family History of Asthma', ['No', 'Yes'])
    HistoryOfAllergies = st.selectbox('History of Allergies', ['No', 'Yes'])
    Eczema = st.selectbox('Eczema', ['No', 'Yes'])
    HayFever = st.selectbox('Hay Fever', ['No', 'Yes'])
    GastroesophagealReflux = st.selectbox('Gastroesophageal Reflux', ['No', 'Yes'])
    LungFunctionFEV1 = st.number_input('Lung Function FEV1 (L)', min_value=0.0, step=0.1)
    LungFunctionFVC = st.number_input('Lung Function FVC (L)', min_value=0.0, step=0.1)
    Wheezing = st.selectbox('Wheezing', ['No', 'Yes'])
    ShortnessOfBreath = st.selectbox('Shortness of Breath', ['No', 'Yes'])
    ChestTightness = st.selectbox('Chest Tightness', ['No', 'Yes'])
    Coughing = st.selectbox('Coughing', ['No', 'Yes'])
    NighttimeSymptoms = st.selectbox('Nighttime Symptoms', ['No', 'Yes'])
    ExerciseInduced = st.selectbox('Exercise-Induced Symptoms', ['No', 'Yes'])

    if Gender == 'None':
        st.warning("Please select all options before proceeding.")
    else:
        gender_mapping = {'Male': 1, 'Female': 0}
        yesno_mapping = {'Yes': 1, 'No': 0}

        Gender = gender_mapping[Gender]
        Smoking = yesno_mapping[Smoking]
        PetAllergy = yesno_mapping[PetAllergy]
        FamilyHistoryAsthma = yesno_mapping[FamilyHistoryAsthma]
        HistoryOfAllergies = yesno_mapping[HistoryOfAllergies]
        Eczema = yesno_mapping[Eczema]
        HayFever = yesno_mapping[HayFever]
        GastroesophagealReflux = yesno_mapping[GastroesophagealReflux]
        Wheezing = yesno_mapping[Wheezing]
        ShortnessOfBreath = yesno_mapping[ShortnessOfBreath]
        ChestTightness = yesno_mapping[ChestTightness]
        Coughing = yesno_mapping[Coughing]
        NighttimeSymptoms = yesno_mapping[NighttimeSymptoms]
        ExerciseInduced = yesno_mapping[ExerciseInduced]
    

    if st.button('Result'):
        input_data = []
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = asthma_model.predict(input_data_reshaped)
        if prediction[0] == 1:
            st.success('Person has Asthma')
        else:
            st.success('Person Doesnt Have Asthma')




if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')

    Age = st.number_input('Age of the person', min_value=0, step=1)
    Sex = st.selectbox('Gender of the person', ['None','Male', 'Female'])
    cptype = st.number_input('Chest Pain Type', min_value=0, step=1)
    restingbloodpressure = st.number_input('Resting Blood Pressure', min_value=0, step=1)
    serumcholestrol = st.number_input('Serum Cholestoral in mg/dl', min_value=0, step=1)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl',['None','Yes', 'No'])
    restingelectrocardiographicresults = st.number_input('Resting Electrocardiographic results', min_value=0, step=1)
    max_heart_rate = st.number_input('Max Heart Rate Achieved',min_value=0, step=1)
    exerciseinducedangina = st.number_input('Exercise induced angina',min_value=0, step=1)
    oldpeak = st.number_input('ST depression induced by exercise relative to rest',min_value=0.0, format="%.1f")
    slope = st.number_input('The slope of the peak exercise ST segment',min_value=0, step=1)
    majorvessels = st.number_input('Number of major vessels colored by flourosopy',min_value=0, step=1)
    thal = st.selectbox('Thal',['None','Normal','Fixed Defect','Reversable Defect'])


    if Sex == 'None' or fbs == 'None' or thal == 'None':
        st.warning("Please select all options before proceeding.")
    else:
        # Map strings to numbers
        sex_mapping = {'Male': 1, 'Female': 0}
        fbs_mapping = {'Yes': 1, 'No': 0}
        thal_mapping = {'Normal': 0, 'Fixed Defect': 1, 'Reversable Defect': 2}

        Sex = sex_mapping[Sex]
        fbs = fbs_mapping[fbs]
        thal = thal_mapping[thal]
    if st.button('Result'):
        input_data = [Age,Sex,cptype,restingbloodpressure,serumcholestrol,fbs,restingelectrocardiographicresults,max_heart_rate,exerciseinducedangina,oldpeak,slope,majorvessels,thal]
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = heart_disease_model.predict(input_data_reshaped)
        if prediction[0] == 1:
            st.success('Person has a Heart Diease')
        else:
            st.success('Person Doesnt Have a Heart Diease')

if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')

    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
    Glucose = st.number_input('Glucose Level', min_value=0)
    BloodPressure = st.number_input('Blood Pressure value', min_value=0)
    SkinThickness = st.number_input('Skin Thickness value', min_value=0)
    Insulin = st.number_input('Insulin Level', min_value=0)
    BMI = st.number_input('BMI', min_value=0.0, format="%.1f")
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, format="%.3f")
    Age = st.number_input('Age of the person', min_value=0, step=1)

    diagnosis = ''

    if st.button('Result'):
        diagnosis = make_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    st.success(diagnosis)

if selected == 'Calories Burn Prediction':
    st.title('Calories Burn Prediction')

    Gender = st.selectbox('Gender of the person', ['None','Male', 'Female'])
    Age = st.number_input('Age of the person', min_value=0, step=1)
    Height = st.number_input('Height of the person',min_value=0,step=1)
    Weight = st.number_input('Weight of the person',min_value=0,step=1)
    Duration = st.number_input('Duration of exercise',min_value=0,step=1)
    Heart_rate = st.number_input('Heart Rate',min_value=0,step=1)
    Body_temp = st.number_input('Body Tempurature ',min_value=0.0,format="%.1f")

    if Gender == 'None':
        st.warning("Please select all options before proceeding.")
    else:
        # Map strings to numbers
        gender_mapping = {'Male': 1, 'Female': 0}
        Gender = gender_mapping[Gender]

    if st.button('Check how many calories you burnt!'):
        input_data = [Gender,Age,Height,Weight,Duration,Heart_rate,Body_temp]
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)

        calories_burnt = calories_burnt_model.predict(input_data_reshaped)
        st.success(f"Calories Burnt: {calories_burnt[0]}")

if selected == 'Medical Insurance':
    st.title('Check cost of your Medical Insurance')

    Age = st.number_input('Age of the person', min_value=0, step=1)
    Sex = st.selectbox('Gender of the person', ['None','Male', 'Female'])
    BMI = st.number_input('BMI', min_value=0.0, format="%.2f")
    Children = st.number_input('No of Children', min_value=0, step=1)
    Smoker = st.selectbox('Smoker status', ['None','Yes', 'No'])
    Region = st.selectbox('Region', ['None','southeast', 'southwest', 'northeast', 'northwest'])

    if Sex == 'None' or Smoker == 'None' or Region == 'None':
        st.warning("Please select all options before proceeding.")
    else:
        # Map strings to numbers
        sex_mapping = {'Male': 1, 'Female': 0}
        smoker_mapping = {'Yes': 1, 'No': 0}
        region_mapping = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}

        # Convert inputs to numerical values
        Sex = sex_mapping[Sex]
        Smoker = smoker_mapping[Smoker]
        Region = region_mapping[Region]

    if st.button('Predict'):
        prediction = medicalinsurance_model.predict([[Age,Sex,BMI,Children,Smoker,Region]])
        st.success(f"Your Insurance Cost Will be: {prediction[0]}")

        # "C:\Users\LAKSHYA PALIWAL\AppData\Local\Programs\Python\Python310\Scripts\streamlit.exe"
