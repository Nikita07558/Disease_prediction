#0-chronic kidney disease , 1- no kidney disease
#numpy version this can work with 1.26.4
#scikit-learn version 1.3.0
#pandas 2.0.2
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_model():
    with open("kidney_model.pkl", 'rb') as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open("encoders.pkl", 'rb') as encoder_file:
        encoders = pickle.load(encoder_file)
    return model, scaler, encoders

# Load model, scaler, and encoders
model, scaler, encoders = load_model()

# Streamlit UI
st.title("Kidney Disease Prediction")
st.write("Enter patient details to predict kidney disease.")

# Define input fields
columns = ['hemo', 'pcv', 'sc', 'rc', 'sg', 'al','sod', 'htn','pot','dm' , 'bu', 'bgr']
input_data = {}
for col in columns:
    if col in encoders:
        input_data[col] = st.selectbox(f"{col}", options=encoders[col].classes_)
    else:
        input_data[col] = st.number_input(f"{col}", min_value=0.0, format="%.2f")

# Convert inputs to dataframe
input_df = pd.DataFrame([input_data])

# Encode categorical features
for col in encoders:
    if col in input_df:
        input_df[col] = encoders[col].transform(input_df[col])

# Scale numerical features
input_df_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df_scaled)
    result = "No Kidney Disease Detected" if prediction[0] == 1 else "Chronic Kidney Disease"
    st.write(f"### Prediction: {result}")





















#nyi ..but oldr version plus input taking in a loop 
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# def load_model():
#     with open('kidney_model.pkl', 'rb') as model_file:
#         model = pickle.load(model_file)
#     with open('scaler.pkl', 'rb') as scaler_file:
#         scaler = pickle.load(scaler_file)
#     with open('encoders.pkl', 'rb') as encoder_file:
#         encoders = pickle.load(encoder_file)
#     return model, scaler, encoders

# # Load model, scaler, and encoders
# model, scaler, encoders = load_model()

# # Define input fields
# categorical_features = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
# numerical_features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
# all_features = categorical_features + numerical_features

# # Streamlit UI
# st.title("Kidney Disease Prediction")
# st.write("Enter patient details to predict kidney disease.")

# # Input collection
# input_data = {}
# for col in numerical_features:
#     input_data[col] = st.number_input(f"{col}", min_value=0.0, format="%.2f")
# for col in categorical_features:
#     input_data[col] = st.selectbox(f"{col}", options=encoders[col].classes_)

# # Convert inputs to dataframe
# input_df = pd.DataFrame([input_data])

# # Encode categorical features
# for col in categorical_features:
#     input_df[col] = encoders[col].transform(input_df[col])

# # Scale numerical features
# input_df_scaled = scaler.transform(input_df)

# # Prediction
# if st.button("Predict"):
#     prediction = model.predict(input_df_scaled)
#     result = "Kidney Disease Detected" if prediction[0] == 1 else "No Kidney Disease"
#     st.write(f"### Prediction: {result}")












# nyi vsli but usind older versions of numpy.pands and sk learn so tha diikkat na aye
# import numpy as np
# import pandas as pd
# import streamlit as st
# import pickle

# # Ensure compatibility with NumPy 1.26.4, Pandas 2.0.2, and Scikit-learn 1.3.0

# # Load trained model, scaler, and label encoder
# with open("kidney_model (1).pkl", "rb") as model_file:
#     loaded_model = pickle.load(model_file)
# with open("scaler (1).pkl", 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)
# with open("encoder (1).pkl", 'rb') as encoder_file:
#     label_encoder = pickle.load(encoder_file)

# # Streamlit UI
# st.title("Kidney Disease Prediction")
# st.markdown("**Enter patient details to get a prediction**")

# # User Inputs
# age = st.number_input("Age", min_value=1, max_value=100, step=1, format="%i")
# gender_str = st.radio("Gender", ['Female', 'Male'])
# bp = st.number_input("Blood Pressure (mmHg)", min_value=0.0)
# sgr = st.number_input("Sugar Level", min_value=0.0)
# al = st.number_input("Albumin Level", min_value=0.0)
# sc = st.number_input("Serum Creatinine", min_value=0.0)
# bgr = st.number_input("Blood Glucose Random", min_value=0.0)
# wbcc = st.number_input("White Blood Cell Count", min_value=0.0)
# rbc_str = st.radio("Red Blood Cells (Normal/Abnormal)", ['Normal', 'Abnormal'])

# # Encode categorical inputs
# try:
#     gender_encoded = label_encoder.transform([gender_str])[0]
#     rbc_encoded = label_encoder.transform([rbc_str])[0]
# except ValueError:
#     gender_encoded = 0  # Default value if unseen category
#     rbc_encoded = 0  # Default value if unseen category

# # Prepare input array
# kidney_array = np.array([[age, gender_encoded, bp, sgr, al, sc, bgr, wbcc, rbc_encoded]], dtype=np.float64)
# kidney_scaled = scaler.transform(kidney_array)  # Apply same scaling as training

# # Make prediction
# if st.button("Predict"):
#     probability = loaded_model.predict_proba(kidney_scaled)[0][1]  # Probability of kidney disease
#     st.success(f"Prediction Probability: {round(probability, 4)}")
#     if probability < 0.5:
#         st.write("Kidney disease is **unlikely**.")
#     else:
#         st.write("Kidney disease is **likely**.")


























#new one but newer version of panda,sklearn,numpy se bna so liver vali me dikkat aa rhi thi agar hm newer versio se krte so not using this se bn gya

# import numpy as np
# import pandas as pd
# import streamlit as st
# import pickle

# # Load trained model, scaler, and label encoder
# with open("kidney_model.pkl", "rb") as model_file:
#     loaded_model = pickle.load(model_file)
# with open("scaler.pkl", 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)
# with open("encoder.pkl", 'rb') as encoder_file:
#     label_encoder = pickle.load(encoder_file)

# # Streamlit UI
# st.title("Kidney Disease Prediction")
# st.markdown("**Enter details to get a prediction**")

# # User Inputs
# age = st.number_input("Age", min_value=1, max_value=100, step=1, format="%i")
# gender_str = st.radio("Gender", ['Female', 'Male'])
# bp = st.number_input("Blood Pressure (mmHg)", min_value=0.0)
# sgr = st.number_input("Sugar Level", min_value=0.0)
# al = st.number_input("Albumin Level", min_value=0.0)
# sc = st.number_input("Serum Creatinine", min_value=0.0)
# bgr = st.number_input("Blood Glucose Random", min_value=0.0)
# wbcc = st.number_input("White Blood Cell Count", min_value=0.0)
# rbc_str = st.radio("Red Blood Cells (Normal/Abnormal)", ['Normal', 'Abnormal'])

# # Encode categorical inputs
# gender_encoded = label_encoder.transform([gender_str])[0]
# rbc_encoded = label_encoder.transform([rbc_str])[0]

# # Prepare input array
# kidney_array = np.array([[age, gender_encoded, bp, sgr, al, sc, bgr, wbcc, rbc_encoded]])
# kidney_scaled = scaler.transform(kidney_array)  # Apply same scaling as training

# # Make prediction
# if st.button("Predict"):
#     probability = loaded_model.predict_proba(kidney_scaled)[0][1]  # Probability of kidney disease
#     st.success(f"Prediction: {round(probability, 4)}")
#     if probability < 0.5:
#         st.write("Kidney disease is **unlikely**.")
#     else:
#         st.write("Kidney disease is **likely**.")























#old vala of no use

# import numpy as np
# import streamlit as st
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import PolynomialFeatures 
# import pickle

# with open('model_filename (5).pkl', 'rb') as model_file:
#     loaded_model = pickle.load(model_file)

# one,two=st.columns((9,2))
# one.title("Kidney Disease Prediction:")
# two.image('https://ourhardworkingkidneys.com/wp-content/uploads/2017/12/91ca8ce1.png',width=170)
# st.markdown("""<br>""",True)

# one,two,three=st.columns(3)
# age=one.number_input("Enter age: ")
# blood_pressure=two.number_input("Enter blood pressure: ")
# specific_gravity = three.number_input("Enter specific gravity: ")
# albumin =one.number_input("Enter albumin: ")
# sugar =two.number_input("Enter sugar: ")
# red_blood_cells =three.number_input("Enter red blood cell status (1 for normal/0 for abnormal): ")
# pus_cell =one.number_input("Enter pus cell status (1 for normal/0 for abnormal): ")
# pus_cell_clumps =two.number_input("Enter pus cell clumps status (1 for notpresent/0 for present): ")
# bacteria =three.number_input("Enter bacteria status (0 for notpresent/1 for present): ")
# blood_glucose_random = one.number_input("Enter blood glucose random: ")
# blood_urea =two.number_input("Enter blood urea: ")
# serum_creatinine =three.number_input("Enter serum creatinine: ")
# sodium = one.number_input("Enter sodium: ")
# potassium =one.number_input("Enter potassium: ")
# haemoglobin =two.number_input("Enter haemoglobin: ")
# packed_cell_volume =three.number_input("Enter packed cell volume: ")
# white_blood_cell_count =one.number_input("Enter white blood cell count: ")
# red_blood_cell_count =two.number_input("Enter red blood cell count: ")
# hypertension =three.number_input("Enter hypertension status (1 for yes/0 for no): ")
# diabetes_mellitus =one.number_input("Enter diabetes mellitus status (2 for yes/1 for no): ")
# coronary_artery_disease =two.number_input("Enter coronary artery disease status (1 for yes/0 for no): ")
# appetite =three.number_input("Enter appetite status (0 for good/1 for poor): ")
# peda_edema =one.number_input("Enter pedal edema status (1 for yes/0 for no): ")
# aanemia =two.number_input("Enter anemia status (1 for yes/0 for no): ")
# user_input = np.array([[red_blood_cells,pus_cell,pus_cell_clumps,bacteria,hypertension,diabetes_mellitus,
#     coronary_artery_disease,appetite,peda_edema,aanemia,age,
#     blood_pressure,specific_gravity,albumin,sugar,
#     blood_glucose_random,blood_urea,serum_creatinine,sodium,potassium,
#     haemoglobin,packed_cell_volume,white_blood_cell_count,red_blood_cell_count]])

# # user_input_np = np.asarray(user_input)
# # res_input_unsc = user_input_np.reshape(1,-1)
# # scaler=StandardScaler()
# # res_input = scaler.fit_transform(res_input_unsc)
# # probability_of_ckd = loaded_model.predict_proba(res_input)[:, 0]
# # prediction = loaded_model.predict(res_input)
# # percentage_probability = probability_of_ckd * 100

# # st.success(f"Prediction {round(prediction)}")
# # st.markdown("""<br>""",True)
# # st.write("The probability of having Chronic Kidney Disease (CKD) is ")
# # st.write(percentage_probability)


