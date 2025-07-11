#1- healthy , 2-cirrohsis
#numpy version this can work with 1.26.4
#scikit-learn version 1.3.0
#pandas 2.0.2
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer

# Load trained model
with open('model_filename (3).pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load dataset
liver_file = pd.read_csv("Liver cirrhosis UCI Dataset.csv")

# Define feature columns
cols = ['AGE', 'GENDER', 'TB(Total Bilirubin)', 'DB(Direct Bilirubin)',
        'Alkphos Alkaline Phosphotase', 'Sgpt  Alamine Aminotransferase',
        'Sgot Aspartate Aminotransferase', 'TP Total Protiens',
        'ALB Albumin', 'A/G Ratio Albumin and Globulin Ratio']

# Encode GENDER column
label_encoder = LabelEncoder()
liver_file['GENDER'] = label_encoder.fit_transform(liver_file['GENDER'].astype(str))

# Handle missing values
imputer = SimpleImputer(strategy="mean")
liver_file[cols] = imputer.fit_transform(liver_file[cols])

# Apply PolynomialFeatures if it was used in training
poly = PolynomialFeatures(degree=1)  # If used during training
liver_poly = poly.fit_transform(liver_file[cols])

# Fit StandardScaler on transformed features
scaler = StandardScaler()
scaler.fit(liver_poly)

# Streamlit UI
st.title("Liver Cirrhosis Prediction")
st.markdown("**Enter details to get a prediction**")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=100, step=1, format="%i")
gender_str = st.radio("Gender", ['Female', 'Male'])
tb = st.number_input("Total Bilirubin", min_value=0.0)
db = st.number_input("Direct Bilirubin", min_value=0.0)
aap = st.number_input("Alkphos Alkaline Phosphotase", min_value=0.0)
saa = st.number_input("Sgpt Alamine Aminotransferase", min_value=0.0)
sgaa = st.number_input("Sgot Aspartate Aminotransferase", min_value=0.0)
ttp = st.number_input("TP Total Proteins", min_value=0.0)
aa = st.number_input("ALB Albumin", min_value=0.0)
ragr = st.number_input("A/G Ratio Albumin and Globulin Ratio", min_value=0.0)
# Encode gender input
gender_encoded = label_encoder.transform([gender_str])[0]

# Prepare input array
liver_array = np.array([[age, gender_encoded, tb, db, aap, saa, sgaa, ttp, aa, ragr]])

# Apply the same transformations as during training
liver_poly_test = poly.transform(liver_array)  # Ensures 11 features
liver_scaled = scaler.transform(liver_poly_test)  # Scales the data correctly

# Make prediction
if st.button("Predict"):
    probability = loaded_model.predict_proba(liver_scaled)[0][1]  # Probability of cirrhosis
    st.success(f"Prediction: {round(probability, 4)}")
    if probability < 0.1204:
        st.write("Liver cirrhosis is **unlikely**.")
    else:
        st.write("Liver cirrhosis is **likely**.")



# used technologies

# StackingClassifier(estimators=[('logistic',
#                                 LogisticRegression(C=1, max_iter=100000,
#                                                    solver='liblinear')),
#                                ('svm', SVC(C=100)),
#                                ('gb',
#                                 GradientBoostingClassifier(max_depth=2,
#                                                            max_features='sqrt',
#                                                            min_samples_split=4)),
#                                ('dt',
#                                 DecisionTreeClassifier(max_depth=2,
#                                                        min_samples_split=4))],
#                    final_estimator=LogisticRegression(C=1, max_iter=100000,
#                                                       solver='liblinear'))



#old model

# import numpy as np
# import pandas as pd
# import streamlit as st
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import PolynomialFeatures 
# import pickle

# with open('model_filename (3).pkl', 'rb') as model_file:
#     loaded_model = pickle.load(model_file)
# one,two=st.columns((9,2.6))

# liver_file=pd.read_csv("Liver cirrhosis UCI Dataset.csv")
# cols=['AGE','GENDER','TB(Total Bilirubin)','DB(Direct Bilirubin)','Alkphos Alkaline Phosphotase','Sgpt  Alamine Aminotransferase','Sgot Aspartate Aminotransferase','TP Total Protiens','ALB Albumin','A/G Ratio Albumin and Globulin Ratio']
# gen_col=['GENDER']


# scaler=StandardScaler()
# poly = PolynomialFeatures(degree=1)


# one.title("Liver Cirrhosis Prediction:")
# st.markdown(":red[**Enter the following values in order to get the prediction------>**]")
# two.image('https://th.bing.com/th/id/R.44bd49dbc357f02fa2ddb72f3ff7d6d6?rik=w0%2b%2bH2o7KMn3VQ&riu=http%3a%2f%2fniroginepal.com%2fwp-content%2fuploads%2f2016%2f04%2fCirrhosis-of-Liver.png&ehk=9d4MFJTn5uJ4A5qbNQJVlX0eSzJkQ63vE3DQ3n4OTGs%3d&risl=&pid=ImgRaw&r=0',width=290)
# st.markdown("""<br>""",True)

# one,two,three=st.columns(3)
# age=one.number_input(" Age",format="%i")
# gender_str = str(two.radio("Gender", ['Female', 'Male']))

# tb=three.number_input("Total Bilirubin")
# db=one.number_input("  Direct Bilirubin")
# aap=two.number_input("Alkphos Alkaline Phosphotase")
# saa=three.number_input("Sgpt Alamine Aminotransferas")
# sgaa=one.number_input("Sgot Aspartate Aminotransferase")
# ttp=two.number_input("TP Total Protiens")
# aa=three.number_input("ALB Albumin")
# ragr=two.number_input("A/G Ratio Albumin and Globulin Ratio")

# label_encoder= LabelEncoder()
# label_encoder.fit(liver_file["GENDER"])
# gender_encoded = label_encoder.transform([gender_str])[0]

# poly.fit(liver_file[cols])
# scaler.fit(liver_file[cols])
# liver_array=np.array([[age,gender_encoded,tb,db,aap,saa,sgaa,ttp,aa,ragr]])


# liver_poly=poly.transform(liver_array)

# liver_scale=scaler.transform(liver_poly)
# probability = loaded_model.predict_proba(liver_scale)[0][1]
# if st.button("Predict: "):
#     st.success(f"Prediction {round(probability)}")
#     st.write("If the value of predicted probability is less than 0.1204 then liver cirrhosis doesnt exists otherwise it does")

