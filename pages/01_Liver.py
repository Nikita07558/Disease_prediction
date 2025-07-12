#1- healthy , 2-cirrohsis
#numpy version this can work with 1.26.4
#scikit-learn version 1.3.0
#pandas 2.0.2
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load trained pipeline model
with open("best_liver_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("🧫 Liver Cirrhosis Prediction")
st.markdown("Enter patient details to predict liver cirrhosis risk.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, step=1)
gender = st.radio("Gender", ["Male", "Female"])
tb = st.number_input("Total Bilirubin")
db = st.number_input("Direct Bilirubin")
alkphos = st.number_input("Alkaline Phosphotase")
sgpt = st.number_input("Sgpt (Alamine Aminotransferase)")
sgot = st.number_input("Sgot (Aspartate Aminotransferase)")
tp = st.number_input("Total Proteins")
alb = st.number_input("Albumin")
agr = st.number_input("Albumin and Globulin Ratio")

# Encode gender manually (as done in training)
gender_encoded = 1 if gender == "Male" else 0

# Final input for prediction
input_data = np.array([[age, gender_encoded, tb, db, alkphos, sgpt, sgot, tp, alb, agr]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.success(f"✅ No Liver Cirrhosis Detected (Probability: {proba:.2f})")
    else:
        st.error(f"⚠️ Liver Cirrhosis Likely (Probability: {proba:.2f})")




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

