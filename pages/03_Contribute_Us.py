import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.title("Contribute to our Dataset 👇")
st.markdown("<br>",True)   
sal=st.radio("**Select the Disease ?**",["Liver","Kidney"])
if sal=="Liver":
    ex=st.number_input("**ENTER YOUR AGE FOR LIVER DISEASE**",0,120)
    if st.button("Submit"):

      to_add= {"Age":ex,"dis":sal}
      to_add = pd.DataFrame([to_add])
      to_add.to_csv("New_Liver_Data.csv",mode="a" ,header=False,index=False)
      st.success("Submitted")

    st.markdown("""<br> ***---------------------GRAPH DEPICTING LIVER DISEASE AMONG DIFFERENT AGE GROUPS----------------*** """,True) 
    data=pd.read_csv("New_Liver_Data.csv")
    plt.figure(figsize=(8,4))
    fig,ax=plt.subplots()
    ax.scatter(data["Disease"],data["Age"])
    plt.ylim(0)
    plt.xlabel("Disease")
    plt.ylabel("Age")
    plt.tight_layout()
    st.pyplot(fig)


if sal=="Kidney":
    ex=st.number_input("ENTER YOUR AGE FOR KIDNEY DISEASE",0,120)
    if st.button("Submit"):

      to_add= {"Age":ex,"dis":sal}
      to_add = pd.DataFrame([to_add])
      to_add.to_csv("New_Kidney_Data.csv",mode="a" ,header=False,index=False)
      st.success("Submitted")
    st.markdown("""<br> ---------------------GRAPH DEPICTING PARK DISEASE AMONG DIFFERENT AGE GROUPS----------------""",True) 
    
    data=pd.read_csv("New_Kidney_Data.csv")
    plt.figure(figsize=(8,4))
    fig,ax=plt.subplots()
    ax.scatter(data["Disease"],data["Age"])
    plt.ylim(0)
    plt.xlabel("Disease")
    plt.ylabel("Age")
    plt.tight_layout()
    st.pyplot(fig)
