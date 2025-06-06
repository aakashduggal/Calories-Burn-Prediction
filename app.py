import streamlit as st
import numpy as np
import pandas as pd
import pickle


rfr = pickle.load(open('rfr.pkl','rb'))
x_train = pd.read_csv('x_train.csv')

def pred(Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp):
    features = np.array([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]])
    prediction = rfr.predict(features).reshape(1,-1)
    return prediction[0]


st.title("Calories Burn Prediction")
Gender = st.selectbox('Gender',x_train['Gender'])
Age = st.selectbox('Age',x_train['Age'])
Height = st.selectbox('Height',x_train['Height'])
Weight = st.selectbox('Weight',x_train['Weight'])
Duration = st.selectbox('Duration (minutes)',x_train['Duration'])
Heart_Rate = st.selectbox('Heart Rate (bpn)',x_train['Heart_Rate'])
Body_Temp = st.selectbox('Body Temp',x_train['Body_Temp'])

result = pred(Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp)

if st.button('predict'):
    if result:
        st.write("You have consumed this calories : ", result)