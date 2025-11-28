import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Titanic Survival Predictor")

df = pd.read_csv("train.csv")

with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.subheader("Input Passenger Features")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=30.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

sex_map = {"male": 0, "female": 1}
embark_map = {"S": 0, "C": 1, "Q": 2}

input_row = np.array([
    pclass,
    sex_map[sex],
    age,
    sibsp,
    parch,
    fare,
    embark_map[embarked]
]).reshape(1, -1)

if st.button("Predict Survival"):
    pred = model.predict(input_row)[0]
    proba = model.predict_proba(input_row)[0][1]

    if pred == 1:
        st.success(f"Survived (probability: {proba:.2f})")
    else:
        st.error(f"Did NOT survive (probability: {proba:.2f})")

st.subheader("Dataset Preview")
st.write(df.head())
