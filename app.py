import streamlit as st
import pandas as pd
import pickle
import re

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("Titanic Survival Predictor")

# Load model + data preview
df = pd.read_csv("train.csv")

with open("titanic-model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature lists (must match training)
categorical_features = ["Sex", "Embarked", "Title"]
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]

all_features = categorical_features + numeric_features

# Helper functions

def extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", name)
    return match.group(1) if match else "Unknown"

def normalize_title(title: str) -> str:
    rare = {
        "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
        "Sir", "Jonkheer", "Dona"
    }
    if title in rare:
        return "Rare"
    if title in ["Mlle", "Ms"]:
        return "Miss"
    if title == "Mme":
        return "Mrs"
    return title

# UI Inputs
st.subheader("Passenger Features")
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)

with col2:
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])
    name_input = st.text_input("Passenger Name (for Title Extraction)", "John Smith")

# Compute engineered features

title_raw = extract_title(name_input)
title_final = normalize_title(title_raw)

family_size = sibsp + parch + 1

# Construct input DataFrame
input_row = pd.DataFrame([{
    "Sex": sex,
    "Embarked": embarked,
    "Title": title_final,
    "Pclass": pclass,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "FamilySize": family_size
}])[all_features]

# Prediction
if st.button("Predict Survival"):
    pred = model.predict(input_row)[0]
    prob = model.predict_proba(input_row)[0][1]

    if pred == 1:
        st.success(f"Prediction: Survived ({prob:.2f} probability)")
    else:
        st.error(f"Prediction: Did NOT survive ({prob:.2f} probability)")

# Training Data Preview
st.subheader("Training Data Preview")
st.write(df.head())
