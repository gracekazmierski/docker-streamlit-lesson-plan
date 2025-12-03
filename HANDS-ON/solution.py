import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

st.set_page_config(page_title="Titanic Survival Predictor (Solution)", layout="wide")

df = pd.read_csv("train.csv")

model_file = "titanic-model.pkl"
with open(model_file, "rb") as f:
    model = pickle.load(f)

categorical_features = ["Sex", "Embarked", "Title"]
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
all_features = categorical_features + numeric_features

def extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", name)
    return match.group(1) if match else "Unknown"

def normalize_title(title: str) -> str:
    rare = {"Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"}
    if title in rare:
        return "Rare"
    if title in ["Mlle", "Ms"]:
        return "Miss"
    if title == "Mme":
        return "Mrs"
    return title

# Solution for Question 6: multipage interface
page = st.sidebar.selectbox("Page", ["Predictor", "Data Exploration"])


# Predictor Page
if page == "Predictor":

    st.title("Titanic Survival Predictor")
    st.sidebar.header("Passenger Inputs")

    # Solution for Question 1: move all inputs into sidebar
    pclass = st.sidebar.selectbox("Pclass", [1, 2, 3], key="pclass")
    sex = st.sidebar.selectbox("Sex", ["male", "female"], key="sex")
    age = st.sidebar.number_input("Age", 0.0, 100.0, 30.0, key="age")
    sibsp = st.sidebar.number_input("SibSp", 0, 10, 0, key="sibsp")
    parch = st.sidebar.number_input("Parch", 0, 10, 0, key="parch")

    # Solution for Question 2: fare cap with visible override
    fare_input = st.sidebar.number_input("Fare", 0.0, 600.0, 32.0, key="fare_input")
    max_fare = st.sidebar.slider("Maximum Allowed Fare", 10, 200, 200, key="max_fare_slider")
    fare = min(fare_input, max_fare)
    st.sidebar.write(f"Effective Fare Used: {fare}")

    embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"], key="embarked")
    name_input = st.sidebar.text_input("Full Name", "Smith, Mr. John", key="name_input")

    # Solution for Question 3: extract and display passenger title
    title_raw = extract_title(name_input)
    title_final = normalize_title(title_raw)
    st.write(f"Extracted Title: **{title_final}**")

    family_size = sibsp + parch + 1

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

    if st.button("Predict Survival", key="predict_button"):
        pred = model.predict(input_row)[0]
        prob = model.predict_proba(input_row)[0][1]

        st.write(f"Prediction Probability: {prob:.2f}")

        # Solution for Question 7: probability interpretation
        if prob >= 0.85:
            st.info("Very strong survival likelihood.")
        elif prob >= 0.50:
            st.info("Moderate survival likelihood.")
        else:
            st.info("Low survival likelihood.")

        # Solution for Question 9: encoded model input display
        encoded = model["preprocess"].transform(input_row)
        encoded = encoded.astype(float)
        st.write("Encoded Model Input:")
        st.dataframe(encoded, width="stretch")

    # Solution for Question 5: training data toggle
    if st.checkbox("Show Training Data", key="show_training"):
        st.write(df.head())

    # Solution for Question 4: feature importance chart
    st.subheader("Feature Importances")
    ohe = model["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names = cat_names + numeric_features
    importance = model["model"].feature_importances_

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)
    st.bar_chart(fi_df.set_index("feature"))


# Data Exploration Page
if page == "Data Exploration":

    st.title("Data Exploration")

    # (part of Question 6) survival summaries on second page
    st.subheader("Survival Rate by Class")
    st.write(df.groupby("Pclass")["Survived"].mean())

    st.subheader("Survival Rate by Sex")
    st.write(df.groupby("Sex")["Survived"].mean())

    st.subheader("Survival Rate by Embarked")
    st.write(df.groupby("Embarked")["Survived"].mean())

    # Solution for Question 8: age histogram
    st.subheader("Age Distribution")
    st.bar_chart(df["Age"])
