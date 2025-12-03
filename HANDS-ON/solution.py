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

# feature lists aligned with model training pipeline
categorical_features = ["Sex", "Embarked", "Title"]
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
all_features = categorical_features + numeric_features

# helper for extracting title from Titanic-style name string
def extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", name)
    return match.group(1) if match else "Unknown"

# helper for normalizing titles into categories used during training
def normalize_title(title: str) -> str:
    rare = {"Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"}
    if title in rare:
        return "Rare"
    if title in ["Mlle", "Ms"]:
        return "Miss"
    if title == "Mme":
        return "Mrs"
    return title

# Solution for Question 6: page selector for predictor vs exploration views
page = st.sidebar.selectbox("Page", ["Predictor", "Data Exploration"])


# Predictor Page
if page == "Predictor":

    # title for main prediction interface
    st.title("Titanic Survival Predictor")

    # section header for all passenger-related inputs
    st.sidebar.header("Passenger Inputs")

    # Solution for Question 1: move inputs into sidebar
    pclass = st.sidebar.selectbox("Pclass", [1, 2, 3], key="pclass")
    sex = st.sidebar.selectbox("Sex", ["male", "female"], key="sex")
    age = st.sidebar.number_input("Age", 0.0, 100.0, 30.0, key="age")
    sibsp = st.sidebar.number_input("SibSp", 0, 10, 0, key="sibsp")
    parch = st.sidebar.number_input("Parch", 0, 10, 0, key="parch")

    # Solution for Question 2: fare cap logic with visible override to show effective fare
    fare_input = st.sidebar.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=32.0,
        key="fare_input"
    )
    max_fare = st.sidebar.slider(
        "Maximum Allowed Fare",
        min_value=10,
        max_value=200,
        value=200,
        key="max_fare_slider"
    )
    fare = min(fare_input, max_fare)
    st.sidebar.write(f"Effective Fare Used: {fare}")

    # embarked and name inputs complete the feature set for this app
    embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"], key="embarked")
    name_input = st.sidebar.text_input("Full Name", "Smith, Mr. John", key="name_input")

    # Solution for Question 3: display extracted and normalized title
    title_raw = extract_title(name_input)
    title_final = normalize_title(title_raw)
    st.write(f"Extracted Title: **{title_final}**")

    # engineered family size feature consistent with training
    family_size = sibsp + parch + 1

    # construct single-row DataFrame in training feature order
    input_row = pd.DataFrame(
        [{
            "Sex": sex,
            "Embarked": embarked,
            "Title": title_final,
            "Pclass": pclass,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "FamilySize": family_size
        }]
    )[all_features]

    # prediction button triggers model call and result display
    if st.button("Predict Survival", key="predict_button"):
        pred = model.predict(input_row)[0]
        prob = model.predict_proba(input_row)[0][1]

        # display raw probability for transparency
        st.write(f"Prediction Probability: {prob:.2f}")

        # Solution for Question 7: qualitative interpretation of probability ranges
        if prob >= 0.85:
            st.info("Very strong survival likelihood.")
        elif prob >= 0.50:
            st.info("Moderate survival likelihood.")
        else:
            st.info("Low survival likelihood.")

        # Solution for Question 9: show encoded representation from preprocessing step
        encoded = model["preprocess"].transform(input_row)
        encoded = encoded.astype(float)
        st.write("Encoded Model Input:")
        st.dataframe(encoded, width="stretch")

    # Solution for Question 5: optional training data preview for context
    if st.checkbox("Show Training Data", key="show_training"):
        st.write(df.head())

    # Solution for Question 4: feature importance plot from RandomForest model
    st.subheader("Feature Importances")
    ohe = model["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names = cat_names + numeric_features
    importance = model["model"].feature_importances_
    fi_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance
        }
    ).sort_values("importance", ascending=False)
    st.bar_chart(fi_df.set_index("feature"))


# Data Exploration Page
if page == "Data Exploration":

    # title for exploratory statistics view
    st.title("Data Exploration")

    # Solution for Question 6 extension: survival summaries by Pclass
    st.subheader("Survival Rate by Class")
    st.write(df.groupby("Pclass")["Survived"].mean())

    # survival summaries by Sex
    st.subheader("Survival Rate by Sex")
    st.write(df.groupby("Sex")["Survived"].mean())

    # survival summaries by Embarked
    st.subheader("Survival Rate by Embarked")
    st.write(df.groupby("Embarked")["Survived"].mean())

    # Solution for Question 8: age distribution visualization
    st.subheader("Age Distribution")
    st.bar_chart(df["Age"])
