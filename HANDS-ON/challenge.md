# Titanic Streamlit + Docker Challenge

This challenge contains a set of small, focused tasks designed to help you explore
Streamlit, Docker, and a real trained machine learning model. Each task modifies
or extends the starter `app.py` file.

You are expected to write code directly in `app.py`, test changes, and rebuild
your Docker image to see the effects.

This document explains each of the required tasks.

> **Quick Reference:**  
> Streamlit Cheat Sheet — https://docs.streamlit.io/library/cheatsheet

---

## Question 1 – Move all inputs into the sidebar
Move every passenger input widget into the sidebar using `st.sidebar.*`.  
Inputs include: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Full Name.

---

## Question 2 – Add a maximum allowed fare control
Add:
- A number input for Fare.
- A slider for “Maximum Allowed Fare”.
Cap the fare using `fare = min(fare_input, max_fare)`  
and display the effective fare used.

---

## Question 3 – Extract and display the passenger title
Extract the title from the full name (e.g., Mr, Miss, Mrs).  
Normalize rare titles into the same groups used during model training.  
Display the final normalized title on the page.

---

## Question 4 – Add a feature importance chart
Retrieve the one-hot encoded feature names.  
Combine them with the numeric feature names.  
Plot feature importances from the RandomForest model using `st.bar_chart`.

---

## Question 5 – Add a training-data preview toggle
Add a checkbox labeled “Show Training Data”.  
When selected, display `df.head()`.

---

## Question 6 – Add a multipage interface
Create a page selector in the sidebar:
- Predictor page  
- Data Exploration page  

The Data Exploration page must show:
- Survival rate by Pclass  
- Survival rate by Sex  
- Survival rate by Embarked  

---

## Question 7 – Add probability interpretation
After prediction, display the raw probability AND an interpretation:
- ≥ 0.85 → very strong likelihood  
- 0.50–0.85 → moderate likelihood  
- < 0.50 → low likelihood  

---

## Question 8 – Add an age distribution plot
On the Data Exploration page, display an Age distribution chart  
(e.g., `st.bar_chart(df["Age"])`).

---

## Question 9 – Display the encoded model input
Transform the input row with:

encoded = model["preprocess"].transform(input_row)

Display the encoded matrix using `st.dataframe(encoded, width="stretch")`.

---

## Completion
After completing Questions 1–9:
1. Test locally with `streamlit run app.py`
2. Build and run the Docker image
3. Confirm both versions match in behavior
