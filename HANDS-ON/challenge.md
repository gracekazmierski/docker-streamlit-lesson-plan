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

## 1. Move all UI inputs into the sidebar

All controls used to collect passenger information should appear in the sidebar
instead of the main page area. Use `st.sidebar.selectbox`, `st.sidebar.number_input`,
and other sidebar widgets.

This cleans up the main page and prepares the UI for expansion.

---

## 2. Add a “maximum allowed fare” control

Create a slider in the sidebar that sets the highest fare the model is allowed
to use. If the user enters a fare above this threshold, cap it.

Example:
- User enters fare = 300  
- Max allowed fare slider = 90  
- Effective fare used by the model = 90  

Display the effective fare back to the user.

---

## 3. Extract and display the passenger’s title

Given a full Titanic-format name like `"Smith, Mr. John"`, extract the title
(“Mr”) and normalize it into one of the categories used during model training.

Show the extracted title on the main page.

---

## 4. Add a feature importance chart

The trained model includes a RandomForestClassifier with built-in feature
importances. Retrieve these values, combine them with the corresponding encoded
feature names, and visualize them using `st.bar_chart`.

This helps illustrate which inputs influence the model most.

---

## 5. Add a checkbox to preview the training dataset

Add a checkbox labeled “Show Training Data”.  
When checked, display the first five rows of the loaded `train.csv`.

This gives users a sense of the underlying dataset and feature structure.

---

## 6. Add a multipage interface

Use a sidebar page selector such as:

```python
page = st.sidebar.selectbox("Page", ["Predictor", "Data Exploration"])
