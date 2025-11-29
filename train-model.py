import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Rename to match canonical Kaggle column names
df.rename(columns={
    "Survived": "Survived",
    "Pclass": "Pclass",
    "Sex": "Sex",
    "Age": "Age",
    "SibSp": "SibSp",
    "Parch": "Parch",
    "Fare": "Fare",
    "Embarked": "Embarked",
    "Name": "Name"
}, inplace=True)

# Feature Engineering

# Title extraction
df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

# Consolidate rare titles
rare_titles = [
    "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
    "Sir", "Jonkheer", "Dona"
]
df["Title"] = df["Title"].replace(rare_titles, "Rare")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

# Family Size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Select clean features
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize"
]

target = "Survived"
df = df[features + [target]]

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Split
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing + Model Pipeline
categorical_features = ["Sex", "Embarked", "Title"]
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])

# Train
pipeline.fit(X_train, y_train)

# Accuracy
preds = pipeline.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Validation Accuracy: {acc:.4f}")

# Save
with open("titanic-model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Saved titanic-model.pkl")
