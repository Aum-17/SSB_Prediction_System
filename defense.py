# Defense Candidate EDA + ML Dashboard

import streamlit as st
import hashlib
import pandas as pd
import os

# --------- File to store user credentials ---------
USER_DB_FILE = "users.csv"

# --------- Utility Functions ---------
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# --------- Create or Load users.csv ---------
if not os.path.exists(USER_DB_FILE):
    df_users = pd.DataFrame(columns=["username", "password"])
    df_users.to_csv(USER_DB_FILE, index=False)

def load_users():
    return pd.read_csv(USER_DB_FILE)

def save_user(username, password):
    new_user = pd.DataFrame([[username, make_hashes(password)]], columns=["username", "password"])
    new_user.to_csv(USER_DB_FILE, mode='a', header=False, index=False)

# --------- User Authentication Panel ---------
st.sidebar.title("🔐 User Authentication")

auth_choice = st.sidebar.radio("Login or Register", ["Login", "Register"])

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""

if not st.session_state.logged_in:
    if auth_choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')
        login_btn = st.sidebar.button("Login")

        if login_btn:
            users_df = load_users()
            user_match = users_df[
                (users_df['username'] == username) &
                (users_df['password'].apply(lambda x: check_hashes(password, x)))
            ]
            if not user_match.empty:
                st.session_state.logged_in = True
                st.session_state.current_user = username
                st.success(f"Welcome back, {username} 👋")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

    elif auth_choice == "Register":
        new_username = st.sidebar.text_input("Create Username")
        new_password = st.sidebar.text_input("Create Password", type='password')
        register_btn = st.sidebar.button("Register")

        if register_btn:
            users_df = load_users()
            if new_username in users_df['username'].values:
                st.warning("⚠️ Username already exists!")
            else:
                save_user(new_username, new_password)
                st.success("🎉 Registered successfully! You can now log in.")
                st.rerun()

else:
    st.sidebar.success(f"✅ Logged in as {st.session_state.current_user}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_user = ""
        st.rerun()

# --------- If not logged in, stop everything below ---------
if not st.session_state.logged_in:
    st.stop()

# --------- EDA + ML Dashboard ---------

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load Dataset
st.title("🎖️ Defense Candidate Analysis Dashboard")

try:
    df = pd.read_csv("defense_candidate_dataset.csv")
except FileNotFoundError:
    st.error("❗ Please make sure 'defense_candidate_dataset.csv' exists in the same folder.")
    st.stop()

df['Rank_Secured'] = pd.to_numeric(df['Rank_Secured'], errors='coerce')

# Sidebar Filters
st.sidebar.header("📊 Filter Options")
region_filter = st.sidebar.multiselect("Select Region", options=df['Region'].unique(), default=df['Region'].unique())
gender_filter = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())

filtered_df = df[(df['Region'].isin(region_filter)) & (df['Gender'].isin(gender_filter))]

# EDA Visuals
st.subheader("📈 SSB Score Distribution")
st.plotly_chart(px.histogram(filtered_df, x="SSB_Score", nbins=30, title="SSB Score Histogram"))

st.subheader("📊 OLQ Score vs Recommendation")
st.plotly_chart(px.box(filtered_df, x="Recommended", y="OLQ_Score", color="Recommended"))

st.subheader("🌍 Region-wise Recommendation")
st.plotly_chart(px.histogram(filtered_df, x="Region", color="Recommended", barmode="group"))

# --------- Machine Learning: Predict Recommendation ---------

st.subheader("🤖 ML Model: Predict Candidate Recommendation")

features = ['Age', 'OLQ_Score', 'Psych_Test', 'GTO_Result', 'PI_Marks']
df_ml = df[df['Recommended'].notnull()]
x = df_ml[features]
y = df_ml['Recommended'].apply(lambda x: 1 if x == 'Yes' else 0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

st.code(classification_report(y_test, predictions), language='text')
st.text("📉 Confusion Matrix:")
st.write(confusion_matrix(y_test, predictions))

# --------- User Input Prediction ---------
st.subheader("🎯 Try It Yourself")

age = st.slider("Age", 18, 30, 22)
olq = st.slider("OLQ Score", 40, 100, 70)
psych = st.slider("Psych Test Score", 20, 80, 60)
gto = st.slider("GTO Result", 10, 50, 30)
pi = st.slider("PI Marks", 10, 50, 35)

input_data = pd.DataFrame([[age, olq, psych, gto, pi]], columns=features)
pred = model.predict(input_data)[0]

if pred == 1:
    st.success("✅ Recommended for selection!")
else:
    st.error("❌ Not Recommended")

# Save processed data
df.to_csv("cleaned_defense_dataset.csv", index=False)

st.success("🎉 Dashboard and Model loaded successfully!")
