# Defense Candidate EDA + ML Dashboard (Realistic Version)

import streamlit as st
import hashlib
import pandas as pd
import os
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --------- File to store user credentials ---------
USER_DB_FILE = "users.csv"

# --------- Utility Functions ---------
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

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
                st.success(f"Welcome back, {username} ")
                st.rerun()
            else:
                st.error("Invalid username or password")

    elif auth_choice == "Register":
        new_username = st.sidebar.text_input("Create Username")
        new_password = st.sidebar.text_input("Create Password", type='password')
        register_btn = st.sidebar.button("Register")

        if register_btn:
            users_df = load_users()
            if new_username in users_df['username'].values:
                st.warning("Username already exists!")
            else:
                save_user(new_username, new_password)
                st.success("Registered successfully! You can now log in.")
                st.rerun()
else:
    st.sidebar.success(f"Logged in as {st.session_state.current_user}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_user = ""
        st.rerun()

if not st.session_state.logged_in:
    st.stop()

# --------- Load Dataset ---------
st.title("🎖 Defense Candidate Analysis Dashboard")
try:
    df = pd.read_csv("defense_candidate_dataset.csv")
except FileNotFoundError:
    st.error("Dataset not found! Please upload or generate 'defense_candidate_dataset.csv'.")
    st.stop()

# Data Cleaning
df = df.dropna(subset=['SSB_Score', 'Psych_Test', 'GTO_Result', 'PI_Marks', 'Conference_Marks', 'Recommended'])
df['Rank_Secured'] = pd.to_numeric(df['Rank_Secured'], errors='coerce')
df['Recommended'] = df['Recommended'].str.strip().str.title()

# Sidebar Filters
st.sidebar.header("📊 Filter Options")
region_filter = st.sidebar.multiselect("Select Region", options=df['Region'].dropna().unique(), default=df['Region'].dropna().unique())
gender_filter = st.sidebar.multiselect("Select Gender", options=df['Gender'].dropna().unique(), default=df['Gender'].dropna().unique())

filtered_df = df[(df['Region'].isin(region_filter)) & (df['Gender'].isin(gender_filter))]

# EDA Visuals
st.subheader("📈 SSB Score Distribution")
st.plotly_chart(px.histogram(filtered_df, x="SSB_Score", nbins=30, title="SSB Score Histogram"))

st.subheader("📊 Psychology Score vs Recommendation")
st.plotly_chart(px.box(filtered_df, x="Recommended", y="Psych_Test", color="Recommended"))

st.subheader("🌍 Region-wise Recommendation")
st.plotly_chart(px.histogram(filtered_df, x="Region", color="Recommended", barmode="group"))

# --------- ML Prediction ---------
st.subheader("🤖 ML Model: Predict Candidate Recommendation")

features = ['Psych_Test', 'GTO_Result', 'PI_Marks', 'Conference_Marks']
df_ml = df[df['Recommended'].isin(['Yes', 'No'])]
x = df_ml[features]
y = df_ml['Recommended'].map({'Yes': 1, 'No': 0})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

st.text("Classification Report:")
st.code(classification_report(y_test, predictions), language='text')

st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, predictions))

# --------- User Input for Prediction ---------
st.subheader("🎯 Try It Yourself")

psych = st.slider("🧠 Psychology Marks (out of 225)", 0, 225, 120)
gto = st.slider("💪 GTO Marks (out of 225)", 0, 225, 120)
pi = st.slider("🗣️ PI Marks (out of 225)", 0, 225, 120)
conference = st.slider("🎓 Conference Marks (out of 225)", 0, 225, 100)

input_data = pd.DataFrame([[psych, gto, pi, conference]], columns=features)
total_score = psych + gto + pi + conference

pred = model.predict(input_data)[0]

st.metric("Total SSB Marks", f"{total_score} / 900")
st.progress(total_score / 900)

if pred == 1:
    st.success("🎉 You are likely to be Recommended!")
else:
    st.error("❌ You may not be Recommended based on current input.")

# Save cleaned data
df.to_csv("cleaned_defense_dataset.csv", index=False)
st.success("✅ Dashboard Loaded Successfully!")
