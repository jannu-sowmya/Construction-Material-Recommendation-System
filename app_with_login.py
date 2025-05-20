
import streamlit as st
import numpy as np
import joblib
import random
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- User Authentication ---
users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "core"}
}

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = users.get(username)
        if user and user["password"] == password:
            st.session_state["logged_in"] = True
            st.session_state["role"] = user["role"]
            st.session_state["username"] = username
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

# --- Load Model and Encoders ---
@st.cache_resource
def load_components():
    model = joblib.load("material_recommender.pkl")
    le_env = joblib.load("env_encoder.pkl")
    le_proj = joblib.load("proj_encoder.pkl")
    le_avail = joblib.load("avail_encoder.pkl")
    le_toxic_lvl = joblib.load("toxic_lvl_encoder.pkl")
    le_toxic_flag = joblib.load("toxic_flag_encoder.pkl")
    le_recycle = joblib.load("recycle_encoder.pkl")
    return model, le_env, le_proj, le_avail, le_toxic_lvl, le_toxic_flag, le_recycle

# --- Admin: Upload & Retrain Model ---
def admin_panel():
    st.sidebar.subheader("🛠 Admin Panel")
    uploaded_file = st.sidebar.file_uploader("Upload new dataset (CSV)", type=["csv"])
    if uploaded_file and st.sidebar.button("Retrain Model"):
        df = pd.read_csv(uploaded_file)
        le_env = LabelEncoder()
        le_proj = LabelEncoder()
        le_avail = LabelEncoder()
        le_toxic_lvl = LabelEncoder()
        le_toxic_flag = LabelEncoder()
        le_recycle = LabelEncoder()

        df['Suitable_Environment'] = le_env.fit_transform(df['Suitable_Environment'])
        df['Project_Type'] = le_proj.fit_transform(df['Project_Type'])
        df['Availability'] = le_avail.fit_transform(df['Availability'])
        df['Toxicity_Level'] = le_toxic_lvl.fit_transform(df['Toxicity_Level'])
        df['Toxicity'] = le_toxic_flag.fit_transform(df['Toxicity'])
        df['Recyclability'] = le_recycle.fit_transform(df['Recyclability'])

        X = df.drop("Material_Name", axis=1)
        y = df["Material_Name"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        joblib.dump(model, "material_recommender.pkl")
        joblib.dump(le_env, "env_encoder.pkl")
        joblib.dump(le_proj, "proj_encoder.pkl")
        joblib.dump(le_avail, "avail_encoder.pkl")
        joblib.dump(le_toxic_lvl, "toxic_lvl_encoder.pkl")
        joblib.dump(le_toxic_flag, "toxic_flag_encoder.pkl")
        joblib.dump(le_recycle, "recycle_encoder.pkl")
        st.sidebar.success("✅ Model retrained and saved!")

# --- Main Application ---
def material_recommendation(role):
    model, le_env, le_proj, le_avail, le_toxic_lvl, le_toxic_flag, le_recycle = load_components()

    st.title("🏗️ Construction Material Recommendation System")

    # Input Fields
    durability = st.slider("Durability Score", 1.0, 10.0, 7.0)
    thermal = st.slider("Thermal Resistance", 0.1, 2.0, 1.0)
    water = st.slider("Water Resistance", 1, 10, 6)
    cost = st.slider("Cost per Unit", 10, 200, 80)
    environment = st.selectbox("Suitable Environment", le_env.classes_)
    project = st.selectbox("Project Type", le_proj.classes_)
    fire = st.slider("Fire Resistance", 1.0, 10.0, 7.0)
    sound = st.slider("Sound Insulation", 1.0, 10.0, 6.0)
    weight = st.slider("Weight per Unit", 10.0, 150.0, 60.0)
    flexural = st.slider("Flexural Strength", 10.0, 60.0, 30.0)
    compressive = st.slider("Compressive Strength", 30.0, 100.0, 70.0)
    carbon = st.slider("Embodied Carbon", 5.0, 50.0, 20.0)
    availability = st.selectbox("Availability", le_avail.classes_)
    toxicity_level = st.selectbox("Toxicity Level", le_toxic_lvl.classes_)
    lifespan = st.slider("Lifespan (Years)", 20, 120, 80)
    toxicity = st.selectbox("Toxicity", le_toxic_flag.classes_)
    recyclability = st.selectbox("Recyclability", le_recycle.classes_)

    # Encode
    encoded_env = le_env.transform([environment])[0]
    encoded_proj = le_proj.transform([project])[0]
    encoded_avail = le_avail.transform([availability])[0]
    encoded_toxic_lvl = le_toxic_lvl.transform([toxicity_level])[0]
    encoded_toxic = le_toxic_flag.transform([toxicity])[0]
    encoded_recycle = le_recycle.transform([recyclability])[0]

    if st.button("Recommend Material"):
        try:
            input_data = np.array([[
                durability, thermal, water, cost, encoded_env, encoded_proj, fire,
                sound, weight, flexural, compressive, carbon,
                encoded_avail, encoded_toxic_lvl, lifespan, encoded_toxic, encoded_recycle
            ]])
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_
            top_indices = np.argsort(probabilities)[::-1][:3]
            st.subheader("🔍 Top 3 Recommended Materials:")
            for i in top_indices:
                st.markdown(f"**{classes[i]}** — {probabilities[i]*100:.2f}% confidence")

            top_material = classes[top_indices[0]]
            if top_material in ["Concrete", "Brick", "Wood", "Stone"]:
                price = random.randint(60, 100)
                avail_status = "High"
            elif top_material in ["Steel", "Glass"]:
                price = random.randint(100, 140)
                avail_status = "Medium"
            else:
                price = random.randint(30, 60)
                avail_status = "Medium"

            st.subheader("📦 Estimated Procurement Info:")
            st.info(f"**Estimated Cost per Unit:** ₹{price}\n\n**Availability:** {avail_status}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# --- App Entry Point ---
if "logged_in" not in st.session_state:
    login()
else:
    if st.session_state["role"] == "admin":
        admin_panel()
    material_recommendation(st.session_state["role"])
