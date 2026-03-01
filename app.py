# --------------------------
# Smart Health Predictor 🚀 - Final Corrected Version
# --------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="Smart Health Predictor 🚀",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------------------------
# Load Model
# --------------------------
model_file = "models/rf.pkl"
labels_file = "models/labels.pkl"

if os.path.exists(model_file) and os.path.exists(labels_file):
    model = pickle.load(open(model_file, "rb"))
    labels = pickle.load(open(labels_file, "rb"))
else:
    st.error("❌ Model files not found! Please upload rf.pkl and labels.pkl in 'models' folder.")
    st.stop()

# --------------------------
# Symptom Columns
# --------------------------
symptom_columns = ['Fever','Cough','Headache','Nausea','Fatigue','Vomiting','Sore throat','Dizziness']

# --------------------------
# Sidebar Login
# --------------------------
st.sidebar.markdown("## 🔐 User Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username and password:
    st.sidebar.success(f"Welcome, {username} ✅")
else:
    st.stop()

# --------------------------
# Sidebar Contact Info
# --------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("## 👩‍💻 Contact Info")
st.sidebar.markdown("""
- **Name:** Sakshi Sharad Jadhav  
- **Email:** sakshi123@example.com  
- **Phone:** +91-XXXXXXXXXX
""")

# --------------------------
# History CSV
# --------------------------
history_file = "history.csv"
if not os.path.exists(history_file):
    pd.DataFrame(columns=["Username","Symptoms","Top1","Top2","Top3","Confidence","Risk"]).to_csv(history_file, index=False)

# --------------------------
# App Title
# --------------------------
st.markdown("<h1 style='text-align:center;'>Smart Health Predictor 🚀</h1>", unsafe_allow_html=True)

selected_symptoms = st.multiselect("Select Your Symptoms:", symptom_columns)

# --------------------------
# Prediction Logic
# --------------------------
if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least 1 symptom!")
    else:

        # Input preparation
        input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptom_columns]
        input_array = np.array([input_data])

        # Predict
        probs = model.predict_proba(input_array)[0]  # 1D array

        # Top 3
        top3_idx = probs.argsort()[-3:][::-1]
        top3_diseases = [labels[i] for i in top3_idx]
        top3_probs = probs[top3_idx] * 100

        # Risk
        top_prob = top3_probs[0]
        if top_prob >= 80:
            risk = "High Risk 🔴"
        elif top_prob >= 50:
            risk = "Moderate Risk 🟠"
        else:
            risk = "Low Risk 🟢"

        # Display Results
        st.markdown("### 🩺 Prediction Results")
        for i in range(3):
            st.write(f"**{top3_diseases[i]}:** {top3_probs[i]:.2f}%")
        st.write(f"**Risk Level:** {risk}")

        # Save history
        history_df = pd.read_csv(history_file)
        new_entry = {
            "Username": username,
            "Symptoms": ",".join(selected_symptoms),
            "Top1": top3_diseases[0],
            "Top2": top3_diseases[1],
            "Top3": top3_diseases[2],
            "Confidence": top3_probs[0],
            "Risk": risk
        }
        history_df = pd.concat([history_df, pd.DataFrame([new_entry])], ignore_index=True)
        history_df.to_csv(history_file, index=False)

        # Bar chart
        st.markdown("### 📊 Confidence Bar Chart")
        fig, ax = plt.subplots()
        ax.bar(top3_diseases, top3_probs, color=['#FF4C4C','#FFA500','#4CAF50'])
        ax.set_ylabel("Confidence %")
        ax.set_ylim(0, 100)
        st.pyplot(fig)

        # Pie chart
        st.markdown("### 🥧 Top 3 Disease Probability Pie Chart")
        fig2, ax2 = plt.subplots()
        ax2.pie(top3_probs, labels=top3_diseases, autopct='%1.1f%%', startangle=140,
                colors=['#FF4C4C','#FFA500','#4CAF50'])
        ax2.set_title("Top 3 Disease Probabilities")
        st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown(
"""
<div style="text-align:center; color: #888; font-size:14px;">
© 2026 Smart Health Predictor | Developed by Sakshi Sharad Jadhav
</div>
""",
unsafe_allow_html=True
)