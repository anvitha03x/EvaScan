import streamlit as st
import os
import pandas as pd
import cv2
import numpy as np
from PIL import Image

st.title("Early Detection of PCOD/PCOS using AI")
# Create folder for saving images
save_folder = "Hair_Analysis_Images"
os.makedirs(save_folder, exist_ok=True)

# File to store responses
file_name = os.path.join(os.getcwd(), "PCOS_Detection_Data.csv")

# Ensure the CSV file exists
if not os.path.exists(file_name):
    df = pd.DataFrame(columns=[
        "Body Weight (kg)", "Height (cm)", "BMI", "Menstrual Cycle", "Period Duration (days)",
        "Heavy Bleeding or Spotting", "Changes in Menstrual Cycle", "Sudden Weight Gain",
        "Insulin Resistance or Diabetes", "Frequent Sugar Cravings or Energy Crashes",
        "Excessive Hunger", "Difficulty Conceiving", "History of Miscarriage",
        "Family History of PCOD/PCOS/Diabetes/Thyroid", "Physical Activity Frequency",
        "Typical Diet", "High Stress or Anxiety", "Dark Patches on Skin",
        "Facial Hair", "Pigmentation on Skin", "Pimples or Acne",
        "Scalp Visibility", "Hair Thinning"
    ])
    df.to_csv(file_name, index=False)


# =============================== #
#     Hair Analysis Section       #
# =============================== #
st.header("Hair Analysis")

# Upload an OLD image from file
old_hair = st.file_uploader("Upload an **OLD** image of your hair", type=["jpg", "jpeg", "png"])

# Capture a NEW image using the camera
new_hair = st.camera_input("Capture a **NEW** image of your hair")

old_hair_path, new_hair_path = None, None

# Save old image (if uploaded)
if old_hair:
    old_img = Image.open(old_hair)
    st.image(old_img, caption="Old Hair Image", use_column_width=True)
    old_hair_path = os.path.join(save_folder, "old_hair_photo.png")
    old_img.save(old_hair_path)

# Save new image (if captured)
if new_hair:
    new_img = Image.open(new_hair)
    st.image(new_img, caption="New Hair Image (Captured)", use_column_width=True)
    new_hair_path = os.path.join(save_folder, "new_hair_photo.png")
    new_img.save(new_hair_path)


# Analyze Hair Thinning & Scalp Visibility
def analyze_hair_changes(old_path, new_path):
    if not old_path or not new_path:
        return None, None  # Skip analysis if images are missing

    # Load images in grayscale
    old_img = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE)
    new_img = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)

    # Resize images to the same size (for accurate comparison)
    new_img = cv2.resize(new_img, (old_img.shape[1], old_img.shape[0]))

    # Convert images to binary (thresholding for better hair contrast)
    _, old_bin = cv2.threshold(old_img, 100, 255, cv2.THRESH_BINARY)
    _, new_bin = cv2.threshold(new_img, 100, 255, cv2.THRESH_BINARY)

    # Compute the difference between old and new hair images
    hair_loss_mask = cv2.absdiff(old_bin, new_bin)
    
    # Count the number of lost pixels
    lost_hair_pixels = np.sum(hair_loss_mask == 255)

    # Calculate total hair area in the old image
    old_hair_pixels = np.sum(old_bin == 255)

    # Hair thinning condition (10-20% loss instead of 30%)
    hair_thinning = (lost_hair_pixels / old_hair_pixels) > 0.15

    # Scalp visibility check (average brightness change)
    old_scalp_brightness = np.mean(old_img)
    new_scalp_brightness = np.mean(new_img)
    scalp_visibility = (new_scalp_brightness > old_scalp_brightness + 10)

    return hair_thinning, scalp_visibility


# Run analysis only if both images are uploaded
hair_thin, scalp_visible = analyze_hair_changes(old_hair_path, new_hair_path) if old_hair and new_hair else (None, None)

# Submit Hair Analysis Data
if st.button("Submit Hair Analysis", key="submit_hair_analysis"):
    new_data = pd.DataFrame([[
        "True" if hair_thin else "False", "True" if scalp_visible else "False"
    ]], columns=["Hair Thinning", "Scalp Visibility"])

    df = pd.read_csv(file_name)
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(file_name, index=False)
    st.success("Hair Analysis Recorded Successfully!")
# =============================== #
#   PCOS/PCOD Symptom Checker     #
# =============================== #
st.header("PCOS/PCOD Detection - Symptom Checker")

# User Inputs
st.subheader("Please answer the following questions:")

weight = st.number_input("Body Weight (kg)", min_value=20.0, max_value=200.0, step=0.1)
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, step=0.1)

# Calculate BMI
bmi = round(weight / ((height / 100) ** 2), 2) if height > 0 else None

menstrual_cycle = st.selectbox("Menstrual Cycle", ["Regular (28-35 days)", "Irregular", "Absent"])
period_duration = st.number_input("Period Duration (days)", min_value=1, max_value=45, step=1)
heavy_bleeding = st.radio("Heavy Bleeding or Spotting", ["Yes", "No"])
cycle_changes = st.radio("Changes in Menstrual Cycle", ["Yes", "No"])
sudden_weight_gain = st.radio("Sudden Weight Gain", ["Yes", "No"])
insulin_resistance = st.radio("Insulin Resistance or Diabetes", ["Yes", "No"])
sugar_cravings = st.radio("Frequent Sugar Cravings or Energy Crashes", ["Yes", "No"])
excessive_hunger = st.radio("Excessive Hunger", ["Yes", "No"])
difficulty_conceiving = st.radio("Difficulty Conceiving", ["Yes", "No"])
miscarriage = st.radio("History of Miscarriage", ["Yes", "No"])
family_history = st.radio("Family History of PCOD/PCOS/Diabetes/Thyroid", ["Yes", "No"])
physical_activity = st.radio("Physical Activity Frequency", ["Frequently", "Rarely", "Never"])
typical_diet = st.radio("Typical Diet", ["Balanced", "Processed Food", "Restaurent Food"])
stress_anxiety = st.radio("High Stress or Anxiety", ["Yes", "No"])
dark_patches = st.radio("Dark Patches on Skin", ["Yes", "No"])
facial_hair = st.radio("Facial Hair", ["Yes", "No"])
pigmentation = st.radio("Pigmentation on Skin", ["Yes", "No"])
pimples_acne = st.radio("Pimples or Acne", ["Yes", "No"])

# Submit Symptom Data
if st.button("Submit Symptoms", key="submit_symptoms"):
    new_data = pd.DataFrame([[  
        weight, height, bmi, menstrual_cycle, period_duration, heavy_bleeding,  
        cycle_changes, sudden_weight_gain, insulin_resistance, sugar_cravings,  
        excessive_hunger, difficulty_conceiving, miscarriage, family_history,  
        physical_activity, typical_diet, stress_anxiety, dark_patches, facial_hair, pigmentation, pimples_acne  
    ]], columns=[  
        "Body Weight (kg)", "Height (cm)", "BMI", "Menstrual Cycle", "Period Duration (days)",  
        "Heavy Bleeding or Spotting", "Changes in Menstrual Cycle", "Sudden Weight Gain",  
        "Insulin Resistance or Diabetes", "Frequent Sugar Cravings or Energy Crashes",  
        "Excessive Hunger", "Difficulty Conceiving", "History of Miscarriage",  
        "Family History of PCOD/PCOS/Diabetes/Thyroid", "Physical Activity Frequency",  
        "Typical Diet", "High Stress or Anxiety", "Dark Patches on Skin", "Facial Hair", "Pigmentation on Skin", "Pimples or Acne"  
    ])  

    df = pd.read_csv(file_name)
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(file_name, index=False)
    st.success("Your responses have been recorded successfully!")

import streamlit as st
import pandas as pd

st.header("PCOS/PCOD Risk Assessment")

file_name = "PCOS_Detection_Data.csv"
df = pd.read_csv(file_name)

if not df.empty:
    st.subheader("Your PCOS/PCOD Risk Score:")

    total_points = 0
    latest_entry = df.iloc[-1]  # Get the last recorded row

    # BMI Calculation
    bmi = latest_entry['BMI']
    if bmi <= 18.5:
        total_points += 5
    elif 18.5 < bmi <= 24.9:
        total_points += 0
    elif 24.9 < bmi <= 29.9:
        total_points += 10
    else:
        total_points += 15

    # Menstrual Cycle
    menstrual_cycle = latest_entry['Menstrual Cycle']
    if menstrual_cycle == "Irregular":
        total_points += 4
    elif menstrual_cycle == "Absent":
        total_points += 8

    # Period Duration
    period_duration = latest_entry['Period Duration (days)']
    if period_duration > 14:
        total_points += 7
    elif period_duration > 7:
        total_points += 4
    elif period_duration > 3:
        total_points += 2

    # Other Symptoms
    risk_factors = {
        "Sudden Weight Gain": 3,
        "Family History of PCOD/PCOS/Diabetes/Thyroid": 4,
        "High Stress or Anxiety": 4,
        "Dark Patches on Skin": 7,
        "Facial Hair": 4,
        "Pigmentation on Skin": 3,
        "Pimples or Acne": 2,
        "Changes in Menstrual Cycle": 2,
        "Insulin Resistance or Diabetes": 4,
        "Frequent Sugar Cravings or Energy Crashes": 2,
        "Excessive Hunger": 2,
        "Difficulty Conceiving": 3,
        "History of Miscarriage": 3,
        "Heavy Bleeding or Spotting": 3,
        "Hair Thinning": 3,
        "Scalp Visibility": 3,
    }

    for key, points in risk_factors.items():
        if latest_entry[key] == "Yes" or latest_entry[key] == "True":
            total_points += points

    # Physical Activity
    activity = latest_entry["Physical Activity Frequency"]
    if activity == "Rarely":
        total_points += 2
    elif activity == "Never":
        total_points += 5

    # Diet
    diet = latest_entry["Typical Diet"]
    if diet == "Processed Food":
        total_points += 8
    elif diet == "Restaurant Food":
        total_points += 5

    # Show Risk Score
    st.write(f"**Total Score: {total_points}**")

    # Risk Category
    if total_points < 25:
        st.success("Low Risk")
    elif 25 <= total_points < 50:
        st.warning("Moderate Risk")
    elif 50 <= total_points < 75:
        st.warning("High Risk")
    else:
        st.error("Very High Risk - Consider consulting a doctor.")

    # Determine PCOS vs PCOD
    if total_points >= 25:
        if latest_entry["Facial Hair"] == "Yes" or latest_entry["Insulin Resistance or Diabetes"] == "Yes":
            st.error("PCOS Detected")
        else:
            st.warning("PCOD Detected")
else:
    st.warning("No data found. Please complete the questionnaire first.")
