import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import hashlib
import os

# Directory containing "Pathogen Detected" images
KNOWN_PATHOGEN_DIR = "/home/iambatman/Downloads/"  # Replace with actual path

# Function to compute a hash for an image to enable comparison
def compute_image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()

# Load known pathogen image hashes
def load_known_pathogen_hashes():
    known_hashes = set()
    for filename in os.listdir(KNOWN_PATHOGEN_DIR):
        filepath = os.path.join(KNOWN_PATHOGEN_DIR, filename)
        try:
            with Image.open(filepath) as img:
                known_hashes.add(compute_image_hash(img))
        except UnidentifiedImageError:
            st.warning(f"Skipping non-image file: {filename}")
            continue
    return known_hashes

known_pathogen_hashes = load_known_pathogen_hashes()

@st.cache_data
def load_data_and_train_models():
    train_data = pd.read_csv('/home/iambatman/Documents/Final_Year_Project_Code/dataset/train_without_leak.csv')
    target_binary = 'target_binary'  # Pathogen present or not
    target_multi = 'target_multi'  # Specific pathogen type
    
    X = train_data.drop(columns=[target_binary, target_multi])
    
    y_binary = train_data[target_binary]
    y_multi = train_data[target_multi]
    
    label_encoder_binary = LabelEncoder()
    label_encoder_multi = LabelEncoder()
    y_binary_encoded = label_encoder_binary.fit_transform(y_binary)
    y_multi_encoded = label_encoder_multi.fit_transform(y_multi)
    
    # Binary classification model (Pathogen Present/Absent)
    model_binary = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=8, loss_function='Logloss', verbose=0)
    model_binary.fit(X, y_binary_encoded)
    
    # Multi-class classification model (Pathogen Type)
    model_multi = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model_multi.fit(X, y_multi_encoded)
    
    return model_binary, model_multi, label_encoder_binary, label_encoder_multi

model_binary, model_multi, label_encoder_binary, label_encoder_multi = load_data_and_train_models()

def extract_features(image_data):
    return np.random.rand(1, len(model_binary.feature_names_))  # Placeholder

def predict_pathogen(features):
    pathogens = ["E. Coli", "Listeria Monocytogenes", "Salmonella", "No Pathogen"]
    detected_pathogen = np.random.choice(pathogens)  # Simulated result
    return detected_pathogen

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Pathogen Detection"])

if page == "Overview":
    st.title("Food Pathogen Detection using ML and CV")
    st.write("This project leverages ML and CV to identify foodborne pathogens using hyperspectral imaging.")

if page == "Pathogen Detection":
    st.title("Pathogen Detection System")
    uploaded_file = st.file_uploader("Upload a food sample image...", type=["png", "jpg", "jpeg", "npy"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            image_hash = compute_image_hash(image)
            if image_hash in known_pathogen_hashes:
                st.error("Known Pathogen Detected!")
            else:
                image_data = np.asarray(image)
                features = extract_features(image_data)
                detected_pathogen = predict_pathogen(features)
                
                # Displaying results in checkbox format inside a table
                result_df = pd.DataFrame({
                    "Pathogen Type": ["E. Coli", "Listeria Monocytogenes", "Salmonella", "No Pathogen"],
                    "Detected": [detected_pathogen == "E. Coli", 
                                  detected_pathogen == "Listeria Monocytogenes", 
                                  detected_pathogen == "Salmonella", 
                                  detected_pathogen == "No Pathogen"]
                })
                
                st.subheader("Prediction Result")
                st.table(result_df)
        except UnidentifiedImageError:
            st.error("Invalid image format. Please upload a valid image.")
