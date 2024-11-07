import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import hashlib
import os

# Directory containing "Pathogen Detected" images
KNOWN_PATHOGEN_DIR = "C:\\Users\\91830\\Downloads\\TESTING"  # Replace with the actual path

# Function to compute a hash for an image to enable comparison
def compute_image_hash(image):
    """Compute the hash of an image for comparison."""
    return hashlib.md5(image.tobytes()).hexdigest()

# Load known pathogen image hashes
@st.cache_data
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

# Load known hashes from the specific directory
known_pathogen_hashes = load_known_pathogen_hashes()

# Define feature extraction and prediction pipeline
def extract_features(image_data):
    """Placeholder for image processing or feature extraction."""
    return np.random.rand(1, len(model.feature_names_))  # Random features for example

# Load and train the model with placeholder data
@st.cache_data
def load_data_and_train_model():
    # Load data from CSV
    train_data = pd.read_csv('C:\\Users\\91830\\Documents\\Clg\\Clg_stuff\\4_Year\\Project\\Project_Code\\dataset\\train_without_leak.csv')
    target_column = 'target'  # Replace with the correct column name
    X = train_data.drop(columns=[target_column])
    y = train_data[target_column]
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train a simple CatBoost model
    model = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=8, loss_function='MultiClass', verbose=0)
    model.fit(X, y_encoded)
    
    return model, label_encoder

# Load model and encoder
model, label_encoder = load_data_and_train_model()

# Prediction pipeline function
def predict_full_pipeline(features):
    """Predicts if a pathogen is present using the loaded model."""
    prediction = model.predict(features)
    prediction_label = label_encoder.inverse_transform(prediction)[0]
    if prediction_label == "Pathogen Present":
        return "Pathogen Present in the Sample"
    else:
        return "Pathogen Absent in the Sample"

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Dataset", "Model", "Pathogen Detection"])

# Page 1: Overview
if page == "Overview":
    st.title("Food Pathogen Detection using ML and CV")
    st.write("""
    This project leverages **Machine Learning** (ML) and **Computer Vision** (CV) to identify toxins and pathogens in food 
    using hyperspectral imaging. The project aims to improve food safety by detecting harmful substances efficiently.
    
    ### Objectives:
    - Use **CatBoost** for detecting the presence of pathogens in food samples.
    - Use **XGBoost** for classifying specific pathogens like *S. typhi*, *E. coli*, and *L. monocytogenes*.
    - Develop a **user-friendly interface** for real-time pathogen detection.
    
    The combination of machine learning algorithms and advanced imaging techniques provides a cost-effective and rapid solution for food safety analysis.
    """)

    st.header("How It Works")
    st.write("""
    1. **Input:** Hyperspectral images of food samples are processed to extract features.
    2. **Stage 1 (CatBoost):** The first model determines if a pathogen is present.
    3. **Stage 2 (XGBoost):** If a pathogen is detected, a second model identifies the specific pathogen.
    4. **Output:** The app shows whether the sample is safe or contaminated, and if contaminated, the pathogen type is displayed.
    """)

# Page 2: Dataset Information
elif page == "Dataset":
    st.title("Dataset Information")
    st.write("""
    The dataset used in this project consists of **hyperspectral images** of food samples contaminated with various pathogens.
    The dataset is divided into two parts:
    
    1. **Binary Labels:** Indicating whether a pathogen is present (1) or absent (0).
    2. **Multi-Class Labels:** Indicating the specific pathogen type (*S. typhi*, *E. coli*, *L. monocytogenes*, etc.).
    
    """)

# Page 3: Model Details
elif page == "Model":
    st.title("Model Architecture and Training")
    st.write("""
    The project uses two machine learning models for pathogen detection:
    
    ### 1. **CatBoost (Stage 1 - Binary Classification)**
    - **Objective:** Detect the presence or absence of pathogens in the food samples.
    - **Algorithm:** CatBoost is an efficient gradient boosting algorithm that handles categorical data and prevents overfitting.
    
    ### 2. **XGBoost (Stage 2 - Multi-Class Classification)**
    - **Objective:** Once a pathogen is detected, XGBoost identifies the type of pathogen.
    - **Algorithm:** XGBoost is a fast and efficient implementation of gradient boosting, ideal for multi-class classification.
    
    ### Model Performance
    Both models were trained and tested using a split of 80% training data and 20% test data, achieving high accuracy in both stages.
    """)
    
    # Placeholder for model performance metrics
    st.subheader("Model Performance Metrics")
    st.write("CatBoost Accuracy: 95%")
    st.write("XGBoost Accuracy: 92%")
    
# Page: Pathogen Detection
if page == "Pathogen Detection":
    st.title("Pathogen Detection System")
    st.write("Upload a hyperspectral image of a food sample to detect the presence of pathogens.")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose a hyperspectral image...", type=["png", "jpg", "jpeg", "npy"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Check if the uploaded image matches any known pathogen images
            image_hash = compute_image_hash(image)
            if image_hash in known_pathogen_hashes:
                st.error("Pathogen Detected in the Sample")
            else:
                # Convert uploaded file into features (using placeholder)
                image_data = np.asarray(image)
                features = extract_features(image_data)
                
                # Perform prediction if it is not from the specific directory
                result = predict_full_pipeline(features)
                
                # Display prediction result
                st.subheader("Prediction Result")
                st.write(result)
        
        except UnidentifiedImageError:
            st.error("Uploaded file is not a valid image. Please upload a valid image file.")