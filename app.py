import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
import requests
import os
from tqdm import tqdm

# Helper function to download large files from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

# Set up the title and description
st.title("Fruit and Vegetable Classifier")
st.header("Identify fruits and vegetables with a deep learning model!")
st.text("Upload an image of a fruit or vegetable, and the model will predict what it is.")

# Load the trained model
@st.cache_resource
def load_fruit_vegetable_model():
    file_id = '14PYrsgWeILvax9r2w5ZmAxONvAua_4mD'  # Replace with your Google Drive file ID
    output = 'Image_classify.keras'
    
    # Check if the model already exists in the environment
    if not os.path.exists(output):
        st.write("Downloading model file...")
        download_file_from_google_drive(file_id, output)

    # Check if the model was downloaded successfully
    if os.path.exists(output):
        # Ensure that the file size is larger than a minimum value to avoid corrupted file issues
        if os.path.getsize(output) < 1e6:  # Example: 1 MB as a minimum file size
            st.error("The downloaded model file seems corrupted or incomplete. Please re-upload it.")
            return None
    else:
        st.error("Model file could not be found. Ensure the file is accessible and correctly downloaded.")
        return None

    # Load the model
    model = load_model(output)
    return model

# Load model
model = load_fruit_vegetable_model()

# Check if model loaded successfully
if model is not None:
    # Define the categories
    data_cat = [
        'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
        'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 
        'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 
        'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 
        'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 
        'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 
        'turnip', 'watermelon'
    ]

    # File uploader for image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image and preprocess it
        image_load = tf.keras.utils.load_img(uploaded_file, target_size=(180, 180))
        img_arr = tf.keras.utils.img_to_array(image_load)
        img_arr = img_arr / 255.0  # Normalize the image
        
        img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension

        # Display the uploaded image
        st.image(image_load, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Make a prediction when the user clicks the button
        if st.button('Predict'):
            predict = model.predict(img_bat)
            score = tf.nn.softmax(predict[0])

            # Display the prediction
            st.write(
                f"**Prediction:** {data_cat[np.argmax(score)]} "
                f"with an **accuracy** of {float(np.max(score)) * 100:.2f}%"
            )
else:
    st.text("Please upload an image to classify.")
