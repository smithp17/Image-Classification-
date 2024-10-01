import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
import gdown  # For downloading the model from Google Drive

# Set up the title and description
st.title("Fruit and Vegetable Classifier")
st.header("Identify fruits and vegetables with a deep learning model!")
st.text("Upload an image of a fruit or vegetable, and the model will predict what it is.")

# Load the trained model
@st.cache_resource
def load_fruit_vegetable_model():
    # URL of the model stored in Google Drive
    url = 'https://drive.google.com/uc?id=14PYrsgWeILvax9r2w5ZmAxONvAua_4mD'
    output = 'Image_classify.keras'
    
    # Download the model if it doesn't exist
    gdown.download(url, output, quiet=False)

    # Load the model
    model = load_model(output)
    return model

# Load model
model = load_fruit_vegetable_model()

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
