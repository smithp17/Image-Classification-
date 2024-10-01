import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

# Set up the title and description
st.title("Fruit and Vegetable Classifier")
st.header("Identify fruits and vegetables with a deep learning model!")
st.text("Upload an image of a fruit or vegetable, and the model will predict what it is.")

# Load the trained model
model = load_model(r'C:\Users\Smith\OneDrive\Desktop\Fruits_Vegetables\Image_classify.keras')

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
    # Load the image
    image_load = tf.keras.utils.load_img(uploaded_file, target_size=(180, 180))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

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
