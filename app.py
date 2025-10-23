
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Fish Image Classifier", page_icon="🐟")
st.title("🐟 Fish Image Classification App")

model = tf.keras.models.load_model("C:/Ganapathy/Proj 5/models/best_model.h5")
class_names = ['animal fish', 'animal fish bass', 'black sea sprat', 'gilt head bream', 'hourse mackerel', 'red mullet', 'red sea bream', 'sea bass', 'shrimp', 'striped red mullet', 'trout']  
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224,224))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)*100

    st.markdown(f"### 🐠 Predicted Fish: **{predicted_class}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}%**")

