import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model/mnist_cnn.h5")

st.title("Handwritten Digit Recognition")
st.write("Upload a digit image (0-9)")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded Image", width=200)

    image = image.resize((28, 28))

    image_array = np.array(image)

    # Invert colors because MNIST is white digit on black background
    if np.mean(image_array) > 127:
         image_array = 255 - image_array

    image_array = image_array.astype("float32") / 255.0
    if image_array.max() > 0:
         image_array = image_array / image_array.max()
  
    st.image(
         image_array,
         caption="Processed Image Used By Model",
         width=150
    )

    image_array = image_array.reshape(1, 28, 28, 1)

    prediction = model.predict(image_array)

    digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(f"Predicted Digit: {digit}")

    st.write(f"Confidence: {confidence:.2f}%")