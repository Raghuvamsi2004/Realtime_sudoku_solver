import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the model
model = load_model('data/model/digit_recognition_model.h5')


def preprocess_image(image):
    # Example preprocessing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Additional preprocessing steps
    return image


def predict_digit(image):
    # Preprocess image
    processed_image = preprocess_image(image)

    # Example prediction logic
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=-1)  # Add channel dimension

    # Predict digits using the loaded model
    predictions = model.predict(processed_image)

    # Post-process predictions
    digit = np.argmax(predictions)

    return digit

