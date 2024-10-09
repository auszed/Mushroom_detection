import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

# Load your model
model = torch.load('model.pth')
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Function to make predictions
def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)

    probabilities = torch.softmax(output, dim=1)
    predicted = torch.argmax(probabilities, dim=1)
    predicted_label = predicted.item()
    predicted_probabilities = probabilities.squeeze().tolist()

    return predicted_label, predicted_probabilities


st.title("Mushroom Classifier Prototype")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
camera_option = st.checkbox("Enable Webcam")

if camera_option:
    # Capture image from webcam
    image_data = st.camera_input("Take a picture")
    if image_data:
        image = Image.open(image_data)
        st.image(image, caption='Captured Image', use_column_width=True)
        if st.button("Predict"):
            predicted_label, predicted_probabilities = predict(image)
            st.write(f"Predicted Class: {predicted_label}")

            st.write("Prediction Probabilities:")
            for i, prob in enumerate(predicted_probabilities):
                st.write(f"Class {i}: {prob:.4f}")

# If an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button("Predict"):
        predicted_label, predicted_probabilities = predict(image)
        st.write(f"Predicted Class: {predicted_label}")

        st.write("Prediction Probabilities:")
        for i, prob in enumerate(predicted_probabilities):
            st.write(f"Class {i}: {prob:.4f}")
