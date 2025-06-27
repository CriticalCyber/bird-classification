import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# Display the logo
logo = Image.open("meta/logo1.png")
st.image(logo, width=200)  # You can adjust the width as needed

# Load class names from file
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Load the trained model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load('bird_cnn_best.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Streamlit UI
st.title('Bird Species Recognition (Image-based)')
st.write('Upload a bird image to predict its species.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_species = class_names[predicted.item()]
    st.write('Predicted Species:', predicted_species)