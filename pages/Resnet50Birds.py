import json
import requests
import torch
import torchvision.models as models
import torch.nn as nn
from io import BytesIO
from PIL import Image
from torchvision.transforms import ToTensor
import streamlit as st

model_resnet50 = models.resnet50()
model_resnet50.fc = nn.Linear(2048, 200)
device = torch.device('cpu')
model_resnet50.load_state_dict(torch.load('model/model_birds.pt', map_location=device))
model_resnet50.eval()

labels = json.load(open('labels_birds.json'))
def decode(x): 
    return labels[str(x)][1]

def process_image_inception(image):
    image = image.resize((224, 224))
    image = ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model_resnet50(image)
        predicted_class = torch.argmax(outputs)
    return predicted_class.item()


st.sidebar.markdown(
    "# Классификация птиц по фотографиям")

st.title("Загрузите изображение или вставьте ссылку на него")

upload_option = st.radio("Выберите способ загрузки:", ("Загрузить изображение", "Вставить ссылку"))

if upload_option == "Загрузить изображение":
    uploaded_image = st.file_uploader("Выберите файл (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        predicted_class = process_image_inception(image)
        class_name = decode(predicted_class)
        st.write(f"Предсказанный класс: {class_name}")

else:
    image_url = st.text_input("Введите URL изображения и нажмите Enter")
    if image_url:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Загруженное изображение', use_column_width=True)
            predicted_class = process_image_inception(image)
            class_name = decode(predicted_class)
            st.write(f"Предсказанный класс: {class_name}")
        else:
            st.write("Не удалось загрузить изображение. Пожалуйста, проверьте URL и повторите попытку.")