import streamlit as st
import requests
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from io import BytesIO
from torchvision.transforms import ToTensor
import streamlit as st

model_resnet18 = models.resnet18()
model_resnet18.fc = nn.Linear(512, 1)
device = torch.device('cpu')
model_resnet18.load_state_dict(torch.load('model/model_resnet18.pt', map_location=device))
model_resnet18.eval()

def process_image_resnet(image):
    image = image.resize((224, 224))
    image = ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model_resnet18(image)
        predicted_prob = torch.sigmoid(outputs)
    return predicted_prob.item()

st.sidebar.markdown(
    "# Классификация изображений котов и собак дообученной моделью ResNet18")

st.title("Загрузите изображение (кота/собаки) или вставьте ссылку на него")

upload_option = st.radio("Выберите способ загрузки:", ("Загрузить изображение", "Вставить ссылку"))

if upload_option == "Загрузить изображение":
    uploaded_file = st.file_uploader("Выберите файл (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        predicted_prob = process_image_resnet(image)
        if predicted_prob >= 0.5:
            st.write("Это ПСИНА")
        else:
            st.write("Это кот")

else:
    image_url = st.text_input("Введите URL изображения и нажмите Enter")
    if image_url:
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.image(image, caption='Загруженное изображение', use_column_width=True)
                predicted_prob = process_image_resnet(image)
                if predicted_prob >= 0.5:
                    st.write("Это ПСИНА")
                else:
                    st.write("Это кот")
            else:
                st.write("Не удалось загрузить изображение. Пожалуйста, проверьте URL и повторите попытку.")
        except requests.exceptions.RequestException as e:
            st.write("Произошла ошибка при запросе. Пожалуйста, проверьте ваше интернет-соединение и URL.")