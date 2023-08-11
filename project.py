import streamlit as st
import streamlit as st
import ssl

# Отключение проверки SSL-сертификата
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(
    page_title='Проект. Введение в нейронные сети',
    layout='wide'
)

st.sidebar.header("Home page")
c1, c2 = st.columns(2)
c2.image('neiro2.jpg')
c1.markdown("""
# Проект. Введение в нейронные сети
Cостоит из 3 частей:
### 1. Классификация произвольного изображения с помощью модели Inception (обученной на датасете ImageNet)
### 2. Классификация изображений котов и собак дообученной моделью ResNet18
### 3. Классификация птиц по фотографиям
""")