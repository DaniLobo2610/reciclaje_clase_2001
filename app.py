import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Reciclaje IA-ISC", layout="centered")
st.title("Modelo Predictivo Reciclaje clase de IA-ISC-Campus Comayagua-2026")
st.write("Suba una imagen para clasificar con el modelo MobilnetV2 pre entrenado")
st.write("ALBERTO DANIEL LOBO CHAVARRIA")

IMG_SIZE=(224,224)
MODEL_DIR=Path("Modelo_reciclaje_mobilnet")
CLASS_PATH=MODEL_DIR/"waste_mobilnet.json"
MODEL_PATHS=[MODEL_DIR/"waste_mobilnet.h5", MODEL_DIR/"waste_mobilnet.keras"]

LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura",
}

@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path, compile=False)
    st.error("No se encontró el modelo. Coloque la carpeta Modelo_reciclaje_mobilnet junto a app.py.")
    st.stop()

@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    top3 = np.argsort(preds)[-3:][::-1]
    return [
        (LABELS_ES.get(clases[i], clases[i]), float(preds[i]) * 100)
        for i in top3
    ]

modelo = cargar_modelo()
clases = cargar_clases()

archivo = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"])

if archivo:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen analizada", use_container_width=True)

    resultados = predecir(imagen)
    st.subheader("Resultado")
    st.success(f"Predicción principal: {resultados[0][0]} ({resultados[0][1]:.2f}%)")

    st.write("Top 3 probabilidades:")
    for clase, prob in resultados:
        st.write(f"{clase}: {prob:.2f}%")
else:
    st.info("Cargue una imagen para iniciar la clasificación.")
