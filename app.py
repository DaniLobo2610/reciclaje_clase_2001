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
MOEL_PATHS=[MODEL_DIR/"waste_mobilnet.h5", MODELO_DIR/"waste_mobilnet.keras"]
