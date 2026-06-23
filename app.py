import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Reciclaje IA-ISC". layout="centered")
st.title("Modelo Predictivo Reciclaje clase de IA-ISC-Campus Comayagua-2026")
st.write("Suba una imagen para clasificar con el modelo MobilnetV2 pre entrenado")
