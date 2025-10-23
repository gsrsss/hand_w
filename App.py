import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os

# --- Configuración de la Página ---
st.set_page_config(page_title='Reconocimiento de Dígitos', layout='centered')

# --- Función de Predicción ---
# Usar st.cache_resource para cargar el modelo solo una vez
@st.cache_resource
def load_app_model():
    try:
        # Añadimos compile=False para ignorar el optimizador,
        # lo que a menudo soluciona errores de carga HDF5.
        model = tf.keras.models.load_model("model/handwritten.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo 'model/handwritten.h5': {e}")
        st.warning("Asegúrate de que el archivo 'model/handwritten.h5' exista y no esté corrupto.")
        return None

model = load_app_model()

def predictDigit(image, model):
    if model is None:
        return None
        
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32')
    img = img / 255.0  # Normalizar
    
    # --- CORRECCIÓN IMPORTANTE ---
    # El modelo MNIST espera dígitos BLANCOS sobre fondo NEGRO.
    # Si dibujamos (naturalmente) NEGRO sobre BLANCO, debemos invertir la imagen.
    img = 1.0 - img # Descomentar si la precisión es baja
    
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# --- Interfaz de la App ---
st.title('Reconocimiento de Dígitos ( ˆ𐃷ˆ) .ᐟ.ᐟ')
st.subheader("Dibuja un dígito en el panel y presiona 'Predecir'")

# Crear directorios si no existen (para guardar imagen temporal)
os.makedirs("prediction", exist_ok=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### 🖌️ Configuración")
    stroke_width = st.slider('Ancho de línea:', 1, 30, 15)
    # Cambiamos los defaults a negro sobre blanco (más natural)
    stroke_color = st.color_picker('Color del trazo:', '#000000')
    bg_color = st.color_picker('Color de fondo:', '#FFFFFF')
    canvas_size = st.number_input('Tamaño del lienzo (px):', min_value=100, max_value=500, value=280)
    st.info("Dibuja un solo dígito (0-9).")

with col2:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", # No se usa en modo 'freedraw'
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=int(canvas_size),
        width=int(canvas_size),
        drawing_mode="freedraw",
        key="canvas",
    )

# Botón de predicción centrado
st.write("") # Espacio
if st.button('Predecir Dígito'):
    if canvas_result.image_data is not None and model is not None:
        # Convertir datos del canvas a imagen PIL
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        
        # Guardar imagen temporalmente (opcional, pero útil para debug)
        input_image.save('prediction/img.png')
        
        # Predecir
        res = predictDigit(input_image, model)
        
        # Mostrar resultado
        st.header(f'El dígito predicho es: {res}')
        
    elif model is None:
        st.error("El modelo no se pudo cargar. No se puede predecir.")
    else:
        st.warning('Por favor, dibuja un dígito en el lienzo.')

