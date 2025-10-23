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

# --- Estilos CSS Personalizados ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 720px;
    }
    h1, h2, h3 {
        text-align: center;
    }
    /* Contenedor para el canvas */
    .canvas-container {
        border: 2px dashed #7f8c8d;
        border-radius: 12px;
        padding: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    /* Estilo del canvas (st_canvas crea su propio iframe/canvas) */
    [key="canvas"] {
        border-radius: 8px;
    }
    /* Botón de Predecir */
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        width: 100%; /* Ocupa todo el ancho del contenedor */
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 6px 15px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    /* Resultado de la predicción */
    .result-box {
        background-color: #e6f7ff; /* Celeste claro */
        border: 1px solid #b3e0ff;
        border-radius: 12px;
        padding: 25px;
        color: #0056b3; /* Azul oscuro */
        text-align: center;
        margin-top: 20px;
    }
    .result-box h2 {
        color: #0056b3;
        margin: 0;
        font-size: 2.5rem; /* Tamaño grande para el número */
    }
    .result-box p {
        margin: 0;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Función de Predicción ---
# Usar st.cache_resource para cargar el modelo solo una vez
@st.cache_resource
def load_app_model():
    try:
        model = tf.keras.models.load_model("model/handwritten.h5")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo 'model/handwritten.h5': {e}")
        st.warning("Asegúrate de que el archivo 'model/handwritten.h5' exista.")
        return None

model = load_app_model()

def predictDigit(image, model):
    if model is None:
        return None
        
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32')
    img = img / 255.0  # Normalizar
    
    # Invertir colores si es necesario (el modelo MNIST espera dígitos blancos sobre fondo negro)
    # st_canvas da blanco sobre negro, así que esto podría no ser necesario
    # Si el fondo del canvas es blanco y el trazo negro, hay que invertir
    # img = 1.0 - img # Descomentar si la precisión es baja
    
    # plt.imshow(img) # Evitar plt.show() en Streamlit
    # plt.show()
    
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# --- Interfaz de la App ---
st.title('✍️ Reconocimiento de Dígitos')
st.subheader("Dibuja un dígito en el panel y presiona 'Predecir'")

# Crear directorios si no existen (para guardar imagen temporal)
os.makedirs("prediction", exist_ok=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### 🖌️ Configuración")
    stroke_width = st.slider('Ancho de línea:', 1, 30, 15)
    st.info("Dibuja un solo dígito (0-9) en el lienzo negro.")

with col2:
    st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", # No se usa en modo 'freedraw'
        stroke_width=stroke_width,
        stroke_color='#FFFFFF', # Color del trazo (blanco)
        background_color='#000000', # Color de fondo (negro)
        height=280, # Tamaño más grande para dibujar mejor
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    st.markdown('</div>', unsafe_allow_html=True)

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
        st.markdown(f"""
        <div class="result-box">
            <p>El dígito predicho es:</p>
            <h2>{res}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    elif model is None:
        st.error("El modelo no se pudo cargar. No se puede predecir.")
    else:
        st.warning('Por favor, dibuja un dígito en el lienzo.')

# "Acerca de" movido a un expander
st.write("") # Espacio
with st.expander("ℹ️ Acerca de esta App"):
    st.text("En esta aplicación se evalúa la capacidad de una Red")
    st.text("Neuronal Artificial (RNA) de reconocer dígitos escritos a mano.")
    st.text("Basado en el desarrollo de Vinay Uniyal.")
    # st.write("[GitHub Repo Link](https://github.com/Vinay2022/Handwritten-Digit-Recognition)")
