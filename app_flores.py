"""
🌸 Clasificador de Flores con Transfer Learning
App de Streamlit que usa el modelo entrenado en el notebook de Colab.

Para ejecutar:
    pip install streamlit tensorflow pillow
    streamlit run app_flores.py

Archivos necesarios (descargados desde Colab):
    - flower_model.keras
    - class_names.json
"""

import streamlit as st
import numpy as np
import json
import os
from PIL import Image

# ─── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Flores",
    page_icon="🌸",
    layout="centered"
)

# ─── Estilos ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { max-width: 700px; margin: auto; }
    .result-box {
        background: #f0f7ff;
        border-left: 4px solid #4A90D9;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 16px;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8em;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Emojis por clase ────────────────────────────────────────────────────────
EMOJIS = {
    "daisy": "🌼",
    "dandelion": "🌱",
    "roses": "🌹",
    "sunflowers": "🌻",
    "tulips": "🌷"
}

DESCRIPCIONES = {
    "daisy": "Margarita — flores blancas con centro amarillo",
    "dandelion": "Diente de León — flor silvestre amarilla",
    "roses": "Rosa — la reina de las flores",
    "sunflowers": "Girasol — siempre mirando al sol",
    "tulips": "Tulipán — elegancia en primavera"
}

# ─── Carga del modelo (con caché para no recargar) ──────────────────────────
@st.cache_resource
def cargar_modelo():
    """Carga el modelo y las clases. Se ejecuta solo una vez gracias al caché."""
    try:
        import tensorflow as tf

        if not os.path.exists("flower_model.keras"):
            return None, None, "❌ No se encontró 'flower_model.keras'. Descárgalo desde Colab."

        if not os.path.exists("class_names.json"):
            return None, None, "❌ No se encontró 'class_names.json'. Descárgalo desde Colab."

        modelo = tf.keras.models.load_model("flower_model.keras")

        with open("class_names.json", "r") as f:
            clases = json.load(f)

        return modelo, clases, None

    except ImportError:
        return None, None, "❌ TensorFlow no está instalado. Ejecuta: pip install tensorflow"
    except Exception as e:
        return None, None, f"❌ Error al cargar el modelo: {e}"


def predecir(imagen_pil, modelo, clases):
    """Preprocesa la imagen y devuelve las predicciones."""
    import tensorflow as tf

    # Redimensionar a 224x224 (tamaño esperado por MobileNetV2)
    img = imagen_pil.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # Añade dimensión de batch: (1, 224, 224, 3)

    predicciones = modelo.predict(arr, verbose=0)[0]
    return predicciones


# ─── Interfaz principal ──────────────────────────────────────────────────────
st.title("🌸 Clasificador de Flores")
st.markdown("*Basado en Transfer Learning con MobileNetV2*")
st.divider()

# Cargar modelo
with st.spinner("Cargando modelo..."):
    modelo, clases, error = cargar_modelo()

if error:
    st.error(error)
    st.info("""
    **¿Cómo obtener los archivos?**
    1. Ejecuta el notebook `transfer_learning_colab.ipynb` en Google Colab
    2. Al final del notebook, los archivos se descargan automáticamente
    3. Coloca `flower_model.keras` y `class_names.json` en la misma carpeta que esta app
    """)
    st.stop()

st.success(f"✅ Modelo listo — {len(clases)} clases: {', '.join(clases)}")
st.divider()

# ─── Subir imagen ────────────────────────────────────────────────────────────
st.subheader("📷 Sube una imagen de flor")

uploaded = st.file_uploader(
    label="Formatos soportados: JPG, PNG, WEBP",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded is not None:
    imagen = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(imagen, caption="Imagen subida", use_container_width=True)

    with col2:
        with st.spinner("Analizando..."):
            predicciones = predecir(imagen, modelo, clases)

        idx_max = int(np.argmax(predicciones))
        clase_pred = clases[idx_max]
        confianza = float(predicciones[idx_max])
        emoji = EMOJIS.get(clase_pred, "🌺")
        desc = DESCRIPCIONES.get(clase_pred, clase_pred)

        st.markdown(f"""
        <div class="result-box">
            <h2 style="margin:0">{emoji} {clase_pred.capitalize()}</h2>
            <p style="color:#555; margin:4px 0 12px">{desc}</p>
            <p style="font-size:1.1em">Confianza: <strong>{confianza:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Barra de progreso de confianza
        st.markdown("**Confianza del modelo:**")
        color = "green" if confianza > 0.7 else "orange" if confianza > 0.4 else "red"
        st.progress(confianza)

    # ─── Probabilidades de todas las clases ──────────────────────────────
    st.divider()
    st.subheader("📊 Probabilidades por clase")

    # Ordenamos de mayor a menor
    indices_ord = np.argsort(predicciones)[::-1]

    for i in indices_ord:
        clase = clases[i]
        prob = float(predicciones[i])
        emoji = EMOJIS.get(clase, "🌺")
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.markdown(f"{emoji} **{clase.capitalize()}**")
        with col_b:
            st.progress(prob, text=f"{prob:.1%}")

# ─── Información educativa (expandible) ──────────────────────────────────────
with st.expander("ℹ️ ¿Cómo funciona este clasificador?"):
    st.markdown("""
    Este modelo usa **Transfer Learning** con la arquitectura **MobileNetV2**:

    1. **MobileNetV2** fue pre-entrenado en ImageNet (1.4M imágenes, 1000 clases)
    2. Se reutilizaron sus capas convolucionales como **extractor de características**
    3. Se añadieron nuevas capas finales entrenadas solo con imágenes de **flores**
    4. Se realizó **fine-tuning** de las últimas capas para adaptar el modelo al dominio

    **Clases que puede reconocer:** Daisy 🌼, Dandelion 🌱, Roses 🌹, Sunflowers 🌻, Tulips 🌷
    """)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Desarrollado con TensorFlow + Streamlit · Transfer Learning con MobileNetV2
</div>
""", unsafe_allow_html=True)
