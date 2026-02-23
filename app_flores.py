"""
🌸 Clasificador de Flores con Transfer Learning
App de Streamlit que permite cargar el modelo entrenado desde la interfaz.

Para ejecutar:
    pip install streamlit tensorflow pillow
    streamlit run app_flores.py
"""

import streamlit as st
import numpy as np
import json
import os
import tempfile
from PIL import Image

# ─── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Flores",
    page_icon="🌸",
    layout="centered"
)

st.markdown("""
    <style>
    .result-box {
        background: #1e3a5f;
        border-left: 4px solid #4A90D9;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 8px;
        color: white;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8em;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Constantes ──────────────────────────────────────────────────────────────
EMOJIS = {
    "daisy": "🌼", "dandelion": "🌱",
    "roses": "🌹", "sunflowers": "🌻", "tulips": "🌷"
}
DESCRIPCIONES = {
    "daisy":      "Margarita — flores blancas con centro amarillo",
    "dandelion":  "Diente de León — flor silvestre amarilla",
    "roses":      "Rosa — la reina de las flores",
    "sunflowers": "Girasol — siempre mirando al sol",
    "tulips":     "Tulipán — elegancia en primavera"
}

# ─── Función de predicción ───────────────────────────────────────────────────
def predecir(imagen_pil, modelo):
    import tensorflow as tf
    img = imagen_pil.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    return modelo.predict(arr, verbose=0)[0]

# ─── Título ──────────────────────────────────────────────────────────────────
st.title("🌸 Clasificador de Flores")
st.markdown("*Basado en Transfer Learning con MobileNetV2*")
st.divider()

# ─── PASO 1: Cargar archivos del modelo ──────────────────────────────────────
st.subheader("⚙️ Paso 1 — Cargar el modelo entrenado")
st.caption("Sube los archivos generados al final del notebook de Colab")

col_m, col_c = st.columns(2)

with col_m:
    archivo_modelo = st.file_uploader(
        "📦 Modelo (`flower_model.keras`)",
        type=["keras", "h5"],
        key="modelo"
    )

with col_c:
    archivo_clases = st.file_uploader(
        "📋 Clases (`class_names.json`)",
        type=["json"],
        key="clases"
    )

# ─── Cargar modelo cuando ambos archivos estén disponibles ───────────────────
modelo = None
clases = None

if archivo_modelo and archivo_clases:
    try:
        import tensorflow as tf

        # Guardamos el modelo en un archivo temporal (Keras necesita leerlo del disco)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
            tmp.write(archivo_modelo.read())
            tmp_path = tmp.name

        with st.spinner("Cargando modelo... esto puede tardar unos segundos"):
            modelo = tf.keras.models.load_model(tmp_path)
        os.unlink(tmp_path)  # Borramos el temporal

        clases = json.load(archivo_clases)

        st.success(f"✅ Modelo listo — {len(clases)} clases: {', '.join(clases)}")

    except Exception as e:
        st.error(f"❌ Error al cargar: {e}")

elif archivo_modelo and not archivo_clases:
    st.warning("⚠️ Falta subir el archivo `class_names.json`")
elif archivo_clases and not archivo_modelo:
    st.warning("⚠️ Falta subir el archivo `flower_model.keras`")
else:
    st.info("👆 Sube los dos archivos para activar el clasificador")

# ─── PASO 2: Clasificar imagen (solo si el modelo está cargado) ──────────────
st.divider()
st.subheader("📷 Paso 2 — Clasificar una imagen")

if modelo is None:
    st.caption("Primero carga el modelo en el Paso 1")
else:
    uploaded = st.file_uploader(
        "Sube una foto de flor (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        imagen = Image.open(uploaded)
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(imagen, caption="Imagen subida", use_container_width=True)

        with col2:
            with st.spinner("Analizando..."):
                predicciones = predecir(imagen, modelo)

            idx_max = int(np.argmax(predicciones))
            clase_pred = clases[idx_max]
            confianza = float(predicciones[idx_max])
            emoji = EMOJIS.get(clase_pred, "🌺")
            desc = DESCRIPCIONES.get(clase_pred, clase_pred)

            st.markdown(f"""
            <div class="result-box">
                <h2 style="margin:0">{emoji} {clase_pred.capitalize()}</h2>
                <p style="margin:4px 0 12px; opacity:0.8">{desc}</p>
                <p style="font-size:1.1em">Confianza: <strong>{confianza:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.progress(confianza)

        # ─── Probabilidades ──────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Probabilidades por clase")

        for i in np.argsort(predicciones)[::-1]:
            clase = clases[i]
            prob = float(predicciones[i])
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.markdown(f"{EMOJIS.get(clase,'🌺')} **{clase.capitalize()}**")
            with col_b:
                st.progress(prob, text=f"{prob:.1%}")

# ─── Info educativa ──────────────────────────────────────────────────────────
with st.expander("ℹ️ ¿Cómo funciona este clasificador?"):
    st.markdown("""
    Este modelo usa **Transfer Learning** con **MobileNetV2**:

    1. MobileNetV2 fue pre-entrenado en ImageNet (1.4M imágenes, 1000 clases)
    2. Se reutilizaron sus capas como **extractor de características** (congeladas)
    3. Se añadieron nuevas capas entrenadas con ~3,700 imágenes de flores
    4. Se realizó **fine-tuning** de las últimas capas para ajuste fino

    **Clases:** Daisy 🌼 · Dandelion 🌱 · Roses 🌹 · Sunflowers 🌻 · Tulips 🌷
    """)

st.markdown('<div class="footer">Transfer Learning con MobileNetV2 · TensorFlow + Streamlit</div>', unsafe_allow_html=True)
