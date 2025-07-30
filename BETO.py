import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# --- Optimización con Caché ---
# @st.cache_resource se usa para objetos pesados como los modelos de ML.
# Se carga solo una vez y permanece en memoria para toda la sesión del usuario.
@st.cache_resource
def cargar_modelo_sbert():
    """Carga el modelo de Sentence-BERT una sola vez y lo mantiene en memoria."""
    try:
        # Modelo SBERT optimizado para similitud en español
        model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")
        return model
    except Exception as e:
        st.error(f"⚠️ Error fatal al cargar el modelo SBERT: {e}")
        return None

# @st.cache_data se usa para datos que no cambian, como el CSV procesado.
# Carga y procesa los datos solo una vez. Si el archivo cambia, lo vuelve a ejecutar.
@st.cache_data
def cargar_y_codificar_datos():
    """Carga el CSV, extrae los textos y los codifica en vectores (embeddings)."""
    model = cargar_modelo_sbert()
    if model is None:
        return [], None # Retorna tuplas vacías si el modelo no cargó

    try:
        df = pd.read_csv("base_expandida.csv")
        # --- CAMBIO CLAVE: Lee la primera columna sin importar el nombre ---
        marca_textos = df.iloc[:, 0].dropna().astype(str).str.lower().tolist()
        
        # Codificar todos los textos a la vez (mucho más rápido)
        # Se usa el dispositivo de la GPU si está disponible (en la nube usualmente es CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        marca_embeddings = model.encode(marca_textos, convert_to_tensor=True, device=device)
        
        return marca_textos, marca_embeddings
    except FileNotFoundError:
        st.error("Error en Módulo Semántico: No se encontró el archivo 'base_expandida.csv'.")
        return [], None
    except IndexError:
        st.error("Error en Módulo Semántico: El archivo 'base_expandida.csv' parece estar vacío.")
        return [], None

def buscar_marcas_similares(input_marca):
    """
    Busca marcas similares usando el modelo SBERT pre-cargado.
    """
    model = cargar_modelo_sbert()
    marca_textos, marca_embeddings = cargar_y_codificar_datos()

    # Si algo falló en la carga, no continuar.
    if model is None or marca_embeddings is None:
        return []

    # Codificar la marca de entrada y calcular similitud
    input_embedding = model.encode(input_marca.lower(), convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(input_embedding, marca_embeddings)[0]

    # Combinar textos y scores, y ordenarlos
    resultados = sorted(
        zip(marca_textos, similitudes),
        key=lambda x: x[1],
        reverse=True
    )

    # Devolver la lista completa de resultados con su score en formato de 0 a 100
    return [(marca, score.item() * 100) for marca, score in resultados]
