import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# --- Optimización con Caché ---
# @st.cache_resource para cargar el modelo pesado una sola vez.
@st.cache_resource
def cargar_modelo_beto():
    """Carga el modelo BETO (como SentenceTransformer) una sola vez."""
    try:
        # BETO es un modelo base para español. Lo usamos a través de un wrapper 
        # que lo hace compatible con sentence-transformers para tareas de similitud.
        # "dccuchile/bert-base-spanish-wwm-cased" es el BETO original.
        # Lo usamos con un modelo de pooling para obtener embeddings de sentencias.
        model = SentenceTransformer("dccuchile/bert-base-spanish-wwm-cased")
        return model
    except Exception as e:
        st.error(f"⚠️ Error fatal al cargar el modelo BETO: {e}")
        return None

# @st.cache_data para procesar y codificar los datos una sola vez.
@st.cache_data
def cargar_y_codificar_datos_beto():
    """Carga el CSV, extrae los textos y los codifica usando BETO."""
    model = cargar_modelo_beto()
    if model is None:
        return [], None

    try:
        # --- CAMBIO CLAVE: Lectura robusta del archivo CSV ---
        df = pd.read_csv(
            "base_expandida.csv",
            header=None,
            names=['marca'],
            quotechar='"',
            engine='python',
            sep=';'
        )
        
        # Lee la primera columna, la limpia y la convierte a lista
        marca_textos = df.iloc[:, 0].dropna().astype(str).str.lower().tolist()
        
        # Determina si usar GPU o CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Codifica todos los textos en un solo paso
        marca_embeddings = model.encode(marca_textos, convert_to_tensor=True, device=device)
        
        return marca_textos, marca_embeddings
    except FileNotFoundError:
        st.error("Error en Módulo BETO: No se encontró el archivo 'base_expandida.csv'.")
        return [], None
    except IndexError:
        st.error("Error en Módulo BETO: El archivo 'base_expandida.csv' parece estar vacío.")
        return [], None

# --- Función Principal de Búsqueda (exportada para la app principal) ---
def buscar_marcas_similares(input_marca):
    """
    Busca marcas similares usando el modelo BETO pre-cargado.
    """
    model = cargar_modelo_beto()
    marca_textos, marca_embeddings = cargar_y_codificar_datos_beto()

    # Si la carga falló, retorna una lista vacía para evitar errores
    if model is None or marca_embeddings is None:
        return []

    # Codifica la marca de entrada y calcula la similitud del coseno
    input_embedding = model.encode(input_marca.lower(), convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(input_embedding, marca_embeddings)[0]

    # Combina los textos con sus puntuaciones y los ordena de mayor a menor
    resultados = sorted(
        zip(marca_textos, similitudes),
        key=lambda x: x[1],
        reverse=True
    )

    # Devuelve la lista de resultados con el puntaje en formato de 0 a 100
    return [(marca, score.item() * 100) for marca, score in resultados]
