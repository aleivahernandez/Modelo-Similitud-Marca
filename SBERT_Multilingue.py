import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

## ----------------- Optimización con Caché de Streamlit -----------------

# @st.cache_resource se usa para cargar el modelo pesado una sola vez por sesión.
@st.cache_resource
def cargar_modelo_sbert():
    """Carga el modelo multilingüe de Sentence-BERT y lo mantiene en memoria."""
    try:
        # Este modelo es eficiente y funciona bien para múltiples idiomas, incluido el español.
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return model
    except Exception as e:
        st.error(f"⚠️ Error fatal al cargar el modelo SBERT: {e}")
        return None

# @st.cache_data se usa para procesar los datos. Se ejecuta solo si los datos cambian.
@st.cache_data
def cargar_y_codificar_datos():
    """Carga el CSV, extrae los textos de la primera columna y los codifica."""
    model = cargar_modelo_sbert()
    if model is None:
        return [], None

    try:
        # --- CAMBIO CLAVE: Se hace la lectura del CSV más robusta ---
        df = pd.read_csv(
            "base_expandida.csv",
            header=None,         # Indica que no hay una fila de encabezado.
            names=['marca'],       # Nombra la única columna como "marca".
            quotechar='"',       # Trata el contenido dentro de comillas como un solo elemento.
            engine='python',      # Usa un motor de lectura más flexible.
            sep=';'
        )
        
        # Se lee la primera columna (índice 0) sin importar su nombre
        marca_textos = df.iloc[:, 0].dropna().astype(str).str.lower().tolist()
        
        # Determinar el dispositivo (GPU si está disponible, si no CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Codificar todos los textos a la vez es más eficiente
        marca_embeddings = model.encode(marca_textos, convert_to_tensor=True, device=device)
        
        return marca_textos, marca_embeddings
    except FileNotFoundError:
        st.error("Error en Módulo SBERT: No se encontró el archivo 'base_expandida.csv'.")
        return [], None
    except IndexError:
        st.error("Error en Módulo SBERT: El archivo 'base_expandida.csv' parece estar vacío.")
        return [], None

## ----------------- Función Principal de Búsqueda -----------------

def buscar_marcas_similares(input_marca):
    """
    Busca marcas similares usando el modelo SBERT multilingüe pre-cargado.
    """
    model = cargar_modelo_sbert()
    marca_textos, marca_embeddings = cargar_y_codificar_datos()

    # Si el modelo o los datos no se cargaron, retorna una lista vacía.
    if model is None or marca_embeddings is None:
        return []

    # Codificar la marca de entrada y calcular la similitud del coseno
    input_embedding = model.encode(input_marca.lower(), convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(input_embedding, marca_embeddings)[0]

    # Combinar los textos con sus puntuaciones de similitud
    resultados = sorted(
        zip(marca_textos, similitudes),
        key=lambda x: x[1],
        reverse=True
    )

    # Retorna la lista completa de resultados, convirtiendo el score a un rango de 0-100.
    return [(marca, score.item() * 100) for marca, score in resultados]

