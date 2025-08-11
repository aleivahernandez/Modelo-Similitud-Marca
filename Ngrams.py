import pandas as pd
import streamlit as st
from thefuzz import fuzz

@st.cache_data
def cargar_base_de_datos():
    """Carga la base de datos desde el CSV de forma robusta y la retorna como una lista."""
    try:
        # --- CAMBIO CLAVE: Lectura robusta del CSV para manejar comas ---
        df = pd.read_csv(
            "base_expandida.csv",
            header=None,
            names=['marca'],
            quotechar='"',
            engine='python',
            sep=',',
        )
        df['marca'] = df['marca'].str.rstrip(';').str.strip()
        return df.iloc[:, 0].dropna().str.lower().tolist()
    except FileNotFoundError:
        st.error("Error en Módulo N-Grams: No se encontró el archivo 'base_expandida.csv'.")
        return []
    except IndexError:
        st.error("Error en Módulo N-Grams: El archivo 'base_expandida.csv' parece estar vacío.")
        return []

def buscar_marcas_similares(marca_input):
    """
    Busca marcas similares usando un ratio de similitud (Levenshtein),
    ideal para encontrar errores de tipeo o pequeñas variaciones en los caracteres.
    """
    marcas_base = cargar_base_de_datos()
    if not marcas_base:
        return []

    # Normalizar la entrada para una comparación insensible a mayúsculas/minúsculas
    input_lower = marca_input.lower()
    
    resultados = []
    for marca in marcas_base:
        # Compara directamente las dos cadenas y devuelve un score de 0 a 100.
        # fuzz.ratio es efectivo para medir la similitud a nivel de caracteres.
        similitud = fuzz.ratio(input_lower, marca)
        
        # Guardamos la marca con su mayúscula original para mostrarla en los resultados
        resultados.append((marca.title(), similitud))
        
    return resultados
