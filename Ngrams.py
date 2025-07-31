import pandas as pd
import streamlit as st
from thefuzz import fuzz # <- Usaremos esta librería para la comparación

@st.cache_data
def cargar_base_de_datos():
    """Carga la base de datos desde el CSV y la retorna como una lista de marcas."""
    try:
        df = pd.read_csv("base_expandida.csv")
        return df.iloc[:, 0].dropna().str.lower().tolist()
    except FileNotFoundError:
        st.error("Error en Módulo N-Grams: No se encontró el archivo 'base_expandida.csv'.")
        return []
    except IndexError:
        st.error("Error en Módulo N-Grams: El archivo 'base_expandida.csv' parece estar vacío.")
        return []

def buscar_marcas_similares(marca_input):
    """
    Busca marcas similares usando un ratio de similitud de Levenshtein,
    que es ideal para encontrar errores de tipeo o pequeñas variaciones.
    """
    marcas_base = cargar_base_de_datos()
    if not marcas_base:
        return []

    # Normalizar la entrada
    input_lower = marca_input.lower()
    
    resultados = []
    for marca in marcas_base:
        # --- CAMBIO CLAVE: Usamos fuzz.ratio en lugar de Jaccard ---
        # Compara directamente las dos palabras y devuelve un score de 0 a 100.
        similitud = fuzz.ratio(input_lower, marca)
        
        resultados.append((marca, similitud))
        
    return resultados
