import pandas as pd
import streamlit as st
import re
from thefuzz import fuzz # Asegúrate de que thefuzz esté importado si lo usas aquí

@st.cache_data
def cargar_base_de_datos():
    """Carga la base de datos desde el CSV y la retorna como una lista de marcas."""
    try:
        df = pd.read_csv("base_expandida.csv")
        # --- CAMBIO CLAVE ---
        # Leemos la primera columna (índice 0), sin importar su nombre.
        return df.iloc[:, 0].dropna().str.lower().tolist()
    except FileNotFoundError:
        st.error("Error en Módulo N-Grams: No se encontró el archivo 'base_expandida.csv'.")
        return []
    except IndexError:
        st.error("Error en Módulo N-Grams: El archivo 'base_expandida.csv' parece estar vacío.")
        return []

def _crear_ngrams(texto, n=3):
    """Función auxiliar para convertir un string en un conjunto de n-gramas."""
    texto_limpio = re.sub(r'[^a-z0-9]+', '', texto.lower())
    if len(texto_limpio) < n:
        return {texto_limpio}
    return {texto_limpio[i:i+n] for i in range(len(texto_limpio) - n + 1)}

def _jaccard_similarity(set_a, set_b):
    """Calcula la similitud Jaccard entre dos conjuntos."""
    interseccion = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if not union:
        return 1.0 if not set_a and not set_b else 0.0
    return len(interseccion) / len(union)

def buscar_marcas_similares(marca_input, n=3):
    """Busca marcas similares usando similitud Jaccard sobre n-gramas."""
    marcas_base = cargar_base_de_datos()
    if not marcas_base:
        return []
        
    ngrams_input = _crear_ngrams(marca_input, n)
    resultados = []
    for marca in marcas_base:
        ngrams_marca = _crear_ngrams(marca, n)
        similitud = _jaccard_similarity(ngrams_input, ngrams_marca)
        resultados.append((marca, similitud * 100))
        
    return resultados
