import pandas as pd
import streamlit as st
import re

# --- Optimización de Rendimiento ---
# Usamos el caché de Streamlit para leer el CSV solo una vez y reutilizarlo.
# Esto hace que las búsquedas posteriores sean casi instantáneas.
@st.cache_data
def cargar_base_de_datos():
    """Carga la base de datos desde el CSV y la retorna como una lista de marcas."""
    try:
        df = pd.read_csv("base_expandida.csv")
        # Asegurarnos de que no hay valores nulos y están en minúsculas
        return df["Marcas"].dropna().str.lower().tolist()
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo 'base_expandida.csv'. Asegúrate de que esté en la carpeta correcta.")
        return []
    except KeyError:
        st.error("Error: El archivo 'base_expandida.csv' no contiene una columna llamada 'Marcas'.")
        return []

def _crear_ngrams(texto, n=3):
    """
    Función auxiliar para convertir un string en un conjunto de n-gramas.
    Ej: 'cocacola' con n=3 -> {'coc', 'oca', 'cac', 'aco', 'col', 'ola'}
    """
    # Limpieza simple para mejorar la calidad de los n-grams
    texto_limpio = re.sub(r'[^a-z0-9]+', '', texto.lower())
    if len(texto_limpio) < n:
        return {texto_limpio}
    
    return {texto_limpio[i:i+n] for i in range(len(texto_limpio) - n + 1)}

def _jaccard_similarity(set_a, set_b):
    """Calcula la similitud Jaccard entre dos conjuntos."""
    interseccion = set_a.intersection(set_b)
    union = set_a.union(set_b)
    
    # Evitar división por cero si ambos sets están vacíos
    if not union:
        return 1.0 if not set_a and not set_b else 0.0
        
    return len(interseccion) / len(union)

def buscar_marcas_similares(marca_input, n=3):
    """
    Busca marcas similares en la base de datos usando similitud Jaccard sobre n-gramas.
    Esta función imita la estructura de tus otros modelos.
    """
    marcas_base = cargar_base_de_datos()
    if not marcas_base:
        return [] # Retorna vacío si no se pudo cargar la base

    # Pre-calcular n-grams de la marca de entrada para eficiencia
    ngrams_input = _crear_ngrams(marca_input, n)
    
    resultados = []
    for marca in marcas_base:
        ngrams_marca = _crear_ngrams(marca, n)
        similitud = _jaccard_similarity(ngrams_input, ngrams_marca)
        
        # El resultado de Jaccard es de 0.0 a 1.0, lo convertimos a porcentaje (0-100)
        # para que sea consistente con tus otros modelos.
        resultados.append((marca, similitud * 100))
        
    return resultados
