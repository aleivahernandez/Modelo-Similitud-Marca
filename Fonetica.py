import pandas as pd
import streamlit as st
import jellyfish # <- CAMBIO: Se usa la librería jellyfish
from thefuzz import fuzz

@st.cache_data
def cargar_base_de_datos():
    """Carga la base de datos desde el CSV y la retorna como una lista de marcas."""
    try:
        df = pd.read_csv("base_expandida.csv")
        return df.iloc[:, 0].dropna().str.lower().tolist()
    except FileNotFoundError:
        st.error("Error en Módulo Fonético: No se encontró el archivo 'base_expandida.csv'.")
        return []
    except IndexError:
        st.error("Error en Módulo Fonético: El archivo 'base_expandida.csv' parece estar vacío.")
        return []

def buscar_marcas_similares(marca_input):
    """
    Busca marcas fonéticamente similares usando el algoritmo Metaphone de jellyfish.
    """
    marcas_base = cargar_base_de_datos()
    if not marcas_base:
        return []

    # Obtener el código fonético de la marca de entrada usando jellyfish
    codigo_input = jellyfish.metaphone(marca_input)
    
    resultados = []
    for marca in marcas_base:
        # Obtener el código fonético de cada marca en la base de datos
        codigo_marca = jellyfish.metaphone(marca)
        
        # Usar thefuzz para obtener un porcentaje de similitud entre los códigos fonéticos
        similitud = fuzz.ratio(codigo_input, codigo_marca)
        
        resultados.append((marca, similitud))
        
    return resultados
