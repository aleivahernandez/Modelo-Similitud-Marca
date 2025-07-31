import pandas as pd
import streamlit as st
from fonetika import Fonetika # Esto funcionará con la librería actualizada
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
    Busca marcas fonéticamente similares usando el algoritmo Metaphone para español.
    """
    marcas_base = cargar_base_de_datos()
    if not marcas_base:
        return []

    # Instancia la clase con el idioma 'es'
    encoder = Fonetika(language='es')

    codigo_input = encoder.transform(marca_input)
    
    resultados = []
    for marca in marcas_base:
        codigo_marca = encoder.transform(marca)
        similitud = fuzz.ratio(codigo_input, codigo_marca)
        resultados.append((marca, similitud))
        
    return resultados
