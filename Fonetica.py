import pandas as pd
import streamlit as st
from fonetika import Fonetika # <- CAMBIO 1: Se importa la clase principal
from thefuzz import fuzz

# Usamos el caché de Streamlit para leer el CSV solo una vez.
@st.cache_data
def cargar_base_de_datos():
    """Carga la base de datos desde el CSV y la retorna como una lista de marcas."""
    try:
        df = pd.read_csv("base_expandida.csv")
        # Leemos la primera columna (índice 0), sin importar su nombre.
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
    Compara los códigos fonéticos resultantes usando un ratio de similitud.
    """
    marcas_base = cargar_base_de_datos()
    if not marcas_base:
        return []

    # <- CAMBIO 2: Se instancia la clase con el idioma 'es'
    encoder = Fonetika(language='es')

    # Obtener el código fonético de la marca de entrada
    codigo_input = encoder.transform(marca_input)
    
    resultados = []
    for marca in marcas_base:
        # Obtener el código fonético de cada marca en la base de datos
        codigo_marca = encoder.transform(marca)
        
        # Usar thefuzz para obtener un porcentaje de similitud entre los códigos fonéticos.
        similitud = fuzz.ratio(codigo_input, codigo_marca)
        
        resultados.append((marca, similitud))
        
    return resultados
