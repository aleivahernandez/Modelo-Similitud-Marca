import pandas as pd
import streamlit as st
import jellyfish
from thefuzz import fuzz
import pyphen

# Inicializamos el diccionario de sílabas en español.
try:
    dic = pyphen.Pyphen(lang='es_ES')
except KeyError:
    st.error("Diccionario de sílabas en español no encontrado. Instala 'pyphen'.")
    dic = None

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

def contar_silabas(palabra):
    """Cuenta las sílabas de una palabra usando el diccionario Pyphen para español."""
    if dic is None:
        return 1
    # Pyphen inserta guiones entre sílabas. Contamos los guiones y sumamos 1.
    return len(dic.inserted(palabra).split('-'))

def buscar_marcas_similares(marca_input):
    """
    Busca marcas fonéticamente similares, ignorando palabras de menos de 4 letras
    y ajustando la puntuación según la diferencia en el número de sílabas.
    """
    marcas_base = cargar_base_de_datos()
    if not marcas_base:
        return []

    codigo_input = jellyfish.metaphone(marca_input)
    silabas_input = contar_silabas(marca_input)
    
    resultados = []
    for marca in marcas_base:
        # --- CAMBIO CLAVE: Ignorar palabras con menos de 4 letras ---
        if len(marca) < 4:
            continue

        # 1. Calcular la similitud fonética
        codigo_marca = jellyfish.metaphone(marca)
        similitud_fonetica = fuzz.ratio(codigo_input, codigo_marca)

        # 2. Calcular la penalización por diferencia de sílabas
        silabas_marca = contar_silabas(marca)
        diferencia_silabas = abs(silabas_input - silabas_marca)
        
        penalizacion = 1.0 - (diferencia_silabas * 0.20)
        factor_penalizacion = max(0, penalizacion)

        # 3. Calcular la puntuación final ajustada
        puntuacion_final = similitud_fonetica * factor_penalizacion
        
        resultados.append((marca, puntuacion_final))
        
    return resultados
