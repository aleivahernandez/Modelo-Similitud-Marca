import pandas as pd
import streamlit as st
import jellyfish
from thefuzz import fuzz
import pyphen # <- LIBRERÍA NUEVA

# Inicializamos el diccionario de sílabas en español
# Lo hacemos una sola vez para mayor eficiencia.
try:
    dic = pyphen.Pyphen(lang='es_ES')
except KeyError:
    st.error("Diccionario de sílabas en español no encontrado. Instala 'pyphen' y los diccionarios necesarios.")
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
    """Cuenta las sílabas de una palabra usando el diccionario Pyphen."""
    if dic is None:
        return 1 # Devuelve 1 para no causar errores si el diccionario falla
    # Contamos los guiones insertados por Pyphen y sumamos 1
    return len(dic.inserted(palabra).split('-'))

def buscar_marcas_similares(marca_input):
    """
    Busca marcas fonéticamente similares, ajustando la puntuación
    según la diferencia en el número de sílabas.
    """
    marcas_base = cargar_base_de_datos()
    if not marcas_base:
        return []

    codigo_input = jellyfish.metaphone(marca_input)
    silabas_input = contar_silabas(marca_input)
    
    resultados = []
    for marca in marcas_base:
        # 1. Calcular la similitud fonética como antes
        codigo_marca = jellyfish.metaphone(marca)
        similitud_fonetica = fuzz.ratio(codigo_input, codigo_marca)

        # 2. Calcular la penalización por diferencia de sílabas
        silabas_marca = contar_silabas(marca)
        diferencia_silabas = abs(silabas_input - silabas_marca)
        
        # Penalización: restamos un 20% de la puntuación por cada sílaba de diferencia
        # Puedes ajustar este valor (0.20) si quieres que la penalización sea más o menos fuerte.
        penalizacion = 1.0 - (diferencia_silabas * 0.20)
        
        # Asegurarnos de que la penalización no sea negativa
        factor_penalizacion = max(0, penalizacion)

        # 3. Calcular la puntuación final
        puntuacion_final = similitud_fonetica * factor_penalizacion
        
        resultados.append((marca, puntuacion_final))
        
    return resultados
