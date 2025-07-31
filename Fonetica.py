import pandas as pd
import streamlit as st
import jellyfish
from thefuzz import fuzz
import pyphen
import re

# Inicializamos el diccionario de sílabas en español
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
    """
    Cuenta las sílabas de una palabra. Si Pyphen falla (porque no es una palabra en español),
    se recurre a un conteo simple de grupos de vocales.
    """
    if dic is None:
        return 1
    try:
        # Pyphen inserta guiones; si no inserta ninguno, falló o la palabra es un monosílabo.
        separado = dic.inserted(palabra)
        if '-' in separado:
            return len(separado.split('-'))
        
        # Si no hay guiones, podría ser un monosílabo real o una palabra desconocida.
        # Contamos los grupos de vocales como un método de respaldo.
        grupos_vocales = len(re.findall('[aeiou]+', palabra, re.IGNORECASE))
        return max(1, grupos_vocales) # Devolvemos al menos 1 sílaba
    except Exception:
        # Si Pyphen lanza un error, usamos el método de respaldo.
        grupos_vocales = len(re.findall('[aeiou]+', palabra, re.IGNORECASE))
        return max(1, grupos_vocales)

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
        codigo_marca = jellyfish.metaphone(marca)
        similitud_fonetica = fuzz.ratio(codigo_input, codigo_marca)

        silabas_marca = contar_silabas(marca)
        diferencia_silabas = abs(silabas_input - silabas_marca)
        
        penalizacion = 1.0 - (diferencia_silabas * 0.20)
        factor_penalizacion = max(0, penalizacion)

        puntuacion_final = similitud_fonetica * factor_penalizacion
        
        resultados.append((marca, puntuacion_final))
        
    return resultados
