import pandas as pd
import streamlit as st
import jellyfish
from thefuzz import fuzz
import pyphen

# Inicializamos el diccionario de sílabas en español.
# Se usa un try-except para manejar el caso en que no esté instalado.
try:
    dic = pyphen.Pyphen(lang='es_ES')
except KeyError:
    st.error("Diccionario de sílabas en español no encontrado. Considera instalar 'pyphen'.")
    dic = None

@st.cache_data
def cargar_base_de_datos():
    """Carga la base de datos desde el CSV de forma robusta."""
    try:
        # --- CAMBIO 1: Lectura robusta del CSV para manejar comas ---
        df = pd.read_csv(
            "base_expandida.csv",
            header=None,
            names=['marca'],
            quotechar='"',
            engine='python'
        )
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
    Busca marcas fonéticamente similares, ajustando la puntuación
    según la diferencia en el número de sílabas.
    """
    marcas_base = cargar_base_de_datos()
    if not marcas_base:
        return []

    # Pre-calcula las propiedades de la palabra de entrada para eficiencia
    codigo_input = jellyfish.metaphone(marca_input)
    silabas_input = contar_silabas(marca_input)
    
    resultados = []
    for marca in marcas_base:
        # Ignora palabras muy cortas que pueden generar ruido
        if len(marca) < 4:
            continue

        # 1. Calcular la similitud fonética
        # --- CAMBIO 2: Se corrige la variable usada para generar el código ---
        codigo_marca = jellyfish.metaphone(marca)
        similitud_fonetica = fuzz.ratio(codigo_input, codigo_marca)

        # 2. Calcular la penalización por diferencia de sílabas
        silabas_marca = contar_silabas(marca)
        diferencia_silabas = abs(silabas_input - silabas_marca)
        
        # Cada sílaba de diferencia resta un 20% a la puntuación
        penalizacion = 1.0 - (diferencia_silabas * 0.20)
        factor_penalizacion = max(0, penalizacion) # Asegura que no sea negativo

        # 3. Calcular la puntuación final ajustada
        puntuacion_final = similitud_fonetica * factor_penalizacion
        
        resultados.append((marca.title(), puntuacion_final))
        
    # --- CAMBIO 3: La sentencia 'return' se mueve fuera del bucle ---
    # Esto asegura que se procesen TODAS las marcas antes de devolver el resultado.
    return resultados
