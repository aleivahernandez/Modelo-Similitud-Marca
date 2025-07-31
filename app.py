import streamlit as st
import pandas as pd

# --- Importación de los modelos ---
from SBERT_Multilingue import buscar_marcas_similares as modelo_sbert
from BETO import buscar_marcas_similares as modelo_beto
from Ngrams import buscar_marcas_similares as modelo_ngrams
from Fonetica import buscar_marcas_similares as modelo_fonetico # Modelo reincorporado

def crear_dataframe_comparativo(marca_input, umbral=80.0):
    """
    Procesa los modelos y devuelve un único DataFrame con los resultados
    en columnas separadas, limitado a un máximo de 5 registros por columna.
    """
    # Se reincorpora la categoría "Fonética"
    modelos_por_categoria = {
        "Semántica": [modelo_beto, modelo_sbert],
        "Ngrama": [modelo_ngrams],
        "Fonética": [modelo_fonetico]
    }
    
    resultados_agrupados = {
        "semantica": {},
        "ngrama": {},
        "fonetica": {} # Se añade el diccionario para resultados fonéticos
    }

    categoria_map = {
        "Semántica": "semantica",
        "Ngrama": "ngrama",
        "Fonética": "fonetica" # Se añade la correspondencia
    }

    # Recopilar y deduplicar resultados dentro de cada categoría
    for categoria_nombre, modelos in modelos_por_categoria.items():
        clave_resultado = categoria_map[categoria_nombre]
        for modelo_func in modelos:
            try:
                salida = modelo_func(marca_input)
                if not salida: continue
                
                for marca, similitud in salida:
                    if similitud >= umbral:
                        marca_lower = marca.strip().lower()
                        if marca_lower not in resultados_agrupados[clave_resultado] or similitud > resultados_agrupados[clave_resultado][marca_lower][1]:
                            resultados_agrupados[clave_resultado][marca_lower] = (marca.title(), similitud)
            except Exception as e:
                st.warning(f"Error en modelo de '{categoria_nombre}': {e}")

    # Convertir, ordenar y limitar a 5 resultados
    lista_semantica = sorted(resultados_agrupados["semantica"].values(), key=lambda item: item[1], reverse=True)[:5]
    lista_semantica_str = [f"{marca} ({similitud:.2f}%)" for marca, similitud in lista_semantica]

    lista_ngrama = sorted(resultados_agrupados["ngrama"].values(), key=lambda item: item[1], reverse=True)[:5]
    lista_ngrama_str = [f"{marca} ({similitud:.2f}%)" for marca, similitud in lista_ngrama]

    # Se procesa la lista de resultados fonéticos
    lista_fonetica = sorted(resultados_agrupados["fonetica"].values(), key=lambda item: item[1], reverse=True)[:5]
    lista_fonetica_str = [f"{marca} ({similitud:.2f}%)" for marca, similitud in lista_fonetica]


    # Crear el DataFrame con columnas independientes
    max_len = max(len(lista_semantica_str), len(lista_ngrama_str), len(lista_fonetica_str))
    
    if max_len == 0:
        return pd.DataFrame()

    def pad_list(lst, length):
        return lst + [""] * (length - len(lst))

    # Se reincorpora la columna "Fonética" al DataFrame final
    data = {
        'Semántica': pad_list(lista_semantica_str, max_len),
        'Ngrama': pad_list(lista_ngrama_str, max_len),
        'Fonética': pad_list(lista_fonetica_str, max_len)
    }
    
    df = pd.DataFrame(data)
    
    df.index = pd.RangeIndex(start=1, stop=len(df) + 1)
    df.index.name = "Índice"
    
    return df

# ------------------ Interfaz de Usuario de Streamlit ------------------

st.set_page_config(page_title="Buscador de Términos", page_icon="🔬", layout="wide")

st.title("🔬 Buscador de Términos Protegidos")

marca_input_text = st.text_input(
    "Ingresa el término que deseas evaluar:",
    placeholder="Ej: Leche o Zapatilla"
)
umbral_input_value = st.slider(
    "Umbral mínimo de similitud (%)",
    min_value=0,
    max_value=100,
    value=75,
    step=5
)

if st.button("Buscar", type="primary"):
    if marca_input_text.strip():
        with st.spinner("Analizando y construyendo tabla..."):
            df_resultados = crear_dataframe_comparativo(
                marca_input_text.strip(),
                umbral=float(umbral_input_value)
            )
        
        st.success("¡Búsqueda completada!")

        if not df_resultados.empty:
            st.dataframe(df_resultados, use_container_width=True)
        else:
            st.warning("No se encontraron coincidencias por encima del umbral seleccionado.")
    else:
        st.error("Por favor, ingresa un término para buscar.")
