import streamlit as st
import pandas as pd

# --- Importación de los modelos ---
# Se asume que estos archivos están en la misma carpeta.
# Cada archivo debe tener una función llamada 'buscar_marcas_similares'.
from SBERT_Multilingue import buscar_marcas_similares as modelo_sbert
from BETO import buscar_marcas_similares as modelo_beto
from Ngrams import buscar_marcas_similares as modelo_ngrams

def combinar_modelos_v2_unicos(marca_input, umbral=80.0):
    """
    Llama a cada modelo, recopila los resultados que superan el umbral
    y los combina en un único DataFrame sin duplicados, priorizando
    el de mayor similitud.
    """
    resultados = []
    
    # Lista de los modelos a ejecutar. Fácil de extender en el futuro.
    lista_modelos = [
        (modelo_beto, "BETO"),
        (modelo_sbert, "SBERT"),
        (modelo_ngrams, "N-Grams") # Modelo ligero de n-grams añadido
    ]
    
    for modelo_func, nombre_modelo in lista_modelos:
        try:
            # Llama a la función de búsqueda de cada modelo
            salida = modelo_func(marca_input)

            # Si el modelo no devuelve nada, continúa al siguiente
            if not salida:
                continue

            # Agrega los resultados que superan el umbral a la lista
            for marca, similitud in salida:
                if similitud >= umbral:
                    resultados.append({
                        "Marca": marca.strip().lower(),
                        "Similitud (%)": round(similitud, 2),
                        "Modelo": nombre_modelo
                    })
        except Exception as e:
            # Muestra un error en la app si un modelo falla, sin detener todo
            st.warning(f"Ocurrió un error al procesar el modelo {nombre_modelo}:")
            st.exception(e)

    # Si no hay resultados tras ejecutar todos los modelos, retorna un DataFrame vacío
    if not resultados:
        return pd.DataFrame()

    # Crea el DataFrame a partir de la lista de resultados
    df = pd.DataFrame(resultados)

    # --- Procesamiento final del DataFrame ---
    # Ordena por similitud de forma descendente para que la más alta quede primero
    df = df.sort_values("Similitud (%)", ascending=False)
    
    # Elimina marcas duplicadas, conservando la primera aparición (la de mayor similitud)
    df = df.drop_duplicates(subset="Marca", keep="first")
    
    # Formatea la columna de marcas para una mejor presentación
    df["Marca"] = df["Marca"].str.title()
    
    # Reinicia el índice para que comience en 1
    df = df.reset_index(drop=True)
    df.index += 1
    df.index.name = "Índice"
    
    return df

# ------------------ Interfaz de Usuario de Streamlit ------------------

st.set_page_config(page_title="Buscador de Marcas", page_icon="🔍")

st.title("🔍 Buscador de Marcas Similares")
st.info(
    "Ingresa una marca para buscar coincidencias en la base de datos. "
    "El sistema utiliza múltiples modelos (semánticos y de caracteres) para encontrar resultados."
)

# --- Entradas del usuario ---
marca_input_text = st.text_input(
    "Ingresa la marca que deseas evaluar:", 
    placeholder="Ej: Cocacola"
)
umbral_input_value = st.slider(
    "Umbral mínimo de similitud (%)", 
    min_value=0, 
    max_value=100, 
    value=80, 
    step=5
)

# --- Botón de búsqueda y lógica de presentación ---
if st.button("Buscar Similitudes", type="primary"):
    if marca_input_text.strip():
        with st.spinner("Analizando... Por favor, espera."):
            # Llama a la función principal que combina los modelos
            df_resultados = combinar_modelos_v2_unicos(
                marca_input_text.strip(), 
                umbral=float(umbral_input_value)
            )
        
        # Muestra los resultados si se encontraron
        if not df_resultados.empty:
            st.success(f"¡Búsqueda completada! Se encontraron {len(df_resultados)} marcas similares.")
            st.dataframe(df_resultados, use_container_width=True)
        else:
            st.warning("No se encontraron coincidencias por encima del umbral seleccionado.")
    else:
        st.error("Por favor, ingresa el nombre de una marca para buscar.")
