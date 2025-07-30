import streamlit as st
import pandas as pd

# --- Importación de los modelos ---
from SBERT_Multilingue import buscar_marcas_similares as modelo_sbert
from BETO import buscar_marcas_similares as modelo_beto
from Ngrams import buscar_marcas_similares as modelo_ngrams
from Fonetica import buscar_marcas_similares as modelo_fonetico

def combinar_modelos_en_listas(marca_input, umbral=80.0):
    """
    Procesa los modelos y devuelve los resultados en listas separadas por categoría.
    """
    modelos_por_categoria = {
        "Semántica": [modelo_beto, modelo_sbert],
        "Ngrama": [modelo_ngrams],
        "Fonética": [modelo_fonetico]
    }
    
    resultados_finales = {
        "semantica": {},
        "ngrama": {},
        "fonetica": {}
    }

    # Asignar claves correctas del diccionario
    categoria_map = {
        "Semántica": "semantica",
        "Ngrama": "ngrama",
        "Fonética": "fonetica"
    }

    for categoria_nombre, modelos in modelos_por_categoria.items():
        clave_resultado = categoria_map[categoria_nombre]
        for modelo_func in modelos:
            try:
                salida = modelo_func(marca_input)
                if not salida:
                    continue
                
                for marca, similitud in salida:
                    if similitud >= umbral:
                        marca_lower = marca.strip().lower()
                        # Guardar o actualizar solo si la similitud es mayor
                        if marca_lower not in resultados_finales[clave_resultado] or similitud > resultados_finales[clave_resultado][marca_lower][1]:
                            resultados_finales[clave_resultado][marca_lower] = (marca.title(), similitud)
            except Exception as e:
                st.warning(f"Error en modelo de '{categoria_nombre}': {e}")

    # Convertir los diccionarios de resultados a listas ordenadas
    listas_ordenadas = {
        "semantica": sorted(resultados_finales["semantica"].values(), key=lambda item: item[1], reverse=True),
        "ngrama": sorted(resultados_finales["ngrama"].values(), key=lambda item: item[1], reverse=True),
        "fonetica": sorted(resultados_finales["fonetica"].values(), key=lambda item: item[1], reverse=True)
    }
    
    return listas_ordenadas

# ------------------ Interfaz de Usuario de Streamlit ------------------

st.set_page_config(page_title="Buscador de Marcas", page_icon="🔬", layout="wide")

st.title("🔬 Comparador de Modelos de Similitud")
st.info(
    "Compara los resultados de modelos semánticos, de caracteres (N-Grams) y fonéticos, presentados en listas separadas."
)

# --- Entradas del usuario ---
marca_input_text = st.text_input(
    "Ingresa la marca que deseas evaluar:",
    placeholder="Ej: Cerveza o Serbesa"
)
umbral_input_value = st.slider(
    "Umbral mínimo de similitud (%)",
    min_value=0,
    max_value=100,
    value=75,
    step=5
)

# --- Botón de búsqueda y lógica de presentación ---
if st.button("Comparar Modelos", type="primary"):
    if marca_input_text.strip():
        with st.spinner("Analizando con todos los modelos..."):
            resultados_listas = combinar_modelos_en_listas(
                marca_input_text.strip(),
                umbral=float(umbral_input_value)
            )
        
        st.success("¡Análisis completado!")

        # Crear dos columnas para mostrar los resultados
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🤖 Similitud Semántica")
            if resultados_listas["semantica"]:
                for marca, similitud in resultados_listas["semantica"]:
                    st.write(f"{marca} ({similitud:.2f}%)")
            else:
                st.write("No se encontraron resultados.")

        with col2:
            st.subheader("✍️ Similitud por Caracteres (N-Grams)")
            if resultados_listas["ngrama"]:
                for marca, similitud in resultados_listas["ngrama"]:
                    st.write(f"{marca} ({similitud:.2f}%)")
            else:
                st.write("No se encontraron resultados.")
            
            st.subheader("🗣️ Similitud Fonética")
            if resultados_listas["fonetica"]:
                for marca, similitud in resultados_listas["fonetica"]:
                    st.write(f"{marca} ({similitud:.2f}%)")
            else:
                st.write("No se encontraron resultados.")

    else:
        st.error("Por favor, ingresa el nombre de una marca para buscar.")
