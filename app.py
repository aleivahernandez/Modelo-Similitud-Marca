import streamlit as st
import pandas as pd

# --- Importación de los modelos ---
from SBERT_Multilingue import buscar_marcas_similares as modelo_sbert
from BETO import buscar_marcas_similares as modelo_beto
from Ngrams import buscar_marcas_similares as modelo_ngrams

def combinar_modelos_por_tipo(marca_input, umbral=80.0):
    """
    Combina los resultados de diferentes modelos en un DataFrame con columnas
    por tipo de modelo ('Semántica', 'Ngrama').
    """
    # 1. Definir qué modelos pertenecen a cada categoría
    modelos_por_categoria = {
        "Semántica": [modelo_beto, modelo_sbert],
        "Ngrama": [modelo_ngrams]
    }
    
    resultados_agrupados = {
        "Semántica": {}, # Usamos dict para manejar duplicados: {'marca': score}
        "Ngrama": {}
    }

    # 2. Ejecutar modelos y recopilar resultados por categoría
    for categoria, modelos in modelos_por_categoria.items():
        for modelo_func in modelos:
            try:
                salida = modelo_func(marca_input)
                if not salida:
                    continue
                
                for marca, similitud in salida:
                    if similitud >= umbral:
                        marca_lower = marca.strip().lower()
                        # Si la marca ya fue encontrada, quedarse con el score más alto
                        if marca_lower not in resultados_agrupados[categoria] or similitud > resultados_agrupados[categoria][marca_lower]:
                            resultados_agrupados[categoria][marca_lower] = similitud
            except Exception as e:
                st.warning(f"Error procesando un modelo en la categoría '{categoria}': {e}")

    # 3. Obtener una lista de todas las marcas únicas encontradas
    todas_las_marcas = set(resultados_agrupados["Semántica"].keys()) | set(resultados_agrupados["Ngrama"].keys())

    if not todas_las_marcas:
        return pd.DataFrame()

    # 4. Construir el DataFrame final
    data = []
    for marca in sorted(list(todas_las_marcas)):
        fila = {"Marca Encontrada": marca.title()}
        
        # Formato "Palabra (Similitud%)" para la columna Semántica
        if marca in resultados_agrupados["Semántica"]:
            score = resultados_agrupados["Semántica"][marca]
            fila["Semántica"] = f"{marca.title()} ({score:.2f}%)"
        else:
            fila["Semántica"] = ""
            
        # Formato para la columna Ngrama
        if marca in resultados_agrupados["Ngrama"]:
            score = resultados_agrupados["Ngrama"][marca]
            fila["Ngrama"] = f"{marca.title()} ({score:.2f}%)"
        else:
            fila["Ngrama"] = ""
            
        data.append(fila)

    df = pd.DataFrame(data)
    df = df.set_index("Marca Encontrada") # Poner las marcas como índice de fila
    
    return df

# ------------------ Interfaz de Usuario de Streamlit ------------------

st.set_page_config(page_title="Buscador de Marcas", page_icon="🔍")

st.title("🔬 Comparador de Modelos de Similitud")
st.info(
    "Compara los resultados de modelos semánticos (BETO/SBERT) contra modelos "
    "basados en caracteres (N-Grams)."
)

# --- Entradas del usuario ---
marca_input_text = st.text_input(
    "Ingresa la marca que deseas evaluar:",
    placeholder="Ej: Adiddas"
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
            # Llama a la nueva función que agrupa por tipo
            df_resultados = combinar_modelos_por_tipo(
                marca_input_text.strip(),
                umbral=float(umbral_input_value)
            )
        
        # Muestra los resultados si se encontraron
        if not df_resultados.empty:
            st.success(f"¡Análisis completado! Se encontraron {len(df_resultados)} marcas únicas.")
            st.dataframe(df_resultados, use_container_width=True)
        else:
            st.warning("No se encontraron coincidencias por encima del umbral seleccionado.")
    else:
        st.error("Por favor, ingresa el nombre de una marca para buscar.")
