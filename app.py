import streamlit as st
import pandas as pd

# --- 1. IMPORTACIÓN DE TUS MODELOS ---
# Se importan las funciones de búsqueda directamente desde tus archivos.
# Asegúrate de que SBERT_Multilingue.py y BETO.py estén en la misma carpeta.
try:
    from SBERT_Multilingue import buscar_marcas_similares as modelo_sbert
    from BETO import buscar_marcas_similares as modelo_beto
    st.sidebar.success("Modelos cargados correctamente.")
except ImportError as e:
    st.error(f"Error al importar los módulos de los modelos: {e}")
    st.info("Asegúrate de que los archivos 'SBERT_Multilingue.py' y 'BETO.py' se encuentren en el mismo directorio que esta aplicación.")
    # Detiene la ejecución si los modelos no se pueden cargar
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error inesperado al cargar los modelos: {e}")
    st.stop()

# --- NUEVO: Carga de datos para comprobación directa ---
@st.cache_data
def load_brand_list():
    """Carga la lista de marcas desde el CSV para comprobación directa y depuración."""
    try:
        df = pd.read_csv("base.csv")
        # Asegurarse de que no haya nulos y todo sea string
        marcas = df["NombreProducto"].dropna().astype(str).tolist()
        return marcas
    except Exception as e:
        st.warning(f"No se pudo cargar 'base.csv' para la comprobación de coincidencias exactas: {e}")
        return []

# Carga la lista de marcas una sola vez
marcas_para_comprobacion = load_brand_list()


# --- 2. LÓGICA PARA COMBINAR RESULTADOS (MODIFICADA) ---
def combinar_modelos_v2_unicos(marca_input, umbral=80.0):
    """
    Primero busca una coincidencia exacta, luego llama a cada modelo, 
    recopila los resultados que superan el umbral y los combina.
    """
    resultados = []
    input_lower = marca_input.strip().lower()

    # --- PASO 1: Búsqueda de Coincidencia Exacta (ignora mayúsculas/minúsculas) ---
    for marca_db in marcas_para_comprobacion:
        if marca_db.strip().lower() == input_lower:
            resultados.append({
                "Marca": marca_db.strip(), # Usar la versión original para mostrar
                "Similitud (%)": 100.0,
                "Modelo": "Coincidencia Exacta"
            })
            break # Encontramos la coincidencia, no es necesario seguir buscando

    # --- PASO 2: Búsqueda por Similitud con los modelos de IA ---
    for modelo_func, nombre_modelo in [
        (modelo_beto, "BETO"),
        (modelo_sbert, "SBERT")
    ]:
        try:
            # Llama a la función de búsqueda de cada modelo
            salida = modelo_func(marca_input)
            for marca, similitud in salida:
                if similitud >= umbral:
                    resultados.append({
                        "Marca": marca.strip().lower(),
                        "Similitud (%)": round(similitud, 2),
                        "Modelo": nombre_modelo
                    })
        except Exception as e:
            st.warning(f"No se pudo obtener resultado del modelo {nombre_modelo}: {e}")

    # Verifica si hay resultados antes de crear y manipular el DataFrame
    if not resultados:
        return pd.DataFrame(columns=["Modelo", "Marca", "Similitud (%)"])

    df = pd.DataFrame(resultados)
    df = df.sort_values("Similitud (%)", ascending=False)
    # Mantiene la primera aparición de una marca (la que tiene mayor similitud)
    df = df.drop_duplicates(subset="Marca", keep="first")
    df["Marca"] = df["Marca"].str.title()
    df = df.reset_index(drop=True)
    df.index += 1
    df.index.name = "Índice"
    return df

# ------------------ 3. INTERFAZ DE STREAMLIT (Tu interfaz original) ------------------
st.title("🔍 Buscador de marcas similares")
marca = st.text_input("Ingresa la marca que deseas evaluar:")
umbral = st.slider("Umbral mínimo de similitud (%)", 0, 100, 80)

if st.button("Buscar"):
    if marca.strip():
        with st.spinner("Buscando en las bases de datos de los modelos..."):
            # Llama a la función que combina los resultados de tus modelos
            df_resultados = combinar_modelos_v2_unicos(marca.strip(), umbral=float(umbral))
        
        if df_resultados.empty:
            st.warning("No se encontraron coincidencias sobre el umbral.")
        else:
            st.success(f"{len(df_resultados)} marcas similares encontradas.")
            st.dataframe(df_resultados, use_container_width=True)
    else:
        st.error("Por favor, ingresa una marca.")
