import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re # Importamos re para limpiar mejor las palabras

# --- PREPARACIÓN INICIAL (SOLO SE EJECUTA UNA VEZ) ---
# Es posible que necesites ejecutar esto en tu terminal o en un script aparte la primera vez:
# import nltk
# nltk.download('stopwords')
# ---------------------------------------------------------

# --- 1. FUNCIÓN DE EXPANSIÓN Y LIMPIEZA DE LA BASE DE DATOS ---
def expandir_y_limpiar_terminos(lista_de_marcas):
    """
    Expande una lista de marcas descomponiendo términos compuestos y eliminando stopwords multilingües.
    """
    # Definir los idiomas para los stopwords
    idiomas = ['spanish', 'english', 'portuguese', 'french', 'german', 'italian']
    
    # Crear un conjunto único de stopwords de todos los idiomas definidos
    stopwords_multilingues = set()
    for idioma in idiomas:
        try:
            stopwords_multilingues.update(stopwords.words(idioma))
        except Exception:
            # Si falla la descarga de un idioma, simplemente lo omite.
            print(f"Advertencia: No se encontraron los stopwords para '{idioma}'.")

    # Agrega conectores o palabras comunes que quieras excluir de forma manual
    stopwords_personalizados = {'y', 'e', 'o', 'u', 'a', 'de', 'el', 'la', 'los', 'las', 'the', 'and', 'of'}
    stopwords_multilingues.update(stopwords_personalizados)

    terminos_expandidos = set()

    for marca in lista_de_marcas:
        # Agregar siempre el término original completo
        terminos_expandidos.add(marca.strip())
        
        # Descomponer la marca y limpiar
        palabras = re.findall(r'\b\w+\b', marca.lower()) # Extrae solo palabras
        
        if len(palabras) > 1:
            for palabra in palabras:
                # Añadir la palabra solo si NO es un stopword y no es puramente un número
                if palabra not in stopwords_multilingues and not palabra.isdigit():
                    terminos_expandidos.add(palabra.strip())

    return list(terminos_expandidos)

# --- 2. SIMULACIÓN DE DATOS Y MODELOS (REEMPLAZAR CON TUS DATOS REALES) ---

# Reemplaza esta lista con la carga de tu archivo (pd.read_csv, etc.)
MARCAS_DB_ORIGINAL = [
    "Tequila Margarita Gold", 
    "The Red Shoes", 
    "Café de la Mañana",
    "Sol y Playa Resort",
    "Marca 123",
    "Le Chat Noir"
]

# Usamos el caché de Streamlit para procesar los datos solo una vez
@st.cache_data
def cargar_y_preparar_datos():
    """
    Carga la base de datos original, la expande y la devuelve.
    En un caso real, aquí leerías tu CSV o base de datos.
    """
    marcas_expandidas = expandir_y_limpiar_terminos(MARCAS_DB_ORIGINAL)
    print(f"Base de datos expandida a {len(marcas_expandidas)} términos.")
    return marcas_expandidas

# Cargamos los datos preprocesados
MARCAS_DB_EXPANDIDA = cargar_y_preparar_datos()

# **IMPORTANTE**: Reemplaza estas funciones MOCK con tus modelos reales.
# Tus modelos SBERT y BETO deberían usar `MARCAS_DB_EXPANDIDA` como su corpus de búsqueda.
def modelo_sbert_mock(marca_input):
    """Función simulada de SBERT. Devuelve coincidencias de la DB expandida."""
    resultados = []
    for marca_db in MARCAS_DB_EXPANDIDA:
        # Simulación simple de similitud: si la palabra está contenida.
        if marca_input.lower() in marca_db.lower():
            similitud = 90.0 + len(marca_input) # Simulación
            resultados.append((marca_db, similitud))
    return sorted(resultados, key=lambda x: x[1], reverse=True)[:5]

def modelo_beto_mock(marca_input):
    """Función simulada de BETO. Devuelve coincidencias de la DB expandida."""
    resultados = []
    for marca_db in MARCAS_DB_EXPANDIDA:
        if marca_input.lower() in marca_db.lower():
            similitud = 85.0 + len(marca_input) # Simulación
            resultados.append((marca_db, similitud))
    return sorted(resultados, key=lambda x: x[1], reverse=True)[:5]
# --- FIN DE LA SIMULACIÓN ---


# --- 3. LÓGICA PRINCIPAL DE LA APP (MODIFICADA) ---
def combinar_modelos_v2_unicos(marca_input, umbral=80.0):
    resultados = []
    
    # **IMPORTANTE**: Aquí se usan los modelos MOCK. Reemplázalos por los tuyos.
    for modelo_func, nombre_modelo in [
        (modelo_beto_mock, "BETO"),
        (modelo_sbert_mock, "SBERT")
    ]:
        try:
            # Llama a tu función de búsqueda (aquí usamos la simulada)
            salida = modelo_func(marca_input)
            for marca, similitud in salida:
                if similitud >= umbral:
                    resultados.append({
                        "Marca": marca.strip().lower(),
                        "Similitud (%)": round(similitud, 2),
                        "Modelo": nombre_modelo
                    })
        except Exception as e:
            st.error(f"Error en el modelo {nombre_modelo}: {e}")

    if not resultados:
        return pd.DataFrame(columns=["Modelo", "Marca", "Similitud (%)"])

    df = pd.DataFrame(resultados)
    df = df.sort_values("Similitud (%)", ascending=False)
    df = df.drop_duplicates(subset="Marca", keep="first")
    df["Marca"] = df["Marca"].str.title()
    df = df.reset_index(drop=True)
    df.index += 1
    df.index.name = "Índice"
    return df

# ------------------ STREAMLIT INTERFAZ (SIN CAMBIOS) ------------------
st.set_page_config(page_title="Buscador de Marcas", layout="wide")
st.title("🔍 Buscador de marcas similares")

st.info(f"La base de datos contiene {len(MARCAS_DB_EXPANDIDA)} términos únicos (originales + descompuestos).")

marca = st.text_input("Ingresa la marca que deseas evaluar:", placeholder="Ej: Tequila, Shoes, Café...")
umbral = st.slider("Umbral mínimo de similitud (%)", 0, 100, 80)

if st.button("Buscar"):
    if marca.strip():
        with st.spinner("Buscando coincidencias..."):
            df_resultados = combinar_modelos_v2_unicos(marca.strip(), umbral=float(umbral))
            if df_resultados.empty:
                st.warning("No se encontraron coincidencias por encima del umbral.")
            else:
                st.success(f"Se encontraron {len(df_resultados)} marcas similares únicas.")
                st.dataframe(df_resultados, use_container_width=True)
    else:
        st.error("Por favor, ingresa una marca.")

