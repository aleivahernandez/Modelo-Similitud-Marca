import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

# --- 0. Descarga de dependencias de NLTK (se ejecuta solo si es necesario) ---
try:
    stopwords.words('spanish')
except LookupError:
    st.info("Descargando recursos de NLTK (stopwords)...")
    nltk.download('stopwords')

# --- 1. Carga de Modelos y Datos (Optimizada para Streamlit Cloud) ---

@st.cache_resource
def cargar_modelos():
    """Carga los modelos de IA una sola vez y los mantiene en caché."""
    sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    beto_model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")
    return {"SBERT": sbert_model, "BETO": beto_model}

@st.cache_data
def expandir_y_limpiar_terminos():
    """
    Carga el CSV, descompone los términos en palabras individuales (tokens),
    elimina palabras vacías (stopwords) y devuelve una lista única y expandida.
    """
    try:
        df = pd.read_csv("base.csv")
        lista_de_marcas = df["NombreProducto"].dropna().astype(str).tolist()
    except Exception as e:
        st.error(f"Error al cargar 'base.csv': {e}")
        return []

    # Lista de idiomas para stopwords
    idiomas = ['spanish', 'english', 'french', 'italian', 'portuguese', 'german']
    stopwords_multilingues = set()
    for idioma in idiomas:
        stopwords_multilingues.update(stopwords.words(idioma))
    
    # Añadir conectores comunes que a veces no están en las listas
    stopwords_multilingues.update(['de', 'la', 'los', 'las', 'le', 'les', 'des', 'the', 'y', 'e', 'o', 'u'])

    terminos_expandidos = set()
    for marca in lista_de_marcas:
        # 1. Añadir siempre el término original completo
        terminos_expandidos.add(marca.strip())
        
        # 2. Descomponer en palabras individuales (tokens)
        # Usamos regex para encontrar palabras, incluyendo las que tienen acentos
        palabras = re.findall(r'\b\w+\b', marca.lower())
        
        if len(palabras) > 1:
            for palabra in palabras:
                # Añadir la palabra solo si no es un stopword y tiene más de 2 letras
                if palabra not in stopwords_multilingues and len(palabra) > 2:
                    terminos_expandidos.add(palabra)

    return sorted(list(terminos_expandidos))

@st.cache_data
def crear_embeddings(_model, _terminos):
    """Crea los embeddings para una lista de términos usando un modelo específico."""
    if not _terminos:
        return None
    return _model.encode(_terminos, convert_to_tensor=True)

# --- Ejecución de la carga ---
st.sidebar.info("Cargando modelos y datos...")
modelos = cargar_modelos()
terminos_expandidos = expandir_y_limpiar_terminos()

if not terminos_expandidos:
    st.error("La base de datos de términos está vacía. La aplicación no puede continuar.")
    st.stop()

embeddings = {
    "SBERT": crear_embeddings(modelos["SBERT"], terminos_expandidos),
    "BETO": crear_embeddings(modelos["BETO"], terminos_expandidos)
}
st.sidebar.success(f"Modelos y {len(terminos_expandidos)} términos cargados.")


# --- 2. Lógica de Búsqueda ---

def buscar_similitud(texto_input, modelo, terminos, embeddings_db, top_n=5):
    """Función genérica para buscar similitud usando un modelo y embeddings precalculados."""
    input_embedding = modelo.encode(texto_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(input_embedding, embeddings_db)[0]
    
    # Emparejar cada término con su puntuación
    top_results = sorted(zip(terminos, cos_scores), key=lambda x: x[1], reverse=True)
    
    return [(marca, score.item() * 100) for marca, score in top_results[:top_n]]


def combinar_resultados(marca_input, umbral=80.0):
    """
    Busca coincidencias exactas y por similitud en los modelos, y combina los resultados.
    """
    resultados = []
    input_lower = marca_input.strip().lower()

    # --- PASO 1: Búsqueda de Coincidencia Exacta ---
    for marca_db in terminos_expandidos:
        if marca_db.lower() == input_lower:
            resultados.append({
                "Marca": marca_db,
                "Similitud (%)": 100.0,
                "Modelo": "Coincidencia Exacta"
            })
            break

    # --- PASO 2: Búsqueda por Similitud con los modelos de IA ---
    for nombre_modelo in ["BETO", "SBERT"]:
        modelo_obj = modelos[nombre_modelo]
        embeddings_obj = embeddings[nombre_modelo]
        
        salida = buscar_similitud(marca_input, modelo_obj, terminos_expandidos, embeddings_obj)
        
        for marca, similitud in salida:
            if similitud >= umbral:
                resultados.append({
                    "Marca": marca,
                    "Similitud (%)": round(similitud, 2),
                    "Modelo": nombre_modelo
                })

    if not resultados:
        return pd.DataFrame(columns=["Modelo", "Marca", "Similitud (%)"])

    df = pd.DataFrame(resultados)
    # Usar una clave temporal para eliminar duplicados sin distinción de mayúsculas/minúsculas
    df['Marca_lower'] = df['Marca'].str.lower()
    df = df.sort_values("Similitud (%)", ascending=False).drop_duplicates(subset="Marca_lower", keep="first")
    df = df.drop(columns=['Marca_lower'])
    
    df["Marca"] = df["Marca"].str.title()
    df = df.reset_index(drop=True)
    df.index += 1
    df.index.name = "Índice"
    return df

# --- 3. Interfaz de Streamlit ---
st.title("🔍 Buscador de marcas similares")
st.info("Esta versión descompone las marcas en términos individuales (ej: 'Côtes De Provence' se busca también como 'Côtes' y 'Provence').")

marca = st.text_input("Ingresa la marca que deseas evaluar:", placeholder="Ej: Provence, Tequila, Shoes...")
umbral = st.slider("Umbral mínimo de similitud (%)", 0, 100, 80)

if st.button("Buscar"):
    if marca.strip():
        with st.spinner("Buscando en la base de datos expandida..."):
            df_resultados = combinar_resultados(marca.strip(), umbral=float(umbral))
        
        if df_resultados.empty:
            st.warning("No se encontraron coincidencias sobre el umbral.")
        else:
            st.success(f"{len(df_resultados)} marcas similares encontradas.")
            st.dataframe(df_resultados, use_container_width=True)
    else:
        st.error("Por favor, ingresa una marca.")

# Herramienta de depuración opcional
if st.sidebar.checkbox("Mostrar base de datos expandida para depuración"):
    st.sidebar.write("La búsqueda se realiza contra los siguientes términos:")
    st.sidebar.dataframe(pd.DataFrame(terminos_expandidos, columns=["Términos"]))
