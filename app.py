import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

# --- 0. Descarga de dependencias de NLTK (se ejecuta solo si es necesario) ---
# Esto asegura que los recursos estén disponibles en Streamlit Cloud.
try:
    stopwords.words('spanish')
except LookupError:
    st.info("Descargando recursos de NLTK (stopwords)...")
    nltk.download('stopwords')

# --- 1. Carga de Modelos y Datos (Optimizada para Streamlit Cloud) ---

@st.cache_resource
def cargar_modelos():
    """
    Carga los modelos de IA una sola vez y los mueve a la CPU.
    Forzar el uso de la CPU es la solución más estable para el RuntimeError en la nube.
    """
    device = 'cpu'
    sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
    beto_model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es", device=device)
    return {"SBERT": sbert_model, "BETO": beto_model}

@st.cache_data
def expandir_y_limpiar_terminos():
    """
    Carga el CSV, descompone los términos en palabras individuales (tokens),
    elimina palabras vacías (stopwords) y devuelve una lista única y expandida.
    """
    try:
        # Se asume que el archivo se llama 'base_expandida.csv'
        df = pd.read_csv("base_expandida.csv")
        # La columna debe llamarse 'NombreProducto' como en el archivo de expansión
        lista_de_marcas = df["NombreProducto"].dropna().astype(str).tolist()
        return sorted(lista_de_marcas)
    except Exception as e:
        st.error(f"Error al cargar 'base_expandida.csv': {e}")
        st.info("Asegúrate de haber ejecutado el script 'expand_database.py' y subido 'base_expandida.csv' a tu repositorio.")
        return []

@st.cache_data
def crear_embeddings(_model, _terminos):
    """Crea los embeddings para una lista de términos usando un modelo específico."""
    if not _terminos or _model is None:
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

def buscar_similitud(texto_input, modelo, terminos, embeddings_db, top_n=10):
    """Función genérica para buscar similitud usando un modelo y embeddings precalculados."""
    if embeddings_db is None:
        return []
        
    input_embedding = modelo.encode(texto_input, convert_to_tensor=True)
    
    # El modelo ya fue forzado a 'cpu', por lo que los tensores que crea
    # estarán en el dispositivo correcto.
    cos_scores = util.pytorch_cos_sim(input_embedding, embeddings_db)[0]
    
    top_results = sorted(zip(terminos, cos_scores), key=lambda x: x[1], reverse=True)
    
    return [(marca, score.item() * 100) for marca, score in top_results[:top_n]]


def combinar_resultados(marca_input, umbral=80.0):
    """
    Busca coincidencias por similitud en los modelos y combina los resultados.
    """
    resultados = []
    
    for nombre_modelo in ["BETO", "SBERT"]:
        modelo_obj = modelos[nombre_modelo]
        embeddings_obj = embeddings[nombre_modelo]
        
        if embeddings_obj is not None:
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
st.info("Esta versión lee desde una base de datos pre-procesada y expandida ('base_expandida.csv').")

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
if st.sidebar.checkbox("Mostrar base de datos para depuración"):
    st.sidebar.write("La búsqueda se realiza contra los siguientes términos:")
    st.sidebar.dataframe(pd.DataFrame(terminos_expandidos, columns=["Términos"]))
