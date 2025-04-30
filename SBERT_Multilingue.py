import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Cargar modelo multilingüe
try:
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
except Exception as e:
    print(f"⚠️ Error cargando modelo SBERT: {e}")
    model = None

# Cargar CSV y preparar marcas
try:
    df = pd.read_csv("IGS - Consolidado.csv")
    marca_textos = df["NombreProducto"].astype(str).str.lower().tolist()
    if model:
        marca_embeddings = model.encode(marca_textos, convert_to_tensor=True)
    else:
        marca_embeddings = None
except Exception as e:
    print(f"⚠️ Error cargando CSV en SBERT: {e}")
    marca_textos = []
    marca_embeddings = None

def buscar_marcas_similares(input_marca, top_n=5):
    input_marca = input_marca.lower()
    input_embedding = model.encode(input_marca, convert_to_tensor=True)

    similitudes = util.pytorch_cos_sim(input_embedding, marca_embeddings)[0]

    top_resultados = sorted(
        zip(marca_textos, similitudes),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    #print(f"📋 Resultados SBERT para '{input_marca}':")
    #for marca, score in top_resultados:
    #    print(f" - {marca}: {float(score):.2f}%")

    return [(marca, float(score) * 100) for marca, score in top_resultados]




