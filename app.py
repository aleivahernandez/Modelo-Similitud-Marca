import streamlit as st
import pandas as pd
from SBERT_Multilingue import buscar_marcas_similares as modelo_sbert
from BETO import buscar_marcas_similares as modelo_beto

def combinar_modelos_v2_unicos(marca_input, umbral=80.0):
    """
    Llama a cada modelo, recopila los resultados que superan el umbral
    y los combina en un único DataFrame sin duplicados.
    """
    resultados = []
    # Llama a cada modelo importado
    for modelo_func, nombre_modelo in [
        (modelo_beto, "BETO"),
        (modelo_sbert, "SBERT")
    ]:
        try:
            # Llama a la función de búsqueda (ej: buscar_marcas_similares)
            salida = modelo_func(marca_input)

            # Si el modelo no devuelve nada, continuar al siguiente
            if not salida:
                continue

            # Simplificamos el bucle, eliminando 'enumerate' que no se usaba
            for marca, similitud in salida:
                if similitud >= umbral:
                    resultados.append({
                        "Marca": marca.strip().lower(),
                        "Similitud (%)": round(similitud, 2),
                        "Modelo": nombre_modelo
                    })
        except Exception as e:
            # Hacemos el error visible en la interfaz de Streamlit
            st.warning(f"Ocurrió un error al procesar el modelo {nombre_modelo}:")
            st.exception(e) # Muestra el traceback completo del error

    # Si después de todos los modelos no hay resultados, devolver un DF vacío
    if not resultados:
        return pd.DataFrame()

    # Crear el DataFrame
    df = pd.DataFrame(resultados)

    # --- FIX PARA KEYERROR ---
    # Antes de ordenar, nos aseguramos de que la columna exista.
    # Si no existe (lo que indica un problema), devolvemos el dataframe como está
    # para poder inspeccionarlo en la interfaz, en lugar de crashear la app.
    if "Similitud (%)" not in df.columns:
        st.error("Se produjo un error inesperado: la columna 'Similitud (%)' no se pudo generar.")
        st.write("Datos sin procesar recibidos:", df)
        return df

    # Ahora es seguro ordenar
    df = df.sort_values("Similitud (%)", ascending=False)
    
    # Mantiene la primera aparición de una marca (la que tiene mayor similitud)
    df = df.drop_duplicates(subset="Marca", keep="first")
    df["Marca"] = df["Marca"].str.title()
    df = df.reset_index(drop=True)
    df.index += 1
    df.index.name = "Índice"
    return df

# ------------------ STREAMLIT INTERFAZ ------------------
# Esta versión asume que tus archivos SBERT_Multilingue.py y BETO.py
# han sido modificados para leer el nuevo archivo "base_expandida.csv".

st.title("🔍 Buscador de marcas similares")
st.info("Esta versión lee desde una base de datos pre-procesada y expandida.")

marca_input_text = st.text_input("Ingresa la marca que deseas evaluar:")
umbral_input_value = st.slider("Umbral mínimo de similitud (%)", 0, 100, 80)

if st.button("Buscar"):
    if marca_input_text.strip():
        with st.spinner("Buscando en las bases de datos de los modelos..."):
            # Llama a la función que combina los resultados de tus modelos
            df_resultados = combinar_modelos_v2_unicos(marca_input_text.strip(), umbral=float(umbral_input_value))
        
        if df_resultados.empty:
            st.warning("No se encontraron coincidencias sobre el umbral.")
        else:
            st.success(f"{len(df_resultados)} marcas similares encontradas.")
            st.dataframe(df_resultados, use_container_width=True)
    else:
        st.error("Por favor, ingresa una marca.")
