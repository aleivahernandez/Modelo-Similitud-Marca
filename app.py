import streamlit as st

import pandas as pd

from SBERT_Multilingue import buscar_marcas_similares as modelo_sbert

from BETO import buscar_marcas_similares as modelo_beto



def combinar_modelos_v2_unicos(marca_input, umbral=80.0):

    resultados = []

    for modelo_func, nombre_modelo in [

        (modelo_beto, "BETO"),

        (modelo_sbert, "SBERT")

    ]:

        try:

            salida = modelo_func(marca_input)

            for i, (marca, similitud) in enumerate(salida, start=1):

                if similitud >= umbral:

                    resultados.append({

                        "Marca": marca.strip().lower(),

                        "Similitud (%)": round(similitud, 2),

                        "Modelo": nombre_modelo

                    })

        except Exception as e:

            print(f"Error en {nombre_modelo}: {e}")



    # Verifica si hay resultados antes de crear y manipular el DataFrame

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



# ------------------ STREAMLIT INTERFAZ ------------------

st.title("🔍 Buscador de marcas similares")

marca = st.text_input("Ingresa la marca que deseas evaluar:")

umbral = st.slider("Umbral mínimo de similitud (%)", 0, 100, 80)



if st.button("Buscar"):

    if marca.strip():

        df_resultados = combinar_modelos_v2_unicos(marca.strip(), umbral=umbral)

        if df_resultados.empty:

            st.warning("No se encontraron coincidencias sobre el umbral.")

        else:

            st.success(f"{len(df_resultados)} marcas similares encontradas.")

            st.dataframe(df_resultados)

    else:

        st.error("Por favor, ingresa una marca.")

