import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Vibely - AI Music Recommender", page_icon="🎵", layout="centered")

# URL dei file su Hugging Face
DF_URL = "https://huggingface.co/MarcoBaiguini/vibely-data/resolve/main/df_with_embeddings.pkl"
KNN_URL = "https://huggingface.co/MarcoBaiguini/vibely-data/resolve/main/knn_model.pkl"

@st.cache_resource(show_spinner=False)
def load_optimized_data():
    df_path = "df_with_embeddings.pkl"
    knn_path = "knn_model.pkl"

    if not os.path.exists(df_path):
        with st.spinner("Scaricamento del dataset..."):
            r = requests.get(DF_URL)
            with open(df_path, 'wb') as f:
                f.write(r.content)

    if not os.path.exists(knn_path):
        with st.spinner("Scaricamento del modello KNN..."):
            r = requests.get(KNN_URL)
            with open(knn_path, 'wb') as f:
                f.write(r.content)

    with open(df_path, "rb") as f:
        df = pickle.load(f)
    with open(knn_path, "rb") as f:
        knn_model = pickle.load(f)

    return df, knn_model

# Caricamento dati con spinner
with st.spinner("Caricamento del sistema di raccomandazione..."):
    try:
        df, knn_model = load_optimized_data()
    except Exception as e:
        st.error(f"Errore durante il caricamento dei dati: {e}")
        st.stop()

# Funzione di raccomandazione
def get_recommendations(input_ids, df, knn_model, k=5):
    input_embeddings = df[df['id'].isin(input_ids)]['embedding_list'].tolist()
    if not input_embeddings:
        return []

    input_vector = np.mean(input_embeddings, axis=0).reshape(1, -1)
    distances, indices = knn_model.kneighbors(input_vector, n_neighbors=k + len(input_ids))
    indices = indices[0]
    recommended = df.iloc[indices]
    recommended = recommended[~recommended['id'].isin(input_ids)]
    return recommended.head(k)

# UI
st.title(":headphones: Vibely")
st.markdown("Espandi i tuoi orizzonti musicali con l'intelligenza artificiale")

user_input = st.text_area("Incolla qui i link delle tue canzoni da Spotify", height=150)

if st.button(":musical_note: Scopri nuova musica"):
    with st.spinner("Analisi della playlist in corso..."):
        links = user_input.strip().split("\n")
        input_ids = [link.split("/")[-1].split("?")[0] for link in links if "spotify.com/track" in link]

        if not input_ids:
            st.warning("Inserisci almeno un link valido di brano Spotify.")
        else:
            recommendations = get_recommendations(input_ids, df, knn_model)

            if recommendations is None or recommendations.empty:
                st.warning("Nessun suggerimento trovato. Riprova con altri brani.")
            else:
                st.subheader(":sparkles: Brani consigliati:")
                for _, row in recommendations.iterrows():
                    st.markdown(f"- **{row['Track']}** di *{row['Artist']}*")
