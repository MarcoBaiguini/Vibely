import streamlit as st
import pandas as pd
import pickle
import requests
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Vibely - AI Music Recommender", page_icon="🎵", layout="centered")

# ⬇️ Link Hugging Face ai file
DF_URL = "https://huggingface.co/MarcoBaiguini/vibely-data/resolve/main/df_with_embeddings.pkl"
KNN_URL = "https://huggingface.co/MarcoBaiguini/vibely-data/resolve/main/knn_model.pkl"
DF_FILE = "df_with_embeddings.pkl"
KNN_FILE = "knn_model.pkl"

# ⬇️ Funzione per scaricare file da Hugging Face
@st.cache_resource(show_spinner=False)
def load_files():
    progress = st.progress(0, text="Scaricamento file da Hugging Face...")
    
    # Scarica df
    if not os.path.exists(DF_FILE):
        with requests.get(DF_URL, stream=True) as r:
            r.raise_for_status()
            with open(DF_FILE, 'wb') as f:
                total = int(r.headers.get('content-length', 0))
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.progress(min(downloaded / total, 1.0))

    progress.progress(0.5, text="Scaricamento modello...")
    # Scarica knn
    if not os.path.exists(KNN_FILE):
        with requests.get(KNN_URL, stream=True) as r:
            r.raise_for_status()
            with open(KNN_FILE, 'wb') as f:
                total = int(r.headers.get('content-length', 0))
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.progress(0.5 + 0.5 * (downloaded / total))

    progress.empty()

    with open(DF_FILE, "rb") as f:
        df = pickle.load(f)
    with open(KNN_FILE, "rb") as f:
        knn_model = pickle.load(f)
    
    return df, knn_model

# ⬇️ Funzione di suggerimento
@st.cache_data(show_spinner=False)
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

# ⬇️ Interfaccia
st.title("🎧 Vibely")
st.markdown("Espandi i tuoi orizzonti musicali con l'intelligenza artificiale")

with st.spinner("Caricamento del modello in corso..."):
    df, knn_model = load_files()

user_input = st.text_area("Incolla qui i link delle tue canzoni da Spotify", height=150)

if st.button("🎵 Scopri nuova musica"):
    links = user_input.strip().split("\n")
    input_ids = [link.split("/")[-1].split("?")[0] for link in links if "spotify.com/track" in link]

    if not input_ids:
        st.warning("Inserisci almeno un link valido di brano Spotify.")
    else:
        with st.spinner("🎶 Analisi in corso..."):
            recommendations = get_recommendations(input_ids, df, knn_model)

        if recommendations is None or recommendations.empty:
            st.warning("Nessun suggerimento trovato. Riprova con altri brani.")
        else:
            st.subheader("✨ Brani consigliati:")
            for _, row in recommendations.iterrows():
                st.markdown(f"- **{row['Track']}** di *{row['Artist']}*")
