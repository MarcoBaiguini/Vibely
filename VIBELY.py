import streamlit as st
import pandas as pd
import pickle
import requests
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Vibely - AI Music Recommender", page_icon="🎵", layout="centered")

# ✅ Funzione aggiornata per scaricare da Google Drive (compatibile con Streamlit Cloud e file di grandi dimensioni)
def download_file_from_google_drive(file_id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# ⬇️ Funzione per caricare i file pkl
@st.cache_resource
def load_optimized_data():
    df_id = "1tTYGQviOX0kIPG3DETFfOB5ci7vtBZnD"
    knn_id = "18xWJn2fegaiUbM_EqBwg5gGrOstxoO6M"
    df_path = "df_with_embeddings.pkl"
    knn_path = "knn_model.pkl"

    # 🔄 Elimina i file corrotti per forzare il nuovo download
    if os.path.exists(df_path):
        os.remove(df_path)
    if os.path.exists(knn_path):
        os.remove(knn_path)

    # ⬇️ Scarica i file da Google Drive
    download_file_from_google_drive(df_id, df_path)
    download_file_from_google_drive(knn_id, knn_path)

    with open(df_path, "rb") as f:
        df = pickle.load(f)
    with open(knn_path, "rb") as f:
        knn_model = pickle.load(f)

    return df, knn_model

# ⬇️ Caricamento dati
df, knn_model = load_optimized_data()

# ⬇️ Funzione di suggerimento
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

# ⬇️ UI
st.title("🎧 Vibely")
st.markdown("Espandi i tuoi orizzonti musicali con l'intelligenza artificiale")

user_input = st.text_area("Incolla qui i link delle tue canzoni da Spotify", height=150)

if st.button("🎵 Scopri nuova musica"):
    links = user_input.strip().split("\n")
    input_ids = [link.split("/")[-1].split("?")[0] for link in links if "spotify.com/track" in link]

    if not input_ids:
        st.warning("Inserisci almeno un link valido di brano Spotify.")
    else:
        recommendations = get_recommendations(input_ids, df, knn_model)

        if recommendations.empty:
            st.warning("Nessun suggerimento trovato. Riprova con altri brani.")
        else:
            st.subheader("✨ Brani consigliati:")
            for _, row in recommendations.iterrows():
                st.markdown(f"- **{row['Track']}** di *{row['Artist']}*")
