import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Vibely - AI Music Recommender", page_icon="🎵", layout="centered")

# ⬇️ Link ai file su Hugging Face (aggiornali se cambiano)
EMBEDDINGS_URL = "https://huggingface.co/MarcoBaiguini/vibely-data/resolve/main/df_with_embeddings.pkl"
MODEL_URL = "https://huggingface.co/MarcoBaiguini/vibely-data/resolve/main/knn_model.pkl"

# ⬇️ Scarica e carica i file .pkl con barra di caricamento
@st.cache_resource
def load_data():
    progress = st.progress(0, text="⏳ Caricamento dei dati...")

    progress.progress(10, "🔄 Download dei dati da Hugging Face...")

    df_response = requests.get(EMBEDDINGS_URL)
    knn_response = requests.get(MODEL_URL)

    progress.progress(50, "📦 Parsing dei dati...")

    df = pickle.loads(df_response.content)
    knn_model = pickle.loads(knn_response.content)

    progress.progress(100, "✅ Dati caricati con successo!")
    return df, knn_model

df, knn_model = load_data()

# ⬇️ Sistema di raccomandazione
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

# ⬇️ Interfaccia utente
st.title("🎧 Vibely")
st.markdown("Espandi i tuoi orizzonti musicali con l'intelligenza artificiale")

user_input = st.text_area("Incolla qui i link delle tue canzoni da Spotify (uno per riga)", height=150)

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
